"""Distributed request coalescing with Redis-backed cache and locks.

For multi-instance deployments (Cloud Run, Kubernetes), this provides
coordinated request coalescing across all instances, ensuring identical
requests to different instances share the same result.

Usage:
    # Create distributed coalescer (falls back to in-memory if Redis unavailable)
    coalescer = DistributedRequestCoalescer(ttl=60, namespace="llm")
    result = await coalescer.execute("key", lambda: expensive_call())

    # Or use decorator
    @distributed_coalesce(ttl=60, namespace="courses")
    async def generate_course(req: CourseRequest) -> Course:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import time
import uuid
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from stampede._json import dumps_bytes, loads
from stampede._logging import get_logger, span
from stampede.hashing import make_cache_key as _make_cache_key
from stampede.lua import LuaScripts

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import redis

log = get_logger(__name__)

# Precompiled Lua scripts (EVALSHA for minimal network overhead)
_lua = LuaScripts(__file__, {"release_lock": "_lua/release_lock.lua", "check_cache_and_lock": "_lua/check_cache_and_lock.lua"})

T = TypeVar("T")


@dataclass(slots=True)
class DistributedCoalesceStats:
    """Statistics for monitoring distributed coalescing effectiveness."""

    requests: int = 0
    cache_hits: int = 0  # Served from distributed cache
    local_coalesce_hits: int = 0  # Coalesced within same instance
    lock_acquired: int = 0  # Times this instance executed (won the lock)
    lock_waited: int = 0  # Times waited for another instance
    errors: int = 0
    redis_failures: int = 0  # Redis operations that failed (fell back to local)

    @property
    def savings_rate(self) -> float:
        """Percentage of requests that avoided execution."""
        return (
            (self.cache_hits + self.local_coalesce_hits + self.lock_waited) / self.requests * 100
        ) if self.requests else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "local_coalesce_hits": self.local_coalesce_hits,
            "lock_acquired": self.lock_acquired,
            "lock_waited": self.lock_waited,
            "errors": self.errors,
            "redis_failures": self.redis_failures,
            "savings_rate": f"{self.savings_rate:.1f}%",
        }


class DistributedRequestCoalescer(Generic[T]):
    """Redis-backed request coalescer for distributed deployments.

    Combines multiple concurrent requests for the same resource into a single
    execution across all instances, using Redis for distributed locking and caching.

    How It Works:
        1. Check Redis cache for existing result
        2. If not cached, try to acquire distributed lock
        3. If lock acquired, execute function and cache result
        4. If lock not acquired, poll cache until result available
        5. Local in-flight tracking prevents duplicate executions within same instance

    Features:
        - Distributed locking with Redis (SET NX EX pattern)
        - Automatic serialization with orjson
        - Graceful fallback to in-memory coalescing if Redis unavailable
        - Lock timeout prevents deadlocks
    """

    def __init__(
        self,
        ttl: float | None = 60,
        namespace: str = "coalesce",
        lock_timeout: float = 300.0,
        poll_interval: float = 0.1,
        max_poll_time: float = 600.0,
        redis_client: redis.Redis | None = None,
    ):
        """Initialize distributed request coalescer.

        Args:
            ttl: Time-to-live for cached results in seconds. None = no caching.
            namespace: Namespace for logging/metrics and Redis key prefixes.
            lock_timeout: How long to hold the distributed lock (seconds).
            poll_interval: How often to check cache while waiting (seconds).
            max_poll_time: Maximum time to wait for another instance (seconds).
            redis_client: Redis client for distributed state (None = in-memory fallback).
        """
        self.ttl = ttl
        self.namespace = namespace
        self.lock_timeout = lock_timeout
        self.poll_interval = poll_interval
        self.max_poll_time = max_poll_time
        self._redis = redis_client

        # Precomputed key prefixes
        self._cache_prefix = f"coalesce:{namespace}:cache:"
        self._lock_prefix = f"coalesce:{namespace}:lock:"

        # Local in-flight tracking
        self._in_flight: dict[str, asyncio.Future[T]] = {}
        self._lock = asyncio.Lock()
        self._sync_lock = RLock()

        # Instance ID for lock ownership
        self._instance_id = uuid.uuid4().hex[:8]

        self.stats = DistributedCoalesceStats()

    def _cache_key(self, key: str) -> str:
        return self._cache_prefix + key

    def _lock_key(self, key: str) -> str:
        return self._lock_prefix + key

    def _get_cached(self, key: str) -> T | None:
        if (redis := self._redis) is None:
            return None
        try:
            if data := cast("bytes | None", redis.get(self._cache_key(key))):
                return loads(data)
            return None
        except Exception as e:
            log.debug(f"Redis cache get failed for '{key}'", extra={"error": str(e)})
            self.stats.redis_failures += 1
            return None

    def _set_cached(self, key: str, value: T, ttl: float | None) -> bool:
        if (redis := self._redis) is None or ttl is None:
            return False
        try:
            redis.setex(self._cache_key(key), int(ttl), dumps_bytes(value))
            return True
        except Exception as e:
            log.debug(f"Redis cache set failed for '{key}'", extra={"error": str(e)})
            self.stats.redis_failures += 1
            return False

    def _check_cache_and_lock(self, key: str) -> tuple[str, T | None]:
        """Atomic cache check + lock acquire in single round-trip (Lua script).

        Returns:
            ("cached", value) — Cache hit
            ("locked", None) — We acquired lock, execute function
            ("wait", None) — Another instance has lock, poll for result
        """
        if (client := self._redis) is None:
            return ("locked", None)

        try:
            result = _lua.call(
                client,
                "check_cache_and_lock",
                keys=[self._cache_key(key), self._lock_key(key)],
                args=[self._instance_id, int(self.lock_timeout)],
            )
            if result == b"__LOCKED__":
                return ("locked", None)
            if result == b"__WAIT__":
                return ("wait", None)
            return ("cached", loads(result))
        except Exception as e:
            log.debug(f"Redis check_cache_and_lock failed for '{key}'", extra={"error": str(e)})
            self.stats.redis_failures += 1
            return ("locked", None)

    def _acquire_lock(self, key: str) -> bool:
        if (client := self._redis) is None:
            return True
        try:
            return bool(client.set(self._lock_key(key), self._instance_id, nx=True, ex=int(self.lock_timeout)))
        except Exception as e:
            log.debug(f"Redis lock acquire failed for '{key}'", extra={"error": str(e)})
            self.stats.redis_failures += 1
            return True

    def _release_lock(self, key: str) -> None:
        if (client := self._redis) is None:
            return
        try:
            _lua.call(client, "release_lock", keys=[self._lock_key(key)], args=[self._instance_id])
        except Exception as e:
            log.debug(f"Redis lock release failed for '{key}'", extra={"error": str(e)})

    async def execute(self, key: str, func: Callable[[], Awaitable[T]], ttl: float | None = None) -> T:
        """Execute a function with distributed coalescing and caching.

        Args:
            key: Unique key identifying this request
            func: Async callable to execute if needed
            ttl: Override TTL for this specific call

        Returns:
            The result of the function (possibly cached or shared)

        Raises:
            Exception: Any exception from the function
            TimeoutError: If waiting for another instance exceeds max_poll_time
        """
        effective_ttl = self.ttl if ttl is None else ttl
        self.stats.requests += 1

        # 1. Check for local in-flight request first
        async with self._lock:
            if key in self._in_flight:
                self.stats.local_coalesce_hits += 1
                log.debug("Coalescing with local in-flight request", extra={"namespace": self.namespace, "key": key[:50]})
                return await self._in_flight[key]

            future: asyncio.Future[T] = asyncio.get_running_loop().create_future()
            self._in_flight[key] = future

        try:
            # 2. Atomic cache check + lock acquire
            status, cached = self._check_cache_and_lock(key)

            if status == "cached" and cached is not None:
                self.stats.cache_hits += 1
                log.debug("Distributed cache hit", extra={"namespace": self.namespace, "key": key[:50]})
                if not future.done():
                    future.set_result(cached)
                return cached

            if status == "locked":
                self.stats.lock_acquired += 1
                return await self._execute_and_cache(key, func, effective_ttl, future)

            # status == "wait"
            self.stats.lock_waited += 1
            return await self._wait_for_result(key, func, future)

        finally:
            async with self._lock:
                self._in_flight.pop(key, None)

    async def _execute_and_cache(
        self, key: str, func: Callable[[], Awaitable[T]], ttl: float | None, future: asyncio.Future[T]
    ) -> T:
        try:
            with span("distributed_coalesce.execute", namespace=self.namespace, key=key[:50], instance=self._instance_id):
                result = await func()

                if ttl is not None:
                    self._set_cached(key, result, ttl)

                if not future.done():
                    future.set_result(result)

                return result

        except Exception as e:
            self.stats.errors += 1
            log.warning(
                "Distributed coalesce execution failed",
                extra={"namespace": self.namespace, "key": key[:50], "error": str(e)},
            )
            if not future.done():
                future.set_exception(e)
            raise

        finally:
            self._release_lock(key)

    async def _wait_for_result(
        self, key: str, func: Callable[[], Awaitable[T]], future: asyncio.Future[T]
    ) -> T:
        deadline = time.perf_counter() + self.max_poll_time

        while time.perf_counter() < deadline:
            if (cached := self._get_cached(key)) is not None:
                if not future.done():
                    future.set_result(cached)
                return cached
            await asyncio.sleep(self.poll_interval)

        # Timeout — execute ourselves
        log.warning(
            "Distributed coalesce wait timeout, executing locally",
            extra={"namespace": self.namespace, "key": key[:50], "waited_seconds": self.max_poll_time},
        )
        return await self._execute_and_cache(key, func, self.ttl, future)

    def invalidate(self, key: str) -> bool:
        if (redis := self._redis) is None:
            return False
        try:
            return cast("int", redis.delete(self._cache_key(key))) > 0
        except Exception:
            return False

    def clear(self) -> int:
        if (redis := self._redis) is None:
            return 0
        try:
            pattern = self._cache_prefix + "*"
            count, cursor = 0, 0
            while True:
                cursor, keys = redis.scan(cursor=cursor, match=pattern, count=100)  # type: ignore[assignment]
                if keys:
                    count += cast("int", redis.delete(*keys))
                if not cursor:
                    break
            return count
        except Exception:
            return 0


# =============================================================================
# Decorator API
# =============================================================================


def distributed_coalesce(
    ttl: float | None = 60,
    namespace: str = "coalesce",
    key_fn: Callable[..., str] | None = None,
    lock_timeout: float = 300.0,
    redis_client: redis.Redis | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for distributed request coalescing on async functions.

    Uses Redis for cross-instance coalescing when available, falls back
    to in-memory coalescing within the same instance.

    Args:
        ttl: Time-to-live for cached results in seconds. None = no caching.
        namespace: Namespace for logging/metrics (defaults to function name).
        key_fn: Custom function to generate cache key from arguments.
        lock_timeout: How long to hold distributed lock.
        redis_client: Redis client instance. If None, distributed coalescing is disabled.

    Examples:
        @distributed_coalesce(ttl=60, redis_client=my_redis)
        async def get_recommendation(user_id: str, topic: str) -> str:
            return await llm.generate(...)
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        effective_namespace = namespace if namespace != "coalesce" else func.__name__

        coalescer: DistributedRequestCoalescer[T] = DistributedRequestCoalescer(
            ttl=ttl,
            namespace=effective_namespace,
            lock_timeout=lock_timeout,
            redis_client=redis_client,
        )

        _key_prefix = f"{func.__name__}:"

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                key_args = args[1:] if args and not isinstance(args[0], str | int | float | bool | type(None)) else args
                key = _key_prefix + _make_cache_key(*key_args, **kwargs)

            return await coalescer.execute(key, lambda: func(*args, **kwargs))

        wrapper.coalescer = coalescer  # type: ignore[attr-defined]
        wrapper.invalidate = coalescer.invalidate  # type: ignore[attr-defined]
        wrapper.clear_cache = coalescer.clear  # type: ignore[attr-defined]
        wrapper.stats = lambda: coalescer.stats.to_dict()  # type: ignore[attr-defined]

        return wrapper

    return decorator


# =============================================================================
# Registry
# =============================================================================

_distributed_coalescers: dict[str, DistributedRequestCoalescer] = {}
_coalescers_lock = RLock()


def get_distributed_coalescer(
    namespace: str,
    ttl: float | None = 60,
    lock_timeout: float = 300.0,
    redis_client: redis.Redis | None = None,
) -> DistributedRequestCoalescer:
    """Get or create a shared distributed coalescer for a namespace.

    Args:
        namespace: Unique identifier for the coalescer
        ttl: Default TTL for cached results
        lock_timeout: How long to hold distributed locks
        redis_client: Redis client for distributed state
    """
    with _coalescers_lock:
        if namespace not in _distributed_coalescers:
            if redis_client:
                log.info(f"Creating distributed coalescer '{namespace}' with Redis backend")
            else:
                log.debug(f"Creating coalescer '{namespace}' with in-memory backend (Redis unavailable)")

            _distributed_coalescers[namespace] = DistributedRequestCoalescer(
                ttl=ttl, namespace=namespace, lock_timeout=lock_timeout, redis_client=redis_client
            )
        return _distributed_coalescers[namespace]


def get_all_distributed_coalescer_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all distributed coalescers."""
    with _coalescers_lock:
        return {ns: c.stats.to_dict() for ns, c in _distributed_coalescers.items()}


def clear_all_distributed_coalescers() -> dict[str, int]:
    """Clear all distributed coalescer caches."""
    with _coalescers_lock:
        return {ns: c.clear() for ns, c in _distributed_coalescers.items()}


__all__ = [
    "DistributedCoalesceStats",
    "DistributedRequestCoalescer",
    "clear_all_distributed_coalescers",
    "distributed_coalesce",
    "get_all_distributed_coalescer_stats",
    "get_distributed_coalescer",
]
