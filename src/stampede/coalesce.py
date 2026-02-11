"""Request Coalescing for Async Operations

Combines multiple concurrent requests for the same resource into a single execution,
reducing duplicate work (especially expensive LLM calls).

Key Features:
- In-flight request tracking: Multiple waiters share one execution
- TTL-based result caching: Subsequent requests get cached results
- Configurable key functions: Hash requests by topic, prompt, etc.
- Metrics integration: Track coalesce hits, cache hits, and execution counts
- Error propagation: Errors are properly propagated to all waiting callers

How It Works:

    Request 1 ---+
    Request 2 ---+--> Single LLM Call --> Result shared by all
    Request 3 ---+

Without coalescing, 3 identical requests = 3 LLM calls = 3x cost
With coalescing, 3 identical requests = 1 LLM call = 1x cost

Usage Examples:

    # 1. Decorator with TTL cache
    @coalesce(ttl=60, key_fn=lambda req: hash(req.topic))
    async def generate_course(self, req: CourseRequest) -> Course:
        return await expensive_llm_call(req)

    # 2. Decorator with auto-key (based on all args)
    @coalesce(ttl=300, namespace="skills")
    async def get_skills_outline(self, career: str) -> SkillsOutline:
        return await llm_generate_skills(career)

    # 3. Programmatic API
    coalescer = RequestCoalescer(ttl=60, namespace="llm")
    result = await coalescer.execute(
        key="career:software_engineer",
        func=lambda: generate_course_impl(request)
    )

    # 4. No TTL cache (coalesce only in-flight requests)
    @coalesce(ttl=None)  # Only coalesce concurrent requests, no caching
    async def generate_response(prompt: str) -> str:
        ...
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass, field
from threading import RLock
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from stampede._logging import get_logger, span
from stampede.hashing import content_fingerprint
from stampede.hashing import make_cache_key as _make_cache_key

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

log = get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Statistics & Metrics
# =============================================================================


@dataclass
class CoalesceStats:
    """Statistics for monitoring coalescing effectiveness."""

    requests: int = 0  # Total requests received
    cache_hits: int = 0  # Requests served from TTL cache
    coalesce_hits: int = 0  # Requests that joined an in-flight execution
    executions: int = 0  # Actual function executions
    errors: int = 0  # Executions that failed

    @property
    def cache_hit_rate(self) -> float:
        """Percentage of requests served from cache."""
        return (self.cache_hits / self.requests * 100) if self.requests > 0 else 0.0

    @property
    def coalesce_rate(self) -> float:
        """Percentage of requests that were coalesced (avoided duplicate execution)."""
        return (self.coalesce_hits / self.requests * 100) if self.requests > 0 else 0.0

    @property
    def savings_rate(self) -> float:
        """Percentage of executions saved by coalescing + caching."""
        avoided = self.cache_hits + self.coalesce_hits
        return (avoided / self.requests * 100) if self.requests > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/metrics."""
        return {
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "coalesce_hits": self.coalesce_hits,
            "executions": self.executions,
            "errors": self.errors,
            "cache_hit_rate": f"{self.cache_hit_rate:.1f}%",
            "coalesce_rate": f"{self.coalesce_rate:.1f}%",
            "savings_rate": f"{self.savings_rate:.1f}%",
        }


# =============================================================================
# Cache Entry for Coalesced Results
# =============================================================================


@dataclass(slots=True)
class CachedResult(Generic[T]):
    """A cached result with TTL support."""

    value: T
    created_at: float = field(default_factory=time.monotonic)
    ttl: float = 0.0

    def is_expired(self) -> bool:
        """Check if this result has expired."""
        return (time.monotonic() - self.created_at) > self.ttl


# =============================================================================
# In-Flight Request Tracking
# =============================================================================


@dataclass(slots=True)
class InFlightRequest(Generic[T]):
    """Tracks an in-flight request with its waiters."""

    future: asyncio.Future[T]
    waiters: int = 1

    def add_waiter(self) -> None:
        """Register another waiter for this request."""
        self.waiters += 1


# =============================================================================
# Request Coalescer
# =============================================================================


class RequestCoalescer(Generic[T]):
    """Coalesces concurrent requests for the same resource into a single execution.

    Thread-safe and async-safe for use in concurrent web servers.

    Example:
        coalescer = RequestCoalescer[Course](ttl=60, namespace="courses")

        # Multiple concurrent calls with same key share one execution
        result = await coalescer.execute(
            key="career:software_engineer",
            func=lambda: generate_course_impl(request)
        )
    """

    def __init__(self, ttl: float | None = 60, namespace: str = "coalesce", maxsize: int = 1000):
        """Initialize a request coalescer.

        Args:
            ttl: Time-to-live for cached results in seconds. None = no caching.
            namespace: Namespace for logging/metrics identification.
            maxsize: Maximum number of cached results to keep in memory.
        """
        self.ttl = ttl
        self.namespace = namespace
        self.maxsize = maxsize

        self._cache: dict[str, CachedResult[T]] = {}
        self._in_flight: dict[str, InFlightRequest[T]] = {}
        self._lock = asyncio.Lock()
        self._sync_lock = RLock()  # For thread-safe cache eviction
        self.stats = CoalesceStats()

    async def execute(self, key: str, func: Callable[[], Awaitable[T]], ttl: float | None = None) -> T:
        """Execute a function with coalescing and caching.

        If a result exists in cache and hasn't expired, return it.
        If another request with the same key is in-flight, wait for it.
        Otherwise, execute the function and share the result.

        Args:
            key: Unique key identifying this request
            func: Async callable to execute if needed
            ttl: Override TTL for this specific call (None uses instance TTL)

        Returns:
            The result of the function (possibly cached or shared)

        Raises:
            Exception: Any exception from the function is propagated to all waiters
        """
        effective_ttl = self.ttl if ttl is None else ttl
        self.stats.requests += 1

        async with self._lock:
            # Check cache first (only when TTL caching enabled)
            if effective_ttl and (cached := self._cache.get(key)) and not cached.is_expired():
                self.stats.cache_hits += 1
                log.debug("Coalesce cache hit", extra={"namespace": self.namespace, "key": key[:50]})
                return cached.value

            # Check for in-flight request
            if in_flight := self._in_flight.get(key):
                in_flight.add_waiter()
                self.stats.coalesce_hits += 1
                log.debug(
                    "Coalescing with in-flight request",
                    extra={"namespace": self.namespace, "key": key[:50], "waiters": in_flight.waiters},
                )
                future = in_flight.future
            else:
                # Create new in-flight request
                future = asyncio.get_running_loop().create_future()
                self._in_flight[key] = InFlightRequest(future=future)
                future = None  # Signal that we're the executor

        # If future is None, we're the executor
        if future is None:
            return await self._execute_and_cache(key, func, effective_ttl)

        # Otherwise, wait for the in-flight request
        return await self._in_flight[key].future

    async def _execute_and_cache(self, key: str, func: Callable[[], Awaitable[T]], ttl: float | None) -> T:
        """Execute the function, cache the result, and notify waiters."""
        self.stats.executions += 1
        in_flight = self._in_flight[key]

        try:
            with span("coalesce.execute", namespace=self.namespace, key=key[:50], waiters=in_flight.waiters) as s:
                result = await func()

                # Cache the result if TTL is set
                if ttl is not None:
                    self._cache_result(key, result, ttl)

                s.info("Coalesced execution complete", waiters=in_flight.waiters, cached=ttl is not None)

                # Notify all waiters
                if not in_flight.future.done():
                    in_flight.future.set_result(result)

                return result

        except Exception as e:
            self.stats.errors += 1
            log.warning(
                "Coalesced execution failed",
                extra={"namespace": self.namespace, "key": key[:50], "waiters": in_flight.waiters, "error": str(e)},
            )
            # Propagate exception to all waiters
            if not in_flight.future.done():
                in_flight.future.set_exception(e)
            raise

        finally:
            # Clean up in-flight tracking
            async with self._lock:
                self._in_flight.pop(key, None)

    def _cache_result(self, key: str, result: T, ttl: float) -> None:
        """Cache a result with FIFO eviction if needed."""
        with self._sync_lock:
            # Evict oldest entries if at capacity (batch eviction for efficiency)
            if (overflow := len(self._cache) - self.maxsize + 1) > 0:
                for k in list(self._cache.keys())[:overflow]:
                    del self._cache[k]
            self._cache[key] = CachedResult(value=result, ttl=ttl)

    def invalidate(self, key: str) -> bool:
        """Invalidate a cached entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was found and removed
        """
        with self._sync_lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all cached entries.

        Returns:
            Number of entries cleared
        """
        with self._sync_lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    @property
    def cache_size(self) -> int:
        """Number of entries currently in cache."""
        return len(self._cache)

    @property
    def in_flight_count(self) -> int:
        """Number of requests currently in-flight."""
        return len(self._in_flight)


# =============================================================================
# Decorator API
# =============================================================================


def coalesce(
    ttl: float | None = 60,
    namespace: str = "coalesce",
    key_fn: Callable[..., str] | None = None,
    maxsize: int = 1000,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for request coalescing on async functions.

    Combines multiple concurrent requests for the same resource into a single
    execution, then optionally caches the result for subsequent requests.

    Args:
        ttl: Time-to-live for cached results in seconds. None = no caching,
             only coalesce in-flight requests.
        namespace: Namespace for logging/metrics (defaults to function name).
        key_fn: Custom function to generate cache key from arguments.
                If None, uses stable hash of all arguments.
        maxsize: Maximum cached entries (for LRU eviction).

    Returns:
        Decorated async function with coalescing behavior.

    Examples:
        @coalesce(ttl=60)
        async def get_recommendation(user_id: str, topic: str) -> str:
            return await llm.generate(...)

        @coalesce(ttl=300, key_fn=lambda req: f"career:{req.career_title}")
        async def generate_course(req: CourseRequest) -> Course:
            ...

        @coalesce(ttl=None)  # Coalesce only, no caching
        async def stream_response(prompt: str) -> AsyncIterator[str]:
            ...
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        effective_namespace = namespace if namespace != "coalesce" else func.__name__
        coalescer: RequestCoalescer[T] = RequestCoalescer(ttl=ttl, namespace=effective_namespace, maxsize=maxsize)

        # Precompute key prefix
        _key_prefix = f"{func.__name__}:"

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Generate cache key
            if key_fn is not None:
                key = key_fn(*args, **kwargs)
            else:
                # Auto-generate key from function name and arguments
                # Skip 'self' if present (first arg of bound methods)
                key_args = args[1:] if args and not isinstance(args[0], str | int | float | bool | type(None)) else args
                key = _key_prefix + _make_cache_key(*key_args, **kwargs)

            return await coalescer.execute(key, lambda: func(*args, **kwargs))

        # Attach utilities for testing and management
        wrapper.coalescer = coalescer  # type: ignore[attr-defined]
        wrapper.invalidate = coalescer.invalidate  # type: ignore[attr-defined]
        wrapper.clear_cache = coalescer.clear  # type: ignore[attr-defined]
        wrapper.stats = lambda: coalescer.stats.to_dict()  # type: ignore[attr-defined]

        return wrapper

    return decorator


def coalesce_by_content(
    ttl: float | None = 60,
    namespace: str = "content",
    content_fields: list[str] | None = None,
    maxsize: int = 1000,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for coalescing by content fingerprint.

    Generates cache keys based on content fingerprinting, useful for requests
    where semantic similarity matters more than exact equality.

    Args:
        ttl: Time-to-live for cached results in seconds.
        namespace: Namespace for logging/metrics.
        content_fields: List of kwarg names to include in fingerprint.
                       If None, uses all kwargs.
        maxsize: Maximum cached entries.

    Returns:
        Decorated async function.

    Example:
        @coalesce_by_content(ttl=300, content_fields=["topic", "context"])
        async def generate_content(topic: str, context: str, user_id: str) -> str:
            # user_id is excluded from cache key, so different users
            # requesting the same topic/context share results
            ...
    """

    _prefix = f"{namespace}:"

    def key_fn(*_args: Any, **kwargs: Any) -> str:
        content = {k: v for k in content_fields if (v := kwargs.get(k)) is not None} if content_fields else kwargs
        return _prefix + content_fingerprint(content, 24)

    return coalesce(ttl=ttl, namespace=namespace, key_fn=key_fn, maxsize=maxsize)


# =============================================================================
# Global Coalescer Registry
# =============================================================================


_coalescers: dict[str, RequestCoalescer] = {}
_coalescers_lock = RLock()


def get_coalescer(namespace: str, ttl: float | None = 60, maxsize: int = 1000) -> RequestCoalescer:
    """Get or create a shared coalescer instance for a namespace.

    Coalescers are singletons per namespace — subsequent calls with
    the same namespace return the same instance.

    Args:
        namespace: Unique identifier for the coalescer
        ttl: Default TTL for cached results (only used on creation)
        maxsize: Maximum cached entries (only used on creation)

    Returns:
        RequestCoalescer instance for the namespace
    """
    with _coalescers_lock:
        if namespace not in _coalescers:
            _coalescers[namespace] = RequestCoalescer(ttl=ttl, namespace=namespace, maxsize=maxsize)
        return _coalescers[namespace]


def get_all_coalescer_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all registered coalescers."""
    with _coalescers_lock:
        return {
            namespace: {
                **coalescer.stats.to_dict(),
                "cache_size": coalescer.cache_size,
                "in_flight": coalescer.in_flight_count,
            }
            for namespace, coalescer in _coalescers.items()
        }


def clear_all_coalescers() -> dict[str, int]:
    """Clear all registered coalescers' caches.

    Returns:
        Dict mapping namespace to count of cleared entries.
    """
    with _coalescers_lock:
        return {namespace: coalescer.clear() for namespace, coalescer in _coalescers.items()}


__all__ = [
    "CachedResult",
    "CoalesceStats",
    "InFlightRequest",
    "RequestCoalescer",
    "clear_all_coalescers",
    "coalesce",
    "coalesce_by_content",
    "get_all_coalescer_stats",
    "get_coalescer",
]
