"""Centralized Cache Utility

Provides lightweight, extensible caching with:
- TTL support (time-to-live)
- Async-native operations
- Multiple backends (memory, Redis/Valkey)
- Namespace isolation
- Cache statistics for debugging
- Decorator-based and programmatic APIs
- Thundering herd prevention via atomic Lua scripts (Valkey backend)

Usage Examples:

    # 1. Simple TTL cache decorator (sync)
    @cached(ttl=300, namespace="youtube")
    def fetch_video_data(video_id: str) -> dict:
        return call_api(video_id)

    # 2. Async TTL cache decorator
    @async_cached(ttl=3600, namespace="search")
    async def search_videos(query: str) -> list[dict]:
        return await call_api(query)

    # 3. Programmatic cache access
    cache = Cache(namespace="my_service", default_ttl=600)
    cache.set("key", value)
    result = cache.get("key")

    # 4. With Valkey/Redis backend
    backend = ValkeyBackend.from_url("redis://localhost:6379/0")
    cache = Cache(namespace="distributed", backend=backend)
"""

from __future__ import annotations

import asyncio
import functools
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter
from time import time as time_now
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from stampede._json import JSONDecodeError, dumps_bytes, loads
from stampede._logging import get_logger
from stampede.hashing import make_cache_key as _make_cache_key
from stampede.lua import LuaScripts

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import redis as redis_lib


log = get_logger(__name__)

# Precompiled Lua scripts for atomic cache operations
_lua = LuaScripts(__file__, {"get_or_set": "_lua/cache_get_or_set.lua", "set_and_unlock": "_lua/cache_set_and_unlock.lua"})

T = TypeVar("T")
P = TypeVar("P")


# =============================================================================
# Cache Entry & Statistics
# =============================================================================


@dataclass(slots=True)
class CacheEntry(Generic[T]):
    """A cache entry with optional TTL. Uses __slots__ for faster attribute access."""

    value: T
    created_at: float = field(default_factory=time_now)
    ttl: float | None = None  # None = no expiration
    hits: int = 0
    _expires_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Pre-compute expiry timestamp to avoid repeated addition."""
        self._expires_at = (self.created_at + self.ttl) if self.ttl else None

    def is_expired(self) -> bool:
        """Check if entry has expired (O(1) comparison vs repeated arithmetic)."""
        return (exp := self._expires_at) is not None and time_now() > exp

    def touch(self) -> None:
        """Record a cache hit."""
        self.hits += 1


@dataclass(slots=True)
class CacheStats:
    """Cache statistics for debugging and monitoring. Uses __slots__ for memory efficiency."""

    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate as a percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary for logging."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "hit_rate": f"{self.hit_rate:.1f}%",
        }


# =============================================================================
# Cache Backends (Pluggable Storage)
# =============================================================================


class CacheBackend(ABC):
    """Abstract cache backend interface. Implement for new storage backends."""

    @abstractmethod
    def get(self, key: str) -> CacheEntry | None: ...

    @abstractmethod
    def set(self, key: str, entry: CacheEntry) -> None: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def clear(self) -> int: ...

    @abstractmethod
    def keys(self) -> list[str]: ...

    def size(self) -> int:
        return len(self.keys())


class MemoryBackend(CacheBackend):
    """Thread-safe in-memory cache backend with optional max size.

    Uses OrderedDict for O(1) LRU tracking.
    """

    def __init__(self, maxsize: int | None = None):
        from collections import OrderedDict

        self._data: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = RLock()
        self._maxsize = maxsize

    def get(self, key: str) -> CacheEntry | None:
        with self._lock:
            try:
                self._data.move_to_end(key)
                return self._data[key]
            except KeyError:
                return None

    def set(self, key: str, entry: CacheEntry) -> None:
        with self._lock:
            self._data.pop(key, None)
            if self._maxsize and len(self._data) >= self._maxsize:
                self._data.popitem(last=False)
            self._data[key] = entry

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def clear(self) -> int:
        with self._lock:
            count = len(self._data)
            self._data.clear()
            return count

    def keys(self) -> list[str]:
        with self._lock:
            return [*self._data]


class ValkeyBackend(CacheBackend):
    """Valkey/Redis-backed cache backend for distributed deployments.

    Features:
        - Drop-in Redis replacement (protocol compatible)
        - Distributed caching across instances
        - Automatic serialization with orjson (fast)
        - Fail-open semantics (falls back gracefully)
        - Connection pooling for performance
        - Thundering herd prevention via atomic Lua scripts

    Usage:
        backend = ValkeyBackend.from_url("redis://localhost:6379/0")
        cache = Cache(namespace="my_service", backend=backend)
    """

    def __init__(self, client: redis_lib.Redis, prefix: str = "cache:", default_ttl: int = 3600):
        """Initialize Valkey backend with client, prefix for namespacing, and default TTL."""
        self._client, self._prefix, self._default_ttl = client, prefix, default_ttl

    @property
    def client(self) -> redis_lib.Redis:
        """Access the underlying Redis/Valkey client for advanced operations."""
        return self._client

    @classmethod
    def from_url(
        cls,
        url: str,
        prefix: str = "cache:",
        default_ttl: int = 3600,
        max_connections: int = 10,
        socket_timeout: float = 1.0,
        socket_connect_timeout: float = 1.0,
    ) -> ValkeyBackend:
        """Create ValkeyBackend from connection URL.

        Supports both valkey:// and redis:// schemes.

        Args:
            url: Connection URL (redis://host:port/db or valkey://host:port/db)
            prefix: Key prefix for namespacing
            default_ttl: Default TTL in seconds when not specified per-entry
            max_connections: Maximum connections in pool
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
        """
        import ssl

        import redis

        # Valkey uses Redis protocol — normalize URL scheme
        normalized_url = url.replace("valkey://", "redis://").replace("valkeys://", "rediss://")
        ssl_opts = {"ssl_cert_reqs": ssl.CERT_NONE} if normalized_url.startswith("rediss://") else {}

        client = redis.from_url(
            normalized_url,
            max_connections=max_connections,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
            decode_responses=False,
            **ssl_opts,
        )
        client.ping()
        log.info("Valkey cache backend connected", extra={"prefix": prefix, "max_connections": max_connections})
        return cls(client, prefix=prefix, default_ttl=default_ttl)

    def _make_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def _serialize(self, entry: CacheEntry) -> bytes:
        return dumps_bytes({"value": entry.value, "created_at": entry.created_at, "ttl": entry.ttl, "hits": entry.hits})

    def _deserialize(self, data: bytes) -> CacheEntry | None:
        try:
            obj = loads(data)
            return CacheEntry(value=obj["value"], created_at=obj["created_at"], ttl=obj["ttl"], hits=obj.get("hits", 0))
        except (JSONDecodeError, KeyError, TypeError):
            return None

    def get(self, key: str) -> CacheEntry | None:
        try:
            if (data := cast("bytes | None", self._client.get(self._make_key(key)))) is None:
                return None
            return self._deserialize(data)
        except Exception as e:
            log.warning("Valkey cache get failed", extra={"key": key, "error": str(e)})
            return None

    def set(self, key: str, entry: CacheEntry) -> None:
        try:
            self._client.setex(
                self._make_key(key),
                int(entry.ttl) if entry.ttl else self._default_ttl,
                self._serialize(entry),
            )
        except Exception as e:
            log.warning("Valkey cache set failed", extra={"key": key, "error": str(e)})

    def delete(self, key: str) -> bool:
        try:
            return cast("int", self._client.delete(self._make_key(key))) > 0
        except Exception as e:
            log.warning("Valkey cache delete failed", extra={"key": key, "error": str(e)})
            return False

    def _scan_all(self, pattern: str) -> list[bytes]:
        """Iterate SCAN cursor to collect all matching keys."""
        client, cursor, result = self._client, 0, []
        while True:
            cursor, keys = client.scan(cursor=cursor, match=pattern, count=100)  # type: ignore[misc]
            result.extend(keys)
            if not cursor:
                return result

    def clear(self) -> int:
        try:
            keys = self._scan_all(f"{self._prefix}*")
            return cast("int", self._client.delete(*keys)) if keys else 0
        except Exception as e:
            log.warning("Valkey cache clear failed", extra={"error": str(e)})
            return 0

    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching a glob pattern via SCAN."""
        try:
            keys = self._scan_all(self._make_key(pattern))
            return cast("int", self._client.delete(*keys)) if keys else 0
        except Exception as e:
            log.warning("Valkey cache delete_pattern failed", extra={"pattern": pattern, "error": str(e)})
            return 0

    def keys(self) -> list[str]:
        try:
            return [k.decode().removeprefix(self._prefix) for k in self._scan_all(f"{self._prefix}*")]
        except Exception as e:
            log.warning("Valkey cache keys failed", extra={"error": str(e)})
            return []

    def atomic_get_or_lock(self, key: str, lock_ttl: int = 30) -> tuple[tuple[str, CacheEntry | None], str | None, str | None]:
        """Atomic cache check + lock acquire for thundering herd prevention.

        Returns:
            (("cached", entry), None, None) — Cache hit, return entry
            (("locked", None), owner_id, lock_key) — We acquired lock, compute value
            (("wait", None), None, None) — Another instance computing, poll cache
        """
        cache_key, lock_key = self._make_key(key), f"{self._prefix}lock:{key}"
        owner_id = uuid.uuid4().hex[:8]

        try:
            result = _lua.call(self._client, "get_or_set", keys=[cache_key, lock_key], args=[owner_id, lock_ttl, "__COMPUTING__"])
            if result == b"__COMPUTE__":
                return ("locked", None), owner_id, lock_key
            if result == b"__WAIT__":
                return ("wait", None), None, None
            if entry := self._deserialize(result):
                return ("cached", entry), None, None
            return ("locked", None), owner_id, lock_key
        except Exception as e:
            log.debug(f"Lua get_or_set failed: {e}")
            return ("locked", None), None, None

    def atomic_set_and_unlock(self, key: str, entry: CacheEntry, owner_id: str, lock_key: str) -> bool:
        """Atomic cache set + lock release (only if we own the lock)."""
        try:
            ttl = int(entry.ttl) if entry.ttl else self._default_ttl
            result = _lua.call(
                self._client,
                "set_and_unlock",
                keys=[self._make_key(key), lock_key],
                args=[self._serialize(entry), ttl, owner_id],
            )
            return result == 1
        except Exception as e:
            log.debug(f"Lua set_and_unlock failed: {e}")
            return False


# =============================================================================
# Core Cache Classes
# =============================================================================


class Cache:
    """General-purpose cache with TTL support, namespace isolation,
    pluggable backends, and statistics tracking."""

    def __init__(
        self,
        namespace: str = "default",
        default_ttl: float | None = None,
        backend: CacheBackend | None = None,
        maxsize: int | None = 1000,
    ):
        self.namespace = namespace
        self.default_ttl = default_ttl
        self.backend = backend or MemoryBackend(maxsize=maxsize)
        self.stats = CacheStats()
        self._ns_prefix = f"{namespace}:"

    def get(self, key: str, default: T | None = None) -> T | None:
        """Get a value from cache, returning default if not found or expired."""
        full_key = self._ns_prefix + key
        backend, stats = self.backend, self.stats

        if (entry := backend.get(full_key)) is None:
            stats.misses += 1
            return default

        if entry.is_expired():
            backend.delete(full_key)
            stats.misses += 1
            stats.expirations += 1
            return default

        entry.touch()
        stats.hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value in cache with optional TTL."""
        self.backend.set(
            self._ns_prefix + key,
            CacheEntry(value=value, ttl=ttl if ttl is not None else self.default_ttl),
        )
        self.stats.sets += 1

    def delete(self, key: str) -> bool:
        return self.backend.delete(self._ns_prefix + key)

    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching a glob pattern."""
        full = self._ns_prefix + pattern
        backend = self.backend
        if isinstance(backend, ValkeyBackend):
            return backend.delete_pattern(full)
        from fnmatch import fnmatch

        return sum(backend.delete(k) for k in backend.keys() if fnmatch(k, full))  # noqa: SIM118

    def clear(self) -> int:
        prefix, backend = self._ns_prefix, self.backend
        count = sum(backend.delete(k) for k in backend.keys() if k.startswith(prefix))  # noqa: SIM118
        self.stats.evictions += count
        return count

    def get_or_set(self, key: str, factory: Callable[[], T], ttl: float | None = None) -> T:
        """Get a value, computing via factory if not cached.

        Uses atomic Lua script on ValkeyBackend to prevent thundering herd.
        """
        if (value := self.get(key)) is not None:
            return value

        # For Valkey, use atomic get-or-lock to prevent thundering herd
        if isinstance(self.backend, ValkeyBackend):
            result, owner_id, lock_key = self.backend.atomic_get_or_lock(self._ns_prefix + key)
            status, entry = result
            if status == "cached" and entry:
                entry.touch()
                self.stats.hits += 1
                return entry.value
            if status == "wait":
                import asyncio as _aio
                import time

                try:
                    loop = _aio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    log.warning(
                        "Sync cache get_or_set called from async context — "
                        "use AsyncCache.aget_or_set() to avoid blocking the event loop",
                        extra={"cache_key": key, "namespace": self.namespace},
                    )

                for _ in range(100):  # Max 10s wait
                    time.sleep(0.1)
                    if (value := self.get(key)) is not None:
                        return value
            # We have the lock — compute
            value = factory()
            entry = CacheEntry(value=value, ttl=ttl if ttl is not None else self.default_ttl)
            if owner_id and lock_key:
                self.backend.atomic_set_and_unlock(self._ns_prefix + key, entry, owner_id, lock_key)
            else:
                self.set(key, value, ttl=ttl)
            self.stats.sets += 1
            return value

        # Fallback for other backends
        value = factory()
        self.set(key, value, ttl=ttl)
        return value

    def has(self, key: str) -> bool:
        return self.get(key) is not None

    @property
    def size(self) -> int:
        prefix = self._ns_prefix
        return sum(1 for k in self.backend.keys() if k.startswith(prefix))  # noqa: SIM118


class TTLCache(Cache):
    """Convenience subclass with required TTL."""

    def __init__(
        self,
        namespace: str = "default",
        ttl: float = 300,
        backend: CacheBackend | None = None,
        maxsize: int | None = 1000,
    ):
        super().__init__(namespace=namespace, default_ttl=ttl, backend=backend, maxsize=maxsize)


class AsyncCache(Cache):
    """Async-native cache with awaitable methods.

    Wraps sync Cache with proper locking for concurrent access.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._async_lock = asyncio.Lock()

    async def aget(self, key: str, default: T | None = None) -> T | None:
        async with self._async_lock:
            return self.get(key, default)

    async def aset(self, key: str, value: Any, ttl: float | None = None) -> None:
        async with self._async_lock:
            self.set(key, value, ttl)

    async def adelete(self, key: str) -> bool:
        async with self._async_lock:
            return self.delete(key)

    async def aclear(self) -> int:
        async with self._async_lock:
            return self.clear()

    async def aget_or_set(
        self,
        key: str,
        factory: Callable[[], T] | Callable[[], Awaitable[T]],
        ttl: float | None = None,
    ) -> T:
        """Async get or set with support for sync/async factories.

        Uses atomic Lua script on ValkeyBackend to prevent thundering herd.
        """
        # For Valkey, use atomic get-or-lock
        if isinstance(self.backend, ValkeyBackend):
            result, owner_id, lock_key = self.backend.atomic_get_or_lock(self._ns_prefix + key)
            status, entry = result
            if status == "cached" and entry:
                entry.touch()
                self.stats.hits += 1
                return entry.value
            if status == "wait":
                for _ in range(100):
                    await asyncio.sleep(0.1)
                    async with self._async_lock:
                        if (value := self.get(key)) is not None:
                            return value
            # We have the lock — compute
            result = factory()
            value = await result if asyncio.iscoroutine(result) else cast("T", result)
            entry = CacheEntry(value=value, ttl=ttl if ttl is not None else self.default_ttl)
            if owner_id and lock_key:
                self.backend.atomic_set_and_unlock(self._ns_prefix + key, entry, owner_id, lock_key)
            else:
                async with self._async_lock:
                    self.set(key, value, ttl=ttl)
            self.stats.sets += 1
            return value

        # Fallback for other backends
        async with self._async_lock:
            if (value := self.get(key)) is not None:
                return value

        result = factory()
        value = await result if asyncio.iscoroutine(result) else cast("T", result)

        async with self._async_lock:
            if (existing := self.get(key)) is not None:
                return existing
            self.set(key, value, ttl=ttl)
        return value


# =============================================================================
# Decorators
# =============================================================================


def cached(
    ttl: float | None = None,
    namespace: str = "default",
    key_fn: Callable[..., str] | None = None,
    cache: Cache | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for caching synchronous function results."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        _cache = cache or Cache(namespace=namespace, default_ttl=ttl)
        func_prefix = f"{func.__name__}:"

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = key_fn(*args, **kwargs) if key_fn else func_prefix + _make_cache_key(*args, **kwargs)
            if (result := _cache.get(key)) is not None:
                return result
            result = func(*args, **kwargs)
            _cache.set(key, result, ttl=ttl)
            return result

        wrapper.cache = _cache  # type: ignore[attr-defined]
        wrapper.cache_clear = _cache.clear  # type: ignore[attr-defined]
        wrapper.cache_stats = lambda: _cache.stats.to_dict()  # type: ignore[attr-defined]
        return wrapper

    return decorator


def async_cached(
    ttl: float | None = None,
    namespace: str = "default",
    key_fn: Callable[..., str] | None = None,
    cache: AsyncCache | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for caching async function results."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _cache = cache or AsyncCache(namespace=namespace, default_ttl=ttl)
        func_prefix = f"{func.__name__}:"

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = key_fn(*args, **kwargs) if key_fn else func_prefix + _make_cache_key(*args, **kwargs)
            if (result := await _cache.aget(key)) is not None:
                return result
            result = await func(*args, **kwargs)
            await _cache.aset(key, result, ttl=ttl)
            return result

        wrapper.cache = _cache  # type: ignore[attr-defined]
        wrapper.cache_clear = lambda: asyncio.create_task(_cache.aclear())  # type: ignore[attr-defined]
        wrapper.cache_stats = lambda: _cache.stats.to_dict()  # type: ignore[attr-defined]
        return wrapper

    return decorator


# =============================================================================
# Convenience Constants
# =============================================================================

TTL_MINUTE = 60
TTL_5_MINUTES = 300
TTL_HOUR = 3600
TTL_DAY = 86400
TTL_WEEK = 604800


__all__ = [
    "TTL_5_MINUTES",
    "TTL_DAY",
    "TTL_HOUR",
    "TTL_MINUTE",
    "TTL_WEEK",
    "AsyncCache",
    "Cache",
    "CacheBackend",
    "CacheEntry",
    "CacheStats",
    "MemoryBackend",
    "TTLCache",
    "ValkeyBackend",
    "async_cached",
    "cached",
]
