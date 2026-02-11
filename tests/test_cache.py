"""Tests for stampede.cache — backends, Cache, AsyncCache, decorators."""

from __future__ import annotations

import asyncio
import time

import pytest

from stampede.cache import (
    TTL_DAY,
    TTL_HOUR,
    TTL_MINUTE,
    AsyncCache,
    Cache,
    CacheEntry,
    CacheStats,
    MemoryBackend,
    TTLCache,
    async_cached,
    cached,
)


# =============================================================================
# CacheEntry
# =============================================================================


class TestCacheEntry:
    def test_not_expired_no_ttl(self):
        entry = CacheEntry(value="test")
        assert not entry.is_expired()

    def test_not_expired_long_ttl(self):
        entry = CacheEntry(value="test", ttl=3600)
        assert not entry.is_expired()

    def test_expired(self):
        entry = CacheEntry(value="test", created_at=time.time() - 100, ttl=1)
        assert entry.is_expired()

    def test_touch_increments_hits(self):
        entry = CacheEntry(value="test")
        assert entry.hits == 0
        entry.touch()
        entry.touch()
        assert entry.hits == 2

    def test_uses_slots(self):
        entry = CacheEntry(value="test")
        assert hasattr(entry, "__slots__")

    def test_expires_at_precomputed(self):
        now = time.time()
        entry = CacheEntry(value="test", created_at=now, ttl=60)
        assert entry._expires_at == pytest.approx(now + 60, abs=0.01)

    def test_expires_at_none_without_ttl(self):
        entry = CacheEntry(value="test")
        assert entry._expires_at is None


# =============================================================================
# CacheStats
# =============================================================================


class TestCacheStats:
    def test_defaults(self):
        stats = CacheStats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.sets == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate(self):
        stats = CacheStats(hits=75, misses=25)
        assert stats.hit_rate == 75.0

    def test_to_dict(self):
        stats = CacheStats(hits=10, misses=5, sets=15)
        d = stats.to_dict()
        assert d["hits"] == 10
        assert d["misses"] == 5
        assert d["sets"] == 15
        assert "%" in d["hit_rate"]


# =============================================================================
# MemoryBackend
# =============================================================================


class TestMemoryBackend:
    def test_get_miss(self):
        backend = MemoryBackend()
        assert backend.get("nonexistent") is None

    def test_set_and_get(self):
        backend = MemoryBackend()
        entry = CacheEntry(value="hello")
        backend.set("key1", entry)
        assert backend.get("key1") is entry

    def test_delete(self):
        backend = MemoryBackend()
        backend.set("key", CacheEntry(value="v"))
        assert backend.delete("key") is True
        assert backend.delete("key") is False
        assert backend.get("key") is None

    def test_clear(self):
        backend = MemoryBackend()
        backend.set("k1", CacheEntry(value="v1"))
        backend.set("k2", CacheEntry(value="v2"))
        cleared = backend.clear()
        assert cleared == 2
        assert backend.size() == 0

    def test_keys(self):
        backend = MemoryBackend()
        backend.set("a", CacheEntry(value=1))
        backend.set("b", CacheEntry(value=2))
        keys = backend.keys()
        assert sorted(keys) == ["a", "b"]

    def test_lru_eviction(self):
        backend = MemoryBackend(maxsize=3)
        backend.set("a", CacheEntry(value=1))
        backend.set("b", CacheEntry(value=2))
        backend.set("c", CacheEntry(value=3))
        backend.set("d", CacheEntry(value=4))  # Should evict 'a'

        assert backend.get("a") is None
        assert backend.get("d") is not None
        assert backend.size() == 3

    def test_lru_access_reorder(self):
        """Accessing a key moves it to end, preventing eviction."""
        backend = MemoryBackend(maxsize=3)
        backend.set("a", CacheEntry(value=1))
        backend.set("b", CacheEntry(value=2))
        backend.set("c", CacheEntry(value=3))

        backend.get("a")  # Access 'a' to move it to end

        backend.set("d", CacheEntry(value=4))  # Should evict 'b' (oldest)

        assert backend.get("a") is not None  # 'a' still here
        assert backend.get("b") is None  # 'b' evicted

    def test_overwrite_existing(self):
        backend = MemoryBackend()
        backend.set("key", CacheEntry(value="v1"))
        backend.set("key", CacheEntry(value="v2"))
        assert backend.get("key").value == "v2"
        assert backend.size() == 1


# =============================================================================
# Cache
# =============================================================================


class TestCache:
    def test_get_set(self):
        cache = Cache(namespace="test")
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_get_miss(self):
        cache = Cache(namespace="test")
        assert cache.get("missing") is None

    def test_get_default(self):
        cache = Cache(namespace="test")
        assert cache.get("missing", default="fallback") == "fallback"

    def test_ttl_expiration(self):
        cache = Cache(namespace="test")
        cache.set("key", "value", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("key") is None
        assert cache.stats.expirations == 1

    def test_default_ttl(self):
        cache = Cache(namespace="test", default_ttl=3600)
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_delete(self):
        cache = Cache(namespace="test")
        cache.set("key", "value")
        assert cache.delete("key") is True
        assert cache.get("key") is None

    def test_clear(self):
        cache = Cache(namespace="test")
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cleared = cache.clear()
        assert cleared == 2
        assert cache.size == 0

    def test_has(self):
        cache = Cache(namespace="test")
        assert cache.has("key") is False
        cache.set("key", "value")
        assert cache.has("key") is True

    def test_namespace_isolation(self):
        cache_a = Cache(namespace="a")
        cache_b = Cache(namespace="b")
        cache_a.set("key", "value_a")
        cache_b.set("key", "value_b")
        assert cache_a.get("key") == "value_a"
        assert cache_b.get("key") == "value_b"

    def test_stats_tracking(self):
        cache = Cache(namespace="test")
        cache.set("k", "v")
        cache.get("k")  # hit
        cache.get("missing")  # miss

        assert cache.stats.sets == 1
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1

    def test_get_or_set(self):
        cache = Cache(namespace="test")
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return "computed"

        r1 = cache.get_or_set("key", factory, ttl=60)
        r2 = cache.get_or_set("key", factory, ttl=60)

        assert r1 == r2 == "computed"
        assert call_count == 1

    def test_delete_pattern(self):
        cache = Cache(namespace="test")
        cache.set("user:1", "alice")
        cache.set("user:2", "bob")
        cache.set("post:1", "hello")

        deleted = cache.delete_pattern("user:*")
        assert deleted == 2
        assert cache.get("post:1") == "hello"


# =============================================================================
# TTLCache
# =============================================================================


class TestTTLCache:
    def test_requires_ttl(self):
        cache = TTLCache(namespace="test", ttl=60)
        cache.set("key", "value")
        assert cache.get("key") == "value"

    def test_default_ttl_applied(self):
        cache = TTLCache(namespace="test", ttl=3600)
        assert cache.default_ttl == 3600


# =============================================================================
# AsyncCache
# =============================================================================


class TestAsyncCache:
    @pytest.mark.asyncio
    async def test_aget_aset(self):
        cache = AsyncCache(namespace="async_test")
        await cache.aset("key", "value")
        assert await cache.aget("key") == "value"

    @pytest.mark.asyncio
    async def test_aget_miss(self):
        cache = AsyncCache(namespace="async_test")
        assert await cache.aget("missing") is None

    @pytest.mark.asyncio
    async def test_adelete(self):
        cache = AsyncCache(namespace="async_test")
        await cache.aset("key", "value")
        assert await cache.adelete("key") is True
        assert await cache.aget("key") is None

    @pytest.mark.asyncio
    async def test_aclear(self):
        cache = AsyncCache(namespace="async_test")
        await cache.aset("k1", "v1")
        await cache.aset("k2", "v2")
        cleared = await cache.aclear()
        assert cleared == 2

    @pytest.mark.asyncio
    async def test_aget_or_set_sync_factory(self):
        cache = AsyncCache(namespace="async_test")
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return "computed"

        r1 = await cache.aget_or_set("key", factory, ttl=60)
        r2 = await cache.aget_or_set("key", factory, ttl=60)

        assert r1 == r2 == "computed"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_aget_or_set_async_factory(self):
        cache = AsyncCache(namespace="async_test")
        call_count = 0

        async def factory():
            nonlocal call_count
            call_count += 1
            return "async_computed"

        r1 = await cache.aget_or_set("key", factory, ttl=60)
        r2 = await cache.aget_or_set("key", factory, ttl=60)

        assert r1 == r2 == "async_computed"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Multiple concurrent aget/aset don't corrupt state."""
        cache = AsyncCache(namespace="concurrent_test")

        async def write(i: int):
            await cache.aset(f"key{i}", f"value{i}")

        async def read(i: int):
            return await cache.aget(f"key{i}")

        await asyncio.gather(*[write(i) for i in range(20)])
        results = await asyncio.gather(*[read(i) for i in range(20)])

        assert all(results[i] == f"value{i}" for i in range(20))


# =============================================================================
# @cached decorator
# =============================================================================


class TestCachedDecorator:
    def test_basic_usage(self):
        call_count = 0

        @cached(ttl=60)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert compute(5) == 10
        assert call_count == 1

    def test_different_args(self):
        call_count = 0

        @cached(ttl=60)
        def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert compute(5) == 10
        assert compute(6) == 12
        assert call_count == 2

    def test_custom_key_fn(self):
        call_count = 0

        @cached(ttl=60, key_fn=lambda x, y: str(x))
        def compute(x: int, y: int) -> int:
            nonlocal call_count
            call_count += 1
            return x + y

        assert compute(1, 2) == 3
        assert compute(1, 99) == 3  # Same key (y ignored)
        assert call_count == 1

    def test_attached_utilities(self):
        @cached(ttl=60)
        def my_func(x: int) -> int:
            return x

        assert hasattr(my_func, "cache")
        assert hasattr(my_func, "cache_clear")
        assert hasattr(my_func, "cache_stats")


# =============================================================================
# @async_cached decorator
# =============================================================================


class TestAsyncCachedDecorator:
    @pytest.mark.asyncio
    async def test_basic_usage(self):
        call_count = 0

        @async_cached(ttl=60)
        async def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert await compute(5) == 10
        assert await compute(5) == 10
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_args(self):
        call_count = 0

        @async_cached(ttl=60)
        async def compute(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        assert await compute(5) == 10
        assert await compute(6) == 12
        assert call_count == 2


# =============================================================================
# TTL Constants
# =============================================================================


class TestTTLConstants:
    def test_values(self):
        assert TTL_MINUTE == 60
        assert TTL_HOUR == 3600
        assert TTL_DAY == 86400
