"""Tests for stampede.coalesce — request coalescing and in-flight dedup."""

from __future__ import annotations

import asyncio

import pytest

from stampede.coalesce import (
    CachedResult,
    CoalesceStats,
    InFlightRequest,
    RequestCoalescer,
    clear_all_coalescers,
    coalesce,
    coalesce_by_content,
    get_all_coalescer_stats,
    get_coalescer,
)


# =============================================================================
# CoalesceStats
# =============================================================================


class TestCoalesceStats:
    def test_defaults(self):
        stats = CoalesceStats()
        assert stats.requests == 0
        assert stats.cache_hits == 0
        assert stats.coalesce_hits == 0
        assert stats.executions == 0
        assert stats.errors == 0

    def test_cache_hit_rate_zero_requests(self):
        assert CoalesceStats().cache_hit_rate == 0.0

    def test_cache_hit_rate(self):
        stats = CoalesceStats(requests=100, cache_hits=25)
        assert stats.cache_hit_rate == 25.0

    def test_coalesce_rate(self):
        stats = CoalesceStats(requests=100, coalesce_hits=40)
        assert stats.coalesce_rate == 40.0

    def test_savings_rate(self):
        stats = CoalesceStats(requests=100, cache_hits=25, coalesce_hits=40)
        assert stats.savings_rate == 65.0

    def test_to_dict(self):
        stats = CoalesceStats(requests=10, cache_hits=3, coalesce_hits=2, executions=5, errors=1)
        d = stats.to_dict()
        assert d["requests"] == 10
        assert d["cache_hits"] == 3
        assert "savings_rate" in d
        assert "%" in d["savings_rate"]


# =============================================================================
# CachedResult
# =============================================================================


class TestCachedResult:
    def test_not_expired(self):
        result = CachedResult(value="test", ttl=60)
        assert not result.is_expired()

    def test_expired(self):
        result = CachedResult(value="test", ttl=0)
        # TTL of 0 means immediately expired
        assert result.is_expired()

    def test_uses_slots(self):
        result = CachedResult(value="test")
        assert hasattr(result, "__slots__")


# =============================================================================
# InFlightRequest
# =============================================================================


class TestInFlightRequest:
    def test_initial_waiters(self):
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        req = InFlightRequest(future=future)
        assert req.waiters == 1
        loop.close()

    def test_add_waiter(self):
        loop = asyncio.new_event_loop()
        future = loop.create_future()
        req = InFlightRequest(future=future)
        req.add_waiter()
        req.add_waiter()
        assert req.waiters == 3
        loop.close()


# =============================================================================
# RequestCoalescer
# =============================================================================


class TestRequestCoalescer:
    @pytest.mark.asyncio
    async def test_single_execution(self):
        """Multiple concurrent requests for same key share one execution."""
        call_count = 0
        coalescer = RequestCoalescer(ttl=10, namespace="test")

        async def expensive():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return "result"

        results = await asyncio.gather(*[coalescer.execute("key", expensive) for _ in range(5)])

        assert all(r == "result" for r in results)
        assert call_count == 1
        assert coalescer.stats.executions == 1
        assert coalescer.stats.coalesce_hits == 4
        assert coalescer.stats.requests == 5

    @pytest.mark.asyncio
    async def test_different_keys_execute_separately(self):
        """Different keys trigger separate executions."""
        call_count = 0
        coalescer = RequestCoalescer(ttl=10)

        async def expensive():
            nonlocal call_count
            call_count += 1
            return f"result-{call_count}"

        r1 = await coalescer.execute("key1", expensive)
        r2 = await coalescer.execute("key2", expensive)

        assert r1 == "result-1"
        assert r2 == "result-2"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Second request for same key served from cache."""
        call_count = 0
        coalescer = RequestCoalescer(ttl=10)

        async def expensive():
            nonlocal call_count
            call_count += 1
            return "cached_value"

        r1 = await coalescer.execute("key", expensive)
        r2 = await coalescer.execute("key", expensive)

        assert r1 == r2 == "cached_value"
        assert call_count == 1
        assert coalescer.stats.cache_hits == 1

    @pytest.mark.asyncio
    async def test_no_cache_when_ttl_none(self):
        """No caching when ttl=None."""
        call_count = 0
        coalescer = RequestCoalescer(ttl=None)

        async def expensive():
            nonlocal call_count
            call_count += 1
            return "value"

        await coalescer.execute("key", expensive)
        await coalescer.execute("key", expensive)

        assert call_count == 2  # Both executed since no caching

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Errors propagate to all waiters."""
        coalescer = RequestCoalescer(ttl=10)

        async def failing():
            await asyncio.sleep(0.05)
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await asyncio.gather(*[coalescer.execute("key", failing) for _ in range(3)])

        assert coalescer.stats.errors == 1

    @pytest.mark.asyncio
    async def test_ttl_override(self):
        """Per-call TTL overrides instance TTL."""
        coalescer = RequestCoalescer(ttl=0)  # 0 TTL = expires immediately

        async def compute():
            return "value"

        await coalescer.execute("key", compute, ttl=60)  # Override with 60s
        result = await coalescer.execute("key", compute, ttl=60)

        # Should be a cache hit because we overrode TTL to 60s
        assert result == "value"
        assert coalescer.stats.cache_hits == 1

    @pytest.mark.asyncio
    async def test_invalidate(self):
        coalescer = RequestCoalescer(ttl=60)

        async def compute():
            return "value"

        await coalescer.execute("key", compute)
        assert coalescer.invalidate("key") is True
        assert coalescer.invalidate("nonexistent") is False

    @pytest.mark.asyncio
    async def test_clear(self):
        coalescer = RequestCoalescer(ttl=60)

        async def compute():
            return "value"

        await coalescer.execute("k1", compute)
        await coalescer.execute("k2", compute)
        assert coalescer.cache_size == 2

        cleared = coalescer.clear()
        assert cleared == 2
        assert coalescer.cache_size == 0

    @pytest.mark.asyncio
    async def test_maxsize_eviction(self):
        """Cache evicts oldest entries when at maxsize."""
        coalescer = RequestCoalescer(ttl=60, maxsize=3)
        counter = 0

        async def compute():
            nonlocal counter
            counter += 1
            return f"v{counter}"

        await coalescer.execute("k1", compute)
        await coalescer.execute("k2", compute)
        await coalescer.execute("k3", compute)
        assert coalescer.cache_size == 3

        await coalescer.execute("k4", compute)
        assert coalescer.cache_size == 3  # Still 3, oldest evicted

    @pytest.mark.asyncio
    async def test_in_flight_count(self):
        coalescer = RequestCoalescer(ttl=10)
        assert coalescer.in_flight_count == 0

    @pytest.mark.asyncio
    async def test_savings_rate(self):
        coalescer = RequestCoalescer(ttl=60)

        async def compute():
            await asyncio.sleep(0.02)
            return "val"

        # 5 concurrent -> 1 execution + 4 coalesced
        await asyncio.gather(*[coalescer.execute("key", compute) for _ in range(5)])
        # 1 more -> cache hit
        await coalescer.execute("key", compute)

        assert coalescer.stats.requests == 6
        assert coalescer.stats.savings_rate == pytest.approx(83.3, abs=0.1)


# =============================================================================
# @coalesce decorator
# =============================================================================


class TestCoalesceDecorator:
    @pytest.mark.asyncio
    async def test_basic_usage(self):
        call_count = 0

        @coalesce(ttl=10)
        async def my_func(x: str) -> str:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.02)
            return f"result:{x}"

        results = await asyncio.gather(*[my_func("hello") for _ in range(3)])

        assert all(r == "result:hello" for r in results)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_custom_key_fn(self):
        call_count = 0

        @coalesce(ttl=10, key_fn=lambda x, y: f"{x}")
        async def my_func(x: str, y: int) -> str:
            nonlocal call_count
            call_count += 1
            return f"{x}:{y}"

        r1 = await my_func("a", 1)
        r2 = await my_func("a", 2)  # Same key as r1 (y ignored)

        assert r1 == "a:1"
        assert r2 == "a:1"  # Cached
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_attached_utilities(self):
        @coalesce(ttl=10)
        async def my_func(x: str) -> str:
            return x

        assert hasattr(my_func, "coalescer")
        assert hasattr(my_func, "invalidate")
        assert hasattr(my_func, "clear_cache")
        assert hasattr(my_func, "stats")
        assert callable(my_func.stats)

    @pytest.mark.asyncio
    async def test_stats_dict(self):
        @coalesce(ttl=10)
        async def my_func() -> str:
            return "val"

        await my_func()
        stats = my_func.stats()
        assert stats["requests"] == 1
        assert stats["executions"] == 1

    @pytest.mark.asyncio
    async def test_method_coalescing(self):
        """Coalesce works on bound methods (skips 'self')."""
        call_count = 0

        class MyService:
            @coalesce(ttl=10)
            async def generate(self, topic: str) -> str:
                nonlocal call_count
                call_count += 1
                return f"result:{topic}"

        svc = MyService()
        r1 = await svc.generate("python")
        r2 = await svc.generate("python")

        assert r1 == r2 == "result:python"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_preserves_function_name(self):
        @coalesce(ttl=10)
        async def my_special_function() -> str:
            return "val"

        assert my_special_function.__name__ == "my_special_function"


# =============================================================================
# @coalesce_by_content
# =============================================================================


class TestCoalesceByContent:
    @pytest.mark.asyncio
    async def test_content_fields(self):
        call_count = 0

        @coalesce_by_content(ttl=10, content_fields=["topic"])
        async def generate(topic: str = "", user_id: str = "") -> str:
            nonlocal call_count
            call_count += 1
            return f"result:{topic}"

        r1 = await generate(topic="python", user_id="user1")
        r2 = await generate(topic="python", user_id="user2")  # Different user, same topic

        assert r1 == r2 == "result:python"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_content_different_execution(self):
        call_count = 0

        @coalesce_by_content(ttl=10, content_fields=["topic"])
        async def generate(topic: str = "") -> str:
            nonlocal call_count
            call_count += 1
            return f"result:{topic}"

        await generate(topic="python")
        await generate(topic="javascript")

        assert call_count == 2


# =============================================================================
# Global Registry
# =============================================================================


class TestRegistry:
    def test_get_coalescer_singleton(self):
        c1 = get_coalescer("test_ns_a", ttl=10)
        c2 = get_coalescer("test_ns_a")
        assert c1 is c2

    def test_get_coalescer_different_namespaces(self):
        c1 = get_coalescer("ns_x")
        c2 = get_coalescer("ns_y")
        assert c1 is not c2

    def test_get_all_stats(self):
        get_coalescer("stat_test_ns")
        stats = get_all_coalescer_stats()
        assert "stat_test_ns" in stats

    def test_clear_all(self):
        result = clear_all_coalescers()
        assert isinstance(result, dict)
