"""Tests for stampede.distributed — distributed coalescing without Redis.

Tests the in-memory fallback path (redis_client=None). Redis integration
tests require a running Redis instance and are in test_integration.py.
"""

from __future__ import annotations

import asyncio

import pytest

from stampede.distributed import (
    DistributedCoalesceStats,
    DistributedRequestCoalescer,
    distributed_coalesce,
    get_distributed_coalescer,
)


# =============================================================================
# DistributedCoalesceStats
# =============================================================================


class TestDistributedCoalesceStats:
    def test_defaults(self):
        stats = DistributedCoalesceStats()
        assert stats.requests == 0
        assert stats.savings_rate == 0.0

    def test_savings_rate(self):
        stats = DistributedCoalesceStats(
            requests=100, cache_hits=30, local_coalesce_hits=20, lock_waited=10
        )
        assert stats.savings_rate == 60.0

    def test_to_dict(self):
        stats = DistributedCoalesceStats(requests=10, cache_hits=3)
        d = stats.to_dict()
        assert d["requests"] == 10
        assert "savings_rate" in d


# =============================================================================
# DistributedRequestCoalescer (in-memory fallback, no Redis)
# =============================================================================


class TestDistributedCoalescerNoRedis:
    """Tests with redis_client=None — falls back to local in-flight tracking."""

    @pytest.mark.asyncio
    async def test_single_execution(self):
        """Multiple concurrent requests share one execution (local coalescing)."""
        call_count = 0
        coalescer = DistributedRequestCoalescer(ttl=10, namespace="test", redis_client=None)

        async def expensive():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return "result"

        results = await asyncio.gather(*[coalescer.execute("key", expensive) for _ in range(5)])

        assert all(r == "result" for r in results)
        assert call_count == 1
        assert coalescer.stats.local_coalesce_hits == 4
        assert coalescer.stats.lock_acquired == 1

    @pytest.mark.asyncio
    async def test_different_keys(self):
        coalescer = DistributedRequestCoalescer(ttl=10, redis_client=None)
        call_count = 0

        async def compute():
            nonlocal call_count
            call_count += 1
            return f"r{call_count}"

        r1 = await coalescer.execute("k1", compute)
        r2 = await coalescer.execute("k2", compute)
        assert r1 != r2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        coalescer = DistributedRequestCoalescer(ttl=10, redis_client=None)

        async def failing():
            await asyncio.sleep(0.02)
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError, match="fail"):
            await asyncio.gather(*[coalescer.execute("key", failing) for _ in range(3)])

        assert coalescer.stats.errors == 1

    @pytest.mark.asyncio
    async def test_invalidate_no_redis(self):
        coalescer = DistributedRequestCoalescer(redis_client=None)
        assert coalescer.invalidate("key") is False

    @pytest.mark.asyncio
    async def test_clear_no_redis(self):
        coalescer = DistributedRequestCoalescer(redis_client=None)
        assert coalescer.clear() == 0

    @pytest.mark.asyncio
    async def test_instance_id_unique(self):
        c1 = DistributedRequestCoalescer(redis_client=None)
        c2 = DistributedRequestCoalescer(redis_client=None)
        assert c1._instance_id != c2._instance_id

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        coalescer = DistributedRequestCoalescer(ttl=10, redis_client=None)

        async def compute():
            await asyncio.sleep(0.02)
            return "val"

        await asyncio.gather(*[coalescer.execute("key", compute) for _ in range(4)])

        assert coalescer.stats.requests == 4
        assert coalescer.stats.lock_acquired == 1
        assert coalescer.stats.local_coalesce_hits == 3
        assert coalescer.stats.savings_rate == 75.0


# =============================================================================
# @distributed_coalesce decorator
# =============================================================================


class TestDistributedCoalesceDecorator:
    @pytest.mark.asyncio
    async def test_basic_usage(self):
        call_count = 0

        @distributed_coalesce(ttl=10, redis_client=None)
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

        @distributed_coalesce(ttl=10, key_fn=lambda x: "fixed", redis_client=None)
        async def my_func(x: str) -> str:
            nonlocal call_count
            call_count += 1
            return x

        await my_func("a")
        await my_func("b")  # Same key

        # Second call returns cached-by-coalescing result since both share key "fixed"
        # But they run sequentially, not concurrently, so both execute
        # (coalescing only merges concurrent calls)
        assert call_count == 2  # Sequential calls each execute

    @pytest.mark.asyncio
    async def test_attached_utilities(self):
        @distributed_coalesce(ttl=10, redis_client=None)
        async def my_func() -> str:
            return "val"

        assert hasattr(my_func, "coalescer")
        assert hasattr(my_func, "invalidate")
        assert hasattr(my_func, "clear_cache")
        assert hasattr(my_func, "stats")

    @pytest.mark.asyncio
    async def test_preserves_name(self):
        @distributed_coalesce(ttl=10, redis_client=None)
        async def my_special_function() -> str:
            return "val"

        assert my_special_function.__name__ == "my_special_function"


# =============================================================================
# Registry
# =============================================================================


class TestDistributedRegistry:
    def test_singleton(self):
        c1 = get_distributed_coalescer("dist_test_ns_a", redis_client=None)
        c2 = get_distributed_coalescer("dist_test_ns_a")
        assert c1 is c2

    def test_different_namespaces(self):
        c1 = get_distributed_coalescer("dist_ns_x", redis_client=None)
        c2 = get_distributed_coalescer("dist_ns_y", redis_client=None)
        assert c1 is not c2
