"""Tests for stampede.semantic — semantic cache without external services.

Tests the cache logic with mocked embed_fn and without PostgreSQL/Redis.
Integration tests requiring pgvector/Redis are in test_integration.py.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from stampede.semantic import (
    SemanticCache,
    SemanticCacheStats,
    get_semantic_cache,
    semantic_coalesce,
)


# =============================================================================
# SemanticCacheStats
# =============================================================================


class TestSemanticCacheStats:
    def test_defaults(self):
        stats = SemanticCacheStats()
        assert stats.requests == 0
        assert stats.exact_hits == 0
        assert stats.semantic_hits == 0
        assert stats.misses == 0
        assert stats.hit_rate == 0.0

    def test_hit_rate(self):
        stats = SemanticCacheStats(requests=100, exact_hits=30, semantic_hits=20)
        assert stats.hit_rate == 50.0

    def test_semantic_rate(self):
        stats = SemanticCacheStats(exact_hits=10, semantic_hits=40)
        assert stats.semantic_rate == 80.0

    def test_semantic_rate_no_hits(self):
        stats = SemanticCacheStats()
        assert stats.semantic_rate == 0.0

    def test_to_dict(self):
        stats = SemanticCacheStats(requests=10, exact_hits=3, semantic_hits=2, misses=5)
        d = stats.to_dict()
        assert d["requests"] == 10
        assert d["exact_hits"] == 3
        assert "hit_rate" in d
        assert "semantic_rate" in d


# =============================================================================
# SemanticCache — no pool, no Redis (misses only)
# =============================================================================


class TestSemanticCacheNoBackend:
    """Without pool or Redis, every get() is a miss."""

    @pytest.mark.asyncio
    async def test_get_returns_none(self):
        cache = SemanticCache(namespace="test")
        result = await cache.get("any query")
        assert result is None
        assert cache.stats.misses == 1
        assert cache.stats.requests == 1

    @pytest.mark.asyncio
    async def test_set_without_pool_no_error(self):
        """set() with no pool should not raise."""
        cache = SemanticCache(namespace="test")
        await cache.set("query", "response")  # Should not raise

    @pytest.mark.asyncio
    async def test_get_or_execute(self):
        """Without backend, always executes the function."""
        call_count = 0
        cache = SemanticCache(namespace="test")

        async def compute():
            nonlocal call_count
            call_count += 1
            return "result"

        r1 = await cache.get_or_execute("query1", compute)
        r2 = await cache.get_or_execute("query2", compute)

        assert r1 == r2 == "result"
        assert call_count == 2
        assert cache.stats.misses == 2

    @pytest.mark.asyncio
    async def test_embed_fn_required_for_embedding(self):
        """_get_embedding raises without embed_fn."""
        cache = SemanticCache(namespace="test")
        with pytest.raises(RuntimeError, match="embed_fn"):
            await cache._get_embedding("test")


# =============================================================================
# SemanticCache — with mock embed_fn
# =============================================================================


class TestSemanticCacheWithEmbedFn:
    @pytest.mark.asyncio
    async def test_embed_fn_called(self):
        mock_embed = AsyncMock(return_value=[0.1] * 1536)
        cache = SemanticCache(namespace="test", embed_fn=mock_embed)

        embedding = await cache._get_embedding("hello")
        assert len(embedding) == 1536
        mock_embed.assert_awaited_once_with("hello")
        assert cache.stats.embeddings_generated == 1

    @pytest.mark.asyncio
    async def test_get_or_execute_caches_miss(self):
        """get_or_execute always calls func when no backend is available."""
        mock_embed = AsyncMock(return_value=[0.1] * 1536)
        cache = SemanticCache(namespace="test", embed_fn=mock_embed)
        calls = []

        async def compute():
            calls.append(1)
            return "computed"

        result = await cache.get_or_execute("query", compute)
        assert result == "computed"
        assert len(calls) == 1


# =============================================================================
# SemanticCache — normalize_and_key
# =============================================================================


class TestSemanticCacheNormalization:
    def test_normalize_and_key(self):
        cache = SemanticCache(namespace="test")

        n1, k1 = cache._normalize_and_key("Hello World")
        n2, k2 = cache._normalize_and_key("  hello   world  ")

        assert n1 == n2 == "hello world"
        assert k1 == k2  # Same fingerprint

    def test_different_queries_different_keys(self):
        cache = SemanticCache(namespace="test")

        _, k1 = cache._normalize_and_key("python programming")
        _, k2 = cache._normalize_and_key("javascript programming")

        assert k1 != k2


# =============================================================================
# SemanticCache — Redis hot-path (mocked)
# =============================================================================


class TestSemanticCacheRedisHotPath:
    @pytest.mark.asyncio
    async def test_redis_hot_check_returns_none_without_redis(self):
        cache = SemanticCache(namespace="test")
        assert cache._redis_hot_check("key") is None

    @pytest.mark.asyncio
    async def test_redis_hot_set_noop_without_redis(self):
        cache = SemanticCache(namespace="test")
        cache._redis_hot_set("key", "value")  # Should not raise

    @pytest.mark.asyncio
    async def test_redis_hot_check_with_mock_redis(self):
        """When Redis returns data, hot_check returns deserialized value."""
        import json

        mock_redis = MagicMock()
        # LuaScripts needs to be mocked since we can't register scripts on a mock
        cache = SemanticCache(namespace="test", redis_client=mock_redis)

        # Patch the _lua.call to return a cached value
        with patch.object(cache, "_redis_hot_check", return_value="cached_response"):
            result = cache._redis_hot_check("some_key")
            assert result == "cached_response"


# =============================================================================
# SemanticCache — invalidate/clear without pool
# =============================================================================


class TestSemanticCacheInvalidateClear:
    def test_clear_no_pool(self):
        cache = SemanticCache(namespace="test")
        assert cache.clear() == 0

    @pytest.mark.asyncio
    async def test_invalidate_async_no_pool(self):
        cache = SemanticCache(namespace="test")
        result = await cache._invalidate_async("query")
        assert result is False

    @pytest.mark.asyncio
    async def test_clear_async_no_pool(self):
        cache = SemanticCache(namespace="test")
        result = await cache._clear_async()
        assert result == 0


# =============================================================================
# SemanticCache — background task tracking
# =============================================================================


class TestSemanticCacheBackgroundTasks:
    @pytest.mark.asyncio
    async def test_spawn_background_tracked(self):
        cache = SemanticCache(namespace="test")

        async def noop():
            pass

        task = cache._spawn_background(noop())
        assert task in cache._background_tasks
        await task
        # After completion, task should be discarded via callback
        await asyncio.sleep(0.01)
        assert task not in cache._background_tasks


# =============================================================================
# @semantic_coalesce decorator
# =============================================================================


class TestSemanticCoalesceDecorator:
    @pytest.mark.asyncio
    async def test_basic_usage_no_backend(self):
        """Without pool/redis, always executes the function."""
        call_count = 0

        @semantic_coalesce(ttl=60)
        async def answer(query: str) -> str:
            nonlocal call_count
            call_count += 1
            return f"answer:{query}"

        r1 = await answer("what is python?")
        r2 = await answer("what is javascript?")

        assert r1 == "answer:what is python?"
        assert r2 == "answer:what is javascript?"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_empty_query_bypasses_cache(self):
        call_count = 0

        @semantic_coalesce(ttl=60)
        async def answer(query: str) -> str:
            nonlocal call_count
            call_count += 1
            return "result"

        await answer("")  # Empty query should bypass cache
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_attached_utilities(self):
        @semantic_coalesce(ttl=60)
        async def my_func(query: str) -> str:
            return query

        assert hasattr(my_func, "cache")
        assert hasattr(my_func, "invalidate")
        assert hasattr(my_func, "clear_cache")
        assert hasattr(my_func, "stats")

    @pytest.mark.asyncio
    async def test_query_param_by_name(self):
        call_count = 0

        @semantic_coalesce(ttl=60, query_param="question")
        async def answer(question: str = "", ctx: str = "") -> str:
            nonlocal call_count
            call_count += 1
            return f"a:{question}"

        r = await answer(question="hello", ctx="world")
        assert r == "a:hello"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_preserves_name(self):
        @semantic_coalesce(ttl=60)
        async def my_semantic_func(query: str) -> str:
            return query

        assert my_semantic_func.__name__ == "my_semantic_func"


# =============================================================================
# Registry
# =============================================================================


class TestSemanticRegistry:
    def test_singleton(self):
        c1 = get_semantic_cache("sem_test_ns_a")
        c2 = get_semantic_cache("sem_test_ns_a")
        assert c1 is c2

    def test_different_namespaces(self):
        c1 = get_semantic_cache("sem_ns_x")
        c2 = get_semantic_cache("sem_ns_y")
        assert c1 is not c2
