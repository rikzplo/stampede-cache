"""stampede — Multi-tier async caching with request coalescing, thundering herd
prevention, and semantic similarity matching.

Quick start (zero deps, pure asyncio):

    from stampede import coalesce

    @coalesce(ttl=60)
    async def expensive_call(query: str) -> str:
        return await llm.generate(query)

With Redis distributed coalescing:

    from stampede import distributed_coalesce

    @distributed_coalesce(ttl=60, redis_client=my_redis)
    async def generate(query: str) -> str: ...

With semantic caching (pgvector):

    from stampede import semantic_coalesce

    @semantic_coalesce(ttl=300, embed_fn=embed, pool=pg_pool)
    async def answer(query: str) -> str: ...
"""

from __future__ import annotations

from typing import Any

from stampede._logging import set_tracer

# Core (zero external deps)
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

# Cache backends
from stampede.cache import (
    TTL_5_MINUTES,
    TTL_DAY,
    TTL_HOUR,
    TTL_MINUTE,
    TTL_WEEK,
    AsyncCache,
    Cache,
    CacheBackend,
    CacheEntry,
    CacheStats,
    MemoryBackend,
    TTLCache,
    ValkeyBackend,
    async_cached,
    cached,
)

# Distributed coalescing (requires redis)
from stampede.distributed import (
    DistributedCoalesceStats,
    DistributedRequestCoalescer,
    clear_all_distributed_coalescers,
    distributed_coalesce,
    get_all_distributed_coalescer_stats,
    get_distributed_coalescer,
)

# Semantic cache (requires asyncpg + pgvector)
from stampede.semantic import (
    SemanticCache,
    SemanticCacheStats,
    clear_all_semantic_caches,
    get_all_semantic_cache_stats,
    get_semantic_cache,
    semantic_coalesce,
)

# Hashing utilities
from stampede.hashing import (
    content_fingerprint,
    make_cache_key,
    normalize_query,
    quick_hash,
)


def configure(*, tracer: Any = None) -> None:
    """Configure stampede with optional integrations.

    Args:
        tracer: OpenTelemetry-compatible tracer for span creation.
                Should support tracer.start_as_current_span(name, attributes={}).
    """
    if tracer is not None:
        set_tracer(tracer)


__version__ = "0.1.0"

__all__ = [
    # Configuration
    "configure",
    # Core coalescing
    "CachedResult",
    "CoalesceStats",
    "InFlightRequest",
    "RequestCoalescer",
    "clear_all_coalescers",
    "coalesce",
    "coalesce_by_content",
    "get_all_coalescer_stats",
    "get_coalescer",
    # Cache
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
    "TTL_MINUTE",
    "TTL_5_MINUTES",
    "TTL_HOUR",
    "TTL_DAY",
    "TTL_WEEK",
    # Distributed
    "DistributedCoalesceStats",
    "DistributedRequestCoalescer",
    "clear_all_distributed_coalescers",
    "distributed_coalesce",
    "get_all_distributed_coalescer_stats",
    "get_distributed_coalescer",
    # Semantic
    "SemanticCache",
    "SemanticCacheStats",
    "clear_all_semantic_caches",
    "get_all_semantic_cache_stats",
    "get_semantic_cache",
    "semantic_coalesce",
    # Hashing
    "content_fingerprint",
    "make_cache_key",
    "normalize_query",
    "quick_hash",
]
