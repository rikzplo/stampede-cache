"""Semantic Response Caching for LLM Queries

Caches responses based on semantic similarity rather than exact match.
"What courses for engineering?" and "Recommend engineering courses"
share the same cached response when sufficiently similar.

Critical Design: High similarity threshold (0.92 default) ensures queries
with actual variance still execute separately. Only near-identical intent matches.

Architecture:
    Query -> Redis exact-hash check (fast) -> pgvector similarity -> Return cached OR execute + cache

Storage:
    - Redis/Valkey: Hot-path for exact hash matches (Lua script for atomic check + stats)
    - PostgreSQL + pgvector: Semantic similarity search via HNSW indexing

Usage:
    # You must provide your own embedding function and database pool
    cache = SemanticCache(
        namespace="advisor",
        threshold=0.92,
        embed_fn=my_embed_function,      # async (str) -> list[float]
        pool=my_asyncpg_pool,            # asyncpg.Pool
        redis_client=my_redis,           # optional, for hot-path
    )
    result = await cache.get_or_execute(query, lambda: llm_call(query))

    # Or use the decorator
    @semantic_coalesce(
        ttl=300,
        threshold=0.92,
        embed_fn=my_embed_function,
        pool=my_asyncpg_pool,
    )
    async def get_recommendation(query: str) -> str:
        return await llm.generate(query)
"""

from __future__ import annotations

import asyncio
import functools
import random
from dataclasses import dataclass, fields
from threading import RLock
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from stampede._json import dumps, loads
from stampede._logging import get_logger, span
from stampede.hashing import content_fingerprint, normalize_query
from stampede.lua import LuaScripts

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    import redis as redis_module
    from asyncpg import Pool

    RedisClient = redis_module.Redis
    EmbedFn = Callable[[str], Awaitable[list[float]]]

log = get_logger(__name__)
T = TypeVar("T")

EMBEDDING_DIM = 1536  # text-embedding-3-small dimensions (override as needed)

# Precompiled Lua script for Redis hot-path
_lua = LuaScripts(__file__, {"hot_check": "_lua/semantic_cache_hot.lua"})


@dataclass(slots=True)
class SemanticCacheStats:
    """Statistics for semantic cache effectiveness."""

    requests: int = 0
    exact_hits: int = 0  # Exact hash match (fastest path)
    semantic_hits: int = 0  # Semantic similarity match
    misses: int = 0  # No match, executed
    embeddings_generated: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        return (self.exact_hits + self.semantic_hits) / self.requests * 100 if self.requests else 0.0

    @property
    def semantic_rate(self) -> float:
        """Percentage of hits that were semantic (vs exact)."""
        return self.semantic_hits / total * 100 if (total := self.exact_hits + self.semantic_hits) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            f.name: getattr(self, f.name) for f in fields(self)
        } | {
            "hit_rate": f"{self.hit_rate:.1f}%",
            "semantic_rate": f"{self.semantic_rate:.1f}%",
        }


class SemanticCache(Generic[T]):
    """PostgreSQL + pgvector backed semantic cache with Redis hot-path.

    Uses HNSW indexing for fast approximate nearest neighbor search, providing:
    - Distributed semantic caching across multiple instances
    - Persistence across restarts
    - Reduced memory pressure (embeddings stored in PostgreSQL)
    - Scalable similarity search via pgvector
    - Redis hot-path for exact matches (avoids PostgreSQL round-trip)

    Tiered lookup strategy (fast to slow):
        1. Redis exact hash check (Lua script, <1ms)
        2. PostgreSQL exact hash match (content fingerprint)
        3. Semantic similarity search (pgvector HNSW)
        4. Cache miss -> execute function

    Required schema (create via migration):

        CREATE EXTENSION IF NOT EXISTS vector;

        CREATE TABLE semantic_cache (
            id BIGSERIAL PRIMARY KEY,
            namespace TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            query_normalized TEXT NOT NULL,
            embedding vector(1536) NOT NULL,
            response TEXT NOT NULL,
            hits INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            expires_at TIMESTAMPTZ,
            UNIQUE(namespace, query_hash)
        );

        CREATE INDEX ON semantic_cache USING hnsw (embedding vector_cosine_ops);
    """

    def __init__(
        self,
        namespace: str = "semantic",
        threshold: float = 0.92,
        ttl: float | None = 300,
        maxsize: int = 500,
        embed_fn: EmbedFn | None = None,
        pool: Pool | None = None,
        redis_client: RedisClient | None = None,
        pool_timeout: float = 3.0,
    ):
        """Initialize pgvector semantic cache.

        Args:
            namespace: Cache namespace for isolation
            threshold: Cosine similarity threshold (0.92 = very conservative)
            ttl: Time-to-live for cached entries in seconds
            maxsize: Max entries per namespace (enforced via LRU eviction)
            embed_fn: Async function that takes a string and returns embedding vector.
                      e.g. async def embed(text: str) -> list[float]: ...
            pool: asyncpg connection pool for PostgreSQL + pgvector
            redis_client: Optional Redis client for hot-path caching
            pool_timeout: Timeout for acquiring pool connections (seconds)
        """
        self.namespace = namespace
        self.threshold = threshold
        self.ttl = ttl
        self.maxsize = maxsize
        self._embed_fn = embed_fn
        self._pool = pool
        self._redis = redis_client
        self._pool_timeout = pool_timeout
        self.stats = SemanticCacheStats()
        self._background_tasks: set[asyncio.Task] = set()

    def _redis_hot_check(self, exact_key: str) -> T | None:
        """Check Redis hot-path for exact match using Lua script."""
        if self._redis is None:
            return None
        try:
            redis_key = f"semhot:{self.namespace}:{exact_key}"
            stats_key = f"semhot:{self.namespace}:stats"
            result = _lua.call(
                self._redis, "hot_check",
                keys=[redis_key, stats_key],
                args=[int(self.ttl) if self.ttl else 0],
            )
            if result:
                return loads(result)
        except Exception as e:
            log.debug(f"Redis hot-path check failed: {e}")
        return None

    def _redis_hot_set(self, exact_key: str, value: T) -> None:
        """Store exact match in Redis hot-path."""
        if self._redis is None:
            return
        try:
            redis_key = f"semhot:{self.namespace}:{exact_key}"
            ttl = int(self.ttl) if self.ttl else 300
            self._redis.setex(redis_key, ttl, dumps(value))
        except Exception as e:
            log.debug(f"Redis hot-path set failed: {e}")

    def _spawn_background(self, coro: Awaitable[Any]) -> asyncio.Task[Any]:
        """Spawn a background task, tracking it to prevent GC."""
        task = asyncio.ensure_future(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    def _normalize_and_key(self, query: str) -> tuple[str, str]:
        """Normalize query and generate exact match key in one pass."""
        return (n := normalize_query(query)), content_fingerprint(n, 24)

    async def _get_embedding(self, text: str) -> list[float]:
        """Generate embedding using the configured embed function."""
        if self._embed_fn is None:
            raise RuntimeError(
                "SemanticCache requires an embed_fn. "
                "Pass embed_fn=my_async_embed_function to the constructor."
            )
        self.stats.embeddings_generated += 1
        return await self._embed_fn(text)

    async def get(self, query: str) -> T | None:
        """Get cached response using tiered lookup:
        Redis -> PostgreSQL exact -> pgvector similarity."""
        self.stats.requests += 1
        normalized, exact_key = self._normalize_and_key(query)

        # Tier 0: Redis hot-path (sub-millisecond)
        if (cached := self._redis_hot_check(exact_key)) is not None:
            self.stats.exact_hits += 1
            log.debug("Semantic cache Redis hot hit", extra={"namespace": self.namespace})
            return cached

        if self._pool is None:
            self.stats.misses += 1
            return None

        try:
            # Tier 1: PostgreSQL exact hash match
            async with self._pool.acquire(timeout=self._pool_timeout) as conn:
                if row := await conn.fetchrow(
                    """SELECT id, response FROM semantic_cache
                    WHERE namespace = $1 AND query_hash = $2
                    AND (expires_at IS NULL OR expires_at > NOW())""",
                    self.namespace,
                    exact_key,
                ):
                    await conn.execute("UPDATE semantic_cache SET hits = hits + 1 WHERE id = $1", row["id"])
                    self.stats.exact_hits += 1
                    response = loads(row["response"])
                    self._redis_hot_set(exact_key, response)
                    log.debug("Semantic cache exact hit (promoted to Redis)", extra={"namespace": self.namespace})
                    return response

            # Tier 2: Semantic similarity search via pgvector
            # Generate embedding outside pool.acquire() to avoid holding connection idle
            embedding = await self._get_embedding(normalized)

            async with self._pool.acquire(timeout=self._pool_timeout) as conn:
                if row := await conn.fetchrow(
                    """SELECT id, query_normalized, response, 1 - (embedding <=> $1) as similarity
                    FROM semantic_cache
                    WHERE namespace = $2
                    AND (expires_at IS NULL OR expires_at > NOW())
                    AND 1 - (embedding <=> $1) > $3
                    ORDER BY embedding <=> $1 LIMIT 1""",
                    embedding,
                    self.namespace,
                    self.threshold,
                ):
                    await conn.execute("UPDATE semantic_cache SET hits = hits + 1 WHERE id = $1", row["id"])
                    self.stats.semantic_hits += 1
                    log.debug(
                        "Semantic cache similarity hit",
                        extra={
                            "namespace": self.namespace,
                            "similarity": float(row["similarity"]),
                            "original": row["query_normalized"][:50],
                        },
                    )
                    return loads(row["response"])

        except Exception as e:
            log.warning(
                "Semantic cache get failed",
                extra={"error": str(e), "error_type": type(e).__name__, "namespace": self.namespace},
            )
            self.stats.errors += 1

        self.stats.misses += 1
        return None

    async def set(self, query: str, response: T, ttl: float | None = None) -> None:
        """Cache a response in both Redis hot-path and pgvector."""
        effective_ttl = ttl if ttl is not None else self.ttl
        normalized, exact_key = self._normalize_and_key(query)

        # Always store in Redis hot-path
        self._redis_hot_set(exact_key, response)

        if self._pool is None:
            return

        try:
            embedding = await self._get_embedding(normalized)
        except Exception as e:
            log.warning("Failed to generate embedding for cache", extra={"error": str(e)})
            return

        try:
            # Coerce embedding to list of floats for pgvector
            emb_list = embedding if isinstance(embedding, list) else list(embedding)
            embedding_vec = [float(x) for x in emb_list]

            async with self._pool.acquire(timeout=self._pool_timeout) as conn:
                expires_at = (
                    await conn.fetchval("SELECT NOW() + $1 * INTERVAL '1 second'", effective_ttl)
                    if effective_ttl
                    else None
                )

                await conn.execute(
                    """INSERT INTO semantic_cache (namespace, query_hash, query_normalized, embedding, response, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (namespace, query_hash) DO UPDATE SET
                        query_normalized = EXCLUDED.query_normalized,
                        embedding = EXCLUDED.embedding,
                        response = EXCLUDED.response,
                        expires_at = EXCLUDED.expires_at,
                        created_at = NOW(),
                        hits = 0""",
                    self.namespace,
                    exact_key,
                    normalized,
                    embedding_vec,
                    dumps(response),
                    expires_at,
                )

                # LRU eviction if at capacity
                await conn.execute(
                    """DELETE FROM semantic_cache WHERE namespace = $1 AND id IN (
                        SELECT id FROM semantic_cache WHERE namespace = $1
                        ORDER BY created_at DESC OFFSET $2 LIMIT $3)""",
                    self.namespace,
                    self.maxsize,
                    self.maxsize,
                )

                # Clean up expired entries periodically (1% chance)
                if random.random() < 0.01:
                    await conn.execute(
                        "DELETE FROM semantic_cache WHERE id IN ("
                        "SELECT id FROM semantic_cache WHERE expires_at IS NOT NULL AND expires_at < NOW() LIMIT 1000)"
                    )

        except Exception as e:
            log.warning(
                "Semantic cache set failed",
                extra={"error": str(e), "error_type": type(e).__name__, "namespace": self.namespace},
            )
            self.stats.errors += 1

    async def get_or_execute(self, query: str, func: Callable[[], Awaitable[T]], ttl: float | None = None) -> T:
        """Get cached response or execute function and cache result.

        Main entry point for semantic caching.
        """
        if (cached := await self.get(query)) is not None:
            return cached

        with span("semantic_cache.execute", namespace=self.namespace):
            result = await func()

        self._spawn_background(self.set(query, result, ttl))  # Non-blocking cache write
        return result

    def invalidate(self, query: str) -> bool:
        """Invalidate a specific cached entry (sync wrapper)."""
        try:
            if asyncio.get_running_loop().is_running():
                self._spawn_background(self._invalidate_async(query))
                return True
        except RuntimeError:
            return asyncio.run(self._invalidate_async(query))
        return False

    async def _invalidate_async(self, query: str) -> bool:
        if self._pool is None:
            return False
        _, exact_key = self._normalize_and_key(query)
        try:
            async with self._pool.acquire(timeout=self._pool_timeout) as conn:
                return "DELETE 1" in await conn.execute(
                    "DELETE FROM semantic_cache WHERE namespace = $1 AND query_hash = $2",
                    self.namespace,
                    exact_key,
                )
        except Exception:
            return False

    def clear(self) -> int:
        """Clear all cached entries for this namespace (sync wrapper)."""
        try:
            if asyncio.get_running_loop().is_running():
                self._spawn_background(self._clear_async())
                return 0
        except RuntimeError:
            return asyncio.run(self._clear_async())
        return 0

    async def _clear_async(self) -> int:
        if self._pool is None:
            return 0
        try:
            async with self._pool.acquire(timeout=self._pool_timeout) as conn:
                return int(
                    (await conn.execute("DELETE FROM semantic_cache WHERE namespace = $1", self.namespace)).split()[-1]
                )
        except Exception:
            return 0


# =============================================================================
# Decorator API
# =============================================================================


def semantic_coalesce(
    ttl: float | None = 300,
    namespace: str = "semantic",
    threshold: float = 0.92,
    query_param: str | int = 0,
    maxsize: int = 500,
    embed_fn: EmbedFn | None = None,
    pool: Pool | None = None,
    redis_client: RedisClient | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for semantic caching on async functions.

    High default threshold (0.92) ensures only near-identical queries share responses.

    Args:
        ttl: Time-to-live for cached results in seconds
        namespace: Cache namespace (defaults to function name)
        threshold: Cosine similarity threshold (0.92 = conservative)
        query_param: Which parameter is the query (name or positional index)
        maxsize: Maximum cached entries
        embed_fn: Async function to generate embeddings
        pool: asyncpg connection pool
        redis_client: Optional Redis client for hot-path

    Examples:
        @semantic_coalesce(ttl=300, embed_fn=embed, pool=pool)
        async def get_course_recommendations(query: str) -> str:
            return await llm.generate(query)

        @semantic_coalesce(ttl=600, threshold=0.85, query_param="question",
                           embed_fn=embed, pool=pool)
        async def answer_faq(question: str, context: str) -> str:
            ...
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        cache: SemanticCache[T] = SemanticCache(
            namespace=func.__name__ if namespace == "semantic" else namespace,
            threshold=threshold,
            ttl=ttl,
            maxsize=maxsize,
            embed_fn=embed_fn,
            pool=pool,
            redis_client=redis_client,
        )

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            if isinstance(query_param, int):
                idx = query_param + (
                    1 if args and hasattr(args[0], "__class__") and not isinstance(args[0], str) else 0
                )
                query = str(args[idx]) if len(args) > idx else ""
            else:
                query = str(kwargs[query_param]) if query_param in kwargs else ""

            return (
                await cache.get_or_execute(query, lambda: func(*args, **kwargs))
                if query
                else await func(*args, **kwargs)
            )

        wrapper.cache = cache  # type: ignore[attr-defined]
        wrapper.invalidate = cache.invalidate  # type: ignore[attr-defined]
        wrapper.clear_cache = cache.clear  # type: ignore[attr-defined]
        wrapper.stats = lambda: cache.stats.to_dict()  # type: ignore[attr-defined]
        return wrapper

    return decorator


# =============================================================================
# Registry
# =============================================================================

_semantic_caches: dict[str, SemanticCache] = {}
_caches_lock = RLock()


def get_semantic_cache(
    namespace: str,
    threshold: float = 0.92,
    ttl: float | None = 300,
    maxsize: int = 500,
    embed_fn: EmbedFn | None = None,
    pool: Pool | None = None,
    redis_client: RedisClient | None = None,
) -> SemanticCache:
    """Get or create a shared semantic cache for a namespace.

    Args only used on creation.
    """
    with _caches_lock:
        if namespace not in _semantic_caches:
            _semantic_caches[namespace] = SemanticCache(
                namespace=namespace,
                threshold=threshold,
                ttl=ttl,
                maxsize=maxsize,
                embed_fn=embed_fn,
                pool=pool,
                redis_client=redis_client,
            )
        return _semantic_caches[namespace]


def get_all_semantic_cache_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all semantic caches."""
    with _caches_lock:
        return {ns: c.stats.to_dict() for ns, c in _semantic_caches.items()}


def clear_all_semantic_caches() -> dict[str, int]:
    """Clear all semantic caches."""
    with _caches_lock:
        return {ns: c.clear() for ns, c in _semantic_caches.items()}


__all__ = [
    "SemanticCache",
    "SemanticCacheStats",
    "clear_all_semantic_caches",
    "get_all_semantic_cache_stats",
    "get_semantic_cache",
    "semantic_coalesce",
]
