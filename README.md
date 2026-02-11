# stampede

Multi-tier async caching with request coalescing, thundering herd prevention, and semantic similarity matching for Python.

Built for LLM-heavy backends where duplicate and near-duplicate requests are expensive.

## What it does

```
Request 1 ─┐
Request 2 ─┼──▶ Single Execution ──▶ Result shared by all
Request 3 ─┘
```

Without stampede, 3 identical requests = 3 LLM calls = 3x cost.
With stampede, 3 identical requests = 1 LLM call = 1x cost.

## Install

```bash
# Core (zero deps, pure asyncio)
pip install stampede

# With Redis distributed caching
pip install stampede[redis]

# With semantic similarity caching (pgvector)
pip install stampede[semantic]

# Everything
pip install stampede[all]
```

## Quick Start

### Request Coalescing (in-flight dedup)

Multiple concurrent calls with the same key share a single execution:

```python
from stampede import coalesce

@coalesce(ttl=60)
async def generate_course(topic: str) -> str:
    return await llm.generate(topic)  # Only called once for concurrent identical requests
```

### Content-Based Coalescing

For LLM workloads where exact-match keys are too strict:

```python
from stampede import coalesce_by_content

@coalesce_by_content(ttl=300, content_fields=["topic", "context"])
async def generate_content(topic: str, context: str, user_id: str) -> str:
    # user_id excluded from cache key — different users share results
    return await llm.generate(topic, context)
```

### TTL Cache with Thundering Herd Prevention

```python
from stampede import cached, async_cached, Cache, ValkeyBackend

# Decorator API
@cached(ttl=300)
def fetch_data(key: str) -> dict:
    return expensive_computation(key)

@async_cached(ttl=3600)
async def search(query: str) -> list:
    return await api_call(query)

# Programmatic API with Redis backend
backend = ValkeyBackend.from_url("redis://localhost:6379/0")
cache = Cache(namespace="my_service", backend=backend)
cache.get_or_set("key", lambda: compute(), ttl=300)  # Atomic, stampede-safe
```

### Distributed Coalescing (multi-instance)

Cross-instance dedup for Cloud Run, Kubernetes, etc:

```python
from stampede import distributed_coalesce
import redis

redis_client = redis.from_url("redis://localhost:6379/0")

@distributed_coalesce(ttl=60, redis_client=redis_client)
async def generate(query: str) -> str:
    return await llm.generate(query)
```

### Semantic Cache (pgvector)

Cache by meaning, not exact text. "What courses for engineering?" hits the same cache as "Recommend engineering courses":

```python
from stampede import semantic_coalesce

@semantic_coalesce(
    ttl=300,
    threshold=0.92,       # Cosine similarity threshold
    embed_fn=my_embed,    # async (str) -> list[float]
    pool=my_pg_pool,      # asyncpg.Pool with pgvector
)
async def answer(query: str) -> str:
    return await llm.generate(query)
```

Tiered lookup: Redis exact hash (<1ms) → PostgreSQL exact hash → pgvector HNSW similarity → cache miss.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      stampede                           │
├──────────────┬──────────────┬──────────────┬────────────┤
│  coalesce    │    cache     │ distributed  │  semantic  │
│              │              │              │            │
│ In-flight    │ Memory +     │ Redis locks  │ pgvector   │
│ dedup        │ Redis/Valkey │ + polling    │ + Redis    │
│ TTL cache    │ Lua scripts  │ Lua scripts  │ hot-path   │
│ Stats        │ Thundering   │ Cross-inst.  │ HNSW       │
│              │ herd prev.   │ fallback     │ Embeddings │
└──────────────┴──────────────┴──────────────┴────────────┘
```

## Observability

Every component tracks stats:

```python
from stampede import get_all_coalescer_stats

stats = get_all_coalescer_stats()
# {'skills': {'requests': 1000, 'cache_hits': 450, 'coalesce_hits': 200,
#             'executions': 350, 'savings_rate': '65.0%', ...}}
```

## Configuration

```python
import stampede

# Optional: plug in OpenTelemetry tracing
stampede.configure(tracer=my_otel_tracer)
```

## Optional Dependencies

| Extra | Package | What it enables |
|-------|---------|----------------|
| `blake3` | `blake3` | 3-5x faster hashing (falls back to SHA256) |
| `orjson` | `orjson` | 10-50x faster JSON serialization |
| `redis` | `redis` | Distributed cache + coalescing + thundering herd Lua scripts |
| `semantic` | `asyncpg` | Semantic similarity caching with pgvector |

## Semantic Cache Schema

If using the semantic cache, create this table (or add to your migrations):

```sql
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
```

## License

MIT
