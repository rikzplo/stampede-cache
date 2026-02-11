"""Hashing utilities for content fingerprinting and cache keys.

Provides:
- Content fingerprinting (deduplication, idempotent cache keys)
- Cache key generation (fast, collision-resistant)
- Stable JSON hashing (deterministic hashing of complex objects)
- Query normalization (for semantic cache dedup)

Performance:
- C extension (_native_hash) provides 10-50x faster structural hashing
  when available. Falls back to pure Python automatically.
- Blake3 provides 3-5x faster byte hashing when installed.
  Falls back to SHA256 automatically.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import struct
from typing import Any

# =============================================================================
# C extension for fast structural hashing (optional, auto-compiled via pip)
# Falls back to pure Python if not available — identical output either way.
# =============================================================================
try:
    from stampede._native_hash import fast_hash as _native_fast_hash
except ImportError:
    _native_fast_hash = None  # type: ignore[assignment]

# Try to import blake3 (faster), fall back to hashlib
try:
    import blake3 as _blake3_mod

    BLAKE3_AVAILABLE = True
    _blake3_hash = _blake3_mod.blake3
except ImportError:
    BLAKE3_AVAILABLE = False
    _blake3_hash = None  # type: ignore[assignment]

# Pre-compiled regex for query normalization
_WHITESPACE_RE = re.compile(r"\s+")

# Type tuple for simple value checks
_SIMPLE_TYPES = (str, int, float, bool)


# =============================================================================
# Core Hashing
# =============================================================================


def blake3_hash(data: bytes, length: int = 32) -> bytes:
    """Hash bytes using Blake3 (or SHA256 fallback).

    Args:
        data: Bytes to hash
        length: Output length in bytes (default 32 = 256 bits)

    Returns:
        Hash digest as bytes
    """
    if _blake3_hash is not None:
        return _blake3_hash(data).digest(length=length)
    digest = hashlib.sha256(data).digest()
    return digest[:length] if length < 32 else digest


def quick_hash(value: str | bytes, length: int = 16) -> str:
    """Fast hash for cache keys and quick lookups.

    Args:
        value: String or bytes to hash
        length: Output length in hex characters (default 16)

    Returns:
        Hex hash string
    """
    data: bytes = value.encode("utf-8") if type(value) is str else value  # type: ignore[assignment]
    byte_length = (length + 1) >> 1
    return blake3_hash(data, byte_length).hex()[:length]


# =============================================================================
# Structural FNV-1a (order-independent dict/set hashing)
# =============================================================================

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _fnv_byte(h: int, b: int) -> int:
    return ((h ^ b) * _FNV_PRIME) & _MASK64


def _fnv_bytes(h: int, data: bytes) -> int:
    for b in data:
        h = ((h ^ b) * _FNV_PRIME) & _MASK64
    return h


def _fnv_u64(h: int, v: int) -> int:
    v = v & _MASK64
    for _ in range(8):
        h = ((h ^ (v & 0xFF)) * _FNV_PRIME) & _MASK64
        v >>= 8
    return h


def _splitmix64(z: int) -> int:
    z = z & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _sh(val: Any, depth: int = 0) -> int:  # noqa: PLR0911, PLR0912, C901
    """Structural hash — produces deterministic hash regardless of dict/set ordering."""
    if depth > 100:
        return 0
    h = _FNV_OFFSET

    if val is None:
        return _fnv_byte(h, 0x00)

    tp = type(val)

    if tp is bool:
        return _fnv_byte(h, 0x01 if val else 0x02)

    if tp is int:
        h = _fnv_byte(h, 0x03)
        if _INT64_MIN <= val <= _INT64_MAX:
            return _fnv_u64(h, val)
        return _fnv_bytes(h, str(val).encode("utf-8"))

    if tp is float:
        import math

        h = _fnv_byte(h, 0x04)
        if val == 0.0:
            bits = 0
        elif math.isnan(val):
            bits = 0x7FF8000000000000
        else:
            bits = struct.unpack("<Q", struct.pack("<d", val))[0]
        return _fnv_u64(h, bits)

    if tp is str:
        h = _fnv_byte(h, 0x05)
        data = val.encode("utf-8")
        h = _fnv_u64(h, len(data))
        return _fnv_bytes(h, data)

    if tp is bytes:
        h = _fnv_byte(h, 0x06)
        h = _fnv_u64(h, len(val))
        return _fnv_bytes(h, val)

    if tp is list:
        h = _fnv_byte(h, 0x07)
        h = _fnv_u64(h, len(val))
        for item in val:
            h = _fnv_u64(h, _sh(item, depth + 1))
        return h

    if tp is tuple:
        h = _fnv_byte(h, 0x08)
        h = _fnv_u64(h, len(val))
        for item in val:
            h = _fnv_u64(h, _sh(item, depth + 1))
        return h

    if tp is dict:
        h = _fnv_byte(h, 0x09)
        h = _fnv_u64(h, len(val))
        acc = 0
        for k, v in val.items():
            kh = _sh(k, depth + 1)
            vh = _sh(v, depth + 1)
            acc ^= kh ^ ((vh * _GOLDEN) & _MASK64)
        return _fnv_u64(h, acc)

    if tp is set or tp is frozenset:
        h = _fnv_byte(h, 0x0A)
        h = _fnv_u64(h, len(val))
        acc = 0
        for item in val:
            acc ^= _sh(item, depth + 1)
        return _fnv_u64(h, acc)

    # Slow path: Pydantic models, dataclasses, arbitrary objects
    if hasattr(val, "model_dump"):
        return _sh(val.model_dump(), depth + 1)
    if hasattr(val, "__dataclass_fields__"):
        return _sh(dataclasses.asdict(val), depth + 1)
    if hasattr(val, "__dict__"):
        return _sh(val.__dict__, depth + 1)
    return 0


def _derive_hex(h: int, length: int) -> str:
    """Derive hex string from 64-bit hash."""
    h = h & _MASK64
    vals = [h]
    vals.extend(_splitmix64((h + _GOLDEN * i) & _MASK64) for i in range(1, 4))
    full = "".join(f"{v:016x}" for v in vals)
    return full[:length]


def fast_hash(obj: Any, length: int = 16) -> str:
    """High-performance structural hash using FNV-1a.

    Hashes the object tree directly without serialization or dict-key
    sorting. Dicts/sets use commutative XOR so insertion order never matters.

    Uses the C extension when available (10-50x faster), falls back to
    pure Python automatically. Output is identical either way.

    Args:
        obj: Any Python object (dict, list, str, Pydantic model, etc.)
        length: Hex output length (1-64, default 16)

    Returns:
        Hex hash string
    """
    if _native_fast_hash is not None:
        return _native_fast_hash(obj, length)
    return _derive_hex(_sh(obj), max(1, min(length, 64)))


def stable_json_hash(obj: Any, length: int = 32) -> str:
    """Generate a deterministic hash for any JSON-serializable object.

    Same object always produces same hash, regardless of dict/set ordering.

    Args:
        obj: Any JSON-serializable object (or Pydantic model, dataclass)
        length: Output length in hex characters (default 32)

    Returns:
        Hex hash string
    """
    return fast_hash(obj, length)


# =============================================================================
# Content Fingerprinting
# =============================================================================


def content_fingerprint(content: str | bytes | dict | Any, length: int = 32) -> str:
    """Generate a content fingerprint for deduplication and caching.

    Args:
        content: Content to fingerprint (string, bytes, dict, or object)
        length: Output length in hex characters (default 32)

    Returns:
        Hex fingerprint string
    """
    ctype = type(content)
    return quick_hash(content, length) if ctype is str or ctype is bytes else stable_json_hash(content, length)  # type: ignore[arg-type]


# =============================================================================
# Cache Key Generation
# =============================================================================


def make_cache_key(*args: Any, namespace: str = "", **kwargs: Any) -> str:
    """Generate a stable, deterministic cache key from function arguments.

    Args:
        *args: Positional arguments to include in key
        namespace: Optional namespace prefix for key isolation
        **kwargs: Keyword arguments to include in key

    Returns:
        Cache key string
    """
    parts: list[str] = [namespace] if namespace else []

    for arg in args:
        if arg is None:
            parts.append("None")
        elif type(arg) in _SIMPLE_TYPES:
            parts.append(str(arg))
        else:
            parts.append(content_fingerprint(arg, 12))

    for key, value in sorted(kwargs.items()):
        parts.append(
            f"{key}={value}"
            if value is None or type(value) in _SIMPLE_TYPES
            else f"{key}={content_fingerprint(value, 12)}"
        )

    combined = ":".join(parts)
    return quick_hash(combined, 32) if len(combined) > 200 else combined


# =============================================================================
# Query Normalization
# =============================================================================


def normalize_query(query: str) -> str:
    """Normalize a search query for consistent caching/deduplication.

    Applies: lowercase, trim whitespace, collapse multiple spaces.

    Args:
        query: Search query string

    Returns:
        Normalized query string
    """
    return _WHITESPACE_RE.sub(" ", query.lower().strip())
