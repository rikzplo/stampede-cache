"""Tests for stampede.hashing — fingerprinting, cache keys, normalization."""

from __future__ import annotations

from dataclasses import dataclass

from stampede.hashing import (
    content_fingerprint,
    fast_hash,
    make_cache_key,
    normalize_query,
    quick_hash,
    stable_json_hash,
)


# =============================================================================
# quick_hash
# =============================================================================


class TestQuickHash:
    def test_returns_hex_string(self):
        result = quick_hash("hello")
        assert isinstance(result, str)
        assert all(c in "0123456789abcdef" for c in result)

    def test_default_length_16(self):
        assert len(quick_hash("hello")) == 16

    def test_custom_length(self):
        assert len(quick_hash("hello", length=8)) == 8
        assert len(quick_hash("hello", length=32)) == 32

    def test_deterministic(self):
        assert quick_hash("hello") == quick_hash("hello")

    def test_different_inputs_different_hashes(self):
        assert quick_hash("hello") != quick_hash("world")

    def test_bytes_input(self):
        result = quick_hash(b"hello")
        assert isinstance(result, str)
        assert len(result) == 16

    def test_empty_string(self):
        result = quick_hash("")
        assert len(result) == 16


# =============================================================================
# fast_hash (structural FNV-1a)
# =============================================================================


class TestFastHash:
    def test_string(self):
        assert isinstance(fast_hash("hello"), str)
        assert len(fast_hash("hello")) == 16

    def test_int(self):
        assert fast_hash(42) == fast_hash(42)
        assert fast_hash(42) != fast_hash(43)

    def test_float(self):
        assert fast_hash(3.14) == fast_hash(3.14)
        assert fast_hash(3.14) != fast_hash(2.71)

    def test_none(self):
        assert fast_hash(None) == fast_hash(None)
        assert fast_hash(None) != fast_hash(0)

    def test_bool(self):
        assert fast_hash(True) == fast_hash(True)
        assert fast_hash(True) != fast_hash(False)
        # bool hashes should differ from int hashes
        assert fast_hash(True) != fast_hash(1)

    def test_list(self):
        assert fast_hash([1, 2, 3]) == fast_hash([1, 2, 3])
        assert fast_hash([1, 2, 3]) != fast_hash([3, 2, 1])

    def test_tuple(self):
        assert fast_hash((1, 2)) == fast_hash((1, 2))

    def test_dict_order_independent(self):
        """Dicts with same keys/values in different order produce same hash."""
        assert fast_hash({"a": 1, "b": 2}) == fast_hash({"b": 2, "a": 1})

    def test_dict_different_values(self):
        assert fast_hash({"a": 1}) != fast_hash({"a": 2})

    def test_set_order_independent(self):
        assert fast_hash({1, 2, 3}) == fast_hash({3, 2, 1})

    def test_nested_structure(self):
        obj = {"users": [{"name": "alice", "scores": [1, 2, 3]}]}
        assert fast_hash(obj) == fast_hash(obj)

    def test_custom_length(self):
        assert len(fast_hash("hello", length=32)) == 32
        assert len(fast_hash("hello", length=8)) == 8

    def test_length_clamped(self):
        assert len(fast_hash("hello", length=0)) == 1  # min 1
        assert len(fast_hash("hello", length=100)) == 64  # max 64

    def test_bytes(self):
        assert fast_hash(b"hello") == fast_hash(b"hello")
        assert fast_hash(b"hello") != fast_hash(b"world")

    def test_dataclass(self):
        @dataclass
        class Point:
            x: int
            y: int

        assert fast_hash(Point(1, 2)) == fast_hash(Point(1, 2))
        assert fast_hash(Point(1, 2)) != fast_hash(Point(3, 4))


# =============================================================================
# stable_json_hash
# =============================================================================


class TestStableJsonHash:
    def test_deterministic(self):
        obj = {"query": "test", "limit": 10}
        assert stable_json_hash(obj) == stable_json_hash(obj)

    def test_order_independent(self):
        assert stable_json_hash({"a": 1, "b": 2}) == stable_json_hash({"b": 2, "a": 1})

    def test_default_length_32(self):
        assert len(stable_json_hash({"x": 1})) == 32


# =============================================================================
# content_fingerprint
# =============================================================================


class TestContentFingerprint:
    def test_string_input(self):
        fp = content_fingerprint("hello world")
        assert isinstance(fp, str)
        assert len(fp) == 32

    def test_bytes_input(self):
        fp = content_fingerprint(b"hello world")
        assert isinstance(fp, str)
        assert len(fp) == 32

    def test_dict_input(self):
        fp = content_fingerprint({"key": "value"})
        assert isinstance(fp, str)
        assert len(fp) == 32

    def test_deterministic(self):
        assert content_fingerprint("test") == content_fingerprint("test")

    def test_custom_length(self):
        assert len(content_fingerprint("test", length=12)) == 12

    def test_different_content_different_fingerprint(self):
        assert content_fingerprint("hello") != content_fingerprint("world")

    def test_dict_order_independent(self):
        assert content_fingerprint({"a": 1, "b": 2}) == content_fingerprint({"b": 2, "a": 1})


# =============================================================================
# make_cache_key
# =============================================================================


class TestMakeCacheKey:
    def test_simple_args(self):
        key = make_cache_key("search", "python")
        assert "search" in key
        assert "python" in key

    def test_kwargs(self):
        key = make_cache_key(query="test", limit=10)
        assert "query=test" in key
        assert "limit=10" in key

    def test_namespace(self):
        key = make_cache_key("arg", namespace="ns")
        assert key.startswith("ns:")

    def test_deterministic(self):
        assert make_cache_key("a", x=1) == make_cache_key("a", x=1)

    def test_different_args_different_keys(self):
        assert make_cache_key("a") != make_cache_key("b")

    def test_none_arg(self):
        key = make_cache_key(None)
        assert "None" in key

    def test_complex_arg_fingerprinted(self):
        key = make_cache_key({"nested": True})
        # Complex args get fingerprinted, so the key is a hex hash
        assert isinstance(key, str)
        assert len(key) > 0

    def test_long_key_hashed(self):
        """Keys longer than 200 chars get hashed down to 32-char hex."""
        key = make_cache_key(*[f"arg{i}" for i in range(100)])
        assert len(key) == 32

    def test_kwargs_sorted(self):
        """Kwargs are sorted by key name for deterministic output."""
        assert make_cache_key(b=2, a=1) == make_cache_key(a=1, b=2)


# =============================================================================
# normalize_query
# =============================================================================


class TestNormalizeQuery:
    def test_lowercase(self):
        assert normalize_query("Hello World") == "hello world"

    def test_trim_whitespace(self):
        assert normalize_query("  hello  ") == "hello"

    def test_collapse_spaces(self):
        assert normalize_query("hello    world") == "hello world"

    def test_combined(self):
        assert normalize_query("  Hello   WORLD  ") == "hello world"

    def test_empty_string(self):
        assert normalize_query("") == ""

    def test_tabs_and_newlines(self):
        assert normalize_query("hello\t\nworld") == "hello world"
