"""Tests for the C extension (_native_hash) — verifies parity with pure Python.

Every test case runs both the C and Python implementations and asserts
they produce identical output. This ensures the C extension is a safe
drop-in acceleration.
"""

from __future__ import annotations

import dataclasses
import math
import sys

import pytest

from stampede.hashing import _derive_hex, _sh

# Import C extension — skip entire module if not available
try:
    from stampede._native_hash import fast_hash as c_fast_hash

    HAS_C_EXTENSION = True
except ImportError:
    HAS_C_EXTENSION = False


pytestmark = pytest.mark.skipif(not HAS_C_EXTENSION, reason="C extension not compiled")


def py_fast_hash(obj: object, length: int = 16) -> str:
    """Pure-Python fast_hash for comparison."""
    return _derive_hex(_sh(obj), max(1, min(length, 64)))


def assert_parity(obj: object, lengths: tuple[int, ...] = (8, 16, 32, 64)) -> None:
    """Assert C and Python produce identical output for all lengths."""
    for length in lengths:
        c = c_fast_hash(obj, length)
        p = py_fast_hash(obj, length)
        assert c == p, f"Mismatch for {obj!r} at length={length}: C={c!r} Py={p!r}"


# =============================================================================
# Primitives
# =============================================================================


class TestPrimitives:
    def test_none(self):
        assert_parity(None)

    def test_true(self):
        assert_parity(True)

    def test_false(self):
        assert_parity(False)

    def test_zero(self):
        assert_parity(0)

    def test_positive_int(self):
        assert_parity(42)

    def test_negative_int(self):
        assert_parity(-1)

    def test_large_int(self):
        assert_parity(2**63 - 1)  # max int64

    def test_negative_large_int(self):
        assert_parity(-(2**63))  # min int64

    def test_bigint_overflow(self):
        """Integers beyond int64 range — fallback to string representation."""
        assert_parity(10**100)
        assert_parity(-(10**100))

    def test_float_positive(self):
        assert_parity(3.14)

    def test_float_negative(self):
        assert_parity(-2.71828)

    def test_float_zero(self):
        assert_parity(0.0)

    def test_float_negative_zero(self):
        # -0.0 should be treated as 0.0
        assert_parity(-0.0)

    def test_float_nan(self):
        assert_parity(float("nan"))

    def test_float_inf(self):
        assert_parity(float("inf"))

    def test_float_neg_inf(self):
        assert_parity(float("-inf"))

    def test_empty_string(self):
        assert_parity("")

    def test_short_string(self):
        assert_parity("hello")

    def test_long_string(self):
        assert_parity("a" * 10_000)

    def test_unicode_string(self):
        assert_parity("hello \u2603 \U0001f600 \u4e16\u754c")

    def test_empty_bytes(self):
        assert_parity(b"")

    def test_bytes(self):
        assert_parity(b"\x00\xff\x80")


# =============================================================================
# Collections
# =============================================================================


class TestCollections:
    def test_empty_list(self):
        assert_parity([])

    def test_list(self):
        assert_parity([1, 2, 3])

    def test_nested_list(self):
        assert_parity([[1, 2], [3, [4, 5]]])

    def test_empty_tuple(self):
        assert_parity(())

    def test_tuple(self):
        assert_parity((1, "two", 3.0))

    def test_empty_dict(self):
        assert_parity({})

    def test_dict(self):
        assert_parity({"a": 1, "b": 2})

    def test_dict_order_independence(self):
        d1 = {"z": 26, "a": 1, "m": 13}
        d2 = {"a": 1, "m": 13, "z": 26}
        c1 = c_fast_hash(d1, 32)
        c2 = c_fast_hash(d2, 32)
        p1 = py_fast_hash(d1, 32)
        p2 = py_fast_hash(d2, 32)
        assert c1 == c2 == p1 == p2

    def test_nested_dict(self):
        assert_parity({"a": {"b": {"c": 1}}})

    def test_empty_set(self):
        assert_parity(set())

    def test_set(self):
        assert_parity({1, 2, 3})

    def test_frozenset(self):
        assert_parity(frozenset([1, 2, 3]))

    def test_set_order_independence(self):
        s = {1, 2, 3}
        c = c_fast_hash(s, 32)
        p = py_fast_hash(s, 32)
        assert c == p


# =============================================================================
# Mixed / nested
# =============================================================================


class TestMixed:
    def test_realistic_payload(self):
        """LLM-style request payload."""
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "metadata": {"user": "u123", "tags": ["test"]},
        }
        assert_parity(payload)

    def test_list_of_mixed_types(self):
        assert_parity([None, True, False, 42, 3.14, "hello", b"bytes", [1], {"a": 2}])

    def test_deeply_nested(self):
        """5 levels deep — well within the 100-depth limit."""
        obj = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        assert_parity(obj)


# =============================================================================
# Dataclasses / objects
# =============================================================================


class TestObjects:
    def test_dataclass(self):
        @dataclasses.dataclass
        class Point:
            x: int
            y: int

        assert_parity(Point(1, 2))

    def test_object_with_dict(self):
        class Foo:
            def __init__(self):
                self.a = 1
                self.b = "hello"

        obj = Foo()
        assert_parity(obj)


# =============================================================================
# Length edge cases
# =============================================================================


class TestLengths:
    def test_length_1(self):
        c = c_fast_hash("test", 1)
        p = py_fast_hash("test", 1)
        assert len(c) == 1
        assert c == p

    def test_length_64(self):
        c = c_fast_hash("test", 64)
        p = py_fast_hash("test", 64)
        assert len(c) == 64
        assert c == p

    def test_length_0_clamped_to_1(self):
        c = c_fast_hash("test", 0)
        p = py_fast_hash("test", 0)
        assert len(c) == 1
        assert c == p

    def test_length_100_clamped_to_64(self):
        c = c_fast_hash("test", 100)
        p = py_fast_hash("test", 100)
        assert len(c) == 64
        assert c == p


# =============================================================================
# Ensures the public fast_hash routes through C
# =============================================================================


class TestIntegration:
    def test_fast_hash_uses_c_extension(self):
        """stampede.hashing.fast_hash should use the C extension when available."""
        from stampede.hashing import _native_fast_hash

        assert _native_fast_hash is not None, "C extension should be loaded"

    def test_fast_hash_returns_same_as_c(self):
        """Public fast_hash() should return same output as direct C call."""
        from stampede.hashing import fast_hash

        obj = {"hello": "world", "n": [1, 2, 3]}
        assert fast_hash(obj) == c_fast_hash(obj, 16)
        assert fast_hash(obj, 32) == c_fast_hash(obj, 32)
