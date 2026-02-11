"""JSON serialization helpers.

Uses orjson when available (10-50x faster), falls back to stdlib json.
"""

from __future__ import annotations

from typing import Any

try:
    import orjson

    def dumps(obj: Any) -> str:
        return orjson.dumps(obj).decode("utf-8")

    def dumps_bytes(obj: Any) -> bytes:
        return orjson.dumps(obj)

    def loads(data: str | bytes) -> Any:
        return orjson.loads(data)

    JSONDecodeError = orjson.JSONDecodeError

except ImportError:
    import json as _json

    def dumps(obj: Any) -> str:  # type: ignore[misc]
        return _json.dumps(obj, separators=(",", ":"), sort_keys=True)

    def dumps_bytes(obj: Any) -> bytes:  # type: ignore[misc]
        return _json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")

    def loads(data: str | bytes) -> Any:  # type: ignore[misc]
        return _json.loads(data)

    JSONDecodeError = _json.JSONDecodeError  # type: ignore[assignment,misc]
