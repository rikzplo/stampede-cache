"""Logging abstraction for stampede.

Uses stdlib logging by default. Provides a pluggable span context manager
for tracing integration (OpenTelemetry, Datadog, etc.).

To integrate with your own tracing:
    import stampede
    stampede.configure(tracer=my_opentelemetry_tracer)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator


def get_logger(name: str) -> logging.Logger:
    """Get a logger namespaced under 'stampede'."""
    return logging.getLogger(f"stampede.{name.split('.')[-1]}")


class _NoOpSpan:
    """No-op span for when no tracer is configured."""

    def info(self, _msg: str, **_kw: Any) -> None:
        pass

    def error(self, _msg: str, **_kw: Any) -> None:
        pass


# Global tracer hook — set via stampede.configure(tracer=...)
_tracer: Any = None


def set_tracer(tracer: Any) -> None:
    """Set a global tracer for span creation.

    The tracer should support: tracer.start_as_current_span(name, attributes={})
    Compatible with OpenTelemetry Tracer interface.
    """
    global _tracer
    _tracer = tracer


@contextmanager
def span(name: str, **attributes: Any) -> Generator[_NoOpSpan, None, None]:
    """Context manager that creates a tracing span if a tracer is configured.

    Falls back to a no-op span if no tracer is set.

    Args:
        name: Span name (e.g. "coalesce.execute")
        **attributes: Span attributes (key=value pairs)
    """
    if _tracer is not None:
        try:
            with _tracer.start_as_current_span(name, attributes=attributes) as s:
                yield s  # type: ignore[misc]
                return
        except Exception:
            pass

    yield _NoOpSpan()
