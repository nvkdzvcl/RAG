"""Lightweight timing helpers for workflow instrumentation."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from time import perf_counter
from typing import Any, AsyncIterator, Iterator, Mapping


RETRIEVAL_TIMING_KEYS: tuple[str, ...] = (
    "retrieval_total_ms",
    "dense_retrieve_ms",
    "sparse_retrieve_ms",
    "hybrid_merge_ms",
)


def elapsed_ms(start: float) -> int:
    """Return elapsed milliseconds since *start* (perf_counter timestamp)."""
    return int((perf_counter() - start) * 1000)


def coerce_ms(value: Any, default: int = 0) -> int:
    """Coerce a timing value to non-negative milliseconds."""
    if isinstance(value, bool):
        return default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def normalize_timing_payload(
    raw: Mapping[str, Any] | None,
    *,
    keys: tuple[str, ...] = RETRIEVAL_TIMING_KEYS,
) -> dict[str, int]:
    """Return a timing payload with all expected keys present."""
    source = raw or {}
    return {key: coerce_ms(source.get(key, 0)) for key in keys}


def has_timing_breakdown(
    raw: Mapping[str, Any] | None,
    *,
    keys: tuple[str, ...] = RETRIEVAL_TIMING_KEYS,
) -> bool:
    """Return true when every expected timing key is present."""
    if not raw:
        return False
    return all(key in raw for key in keys)


class StepTimer:
    """Collect named timing measurements in milliseconds."""

    def __init__(self) -> None:
        self._active: dict[str, float] = {}
        self._metrics: dict[str, int] = {}

    def start_timer(self, name: str) -> float:
        """Start a named timer and return its start timestamp."""
        started = perf_counter()
        self._active[name] = started
        return started

    def stop_timer(self, name: str) -> int:
        """Stop a named timer and return elapsed milliseconds."""
        started = self._active.pop(name, perf_counter())
        duration = elapsed_ms(started)
        self._metrics[name] = duration
        return duration

    def record_ms(self, name: str, milliseconds: int) -> int:
        """Record a precomputed measurement for *name*."""
        duration = int(milliseconds)
        self._metrics[name] = duration
        return duration

    def get_ms(self, name: str, default: int = 0) -> int:
        """Read one measurement, returning *default* when missing."""
        return int(self._metrics.get(name, default))

    def metrics(self) -> dict[str, int]:
        """Return a shallow copy of all measurements."""
        return dict(self._metrics)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Context manager to measure sync code blocks."""
        started = perf_counter()
        try:
            yield
        finally:
            self._metrics[name] = elapsed_ms(started)

    @asynccontextmanager
    async def measure_async(self, name: str) -> AsyncIterator[None]:
        """Async context manager to measure async code blocks."""
        started = perf_counter()
        try:
            yield
        finally:
            self._metrics[name] = elapsed_ms(started)
