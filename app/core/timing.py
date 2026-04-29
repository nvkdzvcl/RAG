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


def coerce_non_negative_int(value: Any, default: int = 0) -> int:
    """Coerce a generic counter/value to non-negative integer."""
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


def safe_ratio(numerator: Any, denominator: Any) -> float:
    """Return a non-negative ratio, defaulting to 0.0 on invalid values."""
    try:
        left = float(numerator)
        right = float(denominator)
    except (TypeError, ValueError):
        return 0.0
    if right <= 0:
        return 0.0
    value = left / right
    if value != value:  # NaN guard
        return 0.0
    return max(0.0, round(value, 6))


def ensure_completed_trace(
    trace: list[dict[str, Any]] | None,
    *,
    total_ms: Any = 0,
) -> list[dict[str, Any]]:
    """Ensure a trace list ends with a stable completion marker."""
    normalized = [dict(item) for item in (trace or [])]
    if (
        normalized
        and str(normalized[-1].get("step", "")).strip().lower() == "completed"
    ):
        normalized[-1]["total_ms"] = coerce_ms(
            normalized[-1].get("total_ms", total_ms),
            coerce_ms(total_ms, 0),
        )
        normalized[-1].setdefault("status", "success")
        return normalized

    normalized.append(
        {
            "step": "completed",
            "status": "success",
            "total_ms": coerce_ms(total_ms, 0),
        }
    )
    return normalized


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
        self._metrics[name] = coerce_ms(duration, 0)
        return duration

    def record_ms(self, name: str, milliseconds: int) -> int:
        """Record a precomputed measurement for *name*."""
        duration = coerce_ms(milliseconds, 0)
        self._metrics[name] = duration
        return duration

    def get_ms(self, name: str, default: int = 0) -> int:
        """Read one measurement, returning *default* when missing."""
        return coerce_ms(self._metrics.get(name, default), coerce_ms(default, 0))

    def metrics(self) -> dict[str, int]:
        """Return a shallow copy of all measurements."""
        return {key: coerce_ms(value, 0) for key, value in self._metrics.items()}

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Context manager to measure sync code blocks."""
        started = perf_counter()
        try:
            yield
        finally:
            self._metrics[name] = coerce_ms(elapsed_ms(started), 0)

    @asynccontextmanager
    async def measure_async(self, name: str) -> AsyncIterator[None]:
        """Async context manager to measure async code blocks."""
        started = perf_counter()
        try:
            yield
        finally:
            self._metrics[name] = coerce_ms(elapsed_ms(started), 0)
