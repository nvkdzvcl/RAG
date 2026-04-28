"""Lightweight timing helpers for workflow instrumentation."""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from time import perf_counter
from typing import Iterator


def elapsed_ms(start: float) -> int:
    """Return elapsed milliseconds since *start* (perf_counter timestamp)."""
    return int((perf_counter() - start) * 1000)


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
    async def measure_async(self, name: str) -> Iterator[None]:
        """Async context manager to measure async code blocks."""
        started = perf_counter()
        try:
            yield
        finally:
            self._metrics[name] = elapsed_ms(started)
