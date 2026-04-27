"""Utilities for bridging async and sync call paths."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable
from typing import Any, TypeVar, cast

T = TypeVar("T")


def run_coro_sync(coro: Awaitable[T]) -> T:
    """Run coroutine from sync code when no event loop is active."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "Cannot call sync wrapper from an active event loop; use async API instead."
    )


async def await_if_needed(value: T | Awaitable[T]) -> T:
    """Await value when needed; return plain value otherwise."""
    if inspect.isawaitable(value):
        return cast(T, await cast(Awaitable[Any], value))
    return cast(T, value)
