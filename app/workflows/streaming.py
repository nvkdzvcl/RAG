"""Shared event-emission helpers for streaming workflow updates."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

StreamEvent = dict[str, Any]
StreamEventHandler = Callable[[StreamEvent], Awaitable[None] | None]


async def emit_stream_event(handler: StreamEventHandler | None, event: StreamEvent) -> None:
    """Emit one stream event when a handler is configured."""
    if handler is None:
        return
    result = handler(event)
    if inspect.isawaitable(result):
        await result
