"""Query routes placeholder for workflow execution."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.core.timing import coerce_ms
from app.schemas.api import QueryRequest, QueryResponse
from app.services import QueryService
from app.services.runtime import get_query_service

router = APIRouter(prefix="/query", tags=["query"])


def _format_sse(event: str, payload: dict[str, Any]) -> str:
    """Serialize one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _extract_response_latency_ms(response: QueryResponse) -> int | None:
    """Extract best-effort response latency in milliseconds."""
    latency = getattr(response, "latency_ms", None)
    if isinstance(latency, (int, float)):
        return coerce_ms(latency, 0)

    for branch in ("standard", "advanced"):
        branch_response = getattr(response, branch, None)
        if branch_response is None:
            continue
        trace = getattr(branch_response, "trace", None)
        if not isinstance(trace, list):
            continue
        for step in reversed(trace):
            if not isinstance(step, dict):
                continue
            compare_total = step.get("compare_total_ms")
            if isinstance(compare_total, (int, float)):
                return coerce_ms(compare_total, 0)
    return None


@router.post("", response_model=QueryResponse)
async def query(
    payload: QueryRequest, query_service: QueryService = Depends(get_query_service)
) -> QueryResponse:
    """Execute selected workflow mode for a user query."""
    try:
        return await query_service.run_request_async(payload)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc


@router.post("/stream")
async def query_stream(
    payload: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
) -> StreamingResponse:
    """Execute query and stream incremental workflow events via SSE."""

    async def _event_stream():
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        stream_started = time.perf_counter()
        first_event_elapsed_ms: int | None = None
        first_token_elapsed_ms: int | None = None

        async def _push(event: dict[str, Any]) -> None:
            nonlocal first_event_elapsed_ms, first_token_elapsed_ms
            elapsed = coerce_ms((time.perf_counter() - stream_started) * 1000, 0)
            if first_event_elapsed_ms is None:
                first_event_elapsed_ms = elapsed
            if (
                first_token_elapsed_ms is None
                and str(event.get("type") or "").strip().lower() == "generation_delta"
            ):
                first_token_elapsed_ms = elapsed
            await queue.put(event)

        async def _worker() -> None:
            try:
                await _push(
                    {
                        "type": "start",
                        "mode": payload.mode.value,
                        "query": payload.query,
                    }
                )
                response = await query_service.run_request_async(
                    payload,
                    event_handler=_push,
                )
                response_latency_ms = _extract_response_latency_ms(response)
                await _push(
                    {
                        "type": "final",
                        "response": response.model_dump(mode="json"),
                        "latency_ms": response_latency_ms,
                        "total_latency_ms": response_latency_ms,
                        "time_to_first_token_ms": first_token_elapsed_ms,
                        "time_to_first_event_ms": first_event_elapsed_ms,
                    }
                )
                await _push({"type": "done"})
            except NotImplementedError as exc:
                await _push(
                    {
                        "type": "error",
                        "error": {
                            "code": "not_implemented",
                            "message": str(exc),
                        },
                    }
                )
            except Exception as exc:
                _ = exc
                await _push(
                    {
                        "type": "error",
                        "error": {
                            "code": "internal_error",
                            "message": "Internal server error.",
                        },
                    }
                )
            finally:
                await queue.put(None)

        worker_task = asyncio.create_task(_worker())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                event_name = str(event.get("type") or "message")
                yield _format_sse(event_name, event)
        finally:
            if not worker_task.done():
                worker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await worker_task

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
