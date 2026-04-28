"""Query routes placeholder for workflow execution."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.schemas.api import QueryRequest, QueryResponse
from app.services import QueryService
from app.services.runtime import get_query_service

router = APIRouter(prefix="/query", tags=["query"])


def _format_sse(event: str, payload: dict[str, Any]) -> str:
    """Serialize one SSE frame."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


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

        async def _push(event: dict[str, Any]) -> None:
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
                await _push(
                    {
                        "type": "final",
                        "response": response.model_dump(mode="json"),
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
                await _push(
                    {
                        "type": "error",
                        "error": {
                            "code": "internal_error",
                            "message": str(exc),
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
