"""Minimal query service used by API routes."""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.async_utils import run_coro_sync
from app.schemas.common import Mode
from app.schemas.documents import RetrievalSettingsRequest, RetrievalSettingsResponse
from app.schemas.api import QueryRequest, QueryResponse
from app.workflows.runner import WorkflowRunner


class QueryService:
    """Application service facade for running query workflows."""

    def __init__(self, runner: WorkflowRunner | None = None) -> None:
        self.runner = runner or WorkflowRunner()

    async def run_async(
        self,
        query: str,
        mode: Mode,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
        query_filters: dict[str, Any] | None = None,
    ) -> QueryResponse:
        """Execute query for selected mode."""
        runner_async = getattr(self.runner, "run_async", None)
        if callable(runner_async):
            return await runner_async(
                query=query,
                mode=mode,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
                query_filters=query_filters,
            )
        return await asyncio.to_thread(
            self.runner.run,
            query=query,
            mode=mode,
            chat_history=chat_history,
            model=model,
            response_language=response_language,
            query_filters=query_filters,
        )

    def run(
        self,
        query: str,
        mode: Mode,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
        query_filters: dict[str, Any] | None = None,
    ) -> QueryResponse:
        """Sync wrapper for legacy callers."""
        return run_coro_sync(
            self.run_async(
                query=query,
                mode=mode,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
                query_filters=query_filters,
            )
        )

    async def run_request_async(self, payload: QueryRequest) -> QueryResponse:
        """Execute query from typed API request payload."""
        query_filters: dict[str, Any] = {}
        if payload.doc_ids:
            query_filters["doc_ids"] = payload.doc_ids
        if payload.filenames:
            query_filters["filenames"] = payload.filenames
        if payload.file_types:
            query_filters["file_types"] = payload.file_types
        if payload.uploaded_after is not None:
            query_filters["uploaded_after"] = payload.uploaded_after
        if payload.uploaded_before is not None:
            query_filters["uploaded_before"] = payload.uploaded_before
        if payload.include_ocr is not None:
            query_filters["include_ocr"] = payload.include_ocr
        runner_async = getattr(self.runner, "run_async", None)
        if callable(runner_async):
            return await runner_async(
                query=payload.query,
                mode=payload.mode,
                chat_history=payload.chat_history,
                model=payload.model,
                query_filters=query_filters or None,
            )
        return await asyncio.to_thread(
            self.runner.run,
            query=payload.query,
            mode=payload.mode,
            chat_history=payload.chat_history,
            model=payload.model,
            query_filters=query_filters or None,
        )

    def run_request(self, payload: QueryRequest) -> QueryResponse:
        """Sync wrapper for typed request execution."""
        return run_coro_sync(self.run_request_async(payload))

    def update_retrieval_settings(self, payload: RetrievalSettingsRequest) -> RetrievalSettingsResponse:
        """Apply retrieval settings to standard/advanced/compare workflows."""
        return self.runner.update_retrieval_settings(payload)

    async def aclose(self) -> None:
        """Close owned async resources."""
        runner_aclose = getattr(self.runner, "aclose", None)
        if callable(runner_aclose):
            await runner_aclose()
