"""Workflow router that dispatches based on selected mode."""

import asyncio
import logging
from typing import Any

from app.core.async_utils import run_coro_sync
from app.core.config import get_settings
from app.schemas.api import QueryResponse
from app.schemas.common import Mode
from app.schemas.documents import (
    RetrievalConfigMode,
    RetrievalMode,
    RetrievalSettingsRequest,
    RetrievalSettingsResponse,
)
from app.services.index_runtime import RuntimeIndexManager
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.compare import CompareWorkflow
from app.workflows.streaming import StreamEventHandler
from app.workflows.standard import StandardWorkflow

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Facade that dispatches user queries to a mode-specific workflow."""

    PRESET_RETRIEVAL: dict[RetrievalMode, int] = {
        "low": 3,
        "balanced": 5,
        "high": 8,
        "custom": 8,
    }

    def __init__(self, *, index_manager: RuntimeIndexManager | None = None) -> None:
        settings = get_settings()
        self._standard = StandardWorkflow(index_manager=index_manager)
        self._advanced = AdvancedWorkflow(standard_workflow=self._standard)
        self._compare = CompareWorkflow(
            standard_workflow=self._standard,
            advanced_workflow=self._advanced,
        )
        configured_mode = (
            str(getattr(settings, "retrieval_mode", "preset")).strip().lower()
        )
        configured_top_k = int(
            getattr(settings, "retrieval_top_k", self._standard.get_retrieval_top_k())
        )
        if configured_mode == "custom":
            self.selected_retrieval_mode: RetrievalMode = "custom"
            self.retrieval_mode: RetrievalConfigMode = "custom"
        else:
            matched = next(
                (
                    mode
                    for mode, value in self.PRESET_RETRIEVAL.items()
                    if mode != "custom" and value == configured_top_k
                ),
                None,
            )
            self.selected_retrieval_mode = matched or "high"
            self.retrieval_mode = "preset" if matched is not None else "custom"

    @staticmethod
    async def _invoke_workflow_async(
        workflow: Any,
        *,
        query: str,
        chat_history: list[dict[str, str]] | None,
        model: str | None,
        response_language: str | None,
        query_filters: dict[str, Any] | None,
        event_handler: StreamEventHandler | None,
        event_context: dict[str, Any] | None,
    ) -> QueryResponse:
        run_async = getattr(workflow, "run_async", None)
        if callable(run_async):
            async_kwargs: dict[str, Any] = {
                "query": query,
                "chat_history": chat_history,
                "model": model,
                "response_language": response_language,
            }
            if query_filters is not None:
                async_kwargs["query_filters"] = query_filters
            if event_handler is not None:
                async_kwargs["event_handler"] = event_handler
            if event_context:
                async_kwargs["event_context"] = dict(event_context)

            for removable in ("event_context", "event_handler", "query_filters"):
                try:
                    return await run_async(**async_kwargs)
                except TypeError:
                    if removable in async_kwargs:
                        async_kwargs.pop(removable, None)
                        continue
                    raise
            return await run_async(**async_kwargs)
        kwargs: dict[str, Any] = {
            "query": query,
            "chat_history": chat_history,
            "model": model,
            "response_language": response_language,
        }
        if query_filters is not None:
            kwargs["query_filters"] = query_filters
        if event_handler is not None:
            kwargs["event_handler"] = event_handler
        if event_context:
            kwargs["event_context"] = dict(event_context)

        for removable in ("event_context", "event_handler", "query_filters"):
            try:
                return await asyncio.to_thread(workflow.run, **kwargs)
            except TypeError:
                if removable in kwargs:
                    kwargs.pop(removable, None)
                    continue
                raise
        return await asyncio.to_thread(workflow.run, **kwargs)

    async def run_async(
        self,
        query: str,
        mode: Mode,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
        query_filters: dict[str, Any] | None = None,
        event_handler: StreamEventHandler | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> QueryResponse:
        if mode == Mode.STANDARD:
            return await self._invoke_workflow_async(
                self._standard,
                query=query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
                query_filters=query_filters,
                event_handler=event_handler,
                event_context=event_context,
            )
        if mode == Mode.ADVANCED:
            return await self._invoke_workflow_async(
                self._advanced,
                query=query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
                query_filters=query_filters,
                event_handler=event_handler,
                event_context=event_context,
            )
        if mode == Mode.COMPARE:
            return await self._invoke_workflow_async(
                self._compare,
                query=query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
                query_filters=query_filters,
                event_handler=event_handler,
                event_context=event_context,
            )
        raise NotImplementedError(f"Unsupported mode: {mode}")

    def run(
        self,
        query: str,
        mode: Mode,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
        query_filters: dict[str, Any] | None = None,
        event_handler: StreamEventHandler | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> QueryResponse:
        """Sync wrapper for CLI/tests."""
        return run_coro_sync(
            self.run_async(
                query=query,
                mode=mode,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
                query_filters=query_filters,
                event_handler=event_handler,
                event_context=event_context,
            )
        )

    def update_retrieval_settings(
        self, payload: RetrievalSettingsRequest
    ) -> RetrievalSettingsResponse:
        """Apply retrieval mode/top_k to active workflows without rebuilding indexes."""
        if payload.mode == "custom":
            if payload.top_k is None:
                raise ValueError("top_k is required when mode=custom.")
            resolved_top_k = int(payload.top_k)
            selected_mode: RetrievalMode = "custom"
            retrieval_mode: RetrievalConfigMode = "custom"
        else:
            resolved_top_k = self.PRESET_RETRIEVAL[payload.mode]
            selected_mode = payload.mode
            retrieval_mode = "preset"

        self._standard.set_retrieval_top_k(resolved_top_k)
        self.selected_retrieval_mode = selected_mode
        self.retrieval_mode = retrieval_mode
        response = RetrievalSettingsResponse(
            status="updated",
            mode=selected_mode,
            retrieval_mode=retrieval_mode,
            top_k=self._standard.get_retrieval_top_k(),
            rerank_top_n=self._standard.get_rerank_top_n(),
            context_top_k=self._standard.context_top_k,
        )
        logger.info(
            (
                "Applied retrieval settings | selected_mode=%s | retrieval_mode=%s | top_k=%s "
                "| rerank_top_n=%s | final_context_size=%s"
            ),
            response.mode,
            response.retrieval_mode,
            response.top_k,
            response.rerank_top_n,
            response.context_top_k,
        )
        return response

    async def aclose(self) -> None:
        """Close workflow-level resources."""
        await self._advanced.aclose()
