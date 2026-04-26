"""Workflow router that dispatches based on selected mode."""

import logging

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
        configured_mode = str(getattr(settings, "retrieval_mode", "preset")).strip().lower()
        configured_top_k = int(getattr(settings, "retrieval_top_k", self._standard.get_retrieval_top_k()))
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

    def run(
        self,
        query: str,
        mode: Mode,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
    ) -> QueryResponse:
        if mode == Mode.STANDARD:
            return self._standard.run(
                query=query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        if mode == Mode.ADVANCED:
            return self._advanced.run(
                query=query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        if mode == Mode.COMPARE:
            return self._compare.run(
                query=query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        raise NotImplementedError(f"Unsupported mode: {mode}")

    def update_retrieval_settings(self, payload: RetrievalSettingsRequest) -> RetrievalSettingsResponse:
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
