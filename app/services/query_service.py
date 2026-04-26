"""Minimal query service used by API routes."""

from __future__ import annotations

from app.schemas.common import Mode
from app.schemas.documents import RetrievalSettingsRequest, RetrievalSettingsResponse
from app.schemas.api import QueryRequest, QueryResponse
from app.workflows.runner import WorkflowRunner


class QueryService:
    """Application service facade for running query workflows."""

    def __init__(self, runner: WorkflowRunner | None = None) -> None:
        self.runner = runner or WorkflowRunner()

    def run(
        self,
        query: str,
        mode: Mode,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
    ) -> QueryResponse:
        """Execute query for selected mode."""
        return self.runner.run(
            query=query,
            mode=mode,
            chat_history=chat_history,
            model=model,
            response_language=response_language,
        )

    def run_request(self, payload: QueryRequest) -> QueryResponse:
        """Execute query from typed API request payload."""
        return self.runner.run(
            query=payload.query,
            mode=payload.mode,
            chat_history=payload.chat_history,
            model=payload.model,
        )

    def update_retrieval_settings(self, payload: RetrievalSettingsRequest) -> RetrievalSettingsResponse:
        """Apply retrieval settings to standard/advanced/compare workflows."""
        return self.runner.update_retrieval_settings(payload)
