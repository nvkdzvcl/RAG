"""Workflow router that dispatches based on selected mode."""

from app.schemas.api import QueryResponse
from app.schemas.common import Mode
from app.services.index_runtime import RuntimeIndexManager
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.compare import CompareWorkflow
from app.workflows.standard import StandardWorkflow


class WorkflowRunner:
    """Facade that dispatches user queries to a mode-specific workflow."""

    def __init__(self, *, index_manager: RuntimeIndexManager | None = None) -> None:
        self._standard = StandardWorkflow(index_manager=index_manager)
        self._advanced = AdvancedWorkflow(standard_workflow=self._standard)
        self._compare = CompareWorkflow(
            standard_workflow=self._standard,
            advanced_workflow=self._advanced,
        )

    def run(
        self,
        query: str,
        mode: Mode,
        chat_history: list[dict[str, str]] | None = None,
    ) -> QueryResponse:
        if mode == Mode.STANDARD:
            return self._standard.run(query=query, chat_history=chat_history)
        if mode == Mode.ADVANCED:
            return self._advanced.run(query=query, chat_history=chat_history)
        if mode == Mode.COMPARE:
            return self._compare.run(query=query, chat_history=chat_history)
        raise NotImplementedError(f"Unsupported mode: {mode}")
