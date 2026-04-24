"""Configuration and schema smoke tests."""

from app.core.config import Settings
from app.schemas.api import QueryRequest
from app.schemas.common import Mode
from app.schemas.workflow import CritiqueResult, WorkflowState


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "Self-RAG"
    assert settings.max_advanced_loops == 2


def test_query_request_mode_default() -> None:
    payload = QueryRequest(query="What is Self-RAG?")
    assert payload.mode == Mode.STANDARD


def test_workflow_state_schema() -> None:
    state = WorkflowState(
        mode=Mode.STANDARD,
        user_query="q",
        normalized_query="q",
    )
    assert state.need_retrieval is True
    assert state.loop_count == 0


def test_critique_schema() -> None:
    critique = CritiqueResult(
        grounded=True,
        enough_evidence=True,
        has_conflict=False,
        should_retry_retrieval=False,
        should_refine_answer=False,
        confidence=0.8,
        note="ok",
    )
    assert critique.grounded is True
