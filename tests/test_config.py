"""Configuration and schema smoke tests."""

from app.core.config import Settings
from app.schemas.api import QueryRequest
from app.schemas.common import Mode
from app.schemas.workflow import CritiqueResult, WorkflowState


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "Self-RAG"
    assert settings.max_advanced_loops == 2
    assert settings.embedding_provider == "sentence_transformers"
    assert settings.embedding_model == "intfloat/multilingual-e5-base"
    assert settings.embedding_device == "cpu"
    assert settings.embedding_batch_size == 16
    assert settings.embedding_normalize is True
    assert settings.reranker_provider == "cross_encoder"
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.reranker_device == "cpu"
    assert settings.reranker_batch_size == 8
    assert settings.reranker_top_n == 6
    assert settings.llm_provider == "stub"
    assert settings.llm_model == "qwen2.5:3b"
    assert settings.llm_api_base == "http://localhost:11434/v1"
    assert settings.llm_api_key == "ollama"
    assert settings.llm_temperature == 0.2
    assert settings.llm_max_tokens == 2048
    assert settings.llm_timeout_seconds == 120


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
