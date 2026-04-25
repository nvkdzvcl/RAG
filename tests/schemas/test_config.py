"""Configuration and schema smoke tests."""

from app.core.config import Settings
from app.schemas.api import QueryRequest
from app.schemas.common import Mode
from app.schemas.workflow import CritiqueResult, WorkflowState


def test_settings_defaults(monkeypatch) -> None:
    for key in [
        "ENVIRONMENT",
        "LOG_LEVEL",
        "LOG_JSON",
        "OPENAI_API_KEY",
        "DATA_DIR",
        "CORPUS_DIR",
        "INDEX_DIR",
        "PROMPT_DIR",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_DEVICE",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_NORMALIZE",
        "EMBEDDING_HASH_DIMENSION",
        "RERANKER_PROVIDER",
        "RERANKER_MODEL",
        "RERANKER_DEVICE",
        "RERANKER_BATCH_SIZE",
        "RERANKER_TOP_N",
        "OCR_ENABLED",
        "OCR_LANGUAGE",
        "OCR_MIN_TEXT_CHARS",
        "OCR_RENDER_DPI",
        "TESSERACT_CMD",
        "OCR_CONFIDENCE_THRESHOLD",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_API_BASE",
        "LLM_API_KEY",
        "LLM_TEMPERATURE",
        "LLM_MAX_TOKENS",
        "LLM_TIMEOUT_SECONDS",
        "MAX_ADVANCED_LOOPS",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)
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
    assert settings.ocr_enabled is False
    assert settings.ocr_language == "vie+eng"
    assert settings.ocr_min_text_chars == 100
    assert settings.ocr_render_dpi == 216
    assert settings.tesseract_cmd == ""
    assert settings.ocr_confidence_threshold == 40.0
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
    assert payload.model is None


def test_query_request_accepts_optional_model_override() -> None:
    payload = QueryRequest(query="What is Self-RAG?", model="qwen2.5:7b")
    assert payload.mode == Mode.STANDARD
    assert payload.model == "qwen2.5:7b"


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
