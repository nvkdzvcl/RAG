"""Configuration and schema smoke tests."""

from datetime import datetime, timezone

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
        "CHUNK_MODE",
        "CHUNK_SIZE",
        "CHUNK_OVERLAP",
        "RETRIEVAL_MODE",
        "RETRIEVAL_TOP_K",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_DEVICE",
        "EMBEDDING_BATCH_SIZE",
        "EMBEDDING_NORMALIZE",
        "EMBEDDING_HASH_DIMENSION",
        "RERANKER_ENABLED",
        "RERANKER_PROVIDER",
        "RERANKER_MODEL",
        "RERANKER_DEVICE",
        "RERANKER_BATCH_SIZE",
        "RERANKER_TOP_K",
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
        "LLM_GATE_MAX_TOKENS",
        "LLM_REWRITE_MAX_TOKENS",
        "LLM_CRITIQUE_MAX_TOKENS",
        "MAX_ADVANCED_LOOPS",
        "MEMORY_WINDOW",
        "GROUNDING_SEMANTIC_ENABLED",
        "GROUNDING_SEMANTIC_MODEL",
        "GROUNDING_SEMANTIC_DEVICE",
        "GROUNDING_SEMANTIC_LOCAL_FILES_ONLY",
        "GROUNDING_SEMANTIC_MAX_CONTEXT_CHUNKS",
        "GROUNDING_SEMANTIC_MIN_SIMILARITY",
        "GROUNDING_SEMANTIC_WEIGHT",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)
    assert settings.app_name == "Self-RAG"
    assert settings.max_advanced_loops == 1
    assert settings.chunk_mode == "preset"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 100
    assert settings.retrieval_mode == "preset"
    assert settings.retrieval_top_k == 8
    assert settings.embedding_provider == "sentence_transformers"
    assert settings.embedding_model == "intfloat/multilingual-e5-base"
    assert settings.embedding_device == "cpu"
    assert settings.embedding_batch_size == 16
    assert settings.embedding_normalize is True
    assert settings.reranker_enabled is True
    assert settings.reranker_provider == "cross_encoder"
    assert settings.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert settings.reranker_device == "cpu"
    assert settings.reranker_batch_size == 8
    assert settings.reranker_top_k == 6
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
    assert settings.llm_gate_max_tokens == 128
    assert settings.llm_rewrite_max_tokens == 256
    assert settings.llm_critique_max_tokens == 384
    assert settings.memory_window == 3


def test_query_request_mode_default() -> None:
    payload = QueryRequest(query="What is Self-RAG?")
    assert payload.mode == Mode.STANDARD
    assert payload.model is None


def test_query_request_accepts_optional_model_override() -> None:
    payload = QueryRequest(query="What is Self-RAG?", model="qwen2.5:7b")
    assert payload.mode == Mode.STANDARD
    assert payload.model == "qwen2.5:7b"


def test_query_request_filters_default_to_none() -> None:
    payload = QueryRequest(query="What is Self-RAG?")
    assert payload.doc_ids is None
    assert payload.filenames is None
    assert payload.file_types is None
    assert payload.uploaded_after is None
    assert payload.uploaded_before is None
    assert payload.include_ocr is None


def test_query_request_accepts_optional_filters_payload() -> None:
    payload = QueryRequest(
        query="What is Self-RAG?",
        doc_ids=["doc_a", "doc_b"],
        filenames=["policy.pdf"],
        file_types=["pdf"],
        uploaded_after=datetime(2026, 1, 1, tzinfo=timezone.utc),
        uploaded_before=datetime(2026, 1, 31, tzinfo=timezone.utc),
        include_ocr=True,
    )
    assert payload.doc_ids == ["doc_a", "doc_b"]
    assert payload.filenames == ["policy.pdf"]
    assert payload.file_types == ["pdf"]
    assert payload.uploaded_after is not None
    assert payload.uploaded_before is not None
    assert payload.include_ocr is True


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


def test_settings_supports_reranker_top_k_and_legacy_top_n(monkeypatch) -> None:
    monkeypatch.setenv("RERANKER_TOP_K", "9")
    monkeypatch.delenv("RERANKER_TOP_N", raising=False)
    settings_top_k = Settings(_env_file=None)
    assert settings_top_k.reranker_top_k == 9
    assert settings_top_k.reranker_top_n == 9

    monkeypatch.delenv("RERANKER_TOP_K", raising=False)
    monkeypatch.setenv("RERANKER_TOP_N", "7")
    settings_top_n = Settings(_env_file=None)
    assert settings_top_n.reranker_top_k == 7
    assert settings_top_n.reranker_top_n == 7
