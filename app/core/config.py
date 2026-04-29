"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for backend services."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Self-RAG"
    app_version: str = "0.1.0"
    environment: str = Field(default="dev", alias="ENVIRONMENT")

    api_prefix: str = "/api/v1"

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=False, alias="LOG_JSON")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    data_dir: str = Field(default="data", alias="DATA_DIR")
    corpus_dir: str = Field(default="docs", alias="CORPUS_DIR")
    index_dir: str = Field(default="data/indexes", alias="INDEX_DIR")
    prompt_dir: str = Field(default="prompts", alias="PROMPT_DIR")
    chunk_mode: str = Field(default="preset", alias="CHUNK_MODE")
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, alias="CHUNK_OVERLAP")
    retrieval_mode: str = Field(default="preset", alias="RETRIEVAL_MODE")
    retrieval_top_k: int = Field(default=8, alias="RETRIEVAL_TOP_K")
    embedding_provider: str = Field(
        default="sentence_transformers", alias="EMBEDDING_PROVIDER"
    )
    embedding_model: str = Field(
        default="intfloat/multilingual-e5-base", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE")
    embedding_normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")
    embedding_hash_dimension: int = Field(default=64, alias="EMBEDDING_HASH_DIMENSION")
    reranker_enabled: bool = Field(default=True, alias="RERANKER_ENABLED")
    reranker_provider: str = Field(default="cross_encoder", alias="RERANKER_PROVIDER")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL"
    )
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")
    reranker_batch_size: int = Field(default=8, alias="RERANKER_BATCH_SIZE")
    reranker_top_k: int = Field(default=6, alias="RERANKER_TOP_K")
    reranker_top_n: int = Field(default=6, alias="RERANKER_TOP_N")
    rerank_cascade_enabled: bool = Field(default=True, alias="RERANK_CASCADE_ENABLED")
    rerank_simple_skip_cross_encoder: bool = Field(
        default=True, alias="RERANK_SIMPLE_SKIP_CROSS_ENCODER"
    )
    rerank_min_candidates_for_cross_encoder: int = Field(
        default=4, alias="RERANK_MIN_CANDIDATES_FOR_CROSS_ENCODER"
    )
    rerank_score_gap_threshold: float = Field(
        default=0.2, alias="RERANK_SCORE_GAP_THRESHOLD"
    )

    ocr_enabled: bool = Field(default=False, alias="OCR_ENABLED")
    ocr_language: str = Field(default="vie+eng", alias="OCR_LANGUAGE")
    ocr_min_text_chars: int = Field(default=100, alias="OCR_MIN_TEXT_CHARS")
    ocr_render_dpi: int = Field(default=216, alias="OCR_RENDER_DPI")
    tesseract_cmd: str = Field(default="", alias="TESSERACT_CMD")
    ocr_confidence_threshold: float = Field(
        default=40.0, alias="OCR_CONFIDENCE_THRESHOLD"
    )

    llm_provider: str = Field(default="stub", alias="LLM_PROVIDER")
    llm_model: str = Field(default="qwen2.5:3b", alias="LLM_MODEL")
    llm_api_base: str = Field(default="http://localhost:11434/v1", alias="LLM_API_BASE")
    llm_api_key: str | None = Field(default="ollama", alias="LLM_API_KEY")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    llm_timeout_seconds: int = Field(default=120, alias="LLM_TIMEOUT_SECONDS")
    llm_gate_max_tokens: int = Field(default=128, alias="LLM_GATE_MAX_TOKENS")
    llm_rewrite_max_tokens: int = Field(default=256, alias="LLM_REWRITE_MAX_TOKENS")
    llm_critique_max_tokens: int = Field(default=384, alias="LLM_CRITIQUE_MAX_TOKENS")
    rag_dynamic_budget_enabled: bool = Field(
        default=True, alias="RAG_DYNAMIC_BUDGET_ENABLED"
    )
    rag_simple_max_tokens: int = Field(default=384, alias="RAG_SIMPLE_MAX_TOKENS")
    rag_normal_max_tokens: int = Field(default=768, alias="RAG_NORMAL_MAX_TOKENS")
    rag_complex_max_tokens: int = Field(default=1536, alias="RAG_COMPLEX_MAX_TOKENS")
    rag_simple_context_chars: int = Field(
        default=1600, alias="RAG_SIMPLE_CONTEXT_CHARS"
    )
    rag_normal_context_chars: int = Field(
        default=3000, alias="RAG_NORMAL_CONTEXT_CHARS"
    )
    grounding_policy: str = Field(default="adaptive", alias="GROUNDING_POLICY")
    grounding_semantic_standard_enabled: bool = Field(
        default=False, alias="GROUNDING_SEMANTIC_STANDARD_ENABLED"
    )
    grounding_semantic_advanced_enabled: bool = Field(
        default=True, alias="GROUNDING_SEMANTIC_ADVANCED_ENABLED"
    )

    max_advanced_loops: int = Field(default=1, alias="MAX_ADVANCED_LOOPS")
    advanced_adaptive_enabled: bool = Field(
        default=True, alias="ADVANCED_ADAPTIVE_ENABLED"
    )
    advanced_force_llm_gate: bool = Field(
        default=False, alias="ADVANCED_FORCE_LLM_GATE"
    )
    advanced_force_llm_critic: bool = Field(
        default=False, alias="ADVANCED_FORCE_LLM_CRITIC"
    )
    memory_window: int = Field(default=3, alias="MEMORY_WINDOW")

    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    embedding_cache_enabled: bool = Field(default=True, alias="EMBEDDING_CACHE_ENABLED")
    retrieval_cache_enabled: bool = Field(default=True, alias="RETRIEVAL_CACHE_ENABLED")
    llm_cache_enabled: bool = Field(default=True, alias="LLM_CACHE_ENABLED")
    rerank_cache_enabled: bool = Field(default=True, alias="RERANK_CACHE_ENABLED")
    cache_embedding_maxsize: int = Field(default=256, alias="CACHE_EMBEDDING_MAXSIZE")
    cache_retrieval_maxsize: int = Field(default=128, alias="CACHE_RETRIEVAL_MAXSIZE")
    cache_llm_maxsize: int = Field(default=64, alias="CACHE_LLM_MAXSIZE")
    cache_rerank_maxsize: int = Field(default=128, alias="CACHE_RERANK_MAXSIZE")

    @model_validator(mode="after")
    def _sync_reranker_top_k_legacy_alias(self) -> "Settings":
        """Keep `RERANKER_TOP_K` and legacy `RERANKER_TOP_N` in sync."""
        fields_set = self.model_fields_set
        has_top_k = "reranker_top_k" in fields_set
        has_top_n = "reranker_top_n" in fields_set

        if has_top_k and not has_top_n:
            self.reranker_top_n = int(self.reranker_top_k)
        elif has_top_n and not has_top_k:
            self.reranker_top_k = int(self.reranker_top_n)
        elif has_top_k and has_top_n:
            # Prefer the new top_k field when both env vars are explicitly set.
            self.reranker_top_n = int(self.reranker_top_k)

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
