"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
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
    reranker_provider: str = Field(default="cross_encoder", alias="RERANKER_PROVIDER")
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL"
    )
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")
    reranker_batch_size: int = Field(default=8, alias="RERANKER_BATCH_SIZE")
    reranker_top_n: int = Field(default=6, alias="RERANKER_TOP_N")

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

    max_advanced_loops: int = Field(default=1, alias="MAX_ADVANCED_LOOPS")
    memory_window: int = Field(default=3, alias="MEMORY_WINDOW")

    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_embedding_maxsize: int = Field(default=256, alias="CACHE_EMBEDDING_MAXSIZE")
    cache_retrieval_maxsize: int = Field(default=128, alias="CACHE_RETRIEVAL_MAXSIZE")
    cache_llm_maxsize: int = Field(default=64, alias="CACHE_LLM_MAXSIZE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
