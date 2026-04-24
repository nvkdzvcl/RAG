"""Application settings loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for backend services."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
    embedding_provider: str = Field(default="sentence_transformers", alias="EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="intfloat/multilingual-e5-base", alias="EMBEDDING_MODEL")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE")
    embedding_normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")
    embedding_hash_dimension: int = Field(default=64, alias="EMBEDDING_HASH_DIMENSION")
    reranker_provider: str = Field(default="cross_encoder", alias="RERANKER_PROVIDER")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")
    reranker_batch_size: int = Field(default=8, alias="RERANKER_BATCH_SIZE")
    reranker_top_n: int = Field(default=6, alias="RERANKER_TOP_N")

    max_advanced_loops: int = Field(default=2, alias="MAX_ADVANCED_LOOPS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
