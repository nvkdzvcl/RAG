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
    index_dir: str = Field(default="data/indexes", alias="INDEX_DIR")
    prompt_dir: str = Field(default="prompts", alias="PROMPT_DIR")

    max_advanced_loops: int = Field(default=2, alias="MAX_ADVANCED_LOOPS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
