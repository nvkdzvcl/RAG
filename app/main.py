"""FastAPI entrypoint for the Self-RAG backend."""

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
    )
    app.include_router(api_router)
    return app


app = create_app()
