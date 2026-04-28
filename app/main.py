"""FastAPI entrypoint for the Self-RAG backend."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.services.runtime import build_app_services


@asynccontextmanager
async def _lifespan(app: FastAPI):
    try:
        yield
    finally:
        await app.state.services.query_service.aclose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=_lifespan,
    )
    app.state.services = build_app_services(settings)
    app.include_router(api_router)
    return app


app = create_app()
