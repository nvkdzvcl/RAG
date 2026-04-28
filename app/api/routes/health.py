"""Health and readiness routes."""

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe endpoint."""
    settings = get_settings()
    return {
        "status": "ok",
        "llm_model": settings.llm_model,
    }
