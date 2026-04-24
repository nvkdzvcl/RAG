"""Health and readiness routes."""

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe endpoint."""
    return {"status": "ok"}
