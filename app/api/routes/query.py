"""Query routes placeholder for workflow execution."""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.api import QueryRequest, QueryResponse
from app.services import QueryService
from app.services.runtime import get_query_service

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query(payload: QueryRequest, query_service: QueryService = Depends(get_query_service)) -> QueryResponse:
    """Execute selected workflow mode for a user query."""
    try:
        return query_service.run_request(payload)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc
