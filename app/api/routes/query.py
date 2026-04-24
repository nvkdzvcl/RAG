"""Query routes placeholder for workflow execution."""

from fastapi import APIRouter, HTTPException, status

from app.schemas.api import QueryRequest, QueryResponse
from app.services import QueryService

router = APIRouter(prefix="/query", tags=["query"])
query_service = QueryService()


@router.post("", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    """Execute selected workflow mode for a user query."""
    try:
        return query_service.run_request(payload)
    except NotImplementedError as exc:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(exc),
        ) from exc
