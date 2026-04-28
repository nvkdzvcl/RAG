"""API request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, Field, TypeAdapter

from app.schemas.common import Citation, Mode


class QueryRequest(BaseModel):
    """Incoming query payload."""

    query: str = Field(min_length=1)
    mode: Mode = Mode.STANDARD
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    model: str | None = Field(default=None, min_length=1)
    doc_ids: list[str] | None = None
    filenames: list[str] | None = None
    file_types: list[str] | None = None
    uploaded_after: datetime | None = None
    uploaded_before: datetime | None = None
    include_ocr: bool | None = None


class ModeResult(BaseModel):
    """Shared answer shape used by standard and advanced outputs."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None
    stop_reason: str | None = None
    status: str = "answered"
    latency_ms: int | None = None
    loop_count: int | None = None
    response_language: str = "en"
    language_mismatch: bool = False
    grounded_score: float = 0.0
    grounding_reason: str = "not_evaluated"
    citation_count: int = 0
    hallucination_detected: bool = False
    llm_fallback_used: bool = False
    trace: list[dict[str, Any]] = Field(default_factory=list)


class StandardQueryResponse(ModeResult):
    """Response payload for standard mode."""

    mode: Literal["standard"] = "standard"


class AdvancedQueryResponse(ModeResult):
    """Response payload for advanced mode."""

    mode: Literal["advanced"] = "advanced"


class ComparisonSummary(BaseModel):
    """Aggregated metrics for compare mode output."""

    winner: Literal["standard", "advanced", "tie", "both_weak"] | None = None
    reasons: list[str] = Field(default_factory=list)
    standard_score: float | None = None
    advanced_score: float | None = None
    confidence_delta: float | None = None
    latency_delta_ms: int | None = None
    citation_delta: int | None = None
    grounded_score_delta: float | None = None
    preferred_mode: str | None = None
    note: str | None = None


class CompareQueryResponse(BaseModel):
    """Combined response payload for compare mode."""

    mode: Literal["compare"] = "compare"
    standard: StandardQueryResponse
    advanced: AdvancedQueryResponse
    comparison: ComparisonSummary


QueryResponse: TypeAlias = Annotated[
    StandardQueryResponse | AdvancedQueryResponse | CompareQueryResponse,
    Field(discriminator="mode"),
]

QueryResponseAdapter: TypeAdapter[QueryResponse] = TypeAdapter(QueryResponse)


def validate_query_response(payload: dict[str, Any]) -> QueryResponse:
    """Validate a raw payload against the API response union schema."""
    return QueryResponseAdapter.validate_python(payload)
