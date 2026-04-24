"""Workflow state and critique schemas."""

from pydantic import BaseModel, Field

from app.schemas.common import Citation, Mode


class CritiqueResult(BaseModel):
    """Structured critique output for advanced mode decisions."""

    grounded: bool
    enough_evidence: bool
    has_conflict: bool
    missing_aspects: list[str] = Field(default_factory=list)
    should_retry_retrieval: bool
    should_refine_answer: bool
    better_queries: list[str] = Field(default_factory=list)
    confidence: float
    note: str


class WorkflowState(BaseModel):
    """Shared state container across workflow orchestration."""

    mode: Mode
    user_query: str
    normalized_query: str
    chat_history: list[dict[str, str]] = Field(default_factory=list)
    need_retrieval: bool = True

    rewritten_queries: list[str] = Field(default_factory=list)
    retrieved_docs: list[dict] = Field(default_factory=list)
    reranked_docs: list[dict] = Field(default_factory=list)
    selected_context: list[dict] = Field(default_factory=list)

    draft_answer: str | None = None
    final_answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)

    critique: CritiqueResult | None = None
    confidence: float | None = None

    loop_count: int = 0
    stop_reason: str | None = None
