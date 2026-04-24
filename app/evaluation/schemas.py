"""Typed schemas for evaluation dataset, outputs, and reports."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import Citation, Mode

EvalExpectedBehavior = Literal["answer", "abstain", "retry"]
EvalCategory = Literal[
    "simple",
    "multi_hop",
    "ambiguous",
    "insufficient_context",
    "conflicting_sources",
    "vietnamese",
]


class EvalExample(BaseModel):
    """One golden evaluation item."""

    id: str
    question: str
    expected_behavior: EvalExpectedBehavior
    reference_answer: str | None = None
    gold_sources: list[str] = Field(default_factory=list)
    category: EvalCategory
    notes: str | None = None


class TraceExtraction(BaseModel):
    """Retrieval/rerank trace fields extracted from mode trace."""

    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    rerank_scores: dict[str, float] = Field(default_factory=dict)
    retrieved_count: int = 0
    selected_context_count: int = 0
    selected_context_texts: list[str] = Field(default_factory=list)


class EvalMetrics(BaseModel):
    """Lightweight and heuristic metrics for one mode output."""

    citation_count: int
    has_citations: bool
    abstain_match: bool
    retry_used: bool
    latency_ms: int | None
    confidence: float | None
    retrieved_count: int
    selected_context_count: int

    answer_non_empty: bool
    answer_contains_reference_keywords: bool | None = None
    cited_gold_source_overlap: float | None = None
    groundedness_proxy: float | None = None
    groundedness_proxy_note: str | None = None


class ModeEvalOutput(BaseModel):
    """Per-question output for one mode branch."""

    example_id: str
    mode: Mode
    question: str
    category: EvalCategory
    expected_behavior: EvalExpectedBehavior

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None
    status: str | None = None
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    rerank_scores: dict[str, float] = Field(default_factory=dict)
    loop_count: int | None = None
    stop_reason: str | None = None
    latency_ms: int | None = None
    retrieved_count: int = 0
    selected_context_count: int = 0
    metrics: EvalMetrics
    trace: list[dict] = Field(default_factory=list)
    run_source: Literal["direct", "compare_branch"] = "direct"
    error: str | None = None


class CompareEvalOutput(BaseModel):
    """Per-question output for compare mode."""

    example_id: str
    mode: Literal["compare"] = "compare"
    question: str
    category: EvalCategory
    expected_behavior: EvalExpectedBehavior
    standard: ModeEvalOutput
    advanced: ModeEvalOutput
    comparison: dict = Field(default_factory=dict)


class CategorySummary(BaseModel):
    """Aggregated metrics grouped by mode and category."""

    mode: Mode
    category: EvalCategory
    count: int
    avg_latency_ms: float | None = None
    avg_confidence: float | None = None
    citation_rate: float
    abstain_rate: float
    retry_rate: float


class ComparativeSummary(BaseModel):
    """Cross-mode summary used in markdown report."""

    paired_count: int
    avg_latency_delta_ms: float | None = None
    avg_confidence_delta: float | None = None
    advanced_retry_rate: float
    abstain_rate_by_mode: dict[str, float] = Field(default_factory=dict)
    citation_rate_by_mode: dict[str, float] = Field(default_factory=dict)
    per_category: list[CategorySummary] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Persisted evaluation report."""

    dataset_path: str
    generated_at: datetime
    modes: list[Mode] = Field(default_factory=list)
    dataset_size: int
    output_count: int
    standard_advanced_summary: ComparativeSummary
    mode_outputs: list[ModeEvalOutput] = Field(default_factory=list)
    compare_outputs: list[CompareEvalOutput] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
