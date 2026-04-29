"""Deterministic cascade reranking policy for CPU-efficient workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from app.schemas.retrieval import RetrievalResult
from app.workflows.query_budget import QueryComplexity

WorkflowMode = Literal["standard", "advanced", "compare"]
RerankerUsed = Literal["score_only", "cross_encoder", "skipped"]


def _normalize_score(value: float | None) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if numeric != numeric:
        return None
    if numeric <= 0.0:
        return 0.0
    if numeric <= 1.0:
        return round(numeric, 4)
    return round(numeric / (1.0 + numeric), 4)


def _top_score(values: list[float | None]) -> float | None:
    normalized: list[float] = []
    for value in values:
        normalized_value = _normalize_score(value)
        if normalized_value is not None:
            normalized.append(normalized_value)
    if not normalized:
        return None
    return max(normalized)


def _top_two_gap(values: list[float | None]) -> float | None:
    normalized: list[float] = []
    for value in values:
        normalized_value = _normalize_score(value)
        if normalized_value is not None:
            normalized.append(normalized_value)
    if len(normalized) < 2:
        return None
    ordered = sorted(normalized, reverse=True)
    return round(max(0.0, ordered[0] - ordered[1]), 4)


@dataclass(frozen=True)
class RerankPolicy:
    """Effective reranking policy and diagnostics for one query."""

    enabled: bool
    mode: WorkflowMode
    query_complexity: QueryComplexity
    candidate_count: int
    top_hybrid_score: float | None
    top_dense_score: float | None
    top_sparse_score: float | None
    top_score_gap: float | None
    use_cross_encoder: bool
    reranker_used: RerankerUsed
    reason: str

    def as_trace_payload(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "query_complexity": self.query_complexity,
            "candidate_count": self.candidate_count,
            "top_hybrid_score": self.top_hybrid_score,
            "top_dense_score": self.top_dense_score,
            "top_sparse_score": self.top_sparse_score,
            "top_score_gap": self.top_score_gap,
            "use_cross_encoder": self.use_cross_encoder,
            "reranker_used": self.reranker_used,
            "reason": self.reason,
        }


def choose_rerank_policy(
    *,
    mode: WorkflowMode,
    query_complexity: QueryComplexity,
    candidates: list[RetrievalResult],
    cascade_enabled: bool,
    simple_skip_cross_encoder: bool,
    min_candidates_for_cross_encoder: int,
    score_gap_threshold: float,
    top_score_threshold: float,
    reranker_supports_cross_encoder: bool,
) -> RerankPolicy:
    """Return cascade policy decision without using LLM calls."""
    candidate_count = len(candidates)
    top_hybrid_score = _top_score([item.score for item in candidates])
    top_dense_score = _top_score([item.dense_score for item in candidates])
    top_sparse_score = _top_score([item.sparse_score for item in candidates])
    top_score_gap = _top_two_gap([item.score for item in candidates])

    if candidate_count == 0:
        return RerankPolicy(
            enabled=cascade_enabled,
            mode=mode,
            query_complexity=query_complexity,
            candidate_count=0,
            top_hybrid_score=top_hybrid_score,
            top_dense_score=top_dense_score,
            top_sparse_score=top_sparse_score,
            top_score_gap=top_score_gap,
            use_cross_encoder=False,
            reranker_used="skipped",
            reason="no_candidates",
        )

    if not reranker_supports_cross_encoder:
        return RerankPolicy(
            enabled=cascade_enabled,
            mode=mode,
            query_complexity=query_complexity,
            candidate_count=candidate_count,
            top_hybrid_score=top_hybrid_score,
            top_dense_score=top_dense_score,
            top_sparse_score=top_sparse_score,
            top_score_gap=top_score_gap,
            use_cross_encoder=False,
            reranker_used="score_only",
            reason="configured_non_cross_encoder",
        )

    if not cascade_enabled:
        return RerankPolicy(
            enabled=False,
            mode=mode,
            query_complexity=query_complexity,
            candidate_count=candidate_count,
            top_hybrid_score=top_hybrid_score,
            top_dense_score=top_dense_score,
            top_sparse_score=top_sparse_score,
            top_score_gap=top_score_gap,
            use_cross_encoder=True,
            reranker_used="cross_encoder",
            reason="cascade_disabled_legacy",
        )

    if simple_skip_cross_encoder and query_complexity == "simple_extractive":
        return RerankPolicy(
            enabled=True,
            mode=mode,
            query_complexity=query_complexity,
            candidate_count=candidate_count,
            top_hybrid_score=top_hybrid_score,
            top_dense_score=top_dense_score,
            top_sparse_score=top_sparse_score,
            top_score_gap=top_score_gap,
            use_cross_encoder=False,
            reranker_used="score_only",
            reason="simple_extractive_skip",
        )

    if candidate_count < max(1, int(min_candidates_for_cross_encoder)):
        return RerankPolicy(
            enabled=True,
            mode=mode,
            query_complexity=query_complexity,
            candidate_count=candidate_count,
            top_hybrid_score=top_hybrid_score,
            top_dense_score=top_dense_score,
            top_sparse_score=top_sparse_score,
            top_score_gap=top_score_gap,
            use_cross_encoder=False,
            reranker_used="score_only",
            reason="few_candidates_skip",
        )

    high_top_score = top_hybrid_score is not None and top_hybrid_score >= max(
        0.0, top_score_threshold
    )
    clear_gap = top_score_gap is not None and top_score_gap >= max(
        0.0, score_gap_threshold
    )
    if high_top_score and clear_gap:
        return RerankPolicy(
            enabled=True,
            mode=mode,
            query_complexity=query_complexity,
            candidate_count=candidate_count,
            top_hybrid_score=top_hybrid_score,
            top_dense_score=top_dense_score,
            top_sparse_score=top_sparse_score,
            top_score_gap=top_score_gap,
            use_cross_encoder=False,
            reranker_used="score_only",
            reason="high_confidence_clear_gap_skip",
        )

    if mode == "advanced":
        return RerankPolicy(
            enabled=True,
            mode=mode,
            query_complexity=query_complexity,
            candidate_count=candidate_count,
            top_hybrid_score=top_hybrid_score,
            top_dense_score=top_dense_score,
            top_sparse_score=top_sparse_score,
            top_score_gap=top_score_gap,
            use_cross_encoder=True,
            reranker_used="cross_encoder",
            reason="advanced_strict_allow_cross_encoder",
        )

    ambiguous_scores = top_score_gap is None or top_score_gap < max(
        0.0, score_gap_threshold
    )
    if query_complexity in {"normal", "complex"} and ambiguous_scores:
        return RerankPolicy(
            enabled=True,
            mode=mode,
            query_complexity=query_complexity,
            candidate_count=candidate_count,
            top_hybrid_score=top_hybrid_score,
            top_dense_score=top_dense_score,
            top_sparse_score=top_sparse_score,
            top_score_gap=top_score_gap,
            use_cross_encoder=True,
            reranker_used="cross_encoder",
            reason="ambiguous_scores_use_cross_encoder",
        )

    return RerankPolicy(
        enabled=True,
        mode=mode,
        query_complexity=query_complexity,
        candidate_count=candidate_count,
        top_hybrid_score=top_hybrid_score,
        top_dense_score=top_dense_score,
        top_sparse_score=top_sparse_score,
        top_score_gap=top_score_gap,
        use_cross_encoder=False,
        reranker_used="score_only",
        reason="score_only_default",
    )
