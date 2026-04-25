"""Compare workflow implementation."""

from __future__ import annotations

import time

from app.schemas.api import AdvancedQueryResponse, CompareQueryResponse, ComparisonSummary, StandardQueryResponse
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.shared import detect_response_language
from app.workflows.standard import StandardWorkflow


class CompareWorkflow:
    """Run standard and advanced workflows and return a comparison payload."""

    def __init__(
        self,
        *,
        standard_workflow: StandardWorkflow,
        advanced_workflow: AdvancedWorkflow,
    ) -> None:
        self.standard_workflow = standard_workflow
        self.advanced_workflow = advanced_workflow

    def _build_summary(
        self,
        standard: StandardQueryResponse,
        advanced: AdvancedQueryResponse,
        *,
        total_latency_ms: int,
        response_language: str,
    ) -> ComparisonSummary:
        confidence_delta = None
        if standard.confidence is not None and advanced.confidence is not None:
            confidence_delta = advanced.confidence - standard.confidence

        latency_delta_ms = None
        if standard.latency_ms is not None and advanced.latency_ms is not None:
            latency_delta_ms = advanced.latency_ms - standard.latency_ms

        standard_citation_count = standard.citation_count if standard.citation_count > 0 else len(standard.citations)
        advanced_citation_count = advanced.citation_count if advanced.citation_count > 0 else len(advanced.citations)
        citation_delta = advanced_citation_count - standard_citation_count

        standard_grounded = standard.grounded_score
        advanced_grounded = advanced.grounded_score
        grounded_score_delta = advanced_grounded - standard_grounded

        def _branch_score(
            *,
            grounded_score: float,
            confidence: float | None,
            citation_count: int,
            hallucination: bool,
            language_mismatch: bool,
            llm_fallback_used: bool,
        ) -> float:
            score = (grounded_score * 0.7) + ((confidence or 0.0) * 0.12) + (min(citation_count, 4) * 0.06)
            if citation_count == 0:
                score -= 0.42
            if hallucination:
                score -= 0.5
            if language_mismatch:
                score -= 0.45
            if llm_fallback_used:
                score -= 0.25
            return round(score, 4)

        standard_score = _branch_score(
            grounded_score=standard_grounded,
            confidence=standard.confidence,
            citation_count=standard_citation_count,
            hallucination=standard.hallucination_detected,
            language_mismatch=standard.language_mismatch,
            llm_fallback_used=standard.llm_fallback_used,
        )
        advanced_score = _branch_score(
            grounded_score=advanced_grounded,
            confidence=advanced.confidence,
            citation_count=advanced_citation_count,
            hallucination=advanced.hallucination_detected,
            language_mismatch=advanced.language_mismatch,
            llm_fallback_used=advanced.llm_fallback_used,
        )

        standard_strong = (
            standard_citation_count > 0
            and not standard.hallucination_detected
            and not standard.language_mismatch
            and not standard.llm_fallback_used
            and standard_grounded >= 0.16
        )
        advanced_strong = (
            advanced_citation_count > 0
            and not advanced.hallucination_detected
            and not advanced.language_mismatch
            and not advanced.llm_fallback_used
            and advanced_grounded >= 0.16
        )

        standard_weak = not standard_strong
        advanced_weak = not advanced_strong

        if standard_weak and advanced_weak:
            preferred_mode = "review"
        elif standard_citation_count > 0 and advanced_citation_count == 0:
            preferred_mode = "standard"
        elif advanced_strong and advanced_score > standard_score + 0.02:
            preferred_mode = "advanced"
        else:
            preferred_mode = "standard"

        if response_language == "vi":
            if preferred_mode == "advanced":
                note = "Nâng cao đáng tin cậy hơn"
            elif preferred_mode == "standard":
                note = "Chuẩn đáng tin cậy hơn"
            else:
                note = "Cả hai cần kiểm tra lại"
        else:
            if preferred_mode == "advanced":
                note = "Advanced is more reliable"
            elif preferred_mode == "standard":
                note = "Standard is more reliable"
            else:
                note = "Both need manual review"

        return ComparisonSummary(
            confidence_delta=confidence_delta,
            latency_delta_ms=latency_delta_ms,
            citation_delta=citation_delta,
            grounded_score_delta=grounded_score_delta,
            preferred_mode=preferred_mode,
            note=note,
        )

    def run(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
    ) -> CompareQueryResponse:
        started = time.perf_counter()
        resolved_language = response_language or detect_response_language(query)

        standard = self.standard_workflow.run(
            query=query,
            chat_history=chat_history,
            model=model,
            response_language=resolved_language,
        )
        advanced = self.advanced_workflow.run(
            query=query,
            chat_history=chat_history,
            model=model,
            response_language=resolved_language,
        )

        total_latency_ms = int((time.perf_counter() - started) * 1000)
        summary = self._build_summary(
            standard=standard,
            advanced=advanced,
            total_latency_ms=total_latency_ms,
            response_language=resolved_language,
        )

        return CompareQueryResponse(
            mode="compare",
            standard=standard,
            advanced=advanced,
            comparison=summary,
        )
