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

        citation_delta = len(advanced.citations) - len(standard.citations)

        if response_language == "vi":
            note = "Đã hoàn tất so sánh."
        else:
            note = "Comparison completed."
        if confidence_delta is not None:
            if confidence_delta > 0:
                note = (
                    "Chế độ nâng cao cho thấy độ tin cậy cao hơn."
                    if response_language == "vi"
                    else "Advanced mode reports higher confidence."
                )
            elif confidence_delta < 0:
                note = (
                    "Chế độ chuẩn cho thấy độ tin cậy cao hơn."
                    if response_language == "vi"
                    else "Standard mode reports higher confidence."
                )
            else:
                note = (
                    "Hai chế độ cho độ tin cậy tương đương."
                    if response_language == "vi"
                    else "Both modes report equal confidence."
                )

        if response_language == "vi":
            note = f"{note} tổng_thời_gian_ms={total_latency_ms}"
        else:
            note = f"{note} total_latency_ms={total_latency_ms}"

        return ComparisonSummary(
            confidence_delta=confidence_delta,
            latency_delta_ms=latency_delta_ms,
            citation_delta=citation_delta,
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
