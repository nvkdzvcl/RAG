"""Compare workflow implementation."""

from __future__ import annotations

import time

from app.schemas.api import AdvancedQueryResponse, CompareQueryResponse, ComparisonSummary, StandardQueryResponse
from app.workflows.advanced import AdvancedWorkflow
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
    ) -> ComparisonSummary:
        confidence_delta = None
        if standard.confidence is not None and advanced.confidence is not None:
            confidence_delta = advanced.confidence - standard.confidence

        latency_delta_ms = None
        if standard.latency_ms is not None and advanced.latency_ms is not None:
            latency_delta_ms = advanced.latency_ms - standard.latency_ms

        citation_delta = len(advanced.citations) - len(standard.citations)

        note = "Comparison completed."
        if confidence_delta is not None:
            if confidence_delta > 0:
                note = "Advanced mode reports higher confidence."
            elif confidence_delta < 0:
                note = "Standard mode reports higher confidence."
            else:
                note = "Both modes report equal confidence."

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
    ) -> CompareQueryResponse:
        started = time.perf_counter()

        standard = self.standard_workflow.run(
            query=query,
            chat_history=chat_history,
            model=model,
        )
        advanced = self.advanced_workflow.run(
            query=query,
            chat_history=chat_history,
            model=model,
        )

        total_latency_ms = int((time.perf_counter() - started) * 1000)
        summary = self._build_summary(
            standard=standard,
            advanced=advanced,
            total_latency_ms=total_latency_ms,
        )

        return CompareQueryResponse(
            mode="compare",
            standard=standard,
            advanced=advanced,
            comparison=summary,
        )
