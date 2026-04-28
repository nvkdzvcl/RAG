"""Compare workflow implementation."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal

from app.core.async_utils import run_coro_sync
from app.schemas.api import (
    AdvancedQueryResponse,
    CompareQueryResponse,
    ComparisonSummary,
    StandardQueryResponse,
)
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.shared import detect_response_language
from app.workflows.streaming import StreamEventHandler, emit_stream_event
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

        standard_citation_count = (
            standard.citation_count
            if standard.citation_count > 0
            else len(standard.citations)
        )
        advanced_citation_count = (
            advanced.citation_count
            if advanced.citation_count > 0
            else len(advanced.citations)
        )
        citation_delta = advanced_citation_count - standard_citation_count

        standard_grounded = standard.grounded_score
        advanced_grounded = advanced.grounded_score
        grounded_score_delta = advanced_grounded - standard_grounded

        def _normalized_status(value: str | None) -> str:
            return (value or "").strip().lower()

        def _branch_score(
            *,
            status: str,
            grounded_score: float,
            confidence: float | None,
            citation_count: int,
            hallucination: bool,
            language_mismatch: bool,
            llm_fallback_used: bool,
        ) -> float:
            # Reliability-first scoring. Confidence is intentionally secondary.
            score = 0.0
            score += min(citation_count, 4) * 1.8
            score += max(0.0, min(grounded_score, 1.0)) * 3.2

            if status == "answered":
                score += 1.4
            elif status == "partial":
                score += 0.5
            elif status == "insufficient_evidence":
                score -= 1.4

            if hallucination:
                score -= 3.0
            else:
                score += 0.9

            if language_mismatch:
                score -= 1.2
            else:
                score += 0.35

            if llm_fallback_used:
                score -= 0.9
            else:
                score += 0.25

            if citation_count == 0:
                score -= 0.8

            score += max(0.0, min(confidence or 0.0, 1.0)) * 0.2
            return round(score, 4)

        standard_status = _normalized_status(standard.status)
        advanced_status = _normalized_status(advanced.status)

        standard_score = _branch_score(
            status=standard_status,
            grounded_score=standard_grounded,
            confidence=standard.confidence,
            citation_count=standard_citation_count,
            hallucination=standard.hallucination_detected,
            language_mismatch=standard.language_mismatch,
            llm_fallback_used=standard.llm_fallback_used,
        )
        advanced_score = _branch_score(
            status=advanced_status,
            grounded_score=advanced_grounded,
            confidence=advanced.confidence,
            citation_count=advanced_citation_count,
            hallucination=advanced.hallucination_detected,
            language_mismatch=advanced.language_mismatch,
            llm_fallback_used=advanced.llm_fallback_used,
        )

        standard_weak = (
            standard_status == "insufficient_evidence"
            or standard_citation_count == 0
            or standard.hallucination_detected
            or standard.language_mismatch
            or standard.llm_fallback_used
            or standard_grounded < 0.08
        )
        advanced_weak = (
            advanced_status == "insufficient_evidence"
            or advanced_citation_count == 0
            or advanced.hallucination_detected
            or advanced.language_mismatch
            or advanced.llm_fallback_used
            or advanced_grounded < 0.08
        )

        winner: Literal["standard", "advanced", "tie", "both_weak"] = "tie"
        reasons: list[str] = []

        if (
            standard_status == "insufficient_evidence"
            and advanced_status != "insufficient_evidence"
            and advanced_citation_count > 0
        ):
            winner = "advanced"
            if response_language == "vi":
                reasons.append(
                    "Nâng cao có trích dẫn trong khi Chuẩn trả về thiếu bằng chứng."
                )
            else:
                reasons.append(
                    "Advanced has citations while Standard returned insufficient evidence."
                )
        elif (
            advanced_status == "insufficient_evidence"
            and standard_status != "insufficient_evidence"
            and standard_citation_count > 0
        ):
            winner = "standard"
            if response_language == "vi":
                reasons.append(
                    "Chuẩn có trích dẫn trong khi Nâng cao trả về thiếu bằng chứng."
                )
            else:
                reasons.append(
                    "Standard has citations while Advanced returned insufficient evidence."
                )
        elif standard.hallucination_detected and not advanced.hallucination_detected:
            winner = "advanced"
            if response_language == "vi":
                reasons.append("Chuẩn có dấu hiệu suy diễn không được hỗ trợ.")
            else:
                reasons.append("Standard shows hallucination risk.")
        elif advanced.hallucination_detected and not standard.hallucination_detected:
            winner = "standard"
            if response_language == "vi":
                reasons.append("Nâng cao có dấu hiệu suy diễn không được hỗ trợ.")
            else:
                reasons.append("Advanced shows hallucination risk.")
        elif standard_weak and advanced_weak:
            winner = "both_weak"
            if response_language == "vi":
                reasons.append("Cả hai nhánh đều thiếu tín hiệu độ tin cậy mạnh.")
            else:
                reasons.append("Both branches lack strong reliability signals.")
        else:
            # Advanced with zero citations cannot win by confidence alone.
            if (
                advanced_citation_count == 0
                and standard_citation_count > 0
                and advanced_score > standard_score
            ):
                winner = "standard"
                if response_language == "vi":
                    reasons.append(
                        "Nâng cao không có trích dẫn nên không thể thắng chỉ nhờ độ tự tin."
                    )
                else:
                    reasons.append(
                        "Advanced cannot win on confidence alone when it has zero citations."
                    )
            else:
                score_gap = standard_score - advanced_score
                if abs(score_gap) <= 0.35:
                    winner = "tie"
                    if response_language == "vi":
                        reasons.append("Hai nhánh có điểm độ tin cậy tương đương.")
                    else:
                        reasons.append("Both branches have similar reliability scores.")
                elif score_gap > 0:
                    winner = "standard"
                else:
                    winner = "advanced"

        if winner in {"standard", "advanced"} and not reasons:
            if winner == "standard":
                if response_language == "vi":
                    if (
                        standard_citation_count > advanced_citation_count
                        and standard_grounded >= advanced_grounded
                    ):
                        reasons.append(
                            "Chuẩn có nhiều trích dẫn và độ bám tài liệu cao hơn."
                        )
                    elif standard_citation_count > advanced_citation_count:
                        reasons.append("Chuẩn có nhiều trích dẫn hơn.")
                    else:
                        reasons.append(
                            "Chuẩn ổn định hơn về groundedness và tín hiệu an toàn."
                        )
                else:
                    if (
                        standard_citation_count > advanced_citation_count
                        and standard_grounded >= advanced_grounded
                    ):
                        reasons.append(
                            "Standard has more citations and higher groundedness."
                        )
                    elif standard_citation_count > advanced_citation_count:
                        reasons.append("Standard has more citations.")
                    else:
                        reasons.append(
                            "Standard is more stable on groundedness and safety signals."
                        )
            else:
                if response_language == "vi":
                    if (
                        advanced_citation_count > standard_citation_count
                        and advanced_grounded >= standard_grounded
                    ):
                        reasons.append(
                            "Nâng cao có nhiều trích dẫn và độ bám tài liệu cao hơn."
                        )
                    elif advanced_citation_count > standard_citation_count:
                        reasons.append("Nâng cao có nhiều trích dẫn hơn.")
                    else:
                        reasons.append(
                            "Nâng cao ổn định hơn về groundedness và tín hiệu an toàn."
                        )
                else:
                    if (
                        advanced_citation_count > standard_citation_count
                        and advanced_grounded >= standard_grounded
                    ):
                        reasons.append(
                            "Advanced has more citations and higher groundedness."
                        )
                    elif advanced_citation_count > standard_citation_count:
                        reasons.append("Advanced has more citations.")
                    else:
                        reasons.append(
                            "Advanced is more stable on groundedness and safety signals."
                        )

        preferred_mode = winner if winner in {"standard", "advanced"} else "review"

        if response_language == "vi":
            if winner == "advanced":
                note = "Nâng cao đáng tin cậy hơn vì có trích dẫn và độ bám tài liệu cao hơn"
            elif winner == "standard":
                note = (
                    "Chuẩn đáng tin cậy hơn vì có trích dẫn và độ bám tài liệu cao hơn"
                )
            elif winner == "both_weak":
                note = "Cả hai cần kiểm tra lại vì thiếu bằng chứng đủ mạnh"
            else:
                note = "Hai chế độ có độ tin cậy tương đương, cần kiểm tra thêm theo ngữ cảnh"
        else:
            if winner == "advanced":
                note = "Advanced is more reliable due to stronger citations and groundedness."
            elif winner == "standard":
                note = "Standard is more reliable due to stronger citations and groundedness."
            elif winner == "both_weak":
                note = "Both branches need review due to weak evidence."
            else:
                note = "Both branches are similarly reliable; review context to choose."

        return ComparisonSummary(
            winner=winner,
            reasons=reasons,
            standard_score=standard_score,
            advanced_score=advanced_score,
            confidence_delta=confidence_delta,
            latency_delta_ms=latency_delta_ms,
            citation_delta=citation_delta,
            grounded_score_delta=grounded_score_delta,
            preferred_mode=preferred_mode,
            note=note,
        )

    async def _run_standard_async(
        self,
        *,
        query: str,
        chat_history: list[dict[str, str]] | None,
        model: str | None,
        response_language: str,
        query_filters: dict[str, Any] | None,
        event_handler: StreamEventHandler | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> StandardQueryResponse:
        run_async = getattr(self.standard_workflow, "run_async", None)
        if callable(run_async):
            async_kwargs: dict[str, Any] = {
                "query": query,
                "chat_history": chat_history,
                "model": model,
                "response_language": response_language,
            }
            if query_filters is not None:
                async_kwargs["query_filters"] = query_filters
            if event_handler is not None:
                async_kwargs["event_handler"] = event_handler
            if event_context:
                async_kwargs["event_context"] = dict(event_context)
            for removable in ("event_context", "event_handler", "query_filters"):
                try:
                    return await run_async(**async_kwargs)
                except TypeError:
                    if removable in async_kwargs:
                        async_kwargs.pop(removable, None)
                        continue
                    raise
            return await run_async(**async_kwargs)
        kwargs: dict[str, Any] = {
            "query": query,
            "chat_history": chat_history,
            "model": model,
            "response_language": response_language,
        }
        if query_filters is not None:
            kwargs["query_filters"] = query_filters
        if event_handler is not None:
            kwargs["event_handler"] = event_handler
        if event_context:
            kwargs["event_context"] = dict(event_context)

        for removable in ("event_context", "event_handler", "query_filters"):
            try:
                return await asyncio.to_thread(self.standard_workflow.run, **kwargs)
            except TypeError:
                if removable in kwargs:
                    kwargs.pop(removable, None)
                    continue
                raise
        return await asyncio.to_thread(self.standard_workflow.run, **kwargs)

    async def _run_advanced_async(
        self,
        *,
        query: str,
        chat_history: list[dict[str, str]] | None,
        model: str | None,
        response_language: str,
        query_filters: dict[str, Any] | None,
        event_handler: StreamEventHandler | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> AdvancedQueryResponse:
        run_async = getattr(self.advanced_workflow, "run_async", None)
        if callable(run_async):
            async_kwargs: dict[str, Any] = {
                "query": query,
                "chat_history": chat_history,
                "model": model,
                "response_language": response_language,
            }
            if query_filters is not None:
                async_kwargs["query_filters"] = query_filters
            if event_handler is not None:
                async_kwargs["event_handler"] = event_handler
            if event_context:
                async_kwargs["event_context"] = dict(event_context)
            for removable in ("event_context", "event_handler", "query_filters"):
                try:
                    return await run_async(**async_kwargs)
                except TypeError:
                    if removable in async_kwargs:
                        async_kwargs.pop(removable, None)
                        continue
                    raise
            return await run_async(**async_kwargs)
        kwargs: dict[str, Any] = {
            "query": query,
            "chat_history": chat_history,
            "model": model,
            "response_language": response_language,
        }
        if query_filters is not None:
            kwargs["query_filters"] = query_filters
        if event_handler is not None:
            kwargs["event_handler"] = event_handler
        if event_context:
            kwargs["event_context"] = dict(event_context)

        for removable in ("event_context", "event_handler", "query_filters"):
            try:
                return await asyncio.to_thread(self.advanced_workflow.run, **kwargs)
            except TypeError:
                if removable in kwargs:
                    kwargs.pop(removable, None)
                    continue
                raise
        return await asyncio.to_thread(self.advanced_workflow.run, **kwargs)

    @staticmethod
    def _branch_event_handler(
        base_handler: StreamEventHandler | None,
        *,
        branch: str,
        event_context: dict[str, Any] | None = None,
    ) -> StreamEventHandler | None:
        if base_handler is None:
            return None
        context_payload = dict(event_context or {})

        async def _handler(event: dict[str, Any]) -> None:
            payload = dict(event)
            payload.setdefault("branch", branch)
            for key, value in context_payload.items():
                payload.setdefault(key, value)
            await emit_stream_event(base_handler, payload)

        return _handler

    async def run_async(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
        query_filters: dict[str, Any] | None = None,
        event_handler: StreamEventHandler | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> CompareQueryResponse:
        started = time.perf_counter()
        resolved_language = response_language or detect_response_language(query)
        standard_event_handler = self._branch_event_handler(
            event_handler,
            branch="standard",
            event_context=event_context,
        )
        advanced_event_handler = self._branch_event_handler(
            event_handler,
            branch="advanced",
            event_context=event_context,
        )
        standard, advanced = await asyncio.gather(
            self._run_standard_async(
                query=query,
                chat_history=chat_history,
                model=model,
                response_language=resolved_language,
                query_filters=query_filters,
                event_handler=standard_event_handler,
                event_context=event_context,
            ),
            self._run_advanced_async(
                query=query,
                chat_history=chat_history,
                model=model,
                response_language=resolved_language,
                query_filters=query_filters,
                event_handler=advanced_event_handler,
                event_context=event_context,
            ),
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

    def run(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
        query_filters: dict[str, Any] | None = None,
        event_handler: StreamEventHandler | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> CompareQueryResponse:
        """Sync wrapper for CLI/tests."""
        return run_coro_sync(
            self.run_async(
                query=query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
                query_filters=query_filters,
                event_handler=event_handler,
                event_context=event_context,
            )
        )

    async def aclose(self) -> None:
        """Close workflow resources."""
        await self.advanced_workflow.aclose()
