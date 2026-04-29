"""Advanced workflow implementation with retrieval gate, critique, retry, and refine."""

from __future__ import annotations

import time
from typing import Any, Protocol

from app.core.async_utils import run_coro_sync
from app.core.config import get_settings
from app.core.timing import coerce_ms, ensure_completed_trace, safe_ratio
from app.generation.citations import CitationBuilder
from app.schemas.api import AdvancedQueryResponse
from app.schemas.common import Citation, Mode
from app.schemas.retrieval import RetrievalResult
from app.schemas.workflow import WorkflowState
from app.workflows.advanced_pipeline import (
    ADVANCED_TIMING_KEYS,
    AdvancedPipelineContext,
    CritiqueLoopStage,
    FinalGroundingStage,
    HallucinationGuardStage,
    LanguageGuardStage,
    Pipeline,
    RefineStage,
    RetrievalGateStage,
)
from app.workflows.advanced_policy import AdvancedPolicy
from app.workflows.critique import HeuristicCritic
from app.workflows.query_rewrite import QueryRewriter
from app.workflows.refine import AnswerRefiner
from app.workflows.retrieval_gate import HeuristicRetrievalGate
from app.workflows.shared import (
    detect_response_language,
    localized_insufficient_evidence,
    normalize_query,
    trim_chat_history,
)
from app.workflows.streaming import StreamEventHandler
from app.workflows.standard import StandardPipelineResult, StandardWorkflow


class RefinerLike(Protocol):
    """Subset of refiner behavior required by advanced pipeline."""

    def refine(
        self,
        query: str,
        draft_answer: str,
        critique: Any,
        context: list[RetrievalResult],
        *,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
    ) -> str: ...

    def refine_strict_grounded(
        self,
        *,
        query: str,
        draft_answer: str,
        context: list[RetrievalResult],
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
    ) -> str: ...


class AdvancedWorkflow:
    """Practical Self-RAG workflow reusing standard retrieval/generation pipeline."""

    STRONG_GROUNDED_THRESHOLD = 0.12
    VERY_LOW_GROUNDED_THRESHOLD = 0.02

    def __init__(
        self,
        *,
        standard_workflow: StandardWorkflow | None = None,
        max_loops: int | None = None,
        retrieval_gate: HeuristicRetrievalGate | None = None,
        query_rewriter: QueryRewriter | None = None,
        critic: HeuristicCritic | None = None,
        refiner: RefinerLike | None = None,
    ) -> None:
        settings = get_settings()
        self.max_loops = (
            max_loops
            if max_loops is not None
            else int(getattr(settings, "max_advanced_loops", 1))
        )
        self.memory_window = max(0, int(getattr(settings, "memory_window", 3)))
        self.adaptive_enabled = bool(
            getattr(settings, "advanced_adaptive_enabled", True)
        )
        self.force_llm_gate = bool(getattr(settings, "advanced_force_llm_gate", False))
        self.force_llm_critic = bool(
            getattr(settings, "advanced_force_llm_critic", False)
        )
        self.policy = AdvancedPolicy(
            adaptive_enabled=self.adaptive_enabled,
            force_llm_gate=self.force_llm_gate,
            force_llm_critic=self.force_llm_critic,
        )

        self.standard_workflow = standard_workflow or StandardWorkflow()
        llm_client = self.standard_workflow.generator.llm_client
        llm_cache = self.standard_workflow.caches.llm
        self.llm_client = llm_client

        self.retrieval_gate = retrieval_gate or HeuristicRetrievalGate(
            llm_client=llm_client,
            prompt_repository=self.standard_workflow.generator.prompt_repository,
            llm_cache=llm_cache,
        )
        self.query_rewriter = query_rewriter or QueryRewriter(
            llm_client=llm_client,
            prompt_repository=self.standard_workflow.generator.prompt_repository,
            llm_cache=llm_cache,
        )
        self.critic = critic or HeuristicCritic(
            llm_client=llm_client,
            prompt_repository=self.standard_workflow.generator.prompt_repository,
            llm_cache=llm_cache,
        )
        self.refiner = refiner or AnswerRefiner(
            llm_client=llm_client,
            prompt_repository=self.standard_workflow.generator.prompt_repository,
            llm_cache=llm_cache,
        )
        self.citation_builder = CitationBuilder()

    @staticmethod
    def _critique_category(note: str | None) -> str | None:
        if not note:
            return None
        lowered = note.strip().lower()
        for category in (
            "no_evidence",
            "weak_evidence",
            "incomplete_answer",
            "hallucination",
            "grounded",
        ):
            if lowered.startswith(f"{category}:"):
                return category
        return None

    @staticmethod
    def _cautious_answer_from_context(
        context: list[RetrievalResult],
        *,
        response_language: str,
    ) -> str:
        if not context:
            return localized_insufficient_evidence(response_language)

        lead = context[0].content.strip().replace("\n", " ")
        if not lead:
            return localized_insufficient_evidence(response_language)
        snippet = lead[:220].strip()
        if response_language == "vi":
            return (
                f"Theo ngữ cảnh hiện có, thông tin liên quan là: {snippet}\n\n"
                "Lưu ý: Bằng chứng còn hạn chế, cần đối chiếu thêm trong tài liệu."
            )
        return (
            f"Based on the available context, relevant information is: {snippet}\n\n"
            "Note: evidence is limited and may need additional verification."
        )

    def _direct_answer_without_retrieval(
        self,
        query: str,
        *,
        model: str | None = None,
        response_language: str = "en",
    ) -> tuple[str, float | None, str, str]:
        _ = query
        _ = model
        return (
            localized_insufficient_evidence(response_language),
            0.0,
            "insufficient_evidence",
            "gate_no_retrieval_no_context",
        )

    def _build_response(
        self,
        *,
        answer: str,
        citations: list[Citation],
        confidence: float | None,
        status: str,
        stop_reason: str,
        start_time: float,
        loop_count: int,
        response_language: str,
        language_mismatch: bool,
        grounded_score: float,
        grounding_reason: str,
        citation_count: int,
        hallucination_detected: bool,
        llm_fallback_used: bool,
        trace: list[dict[str, Any]],
    ) -> AdvancedQueryResponse:
        elapsed_ms = coerce_ms((time.perf_counter() - start_time) * 1000, 0)
        response_trace = list(trace)
        timing_step_index = next(
            (
                idx
                for idx, step in reversed(list(enumerate(response_trace)))
                if step.get("step") == "timing_summary"
            ),
            None,
        )
        if timing_step_index is None:
            summary: dict[str, Any] = {"step": "timing_summary"}
            for key in ADVANCED_TIMING_KEYS:
                summary[key] = 0
            summary["total_ms"] = elapsed_ms
            summary["gate_ms"] = 0
            summary["timing_breakdown_available"] = False
            summary["retrieval_timing_breakdown_available"] = False
            summary["llm_call_count_estimate"] = 0
            summary["retrieval_vs_generation_ratio"] = 0.0
            response_trace.append(summary)
        else:
            summary = dict(response_trace[timing_step_index])
            for key in ADVANCED_TIMING_KEYS:
                summary.setdefault(key, 0)
            for key in ADVANCED_TIMING_KEYS:
                summary[key] = coerce_ms(summary.get(key, 0), 0)
            summary["total_ms"] = elapsed_ms
            summary["gate_ms"] = coerce_ms(
                summary.get("retrieval_gate_ms", summary.get("gate_ms", 0)), 0
            )
            summary["timing_breakdown_available"] = bool(
                summary.get(
                    "timing_breakdown_available",
                    summary.get("retrieval_timing_breakdown_available", False),
                )
            )
            summary["retrieval_timing_breakdown_available"] = bool(
                summary.get(
                    "retrieval_timing_breakdown_available",
                    summary.get("timing_breakdown_available", False),
                )
            )
            summary["llm_call_count_estimate"] = coerce_ms(
                summary.get("llm_call_count_estimate", 0), 0
            )
            summary["retrieval_vs_generation_ratio"] = safe_ratio(
                summary.get("retrieval_total_ms", 0),
                summary.get("llm_generate_ms", 0),
            )
            response_trace[timing_step_index] = summary
        response_trace = ensure_completed_trace(response_trace, total_ms=elapsed_ms)

        return AdvancedQueryResponse(
            mode="advanced",
            answer=answer,
            citations=citations,
            confidence=confidence,
            status=status,
            stop_reason=stop_reason,
            latency_ms=coerce_ms(elapsed_ms, 0),
            loop_count=loop_count,
            response_language=response_language,
            language_mismatch=language_mismatch,
            grounded_score=grounded_score,
            grounding_reason=grounding_reason,
            citation_count=citation_count,
            hallucination_detected=hallucination_detected,
            llm_fallback_used=llm_fallback_used,
            trace=response_trace,
        )

    def _build_pipeline_executor(self, context: AdvancedPipelineContext) -> Pipeline:
        # Keep stage order aligned with existing behavior to avoid regressions.
        return Pipeline(
            stages=[
                RetrievalGateStage(context),
                CritiqueLoopStage(context),
                RefineStage(context),
                LanguageGuardStage(context),
                HallucinationGuardStage(context),
                FinalGroundingStage(context),
            ]
        )

    async def run_async(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
        query_filters: dict[str, Any] | None = None,
        precomputed_pipeline: StandardPipelineResult | None = None,
        event_handler: StreamEventHandler | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> AdvancedQueryResponse:
        start = time.perf_counter()
        normalized_query = normalize_query(query)
        resolved_language = response_language or detect_response_language(query)
        normalized_history = trim_chat_history(
            chat_history, memory_window=self.memory_window
        )
        state = WorkflowState(
            mode=Mode.ADVANCED,
            user_query=query,
            normalized_query=normalized_query,
            response_language=resolved_language,
            chat_history=normalized_history,
        )
        context = AdvancedPipelineContext(
            workflow=self,
            start_time=start,
            model=model,
            query_filters=query_filters,
            normalized_history=normalized_history,
            resolved_language=resolved_language,
            precomputed_pipeline=precomputed_pipeline,
            event_handler=event_handler,
            event_context=dict(event_context or {}),
        )
        await self._build_pipeline_executor(context).run(state)
        if context.terminal_response is not None:
            return context.terminal_response
        generated_confidence = (
            context.pipeline.generated.confidence
            if context.pipeline is not None
            else None
        )
        final_confidence = (
            state.confidence if state.confidence is not None else generated_confidence
        )
        return self._build_response(
            answer=state.final_answer
            or localized_insufficient_evidence(resolved_language),
            citations=state.citations,
            confidence=final_confidence,
            status=context.final_status,
            stop_reason=state.stop_reason or "no_pipeline_result",
            start_time=start,
            loop_count=state.loop_count,
            response_language=resolved_language,
            language_mismatch=state.language_mismatch,
            grounded_score=state.grounded_score,
            grounding_reason=state.grounding_reason,
            citation_count=context.citation_count,
            hallucination_detected=state.hallucination_detected,
            llm_fallback_used=state.llm_fallback_used,
            trace=context.trace,
        )

    def run(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
        query_filters: dict[str, Any] | None = None,
        precomputed_pipeline: StandardPipelineResult | None = None,
        event_handler: StreamEventHandler | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> AdvancedQueryResponse:
        """Sync wrapper for CLI/tests."""
        return run_coro_sync(
            self.run_async(
                query=query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
                query_filters=query_filters,
                precomputed_pipeline=precomputed_pipeline,
                event_handler=event_handler,
                event_context=event_context,
            )
        )

    async def aclose(self) -> None:
        """Close shared resources."""
        await self.standard_workflow.aclose()
