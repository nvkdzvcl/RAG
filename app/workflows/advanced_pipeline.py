"""Pipeline stages for AdvancedWorkflow orchestration."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, TYPE_CHECKING

from app.core.timing import coerce_ms, ensure_completed_trace, safe_ratio
from app.schemas.api import AdvancedQueryResponse
from app.schemas.common import Citation
from app.schemas.workflow import CritiqueResult, WorkflowState
from app.workflows.advanced_policy import AdvancedPolicyInput
from app.workflows.query_budget import classify_query_complexity
from app.workflows.shared import (
    GroundingPolicy,
    assess_grounding_with_policy,
    grounded_overlap_score,
    is_language_mismatch,
    localized_insufficient_evidence,
    normalize_query,
)
from app.workflows.streaming import StreamEventHandler, emit_stream_event
from app.workflows.standard import StandardPipelineResult

if TYPE_CHECKING:
    from app.workflows.advanced import AdvancedWorkflow


ADVANCED_TIMING_KEYS: tuple[str, ...] = (
    "retrieval_gate_ms",
    "retrieval_total_ms",
    "llm_generate_ms",
    "query_rewrite_ms",
    "standard_pipeline_ms",
    "critique_ms",
    "refine_ms",
    "refine_stage_ms",
    "language_guard_ms",
    "hallucination_guard_ms",
    "grounding_ms",
    "final_grounding_ms",
    "total_ms",
)


def _grounding_query_complexity(raw: str | None) -> str:
    normalized = (raw or "normal").strip().lower()
    if normalized == "simple_extractive":
        return "simple"
    if normalized in {"simple", "normal", "complex"}:
        return normalized
    return "normal"


def _estimate_retrieval_confidence_from_pipeline(
    pipeline: StandardPipelineResult,
) -> float | None:
    best_score: float | None = None
    for item in pipeline.reranked or pipeline.retrieved:
        for candidate in (
            item.rerank_score,
            item.score,
            item.dense_score,
            item.sparse_score,
        ):
            if not isinstance(candidate, (int, float)):
                continue
            numeric = float(candidate)
            if numeric != numeric:
                continue
            if best_score is None or numeric > best_score:
                best_score = numeric

    if best_score is None:
        return None
    if best_score <= 0.0:
        return 0.0
    if best_score <= 1.0:
        return round(best_score, 4)
    return round(best_score / (1.0 + best_score), 4)


def _build_grounding_policy(
    *,
    pipeline: StandardPipelineResult,
    status: str,
    answer: str,
    citation_count: int,
) -> GroundingPolicy:
    query_complexity = (
        pipeline.query_budget.complexity
        if pipeline.query_budget is not None
        else "normal"
    )
    return GroundingPolicy(
        mode="advanced",
        query_complexity=_grounding_query_complexity(query_complexity),
        generated_status=status,
        answer_length=len(answer.strip()),
        citation_count=max(0, int(citation_count)),
        retrieval_confidence=_estimate_retrieval_confidence_from_pipeline(pipeline),
        fast_path_used=pipeline.generated.stop_reason.startswith("heuristic_"),
    )


@dataclass
class AdvancedPipelineContext:
    """Run-scoped mutable context shared by advanced pipeline stages."""

    workflow: AdvancedWorkflow
    start_time: float
    model: str | None
    query_filters: dict[str, Any] | None
    normalized_history: list[dict[str, str]]
    resolved_language: str
    event_handler: StreamEventHandler | None = None
    event_context: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, int] = field(default_factory=dict)
    precomputed_pipeline: StandardPipelineResult | None = None

    current_query: str = ""
    pipeline: StandardPipelineResult | None = None
    critique_result: CritiqueResult | None = None
    reused_standard_result: bool = False
    standard_reuse_saved_steps: list[str] = field(default_factory=list)
    standard_reuse_reason: str | None = None

    final_answer: str = ""
    final_citations: list[Citation] = field(default_factory=list)
    final_status: str = "answered"
    stop_reason: str = "critique_pass"

    has_context: bool = False
    has_relevant_context: bool = False
    context_texts: list[str] = field(default_factory=list)
    context_citations: list[Citation] = field(default_factory=list)
    critique_category: str | None = None
    insufficient_answer: str = ""

    citation_count: int = 0
    language_mismatch: bool = False
    grounded_score: float = 0.0
    grounding_reason: str = "not_evaluated"
    hallucination_detected: bool = False
    grounding_policy: str = "not_evaluated"
    grounding_semantic_used: bool = False
    grounding_cache_hit: bool = False
    grounding_ms: int = 0
    llm_fallback_used: bool = False
    embedding_cache_hit: bool = False
    retrieval_cache_hit: bool = False
    rerank_cache_hit: bool = False
    llm_cache_hit: bool = False
    retrieval_timing_breakdown_available: bool = False
    query_complexity: str = "normal"
    gate_strategy: str = "heuristic"
    critique_strategy: str = "skipped"
    refine_used: bool = False
    language_rewrite_used: bool = False
    hallucination_refine_used: bool = False
    llm_call_count_estimate: int = 0

    terminal_response: AdvancedQueryResponse | None = None

    def add_llm_calls(self, count: int = 1) -> None:
        self.llm_call_count_estimate = max(0, self.llm_call_count_estimate + count)

    def timing_summary(self, *, total_ms: int | None = None) -> dict[str, Any]:
        """Build a trace-ready timing summary from measured stage timings."""
        if total_ms is not None:
            self.timings["total_ms"] = coerce_ms(total_ms, 0)
        summary: dict[str, Any] = {"step": "timing_summary"}
        for key in ADVANCED_TIMING_KEYS:
            summary[key] = coerce_ms(self.timings.get(key, 0), 0)
        summary["gate_ms"] = summary["retrieval_gate_ms"]
        summary["timing_breakdown_available"] = bool(
            self.retrieval_timing_breakdown_available
        )
        summary["retrieval_timing_breakdown_available"] = bool(
            self.retrieval_timing_breakdown_available
        )
        summary["embedding_cache_hit"] = bool(self.embedding_cache_hit)
        summary["retrieval_cache_hit"] = bool(self.retrieval_cache_hit)
        summary["rerank_cache_hit"] = bool(self.rerank_cache_hit)
        summary["llm_cache_hit"] = bool(self.llm_cache_hit)
        summary["grounding_cache_hit"] = bool(self.grounding_cache_hit)
        summary["gate_strategy"] = self.gate_strategy
        summary["critique_strategy"] = self.critique_strategy
        summary["refine_used"] = bool(self.refine_used)
        summary["language_rewrite_used"] = bool(self.language_rewrite_used)
        summary["hallucination_refine_used"] = bool(self.hallucination_refine_used)
        summary["llm_call_count_estimate"] = coerce_ms(
            self.llm_call_count_estimate, 0
        )
        summary["retrieval_vs_generation_ratio"] = safe_ratio(
            summary.get("retrieval_total_ms", 0),
            summary.get("llm_generate_ms", 0),
        )
        summary["reused_standard_result"] = bool(self.reused_standard_result)
        summary["standard_reuse_saved_steps"] = list(self.standard_reuse_saved_steps)
        if self.standard_reuse_reason is not None:
            summary["standard_reuse_reason"] = self.standard_reuse_reason
        return summary


class PipelineStage(Protocol):
    """Stage contract for advanced pipeline execution."""

    async def execute(self, state: WorkflowState) -> WorkflowState: ...


class Pipeline:
    """Execute workflow stages in strict sequence."""

    def __init__(self, stages: list[PipelineStage]) -> None:
        self.stages = list(stages)

    async def run(self, state: WorkflowState) -> WorkflowState:
        current_state = state
        for stage in self.stages:
            current_state = await stage.execute(current_state)
        return current_state


class BasePipelineStage:
    """Base class for advanced pipeline stages sharing mutable run context."""

    def __init__(self, context: AdvancedPipelineContext) -> None:
        self.context = context


class RetrievalGateStage(BasePipelineStage):
    """Stage 1: decide whether retrieval is needed."""

    @staticmethod
    async def _decide_with_compat(
        gate: Any,
        query: str,
        *,
        chat_history: list[dict[str, str]],
        model: str | None,
        response_language: str,
        allow_llm: bool,
        force_llm: bool,
    ) -> tuple[bool, str, str]:
        decide_with_strategy = getattr(gate, "decide_with_strategy_async", None)
        if callable(decide_with_strategy):
            return await decide_with_strategy(
                query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
                allow_llm=allow_llm,
                force_llm=force_llm,
            )

        decide_async = getattr(gate, "decide_async", None)
        if callable(decide_async):
            try:
                decision, reason = await decide_async(
                    query,
                    chat_history=chat_history,
                    model=model,
                    response_language=response_language,
                    allow_llm=allow_llm,
                )
            except TypeError:
                decision, reason = await decide_async(
                    query,
                    chat_history=chat_history,
                    model=model,
                    response_language=response_language,
                )
            return decision, reason, "heuristic"

        try:
            decision, reason = gate.decide(
                query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        except TypeError:
            decision, reason = gate.decide(
                query,
                chat_history=chat_history,
            )
        return decision, reason, "heuristic"

    async def execute(self, state: WorkflowState) -> WorkflowState:
        context = self.context
        if context.terminal_response is not None:
            return state
        workflow = context.workflow
        context.query_complexity = classify_query_complexity(state.normalized_query)
        heuristic_decider = getattr(workflow.retrieval_gate, "heuristic_decide", None)
        if callable(heuristic_decider):
            heuristic_need, heuristic_reason = heuristic_decider(
                state.normalized_query,
                chat_history=context.normalized_history,
            )
        else:
            heuristic_need, heuristic_reason = True, "default_retrieval"
        allow_llm_gate = workflow.policy.should_use_llm_gate(
            query=state.normalized_query,
            query_complexity=context.query_complexity,
            heuristic_reason=heuristic_reason,
            user_selected_strictness=None,
        )
        gate_started = time.perf_counter()
        need_retrieval, gate_reason, gate_strategy = await self._decide_with_compat(
            workflow.retrieval_gate,
            state.normalized_query,
            chat_history=context.normalized_history,
            model=context.model,
            response_language=context.resolved_language,
            allow_llm=allow_llm_gate,
            force_llm=workflow.force_llm_gate,
        )
        retrieval_gate_ms = coerce_ms((time.perf_counter() - gate_started) * 1000, 0)
        context.timings["retrieval_gate_ms"] = retrieval_gate_ms
        context.gate_strategy = gate_strategy
        if gate_strategy == "llm":
            context.add_llm_calls(1)
        state.need_retrieval = need_retrieval
        context.trace.append(
            {
                "step": "retrieval_gate",
                "need_retrieval": need_retrieval,
                "reason": gate_reason,
                "heuristic_need_retrieval": heuristic_need,
                "heuristic_reason": heuristic_reason,
                "gate_strategy": gate_strategy,
                "query_complexity": context.query_complexity,
                "retrieval_gate_ms": retrieval_gate_ms,
                "llm_call_count_estimate": context.llm_call_count_estimate,
                "response_language": context.resolved_language,
                "memory_window": workflow.memory_window,
                "memory_messages": len(context.normalized_history),
            }
        )
        await emit_stream_event(
            context.event_handler,
            {
                "type": "advanced_stage",
                "stage": "retrieval_gate",
                "stage_ms": retrieval_gate_ms,
                "need_retrieval": need_retrieval,
                "reason": gate_reason,
                "gate_strategy": gate_strategy,
                **context.event_context,
            },
        )

        if not need_retrieval:
            total_ms = coerce_ms((time.perf_counter() - context.start_time) * 1000, 0)
            context.trace.append(context.timing_summary(total_ms=total_ms))
            answer, confidence, status, stop_reason = (
                workflow._direct_answer_without_retrieval(
                    state.normalized_query,
                    model=context.model,
                    response_language=context.resolved_language,
                )
            )
            context.terminal_response = workflow._build_response(
                answer=answer,
                citations=[],
                confidence=confidence,
                status=status,
                stop_reason=stop_reason,
                start_time=context.start_time,
                loop_count=0,
                response_language=context.resolved_language,
                language_mismatch=False,
                grounded_score=0.0,
                grounding_reason="gate_no_retrieval_no_context",
                citation_count=0,
                hallucination_detected=False,
                llm_fallback_used=False,
                trace=context.trace,
            )
            return state

        context.current_query = state.normalized_query
        state.rewritten_queries = [context.current_query]
        return state


class CritiqueLoopStage(BasePipelineStage):
    """Stage 2: run rewrite/retrieve/generate/critique loop."""

    @staticmethod
    def _rewrite_may_use_llm(rewriter: Any) -> bool:
        return bool(
            getattr(rewriter, "use_llm", False)
            and getattr(rewriter, "llm_client", None) is not None
        )

    @staticmethod
    async def _critique_with_compat(
        critic: Any,
        *,
        query: str,
        draft_answer: str,
        context_docs: list[Any],
        loop_count: int,
        max_loops: int,
        chat_history: list[dict[str, str]],
        model: str | None,
        response_language: str,
        allow_llm: bool,
    ) -> tuple[CritiqueResult, str]:
        critique_with_strategy = getattr(critic, "critique_with_strategy_async", None)
        if callable(critique_with_strategy):
            return await critique_with_strategy(
                query=query,
                draft_answer=draft_answer,
                context=context_docs,
                loop_count=loop_count,
                max_loops=max_loops,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
                allow_llm=allow_llm,
            )

        critique_async = getattr(critic, "critique_async", None)
        if callable(critique_async):
            try:
                result = await critique_async(
                    query=query,
                    draft_answer=draft_answer,
                    context=context_docs,
                    loop_count=loop_count,
                    max_loops=max_loops,
                    chat_history=chat_history,
                    model=model,
                    response_language=response_language,
                    allow_llm=allow_llm,
                )
            except TypeError:
                result = await critique_async(
                    query=query,
                    draft_answer=draft_answer,
                    context=context_docs,
                    loop_count=loop_count,
                    max_loops=max_loops,
                    chat_history=chat_history,
                    model=model,
                    response_language=response_language,
                )
            return result, "heuristic"

        try:
            result = critic.critique(
                query=query,
                draft_answer=draft_answer,
                context=context_docs,
                loop_count=loop_count,
                max_loops=max_loops,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        except TypeError:
            result = critic.critique(
                query=query,
                draft_answer=draft_answer,
                context=context_docs,
                loop_count=loop_count,
                max_loops=max_loops,
            )
        return result, "heuristic"

    @staticmethod
    def _normalize_filter_payload(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): CritiqueLoopStage._normalize_filter_payload(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, list):
            return [CritiqueLoopStage._normalize_filter_payload(item) for item in value]
        return value

    def _consume_precomputed_pipeline(
        self,
        *,
        current_query: str,
    ) -> tuple[StandardPipelineResult | None, str | None]:
        context = self.context
        candidate = context.precomputed_pipeline
        context.precomputed_pipeline = None
        if candidate is None:
            return None, None

        normalized_current_query = normalize_query(current_query)
        if candidate.normalized_query != normalized_current_query:
            return None, "query_mismatch_after_rewrite"

        requested_filters = context.query_filters
        applied_filters = candidate.retrieval_debug.get("applied_filters")
        if (
            isinstance(requested_filters, dict)
            and isinstance(applied_filters, dict)
            and self._normalize_filter_payload(requested_filters)
            != self._normalize_filter_payload(applied_filters)
        ):
            return None, "query_filters_mismatch"

        return candidate, None

    async def execute(self, state: WorkflowState) -> WorkflowState:
        context = self.context
        if context.terminal_response is not None:
            return state
        if not state.need_retrieval:
            return state

        workflow = context.workflow
        current_query = context.current_query or state.normalized_query
        pipeline: StandardPipelineResult | None = None
        critique_result: CritiqueResult | None = None
        critique_strategy: str = "skipped"
        standard_pipeline_total_ms = 0
        critique_total_ms = 0
        query_rewrite_total_ms = 0
        retrieval_total_ms = 0
        llm_generate_total_ms = 0
        retrieval_breakdown_flags: list[bool] = []

        for loop in range(1, workflow.max_loops + 1):
            state.loop_count = loop

            if loop > 1:
                rewrite_started = time.perf_counter()
                rewrites = await workflow.query_rewriter.rewrite_async(
                    current_query,
                    critique=critique_result,
                    loop_count=loop,
                    chat_history=context.normalized_history,
                    model=context.model,
                    response_language=context.resolved_language,
                )
                if self._rewrite_may_use_llm(workflow.query_rewriter):
                    context.add_llm_calls(1)
                rewrite_ms = coerce_ms((time.perf_counter() - rewrite_started) * 1000, 0)
                query_rewrite_total_ms += rewrite_ms
                if rewrites:
                    current_query = rewrites[0]
                    for candidate in rewrites:
                        if candidate not in state.rewritten_queries:
                            state.rewritten_queries.append(candidate)
            else:
                rewrite_ms = 0

            loop_event_context = dict(context.event_context)
            loop_event_context["loop"] = loop
            reused_standard_result = False
            standard_reuse_saved_steps: list[str] = []
            standard_reuse_reason: str | None = None
            if loop == 1:
                pipeline, standard_reuse_reason = self._consume_precomputed_pipeline(
                    current_query=current_query
                )
                if pipeline is not None:
                    reused_standard_result = True
                    standard_reuse_saved_steps = ["retrieve", "rerank", "generate"]
                    context.reused_standard_result = True
                    context.standard_reuse_saved_steps = list(
                        standard_reuse_saved_steps
                    )
                    context.standard_reuse_reason = None
            if pipeline is None:
                if loop == 1 and standard_reuse_reason is not None:
                    context.standard_reuse_reason = standard_reuse_reason
                pipeline_started = time.perf_counter()
                pipeline = await workflow.standard_workflow.run_pipeline(
                    query=current_query,
                    mode=state.mode,
                    model=context.model,
                    response_language=context.resolved_language,
                    chat_history=context.normalized_history,
                    query_filters=context.query_filters,
                    event_handler=context.event_handler,
                    event_context=loop_event_context,
                )
                standard_pipeline_ms = coerce_ms(
                    (time.perf_counter() - pipeline_started) * 1000
                )
            else:
                standard_pipeline_ms = 0
            standard_pipeline_total_ms += standard_pipeline_ms
            retrieval_total_ms += coerce_ms(
                pipeline.timings.get("retrieval_total_ms", 0), 0
            )
            llm_generate_total_ms += coerce_ms(
                pipeline.timings.get("llm_generate_ms", 0), 0
            )
            retrieval_breakdown_flags.append(
                bool(pipeline.retrieval_timing_breakdown_available)
            )

            state.retrieved_docs = [item.model_dump() for item in pipeline.retrieved]
            state.reranked_docs = [item.model_dump() for item in pipeline.reranked]
            state.selected_context = [
                item.model_dump() for item in pipeline.selected_context
            ]
            state.draft_answer = pipeline.generated.answer
            if (
                not reused_standard_result
                and not pipeline.generated.fast_path
                and pipeline.generated.stop_reason != "no_context"
                and not bool(pipeline.generated.llm_cache_hit)
            ):
                context.add_llm_calls(1)

            retrieval_confidence = _estimate_retrieval_confidence_from_pipeline(
                pipeline
            )
            context_texts = [
                item.content
                for item in pipeline.selected_context
                if item.content.strip()
            ]
            lexical_grounding_score = grounded_overlap_score(
                pipeline.generated.answer,
                context_texts,
            )
            policy_signal = AdvancedPolicyInput(
                query_complexity=(
                    pipeline.query_budget.complexity
                    if pipeline.query_budget is not None
                    else context.query_complexity
                ),
                retrieval_confidence=retrieval_confidence,
                citation_count=len(pipeline.generated.citations),
                grounding_lexical_score=lexical_grounding_score,
                answer_length=len(pipeline.generated.answer.strip()),
                fast_path_used=bool(pipeline.generated.fast_path),
                user_selected_strictness=None,
            )
            allow_llm_critic = workflow.policy.should_use_llm_critic(
                query=state.normalized_query,
                signal=policy_signal,
            )

            critique_started = time.perf_counter()
            critique_result, strategy = await self._critique_with_compat(
                workflow.critic,
                query=state.normalized_query,
                draft_answer=pipeline.generated.answer,
                context_docs=pipeline.selected_context,
                loop_count=loop,
                max_loops=workflow.max_loops,
                chat_history=context.normalized_history,
                model=context.model,
                response_language=context.resolved_language,
                allow_llm=allow_llm_critic,
            )
            critique_ms = coerce_ms((time.perf_counter() - critique_started) * 1000, 0)
            critique_total_ms += critique_ms
            critique_strategy = strategy
            if strategy == "llm":
                context.add_llm_calls(1)
            state.critique = critique_result
            state.confidence = critique_result.confidence

            context.trace.append(
                {
                    "step": "loop",
                    "loop": loop,
                    "query": current_query,
                    "chunk_size": workflow.standard_workflow.chunk_size,
                    "chunk_overlap": workflow.standard_workflow.chunk_overlap,
                    "retrieved_count": len(pipeline.retrieved),
                    "applied_filters": pipeline.retrieval_debug.get(
                        "applied_filters", {}
                    ),
                    "candidate_count_before_filter": pipeline.retrieval_debug.get(
                        "candidate_count_before_filter",
                        len(pipeline.retrieved),
                    ),
                    "candidate_count_after_filter": pipeline.retrieval_debug.get(
                        "candidate_count_after_filter",
                        len(pipeline.retrieved),
                    ),
                    "retrieved_docs": [
                        {
                            "chunk_id": item.chunk_id,
                            "doc_id": item.doc_id,
                            "source": item.source,
                            "title": item.title,
                            "section": item.section,
                            "file_name": item.metadata.get("file_name")
                            or item.metadata.get("filename"),
                            "file_type": item.metadata.get("file_type"),
                            "uploaded_at": item.metadata.get("uploaded_at"),
                            "created_at": item.metadata.get("created_at"),
                            "page": item.page,
                            "rank": item.rank,
                            "block_type": item.metadata.get("block_type"),
                            "ocr": bool(item.metadata.get("ocr")),
                            "score": item.score,
                            "dense_score": item.dense_score,
                            "sparse_score": item.sparse_score,
                        }
                        for item in pipeline.retrieved
                    ],
                    "reranked_count": len(pipeline.reranked),
                    "reranked_docs": [
                        {
                            "chunk_id": item.chunk_id,
                            "doc_id": item.doc_id,
                            "source": item.source,
                            "title": item.title,
                            "section": item.section,
                            "file_name": item.metadata.get("file_name")
                            or item.metadata.get("filename"),
                            "file_type": item.metadata.get("file_type"),
                            "uploaded_at": item.metadata.get("uploaded_at"),
                            "created_at": item.metadata.get("created_at"),
                            "page": item.page,
                            "rank": item.rank,
                            "block_type": item.metadata.get("block_type"),
                            "ocr": bool(item.metadata.get("ocr")),
                            "rerank_score": item.rerank_score,
                            "score": item.score,
                            "dense_score": item.dense_score,
                            "sparse_score": item.sparse_score,
                        }
                        for item in pipeline.reranked
                    ],
                    "selected_count": len(pipeline.selected_context),
                    "selected_context_docs": [
                        {
                            "chunk_id": item.chunk_id,
                            "doc_id": item.doc_id,
                            "source": item.source,
                            "title": item.title,
                            "section": item.section,
                            "file_name": item.metadata.get("file_name")
                            or item.metadata.get("filename"),
                            "file_type": item.metadata.get("file_type"),
                            "uploaded_at": item.metadata.get("uploaded_at"),
                            "created_at": item.metadata.get("created_at"),
                            "page": item.page,
                            "block_type": item.metadata.get("block_type"),
                            "ocr": bool(item.metadata.get("ocr")),
                            "content": item.content,
                        }
                        for item in pipeline.selected_context
                    ],
                    "generated_status": pipeline.generated.status,
                    "generated_stop_reason": pipeline.generated.stop_reason,
                    "generated_confidence": pipeline.generated.confidence,
                    "gate_strategy": context.gate_strategy,
                    "critique_strategy": strategy,
                    "query_complexity": policy_signal.query_complexity,
                    "retrieval_confidence": retrieval_confidence,
                    "grounding_lexical_score": lexical_grounding_score,
                    "fast_path_used": bool(pipeline.generated.fast_path),
                    "critique": critique_result.model_dump(),
                    "query_rewrite_ms": rewrite_ms,
                    "standard_pipeline_ms": standard_pipeline_ms,
                    "critique_ms": critique_ms,
                    "reused_standard_result": reused_standard_result,
                    "standard_reuse_saved_steps": standard_reuse_saved_steps,
                    "standard_reuse_reason": standard_reuse_reason,
                    "timing_breakdown_available": (
                        pipeline.retrieval_timing_breakdown_available
                    ),
                    "llm_call_count_estimate": context.llm_call_count_estimate,
                    "retrieval_timing_breakdown_available": (
                        pipeline.retrieval_timing_breakdown_available
                    ),
                    "embedding_cache_hit": pipeline.embedding_cache_hit,
                    "retrieval_cache_hit": pipeline.retrieval_cache_hit,
                    "rerank_cache_hit": pipeline.rerank_cache_hit,
                    "llm_cache_hit": bool(pipeline.generated.llm_cache_hit),
                    "pipeline_step_timings_ms": dict(pipeline.timings),
                }
            )
            await emit_stream_event(
                context.event_handler,
                {
                    "type": "advanced_stage",
                    "stage": "critique_loop",
                    "loop": loop,
                    "standard_pipeline_ms": standard_pipeline_ms,
                    "critique_ms": critique_ms,
                    "critique_strategy": strategy,
                    "query_rewrite_ms": rewrite_ms,
                    "reused_standard_result": reused_standard_result,
                    "standard_reuse_saved_steps": standard_reuse_saved_steps,
                    "standard_reuse_reason": standard_reuse_reason,
                    **loop_event_context,
                },
            )

            if critique_result.should_retry_retrieval and loop < workflow.max_loops:
                continue
            break

        context.timings["standard_pipeline_ms"] = standard_pipeline_total_ms
        context.timings["critique_ms"] = critique_total_ms
        context.timings["query_rewrite_ms"] = query_rewrite_total_ms
        context.timings["retrieval_total_ms"] = retrieval_total_ms
        context.timings["llm_generate_ms"] = llm_generate_total_ms
        context.current_query = current_query
        context.pipeline = pipeline
        context.critique_result = critique_result
        context.critique_strategy = critique_strategy
        context.retrieval_timing_breakdown_available = bool(
            retrieval_breakdown_flags
        ) and all(retrieval_breakdown_flags)
        if pipeline is not None:
            context.embedding_cache_hit = pipeline.embedding_cache_hit
            context.retrieval_cache_hit = pipeline.retrieval_cache_hit
            context.rerank_cache_hit = pipeline.rerank_cache_hit
            context.llm_cache_hit = bool(pipeline.generated.llm_cache_hit)

        if pipeline is None or critique_result is None:
            context.terminal_response = workflow._build_response(
                answer=localized_insufficient_evidence(context.resolved_language),
                citations=[],
                confidence=0.0,
                status="insufficient_evidence",
                stop_reason="no_pipeline_result",
                start_time=context.start_time,
                loop_count=state.loop_count,
                response_language=context.resolved_language,
                language_mismatch=False,
                grounded_score=0.0,
                grounding_reason="no_pipeline_result",
                citation_count=0,
                hallucination_detected=False,
                llm_fallback_used=False,
                trace=context.trace,
            )
        return state


class RefineStage(BasePipelineStage):
    """Stage 3: apply evidence checks + refine/recover decisions."""

    @staticmethod
    async def _refine_with_compat(
        workflow: AdvancedWorkflow,
        *,
        query: str,
        draft_answer: str,
        critique: CritiqueResult,
        context_docs: list[Any],
        chat_history: list[dict[str, str]],
        model: str | None,
        response_language: str,
    ) -> str:
        async_refine = getattr(workflow.refiner, "refine_async", None)
        if callable(async_refine):
            return await async_refine(
                query=query,
                draft_answer=draft_answer,
                critique=critique,
                context=context_docs,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        return await asyncio.to_thread(
            workflow.refiner.refine,
            query,
            draft_answer,
            critique,
            context_docs,
            chat_history=chat_history,
            model=model,
            response_language=response_language,
        )

    @staticmethod
    async def _refine_strict_grounded_with_compat(
        workflow: AdvancedWorkflow,
        *,
        query: str,
        draft_answer: str,
        context_docs: list[Any],
        chat_history: list[dict[str, str]],
        model: str | None,
        response_language: str,
    ) -> str:
        async_refine_strict = getattr(
            workflow.refiner, "refine_strict_grounded_async", None
        )
        if callable(async_refine_strict):
            return await async_refine_strict(
                query=query,
                draft_answer=draft_answer,
                context=context_docs,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        return await asyncio.to_thread(
            workflow.refiner.refine_strict_grounded,
            query=query,
            draft_answer=draft_answer,
            context=context_docs,
            chat_history=chat_history,
            model=model,
            response_language=response_language,
        )

    async def execute(self, state: WorkflowState) -> WorkflowState:
        context = self.context
        if context.terminal_response is not None:
            return state
        if context.pipeline is None or context.critique_result is None:
            return state

        stage_started = time.perf_counter()
        refine_ms_total = 0
        refine_used = False
        workflow = context.workflow
        pipeline = context.pipeline
        critique_result = context.critique_result

        final_answer = pipeline.generated.answer.strip()
        final_citations = list(pipeline.generated.citations)
        final_status = pipeline.generated.status
        stop_reason = "critique_pass"

        context_texts = [
            item.content for item in pipeline.selected_context if item.content.strip()
        ]
        has_context = bool(context_texts)
        critique_category = workflow._critique_category(critique_result.note)
        has_relevant_context = has_context and critique_category != "no_evidence"
        context_citations = workflow.citation_builder.build(pipeline.selected_context)
        if has_context and not final_citations:
            final_citations = context_citations

        insufficient_answer = localized_insufficient_evidence(context.resolved_language)
        force_abstain = "force abstain" in state.normalized_query.lower()
        if force_abstain:
            final_answer = insufficient_answer
            final_citations = []
            final_status = "insufficient_evidence"
            stop_reason = "critic_abstain"
        elif not has_relevant_context:
            final_answer = insufficient_answer
            final_citations = []
            final_status = "insufficient_evidence"
            stop_reason = "no_relevant_context"
        elif (
            critique_result.should_retry_retrieval
            and state.loop_count >= workflow.max_loops
        ):
            stop_reason = "max_loop_reached"

        needs_refine_with_context = bool(critique_result.should_refine_answer)
        if (
            needs_refine_with_context
            and has_relevant_context
            and final_status != "insufficient_evidence"
        ):
            refine_started = time.perf_counter()
            final_answer = await self._refine_with_compat(
                workflow,
                query=state.normalized_query,
                draft_answer=final_answer,
                critique=critique_result,
                context_docs=pipeline.selected_context,
                chat_history=context.normalized_history,
                model=context.model,
                response_language=context.resolved_language,
            )
            refine_ms_total += coerce_ms(
                (time.perf_counter() - refine_started) * 1000, 0
            )
            context.add_llm_calls(1)
            refine_used = True
            if stop_reason != "max_loop_reached":
                stop_reason = "refined_with_context"

        if (
            final_status == "insufficient_evidence"
            and has_relevant_context
            and final_citations
            and critique_result.should_refine_answer
        ):
            strict_refine_started = time.perf_counter()
            refined_from_insufficient = (
                await self._refine_strict_grounded_with_compat(
                    workflow,
                    query=state.normalized_query,
                    draft_answer=final_answer or insufficient_answer,
                    context_docs=pipeline.selected_context,
                    chat_history=context.normalized_history,
                    model=context.model,
                    response_language=context.resolved_language,
                )
            ).strip()
            refine_ms_total += coerce_ms(
                (time.perf_counter() - strict_refine_started) * 1000, 0
            )
            context.add_llm_calls(1)
            refine_used = True
            refined_is_insufficient = refined_from_insufficient == insufficient_answer
            if refined_from_insufficient and not refined_is_insufficient:
                final_answer = refined_from_insufficient
                final_status = "answered"
                if stop_reason != "max_loop_reached":
                    stop_reason = "recovered_from_context"
            else:
                final_answer = workflow._cautious_answer_from_context(
                    pipeline.selected_context,
                    response_language=context.resolved_language,
                )
                final_status = "partial"
                if stop_reason != "max_loop_reached":
                    stop_reason = "weak_evidence_cautious"

        if critique_result.has_conflict and final_status != "insufficient_evidence":
            if context.resolved_language == "vi":
                final_answer = (
                    final_answer + "\n\nPhát hiện khả năng xung đột giữa các nguồn."
                )
            else:
                final_answer = (
                    final_answer + "\n\nPotential conflict detected in sources."
                )
            stop_reason = "conflict_detected"

        stage_ms = coerce_ms((time.perf_counter() - stage_started) * 1000, 0)
        context.timings["refine_ms"] = refine_ms_total
        context.timings["refine_stage_ms"] = stage_ms

        context.final_answer = final_answer
        context.final_citations = final_citations
        context.final_status = final_status
        context.stop_reason = stop_reason
        context.context_texts = context_texts
        context.has_context = has_context
        context.has_relevant_context = has_relevant_context
        context.context_citations = context_citations
        context.critique_category = critique_category
        context.insufficient_answer = insufficient_answer
        context.refine_used = refine_used

        state.final_answer = final_answer
        state.citations = final_citations
        state.stop_reason = stop_reason
        await emit_stream_event(
            context.event_handler,
            {
                "type": "advanced_stage",
                "stage": "refine",
                "stage_ms": stage_ms,
                "refine_ms": refine_ms_total,
                **context.event_context,
            },
        )
        return state


class LanguageGuardStage(BasePipelineStage):
    """Stage 4: enforce response language when needed."""

    async def execute(self, state: WorkflowState) -> WorkflowState:
        context = self.context
        if context.terminal_response is not None:
            return state
        if context.pipeline is None:
            return state

        stage_started = time.perf_counter()
        final_answer = context.final_answer
        final_status = context.final_status
        stop_reason = context.stop_reason

        language_mismatch = is_language_mismatch(
            final_answer, context.resolved_language
        )
        language_rewrite_used = False
        if language_mismatch and final_status != "insufficient_evidence":
            language_rewrite_used = True
            context.add_llm_calls(1)
            rewritten = (
                await context.workflow.standard_workflow._rewrite_answer_language(
                    query=state.normalized_query,
                    answer=final_answer,
                    response_language=context.resolved_language,
                    model=context.model,
                )
            )
            if rewritten:
                final_answer = rewritten
                language_mismatch = is_language_mismatch(
                    final_answer, context.resolved_language
                )
                if not language_mismatch and stop_reason == "critique_pass":
                    stop_reason = "language_refined"

        context.final_answer = final_answer
        context.stop_reason = stop_reason
        context.language_mismatch = language_mismatch
        context.language_rewrite_used = language_rewrite_used
        state.final_answer = final_answer
        state.stop_reason = stop_reason
        state.language_mismatch = language_mismatch
        stage_ms = coerce_ms((time.perf_counter() - stage_started) * 1000, 0)
        context.timings["language_guard_ms"] = stage_ms
        await emit_stream_event(
            context.event_handler,
            {
                "type": "advanced_stage",
                "stage": "language_guard",
                "stage_ms": stage_ms,
                "language_mismatch": language_mismatch,
                **context.event_context,
            },
        )
        return state


class HallucinationGuardStage(BasePipelineStage):
    """Stage 5: enforce grounding thresholds and hallucination mitigation."""

    async def execute(self, state: WorkflowState) -> WorkflowState:
        context = self.context
        if context.terminal_response is not None:
            return state
        if context.pipeline is None:
            return state

        stage_started = time.perf_counter()
        workflow = context.workflow
        pipeline = context.pipeline
        citation_count = len(context.final_citations)
        grounding_started = time.perf_counter()
        grounding_eval = assess_grounding_with_policy(
            context.final_answer,
            context.context_texts,
            citation_count=citation_count,
            has_selected_context=bool(pipeline.selected_context),
            status=context.final_status,
            policy=_build_grounding_policy(
                pipeline=pipeline,
                status=context.final_status,
                answer=context.final_answer,
                citation_count=citation_count,
            ),
        )
        context.grounding_ms = coerce_ms(
            (time.perf_counter() - grounding_started) * 1000, 0
        )
        grounding = grounding_eval.assessment
        grounded_score = grounding.grounded_score
        grounding_reason = grounding.grounding_reason
        hallucination_detected = grounding.hallucination_detected
        context.grounding_policy = grounding_eval.grounding_policy
        context.grounding_semantic_used = grounding_eval.grounding_semantic_used
        context.grounding_cache_hit = grounding_eval.grounding_cache_hit

        if (
            context.has_relevant_context
            and context.final_status != "insufficient_evidence"
        ):
            if grounded_score >= workflow.STRONG_GROUNDED_THRESHOLD:
                context.final_status = "answered"
            elif grounded_score >= workflow.VERY_LOW_GROUNDED_THRESHOLD or (
                citation_count > 0
                and context.has_context
                and not hallucination_detected
            ):
                context.final_status = "partial"
            else:
                context.final_status = "insufficient_evidence"
                context.final_answer = context.insufficient_answer
                context.final_citations = []
                citation_count = 0
                grounded_score = 0.0
                grounding_reason = "very_low_grounding_no_support"
                hallucination_detected = False
                context.stop_reason = "very_low_grounding"

        llm_fallback_used = bool(pipeline.generated.llm_fallback_used)
        llm_cache_hit = bool(pipeline.generated.llm_cache_hit)
        prior_stop_reason = context.stop_reason
        hallucination_refine_used = False

        if hallucination_detected and context.final_status != "insufficient_evidence":
            retrieval_confidence = _estimate_retrieval_confidence_from_pipeline(
                pipeline
            )
            lexical_grounding_score = grounded_overlap_score(
                context.final_answer,
                context.context_texts,
            )
            policy_signal = AdvancedPolicyInput(
                query_complexity=(
                    pipeline.query_budget.complexity
                    if pipeline.query_budget is not None
                    else context.query_complexity
                ),
                retrieval_confidence=retrieval_confidence,
                citation_count=citation_count,
                grounding_lexical_score=lexical_grounding_score,
                answer_length=len(context.final_answer.strip()),
                fast_path_used=bool(pipeline.generated.fast_path),
                user_selected_strictness=None,
            )
            should_run_refine = workflow.policy.should_run_hallucination_refine(
                query=state.normalized_query,
                signal=policy_signal,
                hallucination_detected=hallucination_detected,
            )

            refined_answer_text = ""
            refined_is_insufficient = False
            refined_hallucination = True
            refined_score = 0.0
            refined_reason = "policy_skip"
            refined_grounding_eval = None
            if should_run_refine:
                hallucination_refine_used = True
                context.add_llm_calls(1)
                refined_grounded_answer = (
                    await RefineStage._refine_strict_grounded_with_compat(
                        workflow,
                        query=state.normalized_query,
                        draft_answer=context.final_answer,
                        context_docs=pipeline.selected_context,
                        chat_history=context.normalized_history,
                        model=context.model,
                        response_language=context.resolved_language,
                    )
                )
                refined_answer_text = refined_grounded_answer.strip()
                refined_is_insufficient = (
                    refined_answer_text == context.insufficient_answer
                )
                refined_grounding_eval = assess_grounding_with_policy(
                    refined_answer_text,
                    context.context_texts,
                    citation_count=citation_count,
                    has_selected_context=bool(pipeline.selected_context),
                    status=(
                        "insufficient_evidence"
                        if refined_is_insufficient
                        else context.final_status
                    ),
                    policy=_build_grounding_policy(
                        pipeline=pipeline,
                        status=(
                            "insufficient_evidence"
                            if refined_is_insufficient
                            else context.final_status
                        ),
                        answer=refined_answer_text,
                        citation_count=citation_count,
                    ),
                )
                refined_grounding = refined_grounding_eval.assessment
                refined_score = refined_grounding.grounded_score
                refined_reason = refined_grounding.grounding_reason
                refined_hallucination = refined_grounding.hallucination_detected

            context.trace.append(
                {
                    "step": "hallucination_guard",
                    "triggered": True,
                    "strategy": (
                        "llm_refine" if should_run_refine else "skipped_by_policy"
                    ),
                    "refined_grounded_score": refined_score,
                    "refined_grounding_reason": refined_reason,
                    "refined_hallucination_detected": refined_hallucination,
                    "query_complexity": policy_signal.query_complexity,
                    "retrieval_confidence": retrieval_confidence,
                    "grounding_lexical_score": lexical_grounding_score,
                    "citation_count": citation_count,
                    "critique_category": context.critique_category,
                }
            )

            if should_run_refine and (
                refined_answer_text
                and not refined_is_insufficient
                and not refined_hallucination
            ):
                context.final_answer = refined_answer_text
                grounded_score = refined_score
                grounding_reason = refined_reason
                if refined_grounding_eval is not None:
                    context.grounding_policy = refined_grounding_eval.grounding_policy
                    context.grounding_semantic_used = (
                        refined_grounding_eval.grounding_semantic_used
                    )
                    context.grounding_cache_hit = (
                        refined_grounding_eval.grounding_cache_hit
                    )
                if refined_score >= workflow.STRONG_GROUNDED_THRESHOLD:
                    context.final_status = "answered"
                elif refined_score >= workflow.VERY_LOW_GROUNDED_THRESHOLD:
                    context.final_status = "partial"
                else:
                    context.final_status = "insufficient_evidence"
                    context.final_answer = context.insufficient_answer
                    context.final_citations = []
                    citation_count = 0
                    grounded_score = 0.0
                    grounding_reason = "very_low_grounding_after_refine"
                hallucination_detected = False
                if prior_stop_reason != "max_loop_reached":
                    context.stop_reason = "hallucination_refined"
            else:
                if (
                    context.has_relevant_context
                    and context.final_citations
                    and grounded_score >= workflow.VERY_LOW_GROUNDED_THRESHOLD
                ):
                    context.final_answer = workflow._cautious_answer_from_context(
                        pipeline.selected_context,
                        response_language=context.resolved_language,
                    )
                    context.final_status = "partial"
                    cautious_grounding_eval = assess_grounding_with_policy(
                        context.final_answer,
                        context.context_texts,
                        citation_count=len(context.final_citations),
                        has_selected_context=bool(pipeline.selected_context),
                        status=context.final_status,
                        policy=_build_grounding_policy(
                            pipeline=pipeline,
                            status=context.final_status,
                            answer=context.final_answer,
                            citation_count=len(context.final_citations),
                        ),
                    )
                    cautious_grounding = cautious_grounding_eval.assessment
                    grounded_score = cautious_grounding.grounded_score
                    grounding_reason = cautious_grounding.grounding_reason
                    hallucination_detected = cautious_grounding.hallucination_detected
                    context.grounding_policy = cautious_grounding_eval.grounding_policy
                    context.grounding_semantic_used = (
                        cautious_grounding_eval.grounding_semantic_used
                    )
                    context.grounding_cache_hit = (
                        cautious_grounding_eval.grounding_cache_hit
                    )
                    if prior_stop_reason != "max_loop_reached":
                        context.stop_reason = "weak_evidence_cautious"
                else:
                    context.final_answer = context.insufficient_answer
                    context.final_citations = []
                    context.final_status = "insufficient_evidence"
                    grounded_score = 0.0
                    grounding_reason = "hallucination_fallback_insufficient"
                    hallucination_detected = False
                    citation_count = 0
                    if prior_stop_reason != "max_loop_reached":
                        context.stop_reason = "hallucination_fallback_insufficient"
            context.language_mismatch = is_language_mismatch(
                context.final_answer, context.resolved_language
            )
        context.hallucination_refine_used = hallucination_refine_used

        if (
            context.has_relevant_context
            and context.final_status != "insufficient_evidence"
            and not context.final_citations
        ):
            context.final_citations = context.context_citations
        citation_count = len(context.final_citations)

        context.citation_count = citation_count
        context.grounded_score = grounded_score
        context.grounding_reason = grounding_reason
        context.hallucination_detected = hallucination_detected
        context.llm_fallback_used = llm_fallback_used
        context.llm_cache_hit = llm_cache_hit

        state.final_answer = context.final_answer
        state.citations = context.final_citations
        state.stop_reason = context.stop_reason
        state.language_mismatch = context.language_mismatch
        stage_ms = coerce_ms((time.perf_counter() - stage_started) * 1000, 0)
        context.timings["hallucination_guard_ms"] = stage_ms
        await emit_stream_event(
            context.event_handler,
            {
                "type": "advanced_stage",
                "stage": "hallucination_guard",
                "stage_ms": stage_ms,
                "status": context.final_status,
                **context.event_context,
            },
        )
        return state


class FinalGroundingStage(BasePipelineStage):
    """Stage 6: finalize grounding/citation metadata and emit trace entries."""

    async def execute(self, state: WorkflowState) -> WorkflowState:
        context = self.context
        if context.terminal_response is not None:
            return state
        if context.pipeline is None:
            return state

        stage_started = time.perf_counter()
        citation_count = len(context.final_citations)
        grounding_started = time.perf_counter()
        final_grounding_eval = assess_grounding_with_policy(
            context.final_answer,
            context.context_texts,
            citation_count=citation_count,
            has_selected_context=bool(context.pipeline.selected_context),
            status=context.final_status,
            policy=_build_grounding_policy(
                pipeline=context.pipeline,
                status=context.final_status,
                answer=context.final_answer,
                citation_count=citation_count,
            ),
        )
        context.grounding_ms = coerce_ms(
            (time.perf_counter() - grounding_started) * 1000, 0
        )
        context.timings["grounding_ms"] = context.grounding_ms
        final_grounding = final_grounding_eval.assessment
        context.citation_count = citation_count
        context.grounded_score = final_grounding.grounded_score
        context.grounding_reason = final_grounding.grounding_reason
        context.hallucination_detected = final_grounding.hallucination_detected
        context.grounding_policy = final_grounding_eval.grounding_policy
        context.grounding_semantic_used = final_grounding_eval.grounding_semantic_used
        context.grounding_cache_hit = final_grounding_eval.grounding_cache_hit

        context.trace.append(
            {
                "step": "evidence_decision",
                "has_context": context.has_context,
                "has_relevant_context": context.has_relevant_context,
                "critique_category": context.critique_category,
                "grounded_score": context.grounded_score,
                "grounding_reason": context.grounding_reason,
                "hallucination_detected": context.hallucination_detected,
                "status": context.final_status,
                "grounding_policy": context.grounding_policy,
                "grounding_semantic_used": context.grounding_semantic_used,
                "grounding_cache_hit": context.grounding_cache_hit,
                "grounding_ms": context.grounding_ms,
                "embedding_cache_hit": context.embedding_cache_hit,
                "retrieval_cache_hit": context.retrieval_cache_hit,
                "rerank_cache_hit": context.rerank_cache_hit,
                "llm_cache_hit": context.llm_cache_hit,
                "gate_strategy": context.gate_strategy,
                "critique_strategy": context.critique_strategy,
                "refine_used": context.refine_used,
                "language_rewrite_used": context.language_rewrite_used,
                "hallucination_refine_used": context.hallucination_refine_used,
                "llm_call_count_estimate": context.llm_call_count_estimate,
                "reused_standard_result": context.reused_standard_result,
                "standard_reuse_saved_steps": list(context.standard_reuse_saved_steps),
                "standard_reuse_reason": context.standard_reuse_reason,
            }
        )

        state.final_answer = context.final_answer
        state.citations = context.final_citations
        state.stop_reason = context.stop_reason
        state.language_mismatch = context.language_mismatch
        state.grounded_score = context.grounded_score
        state.grounding_reason = context.grounding_reason
        state.hallucination_detected = context.hallucination_detected
        state.llm_fallback_used = context.llm_fallback_used

        context.trace.append(
            {
                "step": "language_guard",
                "response_language": context.resolved_language,
                "language_mismatch": context.language_mismatch,
                "language_rewrite_used": context.language_rewrite_used,
            }
        )
        context.trace.append(
            {
                "step": "grounding_check",
                "grounded_score": context.grounded_score,
                "grounding_reason": context.grounding_reason,
                "hallucination_detected": context.hallucination_detected,
                "citation_count": context.citation_count,
                "llm_fallback_used": context.llm_fallback_used,
                "llm_cache_hit": context.llm_cache_hit,
                "grounding_policy": context.grounding_policy,
                "grounding_semantic_used": context.grounding_semantic_used,
                "grounding_cache_hit": context.grounding_cache_hit,
                "grounding_ms": context.grounding_ms,
                "embedding_cache_hit": context.embedding_cache_hit,
                "retrieval_cache_hit": context.retrieval_cache_hit,
                "rerank_cache_hit": context.rerank_cache_hit,
                "gate_strategy": context.gate_strategy,
                "critique_strategy": context.critique_strategy,
                "refine_used": context.refine_used,
                "language_rewrite_used": context.language_rewrite_used,
                "hallucination_refine_used": context.hallucination_refine_used,
                "llm_call_count_estimate": context.llm_call_count_estimate,
                "reused_standard_result": context.reused_standard_result,
                "standard_reuse_saved_steps": list(context.standard_reuse_saved_steps),
                "standard_reuse_reason": context.standard_reuse_reason,
            }
        )
        stage_ms = coerce_ms((time.perf_counter() - stage_started) * 1000, 0)
        context.timings["final_grounding_ms"] = stage_ms
        total_ms = coerce_ms((time.perf_counter() - context.start_time) * 1000, 0)
        timing_summary = context.timing_summary(total_ms=total_ms)
        context.trace.append(timing_summary)
        context.trace = ensure_completed_trace(context.trace, total_ms=total_ms)
        await emit_stream_event(
            context.event_handler,
            {
                "type": "advanced_stage",
                "stage": "final_grounding",
                "stage_ms": stage_ms,
                "timings_ms": {
                    key: value for key, value in timing_summary.items() if key != "step"
                },
                **context.event_context,
            },
        )
        return state
