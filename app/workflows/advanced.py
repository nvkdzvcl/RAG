"""Advanced workflow implementation with retrieval gate, critique, retry, and refine."""

from __future__ import annotations

import time

from app.core.config import get_settings
from app.schemas.api import AdvancedQueryResponse
from app.schemas.common import Mode
from app.schemas.workflow import WorkflowState
from app.workflows.critique import HeuristicCritic
from app.workflows.query_rewrite import QueryRewriter
from app.workflows.refine import AnswerRefiner
from app.workflows.retrieval_gate import HeuristicRetrievalGate
from app.workflows.shared import (
    detect_response_language,
    detect_hallucination,
    grounded_overlap_score,
    is_language_mismatch,
    localized_insufficient_evidence,
    normalize_query,
)
from app.workflows.standard import StandardWorkflow


class AdvancedWorkflow:
    """Practical Self-RAG workflow reusing standard retrieval/generation pipeline."""

    def __init__(
        self,
        *,
        standard_workflow: StandardWorkflow | None = None,
        max_loops: int | None = None,
        retrieval_gate: HeuristicRetrievalGate | None = None,
        query_rewriter: QueryRewriter | None = None,
        critic: HeuristicCritic | None = None,
        refiner: AnswerRefiner | None = None,
    ) -> None:
        settings = get_settings()
        self.max_loops = max_loops if max_loops is not None else settings.max_advanced_loops

        self.standard_workflow = standard_workflow or StandardWorkflow()
        llm_client = self.standard_workflow.generator.llm_client
        self.llm_client = llm_client

        self.retrieval_gate = retrieval_gate or HeuristicRetrievalGate(
            llm_client=llm_client,
            prompt_repository=self.standard_workflow.generator.prompt_repository,
        )
        self.query_rewriter = query_rewriter or QueryRewriter(
            llm_client=llm_client,
            prompt_repository=self.standard_workflow.generator.prompt_repository,
        )
        self.critic = critic or HeuristicCritic(
            llm_client=llm_client,
            prompt_repository=self.standard_workflow.generator.prompt_repository,
        )
        self.refiner = refiner or AnswerRefiner(
            llm_client=llm_client,
            prompt_repository=self.standard_workflow.generator.prompt_repository,
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

    def run(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
    ) -> AdvancedQueryResponse:
        start = time.perf_counter()
        normalized_query = normalize_query(query)
        resolved_language = response_language or detect_response_language(query)

        state = WorkflowState(
            mode=Mode.ADVANCED,
            user_query=query,
            normalized_query=normalized_query,
            response_language=resolved_language,
            chat_history=chat_history or [],
        )
        trace: list[dict] = []

        need_retrieval, gate_reason = self.retrieval_gate.decide(
            normalized_query,
            chat_history=chat_history,
            model=model,
            response_language=resolved_language,
        )
        state.need_retrieval = need_retrieval
        trace.append(
            {
                "step": "retrieval_gate",
                "need_retrieval": need_retrieval,
                "reason": gate_reason,
                "response_language": resolved_language,
            }
        )

        if not need_retrieval:
            answer, confidence, status, stop_reason = self._direct_answer_without_retrieval(
                normalized_query,
                model=model,
                response_language=resolved_language,
            )
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return AdvancedQueryResponse(
                mode="advanced",
                answer=answer,
                citations=[],
                confidence=confidence,
                status=status,
                stop_reason=stop_reason,
                latency_ms=elapsed_ms,
                loop_count=0,
                response_language=resolved_language,
                language_mismatch=False,
                grounded_score=0.0,
                citation_count=0,
                hallucination_detected=False,
                llm_fallback_used=False,
                trace=trace,
            )

        current_query = normalized_query
        state.rewritten_queries = [current_query]

        pipeline = None
        critique_result = None

        for loop in range(1, self.max_loops + 1):
            state.loop_count = loop

            if loop > 1:
                rewrites = self.query_rewriter.rewrite(
                    current_query,
                    critique=critique_result,
                    loop_count=loop,
                    model=model,
                    response_language=resolved_language,
                )
                if rewrites:
                    current_query = rewrites[0]
                    for candidate in rewrites:
                        if candidate not in state.rewritten_queries:
                            state.rewritten_queries.append(candidate)

            pipeline = self.standard_workflow.run_pipeline(
                query=current_query,
                mode=Mode.ADVANCED,
                model=model,
                response_language=resolved_language,
            )

            state.retrieved_docs = [item.model_dump() for item in pipeline.retrieved]
            state.reranked_docs = [item.model_dump() for item in pipeline.reranked]
            state.selected_context = [item.model_dump() for item in pipeline.selected_context]
            state.draft_answer = pipeline.generated.answer

            critique_result = self.critic.critique(
                query=normalized_query,
                draft_answer=pipeline.generated.answer,
                context=pipeline.selected_context,
                loop_count=loop,
                max_loops=self.max_loops,
                model=model,
                response_language=resolved_language,
            )
            state.critique = critique_result
            state.confidence = critique_result.confidence

            trace.append(
                {
                    "step": "loop",
                    "loop": loop,
                    "query": current_query,
                    "retrieved_count": len(pipeline.retrieved),
                    "reranked_count": len(pipeline.reranked),
                    "reranked_docs": [
                        {
                            "chunk_id": item.chunk_id,
                            "doc_id": item.doc_id,
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
                            "block_type": item.metadata.get("block_type"),
                            "ocr": bool(item.metadata.get("ocr")),
                            "content": item.content,
                        }
                        for item in pipeline.selected_context
                    ],
                    "critique": critique_result.model_dump(),
                }
            )

            if critique_result.should_retry_retrieval and loop < self.max_loops:
                continue
            break

        if pipeline is None or critique_result is None:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return AdvancedQueryResponse(
                mode="advanced",
                answer=localized_insufficient_evidence(resolved_language),
                citations=[],
                confidence=0.0,
                status="insufficient_evidence",
                stop_reason="no_pipeline_result",
                latency_ms=elapsed_ms,
                loop_count=state.loop_count,
                response_language=resolved_language,
                language_mismatch=False,
                grounded_score=0.0,
                citation_count=0,
                hallucination_detected=False,
                llm_fallback_used=False,
                trace=trace,
            )

        final_answer = pipeline.generated.answer
        final_citations = pipeline.generated.citations
        final_status = pipeline.generated.status
        stop_reason = "critique_pass"

        force_abstain = "force abstain" in normalized_query.lower()
        if force_abstain or (
            not critique_result.enough_evidence
            and not critique_result.should_retry_retrieval
            and not critique_result.should_refine_answer
        ):
            final_answer = localized_insufficient_evidence(resolved_language)
            final_citations = []
            final_status = "insufficient_evidence"
            stop_reason = "critic_abstain"
        elif critique_result.should_retry_retrieval and state.loop_count >= self.max_loops:
            stop_reason = "max_loop_reached"
        elif critique_result.should_refine_answer:
            final_answer = self.refiner.refine(
                query=normalized_query,
                draft_answer=pipeline.generated.answer,
                critique=critique_result,
                context=pipeline.selected_context,
                model=model,
                response_language=resolved_language,
            )
            stop_reason = "refined_after_critique"

        if critique_result.has_conflict and stop_reason == "critique_pass":
            if resolved_language == "vi":
                final_answer = final_answer + "\n\nPhát hiện khả năng xung đột giữa các nguồn."
            else:
                final_answer = final_answer + "\n\nPotential conflict detected in sources."
            stop_reason = "conflict_detected"

        language_mismatch = is_language_mismatch(final_answer, resolved_language)
        if language_mismatch and final_status != "insufficient_evidence":
            rewritten = self.standard_workflow._rewrite_answer_language(
                query=normalized_query,
                answer=final_answer,
                response_language=resolved_language,
                model=model,
            )
            if rewritten:
                final_answer = rewritten
                language_mismatch = is_language_mismatch(final_answer, resolved_language)
                if not language_mismatch and stop_reason == "critique_pass":
                    stop_reason = "language_refined"

        context_texts = [item.content for item in pipeline.selected_context if item.content.strip()]
        grounded_score = grounded_overlap_score(final_answer, context_texts)
        hallucination_detected = detect_hallucination(
            final_answer,
            context_texts,
            status=final_status,
        )
        citation_count = len(final_citations)
        llm_fallback_used = bool(pipeline.generated.llm_fallback_used)

        state.final_answer = final_answer
        state.citations = final_citations
        state.stop_reason = stop_reason
        state.language_mismatch = language_mismatch
        state.grounded_score = grounded_score
        state.hallucination_detected = hallucination_detected
        state.llm_fallback_used = llm_fallback_used
        trace.append(
            {
                "step": "language_guard",
                "response_language": resolved_language,
                "language_mismatch": language_mismatch,
            }
        )
        trace.append(
            {
                "step": "grounding_check",
                "grounded_score": grounded_score,
                "hallucination_detected": hallucination_detected,
                "citation_count": citation_count,
                "llm_fallback_used": llm_fallback_used,
            }
        )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        return AdvancedQueryResponse(
            mode="advanced",
            answer=state.final_answer,
            citations=state.citations,
            confidence=state.confidence if state.confidence is not None else pipeline.generated.confidence,
            status=final_status,
            stop_reason=state.stop_reason,
            latency_ms=elapsed_ms,
            loop_count=state.loop_count,
            response_language=resolved_language,
            language_mismatch=state.language_mismatch,
            grounded_score=state.grounded_score,
            citation_count=citation_count,
            hallucination_detected=state.hallucination_detected,
            llm_fallback_used=state.llm_fallback_used,
            trace=trace,
        )
