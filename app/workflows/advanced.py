"""Advanced workflow implementation with retrieval gate, critique, retry, and refine."""

from __future__ import annotations

import time

from app.core.config import get_settings
from app.generation.citations import CitationBuilder
from app.schemas.api import AdvancedQueryResponse
from app.schemas.common import Mode
from app.schemas.retrieval import RetrievalResult
from app.schemas.workflow import WorkflowState
from app.workflows.critique import HeuristicCritic
from app.workflows.query_rewrite import QueryRewriter
from app.workflows.refine import AnswerRefiner
from app.workflows.retrieval_gate import HeuristicRetrievalGate
from app.workflows.shared import (
    assess_grounding,
    detect_response_language,
    is_language_mismatch,
    localized_insufficient_evidence,
    normalize_query,
    trim_chat_history,
)
from app.workflows.standard import StandardWorkflow


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
        refiner: AnswerRefiner | None = None,
    ) -> None:
        settings = get_settings()
        self.max_loops = max_loops if max_loops is not None else int(getattr(settings, "max_advanced_loops", 1))
        self.memory_window = max(0, int(getattr(settings, "memory_window", 3)))

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
        self.citation_builder = CitationBuilder()

    @staticmethod
    def _critique_category(note: str | None) -> str | None:
        if not note:
            return None
        lowered = note.strip().lower()
        for category in ("no_evidence", "weak_evidence", "incomplete_answer", "hallucination", "grounded"):
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
        normalized_history = trim_chat_history(
            chat_history,
            memory_window=self.memory_window,
        )

        state = WorkflowState(
            mode=Mode.ADVANCED,
            user_query=query,
            normalized_query=normalized_query,
            response_language=resolved_language,
            chat_history=normalized_history,
        )
        trace: list[dict] = []

        need_retrieval, gate_reason = self.retrieval_gate.decide(
            normalized_query,
            chat_history=normalized_history,
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
                "memory_window": self.memory_window,
                "memory_messages": len(normalized_history),
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
                grounding_reason="gate_no_retrieval_no_context",
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
                    chat_history=normalized_history,
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
                chat_history=normalized_history,
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
                chat_history=normalized_history,
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
                    "chunk_size": self.standard_workflow.chunk_size,
                    "chunk_overlap": self.standard_workflow.chunk_overlap,
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
                    "generated_status": pipeline.generated.status,
                    "generated_stop_reason": pipeline.generated.stop_reason,
                    "generated_confidence": pipeline.generated.confidence,
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
                grounding_reason="no_pipeline_result",
                citation_count=0,
                hallucination_detected=False,
                llm_fallback_used=False,
                trace=trace,
            )

        final_answer = pipeline.generated.answer.strip()
        final_citations = list(pipeline.generated.citations)
        final_status = pipeline.generated.status
        stop_reason = "critique_pass"

        context_texts = [item.content for item in pipeline.selected_context if item.content.strip()]
        has_context = bool(context_texts)
        critique_category = self._critique_category(critique_result.note)
        has_relevant_context = has_context and critique_category != "no_evidence"
        context_citations = self.citation_builder.build(pipeline.selected_context)
        if has_context and not final_citations:
            final_citations = context_citations

        insufficient_answer = localized_insufficient_evidence(resolved_language)
        force_abstain = "force abstain" in normalized_query.lower()
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
        elif critique_result.should_retry_retrieval and state.loop_count >= self.max_loops:
            stop_reason = "max_loop_reached"

        needs_refine_with_context = (
            critique_result.should_refine_answer
            or final_status == "insufficient_evidence"
            or critique_category in {"weak_evidence", "incomplete_answer", "hallucination"}
        )
        if needs_refine_with_context and has_relevant_context and final_status != "insufficient_evidence":
            final_answer = self.refiner.refine(
                query=normalized_query,
                draft_answer=final_answer,
                critique=critique_result,
                context=pipeline.selected_context,
                chat_history=normalized_history,
                model=model,
                response_language=resolved_language,
            )
            if stop_reason != "max_loop_reached":
                stop_reason = "refined_with_context"

        if (
            final_status == "insufficient_evidence"
            and has_relevant_context
            and final_citations
        ):
            refined_from_insufficient = self.refiner.refine_strict_grounded(
                query=normalized_query,
                draft_answer=final_answer or insufficient_answer,
                context=pipeline.selected_context,
                chat_history=normalized_history,
                model=model,
                response_language=resolved_language,
            ).strip()
            refined_is_insufficient = refined_from_insufficient == insufficient_answer
            if refined_from_insufficient and not refined_is_insufficient:
                final_answer = refined_from_insufficient
                final_status = "answered"
                if stop_reason != "max_loop_reached":
                    stop_reason = "recovered_from_context"
            else:
                final_answer = self._cautious_answer_from_context(
                    pipeline.selected_context,
                    response_language=resolved_language,
                )
                final_status = "partial"
                if stop_reason != "max_loop_reached":
                    stop_reason = "weak_evidence_cautious"

        if critique_result.has_conflict and final_status != "insufficient_evidence":
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

        citation_count = len(final_citations)
        grounding = assess_grounding(
            final_answer,
            context_texts,
            citation_count=citation_count,
            has_selected_context=bool(pipeline.selected_context),
            status=final_status,
        )
        grounded_score = grounding.grounded_score
        grounding_reason = grounding.grounding_reason
        hallucination_detected = grounding.hallucination_detected

        if has_relevant_context and final_status != "insufficient_evidence":
            if grounded_score >= self.STRONG_GROUNDED_THRESHOLD:
                final_status = "answered"
            elif (
                grounded_score >= self.VERY_LOW_GROUNDED_THRESHOLD
                or (citation_count > 0 and has_context and not hallucination_detected)
            ):
                final_status = "partial"
            else:
                final_status = "insufficient_evidence"
                final_answer = insufficient_answer
                final_citations = []
                citation_count = 0
                grounded_score = 0.0
                grounding_reason = "very_low_grounding_no_support"
                hallucination_detected = False
                stop_reason = "very_low_grounding"

        llm_fallback_used = bool(pipeline.generated.llm_fallback_used)

        prior_stop_reason = stop_reason

        if hallucination_detected and final_status != "insufficient_evidence":
            refined_grounded_answer = self.refiner.refine_strict_grounded(
                query=normalized_query,
                draft_answer=final_answer,
                context=pipeline.selected_context,
                chat_history=normalized_history,
                model=model,
                response_language=resolved_language,
            )
            refined_answer_text = refined_grounded_answer.strip()
            refined_is_insufficient = refined_answer_text == insufficient_answer
            refined_grounding = assess_grounding(
                refined_answer_text,
                context_texts,
                citation_count=citation_count,
                has_selected_context=bool(pipeline.selected_context),
                status="insufficient_evidence" if refined_is_insufficient else final_status,
            )
            refined_score = refined_grounding.grounded_score
            refined_reason = refined_grounding.grounding_reason
            refined_hallucination = refined_grounding.hallucination_detected
            trace.append(
                {
                    "step": "hallucination_guard",
                    "triggered": True,
                    "refined_grounded_score": refined_score,
                    "refined_grounding_reason": refined_reason,
                    "refined_hallucination_detected": refined_hallucination,
                    "critique_category": critique_category,
                }
            )

            if refined_answer_text and not refined_is_insufficient and not refined_hallucination:
                final_answer = refined_answer_text
                grounded_score = refined_score
                grounding_reason = refined_reason
                if refined_score >= self.STRONG_GROUNDED_THRESHOLD:
                    final_status = "answered"
                elif refined_score >= self.VERY_LOW_GROUNDED_THRESHOLD:
                    final_status = "partial"
                else:
                    final_status = "insufficient_evidence"
                    final_answer = insufficient_answer
                    final_citations = []
                    citation_count = 0
                    grounded_score = 0.0
                    grounding_reason = "very_low_grounding_after_refine"
                hallucination_detected = False
                if prior_stop_reason != "max_loop_reached":
                    stop_reason = "hallucination_refined"
            else:
                if has_relevant_context and final_citations and grounded_score >= self.VERY_LOW_GROUNDED_THRESHOLD:
                    final_answer = self._cautious_answer_from_context(
                        pipeline.selected_context,
                        response_language=resolved_language,
                    )
                    final_status = "partial"
                    cautious_grounding = assess_grounding(
                        final_answer,
                        context_texts,
                        citation_count=len(final_citations),
                        has_selected_context=bool(pipeline.selected_context),
                        status=final_status,
                    )
                    grounded_score = cautious_grounding.grounded_score
                    grounding_reason = cautious_grounding.grounding_reason
                    hallucination_detected = cautious_grounding.hallucination_detected
                    if prior_stop_reason != "max_loop_reached":
                        stop_reason = "weak_evidence_cautious"
                else:
                    final_answer = insufficient_answer
                    final_citations = []
                    final_status = "insufficient_evidence"
                    grounded_score = 0.0
                    grounding_reason = "hallucination_fallback_insufficient"
                    hallucination_detected = False
                    citation_count = 0
                    if prior_stop_reason != "max_loop_reached":
                        stop_reason = "hallucination_fallback_insufficient"
            language_mismatch = is_language_mismatch(final_answer, resolved_language)

        if has_relevant_context and final_status != "insufficient_evidence" and not final_citations:
            final_citations = context_citations
        citation_count = len(final_citations)

        final_grounding = assess_grounding(
            final_answer,
            context_texts,
            citation_count=citation_count,
            has_selected_context=bool(pipeline.selected_context),
            status=final_status,
        )
        grounded_score = final_grounding.grounded_score
        grounding_reason = final_grounding.grounding_reason
        hallucination_detected = final_grounding.hallucination_detected

        trace.append(
            {
                "step": "evidence_decision",
                "has_context": has_context,
                "has_relevant_context": has_relevant_context,
                "critique_category": critique_category,
                "grounded_score": grounded_score,
                "grounding_reason": grounding_reason,
                "hallucination_detected": hallucination_detected,
                "status": final_status,
            }
        )

        state.final_answer = final_answer
        state.citations = final_citations
        state.stop_reason = stop_reason
        state.language_mismatch = language_mismatch
        state.grounded_score = grounded_score
        state.grounding_reason = grounding_reason
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
                "grounding_reason": grounding_reason,
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
            grounding_reason=state.grounding_reason,
            citation_count=citation_count,
            hallucination_detected=state.hallucination_detected,
            llm_fallback_used=state.llm_fallback_used,
            trace=trace,
        )
