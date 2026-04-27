"""Metric helpers for practical RAG evaluation."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from app.evaluation.schemas import EvalExpectedBehavior, EvalMetrics, TraceExtraction
from app.schemas.common import Citation

_TERM_PATTERN = re.compile(r"\w+")
_ABSTAIN_STATUSES = {"abstained", "insufficient_evidence"}


def tokenize_terms(text: str) -> set[str]:
    """Extract lowercase lexical terms for heuristic overlap metrics."""
    return {token.lower() for token in _TERM_PATTERN.findall(text) if len(token) > 2}


def extract_trace_fields(trace: list[dict]) -> TraceExtraction:
    """Extract retrieval/rerank details from workflow traces."""
    retrieved_chunk_ids: list[str] = []
    rerank_scores: dict[str, float] = {}
    retrieved_count = 0
    selected_count = 0
    selected_context_texts: list[str] = []
    chunk_size: int | None = None
    chunk_overlap: int | None = None

    def _push_chunk_id(chunk_id: str) -> None:
        if chunk_id and chunk_id not in retrieved_chunk_ids:
            retrieved_chunk_ids.append(chunk_id)

    for step in trace:
        if not isinstance(step, dict):
            continue
        step_name = str(step.get("step", ""))

        if step_name == "retrieve":
            chunk_ids = step.get("chunk_ids", [])
            if isinstance(chunk_ids, list):
                for value in chunk_ids:
                    if isinstance(value, str):
                        _push_chunk_id(value)
            count = step.get("count")
            if isinstance(count, int):
                retrieved_count = count
            step_chunk_size = step.get("chunk_size")
            if isinstance(step_chunk_size, int):
                chunk_size = step_chunk_size
            step_chunk_overlap = step.get("chunk_overlap")
            if isinstance(step_chunk_overlap, int):
                chunk_overlap = step_chunk_overlap

        if step_name == "rerank":
            docs = step.get("docs", [])
            if isinstance(docs, list):
                for doc in docs:
                    if not isinstance(doc, dict):
                        continue
                    chunk_id = doc.get("chunk_id")
                    if isinstance(chunk_id, str):
                        _push_chunk_id(chunk_id)
                        rerank_score = doc.get("rerank_score")
                        if isinstance(rerank_score, (int, float)):
                            rerank_scores[chunk_id] = float(rerank_score)

        if step_name == "context_select":
            count = step.get("count")
            if isinstance(count, int):
                selected_count = count
            docs = step.get("docs", [])
            if isinstance(docs, list):
                for doc in docs:
                    if isinstance(doc, dict) and isinstance(doc.get("content"), str):
                        selected_context_texts.append(doc["content"])

        if step_name == "loop":
            loop_chunk_size = step.get("chunk_size")
            if isinstance(loop_chunk_size, int):
                chunk_size = loop_chunk_size
            loop_chunk_overlap = step.get("chunk_overlap")
            if isinstance(loop_chunk_overlap, int):
                chunk_overlap = loop_chunk_overlap
            loop_retrieved_count = step.get("retrieved_count")
            if isinstance(loop_retrieved_count, int):
                retrieved_count = loop_retrieved_count
            loop_selected_count = step.get("selected_count")
            if isinstance(loop_selected_count, int):
                selected_count = loop_selected_count

            reranked_docs = step.get("reranked_docs", [])
            if isinstance(reranked_docs, list):
                for doc in reranked_docs:
                    if not isinstance(doc, dict):
                        continue
                    chunk_id = doc.get("chunk_id")
                    if isinstance(chunk_id, str):
                        _push_chunk_id(chunk_id)
                        rerank_score = doc.get("rerank_score")
                        if isinstance(rerank_score, (int, float)):
                            rerank_scores[chunk_id] = float(rerank_score)

            selected_docs = step.get("selected_context_docs", [])
            if isinstance(selected_docs, list):
                for doc in selected_docs:
                    if isinstance(doc, dict) and isinstance(doc.get("content"), str):
                        selected_context_texts.append(doc["content"])

    if retrieved_count == 0 and retrieved_chunk_ids:
        retrieved_count = len(retrieved_chunk_ids)

    return TraceExtraction(
        retrieved_chunk_ids=retrieved_chunk_ids,
        rerank_scores=rerank_scores,
        retrieved_count=retrieved_count,
        selected_context_count=selected_count,
        selected_context_texts=selected_context_texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _match_gold_source(
    gold: str, citation_tokens: set[str], citation_compound: list[str]
) -> bool:
    normalized_gold = gold.strip().lower()
    if not normalized_gold:
        return False
    if normalized_gold in citation_tokens:
        return True
    return any(normalized_gold in item for item in citation_compound)


def cited_gold_source_overlap(
    citations: list[Citation], gold_sources: Iterable[str]
) -> float | None:
    """Compute overlap between citations and expected sources."""
    expected = [source for source in gold_sources if source and source.strip()]
    if not expected:
        return None

    citation_tokens: set[str] = set()
    citation_compound: list[str] = []
    for citation in citations:
        citation_tokens.update(
            {
                citation.chunk_id.lower(),
                citation.doc_id.lower(),
                citation.source.lower(),
            }
        )
        citation_compound.append(f"{citation.source}#{citation.chunk_id}".lower())
        citation_compound.append(f"{citation.doc_id}#{citation.chunk_id}".lower())

    matched = sum(
        1
        for source in expected
        if _match_gold_source(source, citation_tokens, citation_compound)
    )
    return matched / len(expected)


def compute_retrieval_metrics(
    retrieved_chunk_ids: list[str], gold_sources: list[str]
) -> tuple[bool, float, float]:
    """Compute Hit@K, MRR@K, and nDCG@K against gold sources."""
    expected = [source for source in gold_sources if source and source.strip()]
    if not expected or not retrieved_chunk_ids:
        return False, 0.0, 0.0

    hit = False
    mrr = 0.0
    dcg = 0.0

    for i, chunk_id in enumerate(retrieved_chunk_ids):
        # Extract base doc_id if chunk_id is standard deterministic format
        doc_id = chunk_id.split("_chunk_")[0] if "_chunk_" in chunk_id else ""

        citation_tokens = {chunk_id.lower(), doc_id.lower()}
        citation_compound = [f"{doc_id}#{chunk_id}".lower()]

        is_relevant = any(
            _match_gold_source(gold, citation_tokens, citation_compound)
            for gold in expected
        )

        if is_relevant:
            hit = True
            if mrr == 0.0:
                mrr = 1.0 / (i + 1)
            dcg += 1.0 / math.log2(i + 2)

    if not hit:
        return False, 0.0, 0.0

    idcg = 0.0
    num_expected = min(len(expected), len(retrieved_chunk_ids))
    for i in range(num_expected):
        idcg += 1.0 / math.log2(i + 2)

    ndcg = dcg / idcg if idcg > 0 else 0.0
    return hit, mrr, ndcg


def compute_metrics(
    *,
    expected_behavior: EvalExpectedBehavior,
    answer: str,
    citations: list[Citation],
    confidence: float | None,
    grounded_score: float | None = None,
    status: str | None,
    loop_count: int | None,
    stop_reason: str | None,
    latency_ms: int | None,
    trace_fields: TraceExtraction,
    reference_answer: str | None,
    gold_sources: list[str],
) -> EvalMetrics:
    """Compute lightweight + heuristic metrics without external judge models."""
    normalized_status = (status or "answered").lower()
    did_abstain = normalized_status in _ABSTAIN_STATUSES

    retry_used = False
    if loop_count is not None and loop_count > 1:
        retry_used = True
    if stop_reason and "retry" in stop_reason.lower():
        retry_used = True
    if expected_behavior == "retry" and loop_count is not None and loop_count >= 2:
        retry_used = True

    if expected_behavior == "abstain":
        abstain_match = did_abstain
    elif expected_behavior == "retry":
        abstain_match = not did_abstain
    else:
        abstain_match = not did_abstain

    answer_non_empty = bool(answer.strip())

    answer_contains_reference_keywords: bool | None = None
    if reference_answer and reference_answer.strip():
        ref_terms = tokenize_terms(reference_answer)
        answer_terms = tokenize_terms(answer)
        answer_contains_reference_keywords = bool(ref_terms.intersection(answer_terms))

    gold_overlap = cited_gold_source_overlap(citations, gold_sources)

    hit, mrr, ndcg = compute_retrieval_metrics(
        trace_fields.retrieved_chunk_ids, gold_sources
    )

    groundedness_proxy: float | None = None
    groundedness_proxy_note: str | None = None
    if trace_fields.selected_context_texts:
        answer_terms = tokenize_terms(answer)
        context_terms = tokenize_terms(" ".join(trace_fields.selected_context_texts))
        groundedness_proxy = (
            (len(answer_terms.intersection(context_terms)) / len(answer_terms))
            if answer_terms
            else 0.0
        )
        groundedness_proxy_note = (
            "Proxy only: lexical overlap between answer and selected context."
        )
    else:
        groundedness_proxy_note = (
            "Proxy unavailable: selected context text not present in trace."
        )

    return EvalMetrics(
        citation_count=len(citations),
        has_citations=len(citations) > 0,
        abstain_match=abstain_match,
        retry_used=retry_used,
        latency_ms=latency_ms,
        confidence=confidence,
        grounded_score=grounded_score,
        retrieved_count=trace_fields.retrieved_count,
        selected_context_count=trace_fields.selected_context_count,
        chunk_size=trace_fields.chunk_size,
        chunk_overlap=trace_fields.chunk_overlap,
        retrieval_hit=hit,
        retrieval_mrr=mrr,
        retrieval_ndcg=ndcg,
        answer_non_empty=answer_non_empty,
        answer_contains_reference_keywords=answer_contains_reference_keywords,
        cited_gold_source_overlap=gold_overlap,
        groundedness_proxy=groundedness_proxy,
        groundedness_proxy_note=groundedness_proxy_note,
    )
