"""Metric helpers for practical RAG evaluation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable

from app.evaluation.schemas import (
    EvalExpectedBehavior,
    EvalMetrics,
    RetrievedSourceTrace,
    TraceExtraction,
)
from app.schemas.common import Citation

_TERM_PATTERN = re.compile(r"\w+")
_ABSTAIN_STATUSES = {"abstained", "insufficient_evidence"}
_GOLD_FIELD_PATTERN = re.compile(r"^\s*([a-zA-Z_]+)\s*[:=]\s*(.+?)\s*$")
_GOLD_FIELD_ALIASES = {
    "chunk_id": "chunk_id",
    "chunk": "chunk_id",
    "doc_id": "doc_id",
    "doc": "doc_id",
    "source": "source",
    "path": "path",
    "title": "title",
    "section": "section",
}


def _safe_normalize(value: str | None, *, is_path: bool = False) -> str:
    if value is None:
        return ""
    normalized = value.strip().lower()
    if not normalized:
        return ""
    normalized = " ".join(normalized.split())
    if is_path:
        normalized = normalized.replace("\\", "/")
        while "//" in normalized:
            normalized = normalized.replace("//", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
    return normalized


def _normalize_text(value: str | None) -> str:
    return _safe_normalize(value)


def _normalize_path(value: str | None) -> str:
    return _safe_normalize(value, is_path=True)


def _basename(path: str) -> str:
    if not path:
        return ""
    return path.rsplit("/", 1)[-1]


def _basename_stem(path: str) -> str:
    basename = _basename(path)
    if "." not in basename:
        return basename
    return basename.rsplit(".", 1)[0]


def _looks_like_path(text: str) -> bool:
    candidate = _normalize_path(text)
    return "/" in candidate or "\\" in text or "." in candidate


def _split_hash(text: str) -> tuple[str, str] | None:
    if "#" not in text:
        return None
    left, right = text.split("#", 1)
    left_norm = _normalize_text(left)
    right_norm = _normalize_text(right)
    if not left_norm or not right_norm:
        return None
    return left_norm, right_norm


def _candidate_chunk_id(candidate: RetrievedSourceTrace) -> str:
    return _normalize_text(candidate.chunk_id)


def _candidate_doc_id(candidate: RetrievedSourceTrace) -> str:
    return _normalize_text(candidate.doc_id)


def _candidate_title(candidate: RetrievedSourceTrace) -> str:
    return _normalize_text(candidate.title)


def _candidate_section(candidate: RetrievedSourceTrace) -> str:
    return _normalize_text(candidate.section)


def _candidate_paths(candidate: RetrievedSourceTrace) -> list[str]:
    values: list[str] = []
    for value in (candidate.source, candidate.path):
        normalized = _normalize_path(value)
        if normalized and normalized not in values:
            values.append(normalized)
    return values


def _candidate_path_basenames(candidate: RetrievedSourceTrace) -> set[str]:
    basenames: set[str] = set()
    for path in _candidate_paths(candidate):
        base = _basename(path)
        if base:
            basenames.add(base)
    return basenames


def _candidate_source_tokens(candidate: RetrievedSourceTrace) -> set[str]:
    tokens = {
        token
        for token in (
            _candidate_doc_id(candidate),
            _candidate_title(candidate),
            _candidate_section(candidate),
            _candidate_chunk_id(candidate),
        )
        if token
    }
    for path in _candidate_paths(candidate):
        tokens.add(path)
        basename = _basename(path)
        if basename:
            tokens.add(basename)
            stem = _basename_stem(path)
            if stem:
                tokens.add(stem)
    return tokens


def _path_basename_is_unambiguous(
    basename: str, candidates: list[RetrievedSourceTrace]
) -> bool:
    if not basename:
        return False
    matches = 0
    for candidate in candidates:
        if basename in _candidate_path_basenames(candidate):
            matches += 1
            if matches > 1:
                return False
    return matches == 1


def _match_path_value(
    expected: str,
    candidate: RetrievedSourceTrace,
    candidates: list[RetrievedSourceTrace],
) -> bool:
    expected_path = _normalize_path(expected)
    if not expected_path:
        return False
    candidate_paths = _candidate_paths(candidate)
    if expected_path in candidate_paths:
        return True

    expected_base = _basename(expected_path)
    if not expected_base:
        return False
    candidate_bases = _candidate_path_basenames(candidate)
    if expected_base not in candidate_bases:
        return False

    # Only allow basename fallback when one side is basename-only.
    expected_is_base_only = expected_path == expected_base
    candidate_has_base_only = any(path == _basename(path) for path in candidate_paths)
    if not (expected_is_base_only or candidate_has_base_only):
        return False

    return _path_basename_is_unambiguous(expected_base, candidates)


def _is_title_unambiguous(
    title: str, candidates: list[RetrievedSourceTrace]
) -> bool:
    expected = _normalize_text(title)
    if not expected:
        return False
    matches = sum(
        1 for candidate in candidates if _candidate_title(candidate) == expected
    )
    return matches == 1


def _is_section_unambiguous(
    section: str, candidates: list[RetrievedSourceTrace]
) -> bool:
    expected = _normalize_text(section)
    if not expected:
        return False
    matches = sum(
        1 for candidate in candidates if _candidate_section(candidate) == expected
    )
    return matches == 1


def _is_title_section_unambiguous(
    title: str, section: str, candidates: list[RetrievedSourceTrace]
) -> bool:
    expected_title = _normalize_text(title)
    expected_section = _normalize_text(section)
    if not expected_title or not expected_section:
        return False
    matches = sum(
        1
        for candidate in candidates
        if _candidate_title(candidate) == expected_title
        and _candidate_section(candidate) == expected_section
    )
    return matches == 1


def _derive_doc_id_from_chunk_id(chunk_id: str | None) -> str | None:
    if not isinstance(chunk_id, str) or not chunk_id:
        return None
    if "_chunk_" not in chunk_id:
        return None
    return chunk_id.split("_chunk_", 1)[0]


def _source_fingerprint(source: RetrievedSourceTrace) -> tuple[str, str, str, str, str, str]:
    return (
        _normalize_text(source.chunk_id),
        _normalize_text(source.doc_id),
        _normalize_path(source.source),
        _normalize_path(source.path),
        _normalize_text(source.title),
        _normalize_text(source.section),
    )


def _to_retrieved_source(doc: dict) -> RetrievedSourceTrace | None:
    chunk_id = doc.get("chunk_id") if isinstance(doc.get("chunk_id"), str) else None
    doc_id = doc.get("doc_id") if isinstance(doc.get("doc_id"), str) else None
    source = doc.get("source") if isinstance(doc.get("source"), str) else None
    title = doc.get("title") if isinstance(doc.get("title"), str) else None
    section = doc.get("section") if isinstance(doc.get("section"), str) else None
    path = doc.get("path") if isinstance(doc.get("path"), str) else None

    if doc_id is None:
        doc_id = _derive_doc_id_from_chunk_id(chunk_id)
    if path is None:
        path = source

    candidate = RetrievedSourceTrace(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source=source,
        path=path,
        title=title,
        section=section,
    )
    if not any(_source_fingerprint(candidate)):
        return None
    return candidate


def _parse_gold_source_fields(gold: str) -> dict[str, str]:
    raw = gold.strip()
    if not raw:
        return {}

    if raw.startswith("{") and raw.endswith("}"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            parsed: dict[str, str] = {}
            for key, value in payload.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    continue
                alias = _GOLD_FIELD_ALIASES.get(key.strip().lower())
                if alias:
                    parsed[alias] = value.strip()
            if parsed:
                return parsed

    parsed: dict[str, str] = {}
    for part in re.split(r"[|,]", raw):
        match = _GOLD_FIELD_PATTERN.match(part)
        if match is None:
            continue
        raw_key, raw_value = match.groups()
        key = _GOLD_FIELD_ALIASES.get(raw_key.strip().lower())
        if key is None:
            continue
        value = raw_value.strip()
        if value:
            parsed[key] = value
    return parsed


def _match_structured_gold_source(
    fields: dict[str, str],
    candidate: RetrievedSourceTrace,
    candidates: list[RetrievedSourceTrace],
) -> bool:
    # 1) exact chunk_id
    if "chunk_id" in fields:
        return _candidate_chunk_id(candidate) == _normalize_text(fields["chunk_id"])

    # 2) exact doc_id
    if "doc_id" in fields:
        return _candidate_doc_id(candidate) == _normalize_text(fields["doc_id"])

    # 3) source/path match
    if "source" in fields or "path" in fields:
        source_ok = True
        path_ok = True
        if "source" in fields:
            source_ok = _match_path_value(fields["source"], candidate, candidates)
        if "path" in fields:
            path_ok = _match_path_value(fields["path"], candidate, candidates)
        return source_ok and path_ok

    title_value = fields.get("title")
    section_value = fields.get("section")

    # 4) title + section combined
    if title_value and section_value:
        return (
            _candidate_title(candidate) == _normalize_text(title_value)
            and _candidate_section(candidate) == _normalize_text(section_value)
            and _is_title_section_unambiguous(title_value, section_value, candidates)
        )

    # 5) title-only / section-only if unambiguous
    if title_value:
        return (
            _candidate_title(candidate) == _normalize_text(title_value)
            and _is_title_unambiguous(title_value, candidates)
        )
    if section_value:
        return (
            _candidate_section(candidate) == _normalize_text(section_value)
            and _is_section_unambiguous(section_value, candidates)
        )

    return False


def _match_legacy_fallback(gold: str, candidate: RetrievedSourceTrace) -> bool:
    normalized_gold = _normalize_text(gold)
    if not normalized_gold:
        return False

    source_tokens = _candidate_source_tokens(candidate)
    if normalized_gold in source_tokens:
        return True

    normalized_gold_path = _normalize_path(gold)
    if normalized_gold_path and normalized_gold_path in source_tokens:
        return True

    chunk_id = _candidate_chunk_id(candidate)
    split = _split_hash(gold)
    if split is None:
        return False

    left, right = split
    left_match = left in source_tokens or _normalize_path(left) in source_tokens
    if not left_match:
        return False
    if not chunk_id:
        return False

    return chunk_id == right or chunk_id.startswith(right)


def _match_legacy_gold_source(
    gold: str,
    candidate: RetrievedSourceTrace,
    candidates: list[RetrievedSourceTrace],
) -> bool:
    normalized_gold = _normalize_text(gold)
    if not normalized_gold:
        return False

    # 1) exact chunk_id
    if normalized_gold == _candidate_chunk_id(candidate):
        return True

    # 2) exact doc_id
    if normalized_gold == _candidate_doc_id(candidate):
        return True

    # 3) source/path match
    if _looks_like_path(gold) and _match_path_value(gold, candidate, candidates):
        return True

    split = _split_hash(gold)
    if split is not None:
        left, right = split
        # 4) title + section combined
        if "chunk" not in right and not _looks_like_path(left):
            if (
                _candidate_title(candidate) == left
                and _candidate_section(candidate) == right
                and _is_title_section_unambiguous(left, right, candidates)
            ):
                return True

    # 5) title-only / section-only if unambiguous
    if (
        _candidate_title(candidate) == normalized_gold
        and _is_title_unambiguous(normalized_gold, candidates)
    ):
        return True
    if (
        _candidate_section(candidate) == normalized_gold
        and _is_section_unambiguous(normalized_gold, candidates)
    ):
        return True

    # 6) backward-compatible fallback string match
    return _match_legacy_fallback(gold, candidate)


def tokenize_terms(text: str) -> set[str]:
    """Extract lowercase lexical terms for heuristic overlap metrics."""
    return {token.lower() for token in _TERM_PATTERN.findall(text) if len(token) > 2}


def extract_trace_fields(trace: list[dict]) -> TraceExtraction:
    """Extract retrieval/rerank details from workflow traces."""
    retrieved_chunk_ids: list[str] = []
    retrieved_sources: list[RetrievedSourceTrace] = []
    rerank_scores: dict[str, float] = {}
    retrieved_count = 0
    selected_count = 0
    selected_context_texts: list[str] = []
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    source_fingerprints: set[tuple[str, str, str, str, str, str]] = set()

    def _push_chunk_id(chunk_id: str) -> None:
        if chunk_id and chunk_id not in retrieved_chunk_ids:
            retrieved_chunk_ids.append(chunk_id)

    def _push_source_doc(doc: dict) -> None:
        source = _to_retrieved_source(doc)
        if source is None:
            return
        fingerprint = _source_fingerprint(source)
        if fingerprint in source_fingerprints:
            return
        source_fingerprints.add(fingerprint)
        retrieved_sources.append(source)
        if source.chunk_id:
            _push_chunk_id(source.chunk_id)

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
            docs = step.get("docs", [])
            if isinstance(docs, list):
                for doc in docs:
                    if isinstance(doc, dict):
                        _push_source_doc(doc)
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
                    _push_source_doc(doc)
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
                    if isinstance(doc, dict):
                        _push_source_doc(doc)
                        if isinstance(doc.get("content"), str):
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

            retrieved_docs = step.get("retrieved_docs", [])
            if isinstance(retrieved_docs, list):
                for doc in retrieved_docs:
                    if isinstance(doc, dict):
                        _push_source_doc(doc)

            reranked_docs = step.get("reranked_docs", [])
            if isinstance(reranked_docs, list):
                for doc in reranked_docs:
                    if not isinstance(doc, dict):
                        continue
                    _push_source_doc(doc)
                    chunk_id = doc.get("chunk_id")
                    if isinstance(chunk_id, str):
                        _push_chunk_id(chunk_id)
                        rerank_score = doc.get("rerank_score")
                        if isinstance(rerank_score, (int, float)):
                            rerank_scores[chunk_id] = float(rerank_score)

            selected_docs = step.get("selected_context_docs", [])
            if isinstance(selected_docs, list):
                for doc in selected_docs:
                    if isinstance(doc, dict):
                        _push_source_doc(doc)
                        if isinstance(doc.get("content"), str):
                            selected_context_texts.append(doc["content"])

    if retrieved_count == 0 and retrieved_chunk_ids:
        retrieved_count = len(retrieved_chunk_ids)

    return TraceExtraction(
        retrieved_chunk_ids=retrieved_chunk_ids,
        retrieved_sources=retrieved_sources,
        rerank_scores=rerank_scores,
        retrieved_count=retrieved_count,
        selected_context_count=selected_count,
        selected_context_texts=selected_context_texts,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _match_gold_source(
    gold: str,
    candidate: RetrievedSourceTrace,
    candidates: list[RetrievedSourceTrace],
) -> bool:
    normalized_gold = _normalize_text(gold)
    if not normalized_gold:
        return False

    structured_fields = _parse_gold_source_fields(gold)
    if structured_fields:
        return _match_structured_gold_source(structured_fields, candidate, candidates)
    return _match_legacy_gold_source(gold, candidate, candidates)


def cited_gold_source_overlap(
    citations: list[Citation], gold_sources: Iterable[str]
) -> float | None:
    """Compute overlap between citations and expected sources."""
    expected = [source for source in gold_sources if source and source.strip()]
    if not expected:
        return None

    citation_sources: list[RetrievedSourceTrace] = []
    for citation in citations:
        citation_sources.append(
            RetrievedSourceTrace(
                chunk_id=citation.chunk_id,
                doc_id=citation.doc_id,
                source=citation.source,
                path=citation.source,
                title=citation.title,
                section=citation.section,
            )
        )

    matched = sum(
        1 for source in expected if any(
            _match_gold_source(source, citation, citation_sources)
            for citation in citation_sources
        )
    )
    return matched / len(expected)


def compute_retrieval_metrics(
    retrieved_chunk_ids: list[str],
    gold_sources: list[str],
    retrieved_sources: list[RetrievedSourceTrace] | None = None,
) -> tuple[bool, float, float]:
    """Compute Hit@K, MRR@K, and nDCG@K against gold sources."""
    expected = [source for source in gold_sources if source and source.strip()]
    if not expected:
        return False, 0.0, 0.0

    ranked_sources: list[RetrievedSourceTrace] = []
    source_by_chunk_id: dict[str, RetrievedSourceTrace] = {}
    for source in retrieved_sources or []:
        chunk_id = _normalize_text(source.chunk_id)
        if chunk_id and chunk_id not in source_by_chunk_id:
            source_by_chunk_id[chunk_id] = source

    seen_chunk_ids: set[str] = set()
    for chunk_id in retrieved_chunk_ids:
        normalized_chunk_id = _normalize_text(chunk_id)
        if not normalized_chunk_id or normalized_chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(normalized_chunk_id)
        ranked_sources.append(
            source_by_chunk_id.get(normalized_chunk_id)
            or RetrievedSourceTrace(
                chunk_id=chunk_id,
                doc_id=_derive_doc_id_from_chunk_id(chunk_id),
            )
        )

    if not ranked_sources:
        ranked_sources = list(retrieved_sources or [])

    if not ranked_sources:
        return False, 0.0, 0.0

    hit = False
    mrr = 0.0
    dcg = 0.0
    matched_golds: set[str] = set()

    for i, source in enumerate(ranked_sources):
        newly_matched = False
        for gold in expected:
            if gold not in matched_golds:
                if _match_gold_source(gold, source, ranked_sources):
                    matched_golds.add(gold)
                    newly_matched = True
                    break

        if newly_matched:
            hit = True
            if mrr == 0.0:
                mrr = 1.0 / (i + 1)
            dcg += 1.0 / math.log2(i + 2)

    if not hit:
        return False, 0.0, 0.0

    idcg = 0.0
    num_expected = min(len(expected), len(ranked_sources))
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
        trace_fields.retrieved_chunk_ids,
        gold_sources,
        trace_fields.retrieved_sources,
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
