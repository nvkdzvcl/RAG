"""Reranker contract and baseline implementation hook."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

from app.schemas.retrieval import RetrievalResult


class BaseReranker(Protocol):
    """Reranker interface for post-retrieval ordering."""

    def rerank(self, query: str, docs: list[RetrievalResult], top_k: int | None = None) -> list[RetrievalResult]:
        """Return reranked docs with updated scores."""


class KeywordOverlapReranker:
    """Baseline reranker using lexical overlap, with pluggable scorer hook."""

    token_pattern = re.compile(r"\w+")

    def __init__(
        self,
        scorer: Callable[[str, RetrievalResult], float] | None = None,
    ) -> None:
        self._scorer = scorer or self._keyword_overlap_score

    def _keyword_overlap_score(self, query: str, doc: RetrievalResult) -> float:
        query_terms = set(self.token_pattern.findall(query.lower()))
        doc_terms = set(self.token_pattern.findall(doc.content.lower()))
        if not query_terms:
            return 0.0

        overlap = len(query_terms.intersection(doc_terms))
        ratio = overlap / len(query_terms)
        # Blend overlap with retrieval score so tie-breaking remains stable.
        return (ratio * 0.8) + (doc.score * 0.2)

    def rerank(self, query: str, docs: list[RetrievalResult], top_k: int | None = None) -> list[RetrievalResult]:
        if not docs:
            return []

        scored: list[tuple[float, RetrievalResult]] = []
        for doc in docs:
            rerank_score = float(self._scorer(query, doc))
            updated = doc.model_copy(update={
                "score": rerank_score,
                "rerank_score": rerank_score,
                "score_type": "rerank",
            })
            scored.append((rerank_score, updated))

        scored.sort(key=lambda row: row[0], reverse=True)

        ranked: list[RetrievalResult] = []
        limit = top_k if top_k is not None else len(scored)
        for rank, (_, item) in enumerate(scored[:limit], start=1):
            item.rank = rank
            ranked.append(item)
        return ranked
