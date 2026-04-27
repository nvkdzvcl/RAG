"""Sparse retrieval using BM25 scoring."""

from __future__ import annotations

from collections import Counter

from app.indexing.bm25_index import BM25Index, tokenize_bm25
from app.schemas.retrieval import RetrievalResult


class SparseRetriever:
    """BM25-based sparse retriever."""

    def __init__(self, bm25_index: BM25Index) -> None:
        self.bm25_index = bm25_index

    def _tokenize(self, text: str) -> list[str]:
        return tokenize_bm25(text)

    def _score_doc(self, query_terms: list[str], doc_idx: int) -> float:
        tf = self.bm25_index.term_freqs[doc_idx]
        doc_len = self.bm25_index.doc_lengths[doc_idx]
        avg_len = self.bm25_index.avg_doc_len
        k1 = self.bm25_index.k1
        b = self.bm25_index.b

        if doc_len == 0 or avg_len == 0:
            return 0.0

        score = 0.0
        query_tf = Counter(query_terms)

        for term, qf in query_tf.items():
            term_freq = tf.get(term, 0)
            if term_freq == 0:
                continue
            idf = self.bm25_index.idf.get(term, 0.0)
            numerator = term_freq * (k1 + 1)
            denominator = term_freq + k1 * (1 - b + b * (doc_len / avg_len))
            score += idf * (numerator / denominator) * qf

        return score

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        if self.bm25_index.doc_count == 0:
            return []

        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[float, int]] = []
        for idx, chunk in enumerate(self.bm25_index.chunks):
            score = self._score_doc(query_terms, idx)
            if score <= 0:
                continue
            scored.append((score, idx))

        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(scored[:top_k], start=1):
            chunk = self.bm25_index.chunks[idx]
            results.append(
                RetrievalResult.from_chunk(
                    chunk,
                    score=score,
                    sparse_score=score,
                    score_type="sparse",
                    rank=rank,
                )
            )
        return results
