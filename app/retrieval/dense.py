"""Dense retrieval using vector similarity over indexed chunk embeddings."""

from __future__ import annotations

import math

from app.indexing.embeddings import EmbeddingProvider
from app.indexing.vector_index import InMemoryVectorIndex
from app.schemas.retrieval import RetrievalResult


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vector dimension mismatch for cosine similarity")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class DenseRetriever:
    """Dense retriever backed by in-memory vector index."""

    def __init__(self, vector_index: InMemoryVectorIndex, embedding_provider: EmbeddingProvider) -> None:
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        if self.vector_index.size == 0:
            return []

        query_vector = self.embedding_provider.embed_query(query)
        scored: list[tuple[float, int]] = []

        for idx, vector in enumerate(self.vector_index.vectors):
            score = _cosine_similarity(query_vector, vector)
            scored.append((score, idx))

        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(scored[:top_k], start=1):
            chunk = self.vector_index.chunks[idx]
            results.append(
                RetrievalResult.from_chunk(
                    chunk,
                    score=score,
                    dense_score=score,
                    score_type="dense",
                    rank=rank,
                )
            )
        return results
