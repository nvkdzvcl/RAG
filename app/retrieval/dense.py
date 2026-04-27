"""Dense retrieval using vector similarity over indexed chunk embeddings."""

from __future__ import annotations

import numpy as np

from app.core.cache import QueryCache
from app.indexing.embeddings import EmbeddingProvider
from app.indexing.vector_index import InMemoryVectorIndex
from app.schemas.retrieval import RetrievalResult


def _rank_top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or scores.size == 0:
        return np.empty(0, dtype=np.int64)

    k = min(top_k, scores.size)
    if k == scores.size:
        candidate_indices = np.arange(scores.size, dtype=np.int64)
    else:
        split_point = scores.size - k
        kth_score = float(np.partition(scores, split_point)[split_point])
        higher_indices = np.flatnonzero(scores > kth_score)
        needed_from_ties = k - int(higher_indices.size)

        tie_indices = np.flatnonzero(scores == kth_score)
        selected_ties = tie_indices[:needed_from_ties]
        candidate_indices = np.concatenate((higher_indices, selected_ties))

    # Keep previous tie behavior: higher score first, then lower original index.
    order = np.lexsort((candidate_indices, -scores[candidate_indices]))
    return candidate_indices[order]


def _cosine_similarity_matrix(
    query_vector: np.ndarray,
    vector_matrix: np.ndarray,
    vector_norms: np.ndarray,
) -> np.ndarray:
    query_norm = float(np.linalg.norm(query_vector))
    if query_norm == 0.0:
        return np.zeros(vector_matrix.shape[0], dtype=np.float64)

    dot_products = vector_matrix @ query_vector
    denominator = vector_norms * query_norm
    return np.divide(
        dot_products,
        denominator,
        out=np.zeros_like(dot_products, dtype=np.float64),
        where=denominator > 0.0,
    )


class DenseRetriever:
    """Dense retriever backed by in-memory vector index."""

    def __init__(
        self,
        vector_index: InMemoryVectorIndex,
        embedding_provider: EmbeddingProvider,
        *,
        embedding_cache: QueryCache | None = None,
    ) -> None:
        self.vector_index = vector_index
        self.embedding_provider = embedding_provider
        self._embedding_cache = embedding_cache
        self._cached_size = -1
        self._cached_dimension = -1
        self._cached_revision = -1
        self._vector_matrix = np.empty((0, 0), dtype=np.float64)
        self._vector_norms = np.empty(0, dtype=np.float64)

    def _refresh_index_cache_if_needed(self) -> None:
        revision_changed = self._cached_revision != self.vector_index.revision
        if (
            not revision_changed
            and self._cached_size == self.vector_index.size
            and self._cached_dimension == self.vector_index.dimension
        ):
            return

        vectors = np.asarray(self.vector_index.vectors, dtype=np.float64)
        if vectors.ndim != 2:
            raise ValueError("Vector index must contain a 2D matrix of vectors")

        self._vector_matrix = vectors
        self._vector_norms = np.linalg.norm(vectors, axis=1)
        self._cached_size = self.vector_index.size
        self._cached_dimension = self.vector_index.dimension
        self._cached_revision = self.vector_index.revision

        # Invalidate embedding cache when index changes (new corpus).
        if revision_changed and self._embedding_cache is not None:
            self._embedding_cache.invalidate()

    def _embed_query(self, query: str) -> np.ndarray:
        """Return query embedding, using cache when available."""
        if self._embedding_cache is not None:
            hit, cached = self._embedding_cache.get(query)
            if hit:
                return np.asarray(cached, dtype=np.float64)

        raw = self.embedding_provider.embed_query(query)
        vector = np.asarray(raw, dtype=np.float64)

        if self._embedding_cache is not None:
            self._embedding_cache.put(query, vector.tolist())
        return vector

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        if self.vector_index.size == 0:
            return []

        self._refresh_index_cache_if_needed()
        query_vector = self._embed_query(query)
        if query_vector.ndim != 1 or query_vector.shape[0] != self._vector_matrix.shape[1]:
            raise ValueError("Vector dimension mismatch for cosine similarity")

        scores = _cosine_similarity_matrix(query_vector, self._vector_matrix, self._vector_norms)
        ranked_indices = _rank_top_k_indices(scores, top_k)
        chunks = self.vector_index.chunks

        results: list[RetrievalResult] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            chunk = chunks[idx]
            score = float(scores[idx])
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
