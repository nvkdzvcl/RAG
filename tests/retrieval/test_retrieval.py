"""Retrieval baseline tests."""

import time

import numpy as np
import pytest

from app.indexing import HashEmbeddingProvider, IndexBuilder
from app.indexing.vector_index import InMemoryVectorIndex
from app.ingestion.chunker import Chunker
from app.ingestion.cleaner import TextCleaner
from app.retrieval import DenseRetriever, HybridRetriever, ScoreOnlyReranker, SparseRetriever
from app.retrieval.dense import _cosine_similarity, _cosine_similarity_matrix, _rank_top_k_indices
from app.schemas.ingestion import DocumentChunk, LoadedDocument
from app.schemas.retrieval import RetrievalResult


class _StaticEmbeddingProvider:
    """Test helper provider that always returns a predefined query embedding."""

    def __init__(self, query_vector: list[float]) -> None:
        self.name = "static-test-provider"
        self.dimension = len(query_vector)
        self._query_vector = list(query_vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self._query_vector) for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return list(self._query_vector)


def _sample_chunks() -> list[DocumentChunk]:
    docs = [
        LoadedDocument(
            doc_id="doc_alpha",
            source="memory://alpha",
            title="Alpha",
            section="Intro",
            content="Alpha system uses retrieval and citations for grounding.",
            metadata={"kind": "note"},
        ),
        LoadedDocument(
            doc_id="doc_beta",
            source="memory://beta",
            title="Beta",
            section="Details",
            content="BM25 sparse retrieval handles exact keyword matching.",
            metadata={"kind": "note"},
        ),
        LoadedDocument(
            doc_id="doc_gamma",
            source="memory://gamma",
            title="Gamma",
            section="Advanced",
            content="Dense retrievers compare embedding vectors for semantic similarity.",
            metadata={"kind": "note"},
        ),
    ]
    cleaner = TextCleaner()
    chunker = Chunker(chunk_size=120, chunk_overlap=20)
    return chunker.chunk_documents(cleaner.clean_documents(docs))


def _build_retrievers() -> tuple[DenseRetriever, SparseRetriever, HybridRetriever]:
    provider = HashEmbeddingProvider(dimension=48)
    chunks = _sample_chunks()
    built = IndexBuilder(embedding_provider=provider).build(chunks)

    dense = DenseRetriever(built.vector_index, provider)
    sparse = SparseRetriever(built.bm25_index)
    hybrid = HybridRetriever(dense, sparse)
    return dense, sparse, hybrid


def _build_synthetic_chunks(count: int) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"chunk_{idx}",
            doc_id=f"doc_{idx}",
            source=f"memory://doc_{idx}",
            content=f"Synthetic content {idx}",
        )
        for idx in range(count)
    ]


def _legacy_rank(
    query_vector: list[float],
    vectors: list[list[float]],
    *,
    top_k: int,
) -> list[tuple[float, int]]:
    scored: list[tuple[float, int]] = []
    for idx, vector in enumerate(vectors):
        scored.append((_cosine_similarity(query_vector, vector), idx))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


def test_dense_retrieval_output_shape() -> None:
    dense, _, _ = _build_retrievers()
    results = dense.retrieve("embedding semantic retrieval", top_k=2)

    assert results
    assert len(results) == 2
    first = results[0]
    assert isinstance(first, RetrievalResult)
    assert first.chunk_id
    assert first.doc_id
    assert first.source
    assert first.content
    assert isinstance(first.score, float)
    assert first.score_type == "dense"


def test_sparse_retrieval_output_shape() -> None:
    _, sparse, _ = _build_retrievers()
    results = sparse.retrieve("bm25 keyword matching", top_k=2)

    assert results
    assert len(results) <= 2
    first = results[0]
    assert isinstance(first, RetrievalResult)
    assert first.chunk_id
    assert first.doc_id
    assert first.source
    assert isinstance(first.score, float)
    assert first.score_type == "sparse"


def test_hybrid_retrieval_merging() -> None:
    _, _, hybrid = _build_retrievers()
    results = hybrid.retrieve("retrieval keyword semantic", top_k=3)

    assert results
    assert len(results) == 3
    assert all(item.score_type == "hybrid" for item in results)
    chunk_ids = [item.chunk_id for item in results]
    assert len(set(chunk_ids)) == len(chunk_ids)


def test_score_only_reranker_output_shape() -> None:
    reranker = ScoreOnlyReranker()
    docs = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            source="s1",
            content="alpha",
            score=0.4,
            score_type="hybrid",
        ),
        RetrievalResult(
            chunk_id="c2",
            doc_id="d2",
            source="s2",
            content="alpha beta gamma",
            score=0.3,
            score_type="hybrid",
        ),
        RetrievalResult(
            chunk_id="c3",
            doc_id="d3",
            source="s3",
            content="theta",
            score=0.9,
            score_type="hybrid",
        ),
    ]

    reranked = reranker.rerank("alpha beta", docs)

    assert reranked[0].chunk_id == "c3"
    assert all(item.score_type == "rerank" for item in reranked)
    assert all(item.rerank_score is not None for item in reranked)
    assert reranked[0].score >= reranked[1].score


def test_dense_retrieval_matches_legacy_ranking_and_scores() -> None:
    rng = np.random.default_rng(7)
    vector_count = 512
    dimension = 64

    vectors_np = rng.standard_normal((vector_count, dimension), dtype=np.float64)
    vectors_np[11] = 0.0
    vectors = vectors_np.tolist()

    chunks = _build_synthetic_chunks(vector_count)
    vector_index = InMemoryVectorIndex()
    vector_index.build(chunks, vectors)

    query_vector = rng.standard_normal(dimension, dtype=np.float64).tolist()
    retriever = DenseRetriever(vector_index, _StaticEmbeddingProvider(query_vector))

    top_k = 40
    got = retriever.retrieve("query ignored by static provider", top_k=top_k)
    expected = _legacy_rank(query_vector, vectors, top_k=top_k)

    assert len(got) == len(expected)
    for result, (expected_score, expected_idx) in zip(got, expected):
        assert result.chunk_id == chunks[expected_idx].chunk_id
        assert result.score == pytest.approx(expected_score, abs=1e-12, rel=1e-12)


def test_dense_retrieval_handles_zero_norm_query_with_stable_order() -> None:
    vectors = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
    ]
    chunks = _build_synthetic_chunks(len(vectors))
    vector_index = InMemoryVectorIndex()
    vector_index.build(chunks, vectors)

    retriever = DenseRetriever(vector_index, _StaticEmbeddingProvider([0.0, 0.0, 0.0]))
    results = retriever.retrieve("zero query", top_k=3)

    assert [item.chunk_id for item in results] == ["chunk_0", "chunk_1", "chunk_2"]
    assert all(item.score == 0.0 for item in results)


@pytest.mark.slow
def test_dense_vectorized_similarity_is_at_least_10x_faster_than_legacy_loop() -> None:
    rng = np.random.default_rng(123)
    vector_count = 20_000
    dimension = 256
    top_k = 10

    vector_matrix = rng.standard_normal((vector_count, dimension), dtype=np.float64)
    vector_norms = np.linalg.norm(vector_matrix, axis=1)
    query_vector = rng.standard_normal(dimension, dtype=np.float64)

    vectors_list = vector_matrix.tolist()
    query_list = query_vector.tolist()

    # Warm up to reduce one-off startup overhead in timing.
    _ = _rank_top_k_indices(_cosine_similarity_matrix(query_vector, vector_matrix, vector_norms), top_k)

    legacy_start = time.perf_counter()
    legacy_ranked = _legacy_rank(query_list, vectors_list, top_k=top_k)
    legacy_elapsed = time.perf_counter() - legacy_start

    vectorized_start = time.perf_counter()
    scores = _cosine_similarity_matrix(query_vector, vector_matrix, vector_norms)
    vectorized_indices = _rank_top_k_indices(scores, top_k)
    vectorized_elapsed = time.perf_counter() - vectorized_start

    assert [idx for _, idx in legacy_ranked] == vectorized_indices.tolist()
    assert vectorized_elapsed > 0.0
    assert legacy_elapsed / vectorized_elapsed >= 10.0
