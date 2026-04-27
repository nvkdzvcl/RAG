"""Retrieval baseline tests."""

import logging
import threading
import time
from typing import Literal

import numpy as np
import pytest

from app.indexing import HashEmbeddingProvider, IndexBuilder
from app.indexing.vector_index import InMemoryVectorIndex
from app.ingestion.chunker import Chunker
from app.ingestion.cleaner import TextCleaner
from app.retrieval import (
    DenseRetriever,
    FusionConfig,
    HybridRetriever,
    ScoreOnlyReranker,
    SparseRetriever,
    reciprocal_rank_fusion,
)
from app.core.math_utils import cosine_similarity
from app.retrieval.dense import _cosine_similarity_matrix, _rank_top_k_indices
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


def _retrieval_result(
    chunk_id: str,
    *,
    score: float,
    score_type: Literal["dense", "sparse", "hybrid", "rerank"],
    doc_id: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id or f"doc_{chunk_id}",
        source=f"memory://{chunk_id}",
        content=f"content-{chunk_id}",
        score=score,
        score_type=score_type,
    )


def _legacy_rank(
    query_vector: list[float],
    vectors: list[list[float]],
    *,
    top_k: int,
) -> list[tuple[float, int]]:
    scored: list[tuple[float, int]] = []
    for idx, vector in enumerate(vectors):
        scored.append((cosine_similarity(query_vector, vector), idx))
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


def test_reciprocal_rank_fusion_combines_dense_and_sparse_rankings() -> None:
    dense = [
        _retrieval_result("a", score=0.99, score_type="dense"),
        _retrieval_result("b", score=0.90, score_type="dense"),
        _retrieval_result("c", score=0.80, score_type="dense"),
    ]
    sparse = [
        _retrieval_result("b", score=8.0, score_type="sparse"),
        _retrieval_result("c", score=7.0, score_type="sparse"),
        _retrieval_result("d", score=6.0, score_type="sparse"),
    ]

    fused = reciprocal_rank_fusion(
        dense,
        sparse,
        top_k=4,
        config=FusionConfig(rrf_k=0, dense_weight=1.0, sparse_weight=1.0),
    )

    assert [item.chunk_id for item in fused] == ["b", "a", "c", "d"]
    assert fused[0].dense_score == pytest.approx(0.90)
    assert fused[0].sparse_score == pytest.approx(8.0)
    assert fused[1].dense_score == pytest.approx(0.99)
    assert fused[1].sparse_score is None
    assert fused[3].dense_score is None
    assert fused[3].sparse_score == pytest.approx(6.0)
    assert [item.rank for item in fused] == [1, 2, 3, 4]


def test_reciprocal_rank_fusion_deduplicates_documents_across_retrievers() -> None:
    dense = [
        _retrieval_result("a_chunk_1", score=0.99, score_type="dense", doc_id="doc_a"),
        _retrieval_result("b_chunk_1", score=0.95, score_type="dense", doc_id="doc_b"),
        _retrieval_result("a_chunk_2", score=0.90, score_type="dense", doc_id="doc_a"),
    ]
    sparse = [
        _retrieval_result("c_chunk_1", score=8.2, score_type="sparse", doc_id="doc_c"),
        _retrieval_result("a_chunk_3", score=8.0, score_type="sparse", doc_id="doc_a"),
        _retrieval_result("b_chunk_2", score=7.8, score_type="sparse", doc_id="doc_b"),
    ]

    fused = reciprocal_rank_fusion(
        dense,
        sparse,
        top_k=3,
        config=FusionConfig(rrf_k=0, dense_weight=1.0, sparse_weight=1.0),
    )

    assert [item.doc_id for item in fused] == ["doc_a", "doc_c", "doc_b"]
    assert len({item.doc_id for item in fused}) == len(fused)
    assert fused[0].chunk_id == "a_chunk_1"
    assert fused[0].dense_score == pytest.approx(0.99)
    assert fused[0].sparse_score == pytest.approx(8.0)


def test_hybrid_retriever_runs_dense_and_sparse_then_returns_top_k_rrf() -> None:
    class _FakeRetriever:
        def __init__(self, results: list[RetrievalResult]) -> None:
            self._results = list(results)
            self.calls: list[tuple[str, int]] = []

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            self.calls.append((query, top_k))
            return list(self._results)[:top_k]

    dense = _FakeRetriever(
        [
            _retrieval_result("dense_only", score=0.91, score_type="dense"),
            _retrieval_result("shared", score=0.87, score_type="dense"),
        ]
    )
    sparse = _FakeRetriever(
        [
            _retrieval_result("shared", score=6.1, score_type="sparse"),
            _retrieval_result("sparse_only", score=5.8, score_type="sparse"),
        ]
    )
    fusion = FusionConfig(
        rrf_k=60,
        dense_weight=1.0,
        sparse_weight=1.0,
        candidate_multiplier=2,
        min_candidates_per_retriever=4,
    )
    hybrid = HybridRetriever(dense, sparse, fusion_config=fusion)  # type: ignore[arg-type]

    results = hybrid.retrieve("mixed query", top_k=2)

    assert dense.calls == [("mixed query", 4)]
    assert sparse.calls == [("mixed query", 4)]
    assert len(results) == 2
    assert results[0].chunk_id == "shared"
    assert {item.chunk_id for item in results}.issubset({"dense_only", "shared", "sparse_only"})
    assert any(item.chunk_id in {"dense_only", "sparse_only"} for item in results[1:])


def test_hybrid_retriever_logs_dense_sparse_and_merged_debug_views(caplog: pytest.LogCaptureFixture) -> None:
    class _FakeRetriever:
        def __init__(self, results: list[RetrievalResult]) -> None:
            self._results = list(results)

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            return list(self._results)[:top_k]

    dense = _FakeRetriever(
        [
            _retrieval_result("dense_1", score=0.93, score_type="dense"),
            _retrieval_result("dense_2", score=0.88, score_type="dense"),
        ]
    )
    sparse = _FakeRetriever(
        [
            _retrieval_result("sparse_1", score=7.2, score_type="sparse"),
            _retrieval_result("sparse_2", score=6.9, score_type="sparse"),
        ]
    )
    hybrid = HybridRetriever(dense, sparse)  # type: ignore[arg-type]
    caplog.set_level(logging.DEBUG, logger="app.retrieval.hybrid")

    results = hybrid.retrieve("debug query", top_k=2)

    assert len(results) == 2
    messages = [record.getMessage() for record in caplog.records if record.name == "app.retrieval.hybrid"]
    assert any("Hybrid dense results" in message for message in messages)
    assert any("Hybrid bm25 results" in message for message in messages)
    assert any("Hybrid merged results" in message for message in messages)


def test_hybrid_retriever_runs_dense_and_sparse_concurrently() -> None:
    barrier = threading.Barrier(2)

    class _BarrierRetriever:
        def __init__(self, item: RetrievalResult) -> None:
            self._item = item
            self.calls: list[tuple[str, int]] = []

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            self.calls.append((query, top_k))
            barrier.wait(timeout=3.0)
            return [self._item]

    dense = _BarrierRetriever(_retrieval_result("dense", score=0.9, score_type="dense"))
    sparse = _BarrierRetriever(_retrieval_result("sparse", score=6.0, score_type="sparse"))
    hybrid = HybridRetriever(dense, sparse)  # type: ignore[arg-type]

    results = hybrid.retrieve("parallel query", top_k=2)

    assert len(dense.calls) == 1
    assert len(sparse.calls) == 1
    assert [item.chunk_id for item in results] == ["dense", "sparse"]


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


def test_dense_retrieval_rebuild_same_shape_invalidates_cache() -> None:
    chunks = _build_synthetic_chunks(2)
    vector_index = InMemoryVectorIndex()
    query_provider = _StaticEmbeddingProvider([1.0, 0.0])
    retriever = DenseRetriever(vector_index, query_provider)

    vectors_a = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    vector_index.build(chunks, vectors_a)
    first_results = retriever.retrieve("query ignored by static provider", top_k=1)
    assert first_results[0].chunk_id == "chunk_0"

    vectors_b = [
        [0.0, 1.0],
        [1.0, 0.0],
    ]
    vector_index.build(chunks, vectors_b)
    second_results = retriever.retrieve("query ignored by static provider", top_k=1)
    assert second_results[0].chunk_id == "chunk_1"


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
