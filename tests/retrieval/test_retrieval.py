"""Retrieval baseline tests."""

from app.indexing import HashEmbeddingProvider, IndexBuilder
from app.ingestion.chunker import Chunker
from app.ingestion.cleaner import TextCleaner
from app.retrieval import DenseRetriever, HybridRetriever, KeywordOverlapReranker, SparseRetriever
from app.schemas.ingestion import DocumentChunk, LoadedDocument
from app.schemas.retrieval import RetrievalResult


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


def test_reranker_orders_by_overlap() -> None:
    reranker = KeywordOverlapReranker()
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

    assert reranked[0].chunk_id == "c2"
    assert all(item.score_type == "rerank" for item in reranked)
    assert reranked[0].score >= reranked[1].score
