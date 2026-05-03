"""Tests for vector index backend factory selection."""

from __future__ import annotations

from types import SimpleNamespace

from app.indexing.faiss_index import FaissVectorIndex
from app.indexing.vector_factory import create_vector_index
from app.indexing.vector_index import InMemoryVectorIndex
from app.retrieval.dense import DenseRetriever
from app.schemas.ingestion import DocumentChunk


def test_create_vector_index_defaults_to_inmemory_when_backend_missing() -> None:
    settings = SimpleNamespace()
    index = create_vector_index(settings)
    assert isinstance(index, InMemoryVectorIndex)


def test_create_vector_index_selects_inmemory_backend() -> None:
    settings = SimpleNamespace(vector_index_backend="inmemory")
    index = create_vector_index(settings)
    assert isinstance(index, InMemoryVectorIndex)


def test_create_vector_index_selects_faiss_backend() -> None:
    settings = SimpleNamespace(vector_index_backend="faiss")
    index = create_vector_index(settings)
    assert isinstance(index, FaissVectorIndex)


def test_create_vector_index_falls_back_for_unknown_backend() -> None:
    settings = SimpleNamespace(vector_index_backend="unknown")
    index = create_vector_index(settings)
    assert isinstance(index, InMemoryVectorIndex)


def test_inmemory_backend_runs_when_faiss_dependency_is_missing(
    monkeypatch,
) -> None:
    def _missing_faiss(name: str, *_args, **_kwargs):
        if name == "faiss":
            raise ModuleNotFoundError("No module named 'faiss'")
        return __import__(name)

    monkeypatch.setattr(
        "app.indexing.faiss_index.importlib.import_module",
        _missing_faiss,
    )

    settings = SimpleNamespace(vector_index_backend="inmemory")
    index = create_vector_index(settings)
    assert isinstance(index, InMemoryVectorIndex)

    chunks = [
        DocumentChunk(
            chunk_id="chunk_0",
            doc_id="doc_0",
            source="memory://doc_0",
            content="inmemory fallback should still retrieve",
        )
    ]
    index.build(chunks, [[1.0, 0.0, 0.0]])

    class _StaticEmbeddingProvider:
        name = "static"
        dimension = 3

        @staticmethod
        def embed_documents(texts: list[str]) -> list[list[float]]:
            _ = texts
            return [[1.0, 0.0, 0.0]]

        @staticmethod
        def embed_query(_text: str) -> list[float]:
            return [1.0, 0.0, 0.0]

    retriever = DenseRetriever(index, _StaticEmbeddingProvider())
    results = retriever.retrieve("test", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "chunk_0"
