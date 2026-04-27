"""Indexing layer tests."""

from pathlib import Path

import pytest

from app.indexing import HashEmbeddingProvider, InMemoryVectorIndex, IndexBuilder, LocalIndexStore
from app.ingestion.chunker import Chunker
from app.ingestion.cleaner import TextCleaner
from app.schemas.ingestion import DocumentChunk, LoadedDocument


def _build_test_chunks() -> list[DocumentChunk]:
    documents = [
        LoadedDocument(
            doc_id="doc_01",
            source="memory://doc_01",
            title="Doc 01",
            section="Intro",
            page=None,
            content="Self-RAG improves reliability with critique and retrieval retry.",
            metadata={"source_type": "unit_test"},
        ),
        LoadedDocument(
            doc_id="doc_02",
            source="memory://doc_02",
            title="Doc 02",
            section="Details",
            page=None,
            content="Hybrid retrieval uses dense vectors and BM25 sparse matching.",
            metadata={"source_type": "unit_test"},
        ),
    ]

    cleaner = TextCleaner()
    chunker = Chunker(chunk_size=60, chunk_overlap=10)
    return chunker.chunk_documents(cleaner.clean_documents(documents))


def test_index_builder_builds_vector_and_bm25_from_chunks() -> None:
    chunks = _build_test_chunks()
    provider = HashEmbeddingProvider(dimension=32)
    builder = IndexBuilder(embedding_provider=provider)

    built = builder.build(chunks)

    assert built.chunk_count == len(chunks)
    assert built.embedding_provider == "hash-embedding"
    assert built.vector_index.size == len(chunks)
    assert built.vector_index.dimension == 32
    assert built.bm25_index.doc_count == len(chunks)


def test_index_builder_rejects_empty_chunks() -> None:
    provider = HashEmbeddingProvider(dimension=16)
    builder = IndexBuilder(embedding_provider=provider)

    with pytest.raises(ValueError, match="empty chunks"):
        builder.build([])


def test_local_persistence_roundtrip_for_indexes(tmp_path: Path) -> None:
    chunks = _build_test_chunks()
    provider = HashEmbeddingProvider(dimension=24)
    builder = IndexBuilder(embedding_provider=provider)
    built = builder.build(chunks)

    store = LocalIndexStore(tmp_path / "indexes")
    vector_path = store.save_vector_index(built.vector_index)
    bm25_path = store.save_bm25_index(built.bm25_index)

    assert vector_path.exists()
    assert bm25_path.exists()

    loaded_vector = store.load_vector_index()
    loaded_bm25 = store.load_bm25_index()

    assert loaded_vector.size == built.vector_index.size
    assert loaded_vector.dimension == built.vector_index.dimension
    assert [chunk.chunk_id for chunk in loaded_vector.chunks] == [
        chunk.chunk_id for chunk in built.vector_index.chunks
    ]

    assert loaded_bm25.doc_count == built.bm25_index.doc_count
    assert loaded_bm25.chunk_ids == built.bm25_index.chunk_ids


def test_inmemory_vector_index_revision_increments_on_successful_build() -> None:
    chunks = _build_test_chunks()
    provider = HashEmbeddingProvider(dimension=16)
    vectors = provider.embed_documents([chunk.content for chunk in chunks])

    index = InMemoryVectorIndex()
    assert index.revision == 0

    index.build(chunks, vectors)
    first_revision = index.revision
    assert first_revision == 1

    index.build(chunks, vectors)
    assert index.revision == first_revision + 1
