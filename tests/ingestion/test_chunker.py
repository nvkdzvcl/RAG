"""Tests for chunk generation and metadata preservation."""

from app.ingestion.chunker import Chunker
from app.schemas.ingestion import LoadedDocument


def test_chunker_preserves_required_metadata() -> None:
    doc = LoadedDocument(
        doc_id="doc_abc",
        source="/tmp/doc.md",
        title="Doc Title",
        section="Intro",
        page=3,
        content="A" * 120,
        metadata={"author": "team"},
    )

    chunker = Chunker(chunk_size=80, chunk_overlap=20)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 2
    first = chunks[0]
    assert first.doc_id == "doc_abc"
    assert first.source == "/tmp/doc.md"
    assert first.title == "Doc Title"
    assert first.section == "Intro"
    assert first.page == 3
    assert first.metadata["author"] == "team"


def test_chunk_id_generation_is_deterministic_and_unique() -> None:
    chunker = Chunker(chunk_size=30, chunk_overlap=5)
    text = "0123456789" * 10

    doc = LoadedDocument(
        doc_id="doc_xyz",
        source="memory://doc",
        title="Synthetic",
        section=None,
        page=None,
        content=text,
        metadata={},
    )

    first_run = chunker.chunk_document(doc)
    second_run = chunker.chunk_document(doc)

    first_ids = [chunk.chunk_id for chunk in first_run]
    second_ids = [chunk.chunk_id for chunk in second_run]

    assert first_ids == second_ids
    assert len(set(first_ids)) == len(first_ids)
