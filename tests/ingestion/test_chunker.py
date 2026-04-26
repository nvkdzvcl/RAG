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
        content="Đây là đoạn văn tiếng Việt có dấu và có đủ số lượng từ để kiểm tra chunking.",
        block_type="text",
        language="vi",
        metadata={"author": "team"},
    )

    chunker = Chunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 1
    first = chunks[0]
    assert first.doc_id == "doc_abc"
    assert first.source == "/tmp/doc.md"
    assert first.title == "Doc Title"
    assert first.section == "Intro"
    assert first.page == 3
    assert first.metadata["author"] == "team"
    assert first.metadata["doc_id"] == "doc_abc"
    assert first.metadata["file_name"] == "doc.md"
    assert first.metadata["file_type"] == "md"
    assert first.block_type == "text"
    assert first.language == "vi"
    assert first.metadata["page"] == 3
    assert first.metadata["ocr"] is False
    assert first.metadata["block_type"] == "text"
    assert first.metadata["language"] == "vi"


def test_chunk_id_generation_is_deterministic_and_unique() -> None:
    chunker = Chunker(chunk_size=8, chunk_overlap=2)
    text = " ".join(f"token{i}" for i in range(40))

    doc = LoadedDocument(
        doc_id="doc_xyz",
        source="memory://doc",
        title="Synthetic",
        section=None,
        page=None,
        content=text,
        block_type="text",
        language="auto",
        metadata={},
    )

    first_run = chunker.chunk_document(doc)
    second_run = chunker.chunk_document(doc)

    first_ids = [chunk.chunk_id for chunk in first_run]
    second_ids = [chunk.chunk_id for chunk in second_run]

    assert first_ids == second_ids
    assert len(set(first_ids)) == len(first_ids)


def test_chunker_does_not_split_table_blocks() -> None:
    table_doc = LoadedDocument(
        doc_id="doc_table",
        source="memory://table",
        title="Scores",
        section="Results",
        page=1,
        content="| Name | Score |\n| --- | --- |\n| A | 10 |",
        block_type="table",
        language="vi",
        metadata={},
    )

    chunker = Chunker(chunk_size=5, chunk_overlap=1)
    chunks = chunker.chunk_document(table_doc)

    assert len(chunks) == 1
    assert chunks[0].content == table_doc.content
    assert chunks[0].block_type == "table"


def test_chunker_preserves_ocr_block_type_in_metadata() -> None:
    doc = LoadedDocument(
        doc_id="doc_ocr",
        source="memory://ocr",
        title="OCR Doc",
        section="Page 1",
        page=1,
        content="Nội dung OCR tiếng Việt có dấu với token ocr-keep-meta.",
        block_type="text",
        language="vi",
        metadata={"block_type": "ocr_text", "ocr": True},
    )

    chunks = Chunker(chunk_size=16, chunk_overlap=2).chunk_document(doc)

    assert chunks
    assert all(chunk.block_type == "text" for chunk in chunks)
    assert all(chunk.metadata.get("block_type") == "ocr_text" for chunk in chunks)
    assert all(bool(chunk.metadata.get("ocr")) is True for chunk in chunks)
