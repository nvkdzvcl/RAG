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

    chunker = Chunker(chunk_size=5, chunk_overlap=1, include_heading_context=False)
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


def test_short_paragraph_merging() -> None:
    doc = LoadedDocument(
        doc_id="doc_p",
        source="memory://p",
        title="Title",
        section="Section",
        page=1,
        content="Short paragraph 1.\n\nShort paragraph 2.\n\nShort paragraph 3.",
        block_type="text",
        language="en",
        metadata={},
    )
    chunker = Chunker(chunk_size=100, chunk_overlap=10, include_heading_context=False)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert "Short paragraph 1." in chunks[0].content
    assert "Short paragraph 3." in chunks[0].content


def test_heading_paragraph_grouping() -> None:
    doc1 = LoadedDocument(
        doc_id="doc_group",
        source="memory://group",
        title="Title",
        section="Group A",
        page=1,
        content="# Group A",
        block_type="text",
        language="en",
        metadata={"is_heading": True},
    )
    doc2 = LoadedDocument(
        doc_id="doc_group",
        source="memory://group",
        title="Title",
        section="Group A",
        page=1,
        content="Detail 1",
        block_type="text",
        language="en",
        metadata={},
    )
    doc3 = LoadedDocument(
        doc_id="doc_group",
        source="memory://group",
        title="Title",
        section="Group A",
        page=1,
        content="Detail 2",
        block_type="text",
        language="en",
        metadata={},
    )

    chunker = Chunker(chunk_size=100, chunk_overlap=10, include_heading_context=False)
    chunks = chunker.chunk_documents([doc1, doc2, doc3])
    # Should be merged into 1 chunk because they share same doc_id and section
    assert len(chunks) == 1
    content = chunks[0].content
    assert "# Group A" in content
    assert "Detail 1" in content
    assert "Detail 2" in content


def test_heading_context_prefix_injection() -> None:
    doc = LoadedDocument(
        doc_id="doc_prefix",
        source="memory://prefix",
        title="Doc Title",
        section="Sub Section",
        page=1,
        content="This is the main content.",
        block_type="text",
        language="en",
        metadata={},
    )
    chunker = Chunker(chunk_size=100, chunk_overlap=10, include_heading_context=True)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].content.startswith("[Title: Doc Title | Section: Sub Section]\n")
    assert "This is the main content." in chunks[0].content
    assert chunks[0].metadata.get("heading_context_injected") is True


def test_chunk_size_enforcement_with_prefix() -> None:
    doc = LoadedDocument(
        doc_id="doc_size",
        source="memory://size",
        title="Doc",
        section="Sec",
        page=1,
        content="A " * 40,  # 40 tokens
        block_type="text",
        language="en",
        metadata={},
    )
    # Prefix is "[Title: Doc | Section: Sec]\n" which is around 6-7 tokens.
    # Set chunk_size to 20, which is enough for prefix (7) + 13 tokens.
    chunker = Chunker(chunk_size=20, chunk_overlap=5, include_heading_context=True)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        # ensure prefix is present
        assert chunk.content.startswith("[Title: Doc | Section: Sec]\n")
        # ensure total tokens is <= chunk_size (20)
        tokens = len(chunker._token_spans(chunk.content))
        assert tokens <= 20
