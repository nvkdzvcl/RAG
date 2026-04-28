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
    assert chunks[0].content.startswith("Title: Doc Title. Section: Sub Section.\n\n")
    assert "This is the main content." in chunks[0].content
    assert chunks[0].metadata.get("heading_context_injected") is True
    assert int(chunks[0].metadata.get("heading_context_prefix_length", 0)) > 0


def test_empty_title_section_not_injected() -> None:
    doc = LoadedDocument(
        doc_id="doc_empty_prefix",
        source="memory://empty-prefix",
        title="   ",
        section=None,
        page=1,
        content="No heading metadata should be injected.",
        block_type="text",
        language="en",
        metadata={"owner": "ingestion"},
    )

    chunk = Chunker(
        chunk_size=40, chunk_overlap=5, include_heading_context=True
    ).chunk_document(doc)[0]

    assert chunk.content == "No heading metadata should be injected."
    assert chunk.metadata["owner"] == "ingestion"
    assert chunk.metadata.get("heading_context_injected") is False
    assert chunk.metadata.get("heading_context_prefix_length") == 0


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
    # Prefix is compact heading context ("Title: ... Section: ...") with small token cost.
    # chunk_size=20 should preserve the prefix while keeping total token count bounded.
    chunker = Chunker(chunk_size=20, chunk_overlap=5, include_heading_context=True)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for chunk in chunks:
        # ensure prefix is present
        assert chunk.content.startswith("Title: Doc. Section: Sec.\n\n")
        # ensure total tokens is <= chunk_size (20)
        tokens = len(chunker._token_spans(chunk.content))
        assert tokens <= 20


def test_chunk_overlap_applied_for_long_text() -> None:
    content = " ".join(f"token{i}" for i in range(60))
    doc = LoadedDocument(
        doc_id="doc_overlap",
        source="memory://overlap",
        title=None,
        section=None,
        page=1,
        content=content,
        block_type="text",
        language="en",
        metadata={},
    )

    chunker = Chunker(chunk_size=20, chunk_overlap=5, include_heading_context=False)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) >= 3
    for previous, current in zip(chunks, chunks[1:]):
        prev_tokens = previous.content.split()
        cur_tokens = current.content.split()
        assert prev_tokens[-5:] == cur_tokens[:5]


def test_chunk_overlap_does_not_create_duplicate_tiny_chunks() -> None:
    content = " ".join(f"overlap{i}" for i in range(70))
    doc = LoadedDocument(
        doc_id="doc_overlap_guard",
        source="memory://overlap-guard",
        title=None,
        section=None,
        page=1,
        content=content,
        block_type="text",
        language="en",
        metadata={},
    )

    chunker = Chunker(chunk_size=18, chunk_overlap=15, include_heading_context=False)
    chunks = chunker.chunk_document(doc)

    contents = [chunk.content for chunk in chunks]
    assert len(contents) == len(set(contents))
    assert all(len(chunk.content.split()) >= 2 for chunk in chunks)


def test_heading_only_chunk_avoided_when_following_paragraph_exists() -> None:
    heading = LoadedDocument(
        doc_id="doc_heading_follow",
        source="memory://heading-follow",
        title="Guide",
        section="Start",
        page=1,
        content="# Start",
        block_type="text",
        language="en",
        metadata={"is_heading": True},
    )
    body = LoadedDocument(
        doc_id="doc_heading_follow",
        source="memory://heading-follow",
        title="Guide",
        section="Start",
        page=1,
        content=" ".join(f"detail{i}" for i in range(40)),
        block_type="text",
        language="en",
        metadata={},
    )

    chunks = Chunker(
        chunk_size=12, chunk_overlap=3, include_heading_context=False
    ).chunk_documents([heading, body])

    assert chunks
    assert all(chunk.content.strip() != "# Start" for chunk in chunks)
    assert "# Start" in chunks[0].content
    assert "detail0" in chunks[0].content


def test_no_merge_across_different_doc_id() -> None:
    doc_a = LoadedDocument(
        doc_id="doc_a",
        source="memory://a",
        title="A",
        section="Shared",
        page=1,
        content="Alpha details.",
        block_type="text",
        language="en",
        metadata={},
    )
    doc_b = LoadedDocument(
        doc_id="doc_b",
        source="memory://b",
        title="B",
        section="Shared",
        page=1,
        content="Beta details.",
        block_type="text",
        language="en",
        metadata={},
    )

    chunks = Chunker(
        chunk_size=100, chunk_overlap=10, include_heading_context=False
    ).chunk_documents([doc_a, doc_b])

    assert len(chunks) == 2
    assert chunks[0].doc_id == "doc_a"
    assert chunks[1].doc_id == "doc_b"


def test_no_merge_across_different_section() -> None:
    doc_1 = LoadedDocument(
        doc_id="doc_sec",
        source="memory://sec",
        title="Doc",
        section="Section A",
        page=1,
        content="Section A details.",
        block_type="text",
        language="en",
        metadata={},
    )
    doc_2 = LoadedDocument(
        doc_id="doc_sec",
        source="memory://sec",
        title="Doc",
        section="Section B",
        page=1,
        content="Section B details.",
        block_type="text",
        language="en",
        metadata={},
    )

    chunks = Chunker(
        chunk_size=100, chunk_overlap=10, include_heading_context=False
    ).chunk_documents([doc_1, doc_2])

    assert len(chunks) == 2
    assert chunks[0].section == "Section A"
    assert chunks[1].section == "Section B"


def test_grouping_guard_prevents_over_merging_many_blocks() -> None:
    docs = [
        LoadedDocument(
            doc_id="doc_group_guard",
            source="memory://group-guard",
            title="Guard",
            section="S",
            page=1,
            content=("x" * 120) + f" {i}",
            block_type="text",
            language="en",
            metadata={},
        )
        for i in range(6)
    ]

    chunks = Chunker(
        chunk_size=400,
        chunk_overlap=20,
        include_heading_context=False,
        max_grouped_chars=260,
    ).chunk_documents(docs)

    assert len(chunks) >= 2
