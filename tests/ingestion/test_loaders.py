"""Tests for ingestion loaders output shape and metadata handling."""

from pathlib import Path

from app.ingestion.docx_loader import DocxLoader
from app.ingestion.markdown_loader import MarkdownLoader
from app.ingestion.text_loader import TextLoader


def test_text_loader_output_shape(tmp_path: Path) -> None:
    source = tmp_path / "sample.txt"
    source.write_text("Line 1\nLine 2\n", encoding="utf-8")

    loader = TextLoader()
    docs = loader.load(source, metadata={"source_type": "notes"})

    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id
    assert doc.source == str(source)
    assert doc.title == "sample"
    assert doc.section is None
    assert doc.page is None
    assert doc.content.startswith("Line 1")
    assert doc.block_type == "text"
    assert doc.language == "auto"
    assert doc.metadata["source_type"] == "notes"


def test_markdown_loader_preserves_section_metadata(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    source.write_text(
        "# Getting Started\n\nOverview text.\n\n## Setup\n\nSetup steps.",
        encoding="utf-8",
    )

    loader = MarkdownLoader()
    docs = loader.load(source, metadata={"lang": "en"})

    assert len(docs) >= 4
    assert all(doc.doc_id == docs[0].doc_id for doc in docs)
    assert all(doc.source == str(source) for doc in docs)
    assert all(doc.title == "Getting Started" for doc in docs)
    assert any(doc.section == "Getting Started" for doc in docs)
    assert any(doc.section == "Setup" for doc in docs)
    assert all(doc.page is None for doc in docs)
    assert all(doc.block_type in {"text", "table", "image"} for doc in docs)
    assert all(doc.metadata["lang"] == "en" for doc in docs)


def test_docx_loader_supports_vietnamese_text(tmp_path: Path) -> None:
    doc_path = tmp_path / "vietnamese.docx"

    from docx import Document

    doc = Document()
    doc.add_heading("Giới thiệu", level=1)
    doc.add_paragraph("Hệ thống hỗ trợ tiếng Việt đầy đủ dấu.")
    doc.save(str(doc_path))

    loader = DocxLoader()
    docs = loader.load(doc_path, metadata={"lang": "vi"})

    assert docs
    assert any("tiếng Việt đầy đủ dấu" in doc.content for doc in docs)
    assert all(doc.metadata["lang"] == "vi" for doc in docs)
