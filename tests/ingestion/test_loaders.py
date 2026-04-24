"""Tests for ingestion loaders output shape and metadata handling."""

from pathlib import Path

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
    assert doc.metadata["source_type"] == "notes"


def test_markdown_loader_preserves_section_metadata(tmp_path: Path) -> None:
    source = tmp_path / "guide.md"
    source.write_text(
        "# Getting Started\n\nOverview text.\n\n## Setup\n\nSetup steps.",
        encoding="utf-8",
    )

    loader = MarkdownLoader()
    docs = loader.load(source, metadata={"lang": "en"})

    assert len(docs) == 2
    first, second = docs

    assert first.doc_id == second.doc_id
    assert first.source == str(source)
    assert second.source == str(source)
    assert first.title == "Getting Started"
    assert second.title == "Getting Started"
    assert first.section == "Getting Started"
    assert second.section == "Setup"
    assert first.page is None
    assert second.page is None
    assert first.metadata["lang"] == "en"
    assert second.metadata["lang"] == "en"
