"""Tests for directory-level ingestion pipeline."""

from pathlib import Path

import pytest

from app.ingestion.directory_ingestor import DirectoryIngestor


def test_directory_ingestor_loads_markdown_and_text(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)

    (corpus / "a.txt").write_text("plain text content", encoding="utf-8")
    (corpus / "guide.md").write_text("# Guide\n\nIntro section.\n", encoding="utf-8")
    from docx import Document

    doc = Document()
    doc.add_paragraph("Nội dung DOCX thử nghiệm.")
    doc.save(str(corpus / "note.docx"))

    ingestor = DirectoryIngestor()
    docs = ingestor.ingest_directory(corpus)

    assert len(docs) >= 2
    assert any(doc.source.endswith("a.txt") for doc in docs)
    assert any(doc.source.endswith("guide.md") for doc in docs)
    assert any(doc.source.endswith("note.docx") for doc in docs)
    assert all("relative_path" in doc.metadata for doc in docs)


def test_directory_ingestor_rejects_empty_supported_files(tmp_path: Path) -> None:
    corpus = tmp_path / "empty_corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "ignore.json").write_text('{"k":"v"}', encoding="utf-8")

    ingestor = DirectoryIngestor()

    with pytest.raises(ValueError, match="No supported documents found"):
        ingestor.ingest_directory(corpus)
