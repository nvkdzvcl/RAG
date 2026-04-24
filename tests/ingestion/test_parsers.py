"""Tests for parser abstraction and mixed-content block extraction."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.ingestion.chunker import Chunker
from app.ingestion.parsers.docx_parser import DocxParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.text_parser import TextParser
from app.ingestion.text_loader import TextLoader


def test_pdf_parser_extracts_text_table_and_image_blocks(monkeypatch) -> None:
    class FakePage:
        images = [{"name": "img_1", "x0": 1, "top": 2, "x1": 3, "bottom": 4}]

        @staticmethod
        def extract_text() -> str:
            return "Muc Luc\n\nDoan van thu nhat.\n\nDoan van thu hai."

        @staticmethod
        def extract_tables() -> list[list[list[str]]]:
            return [[["Name", "Score"], ["A", "10"]]]

    class FakePDF:
        def __init__(self) -> None:
            self.pages = [FakePage()]

        def __enter__(self) -> "FakePDF":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            _ = exc_type
            _ = exc
            _ = tb

    monkeypatch.setattr(
        "app.ingestion.parsers.pdf_parser.pdfplumber",
        SimpleNamespace(open=lambda _: FakePDF()),
    )

    parser = PDFParser()
    blocks = parser.parse(Path("sample.pdf"))

    assert any(block.type == "text" for block in blocks)
    assert any(block.type == "table" for block in blocks)
    assert any(block.type == "image" for block in blocks)
    assert all("page" in block.metadata for block in blocks)


def test_docx_parser_extracts_heading_paragraph_and_table(tmp_path: Path) -> None:
    from docx import Document

    doc_path = tmp_path / "mixed.docx"
    doc = Document()
    doc.add_heading("Bao cao", level=1)
    doc.add_paragraph("Noi dung chinh cua tai lieu.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Score"
    table.cell(1, 0).text = "A"
    table.cell(1, 1).text = "10"
    doc.save(str(doc_path))

    parser = DocxParser()
    blocks = parser.parse(doc_path)

    assert any(block.type == "text" and block.metadata.get("is_heading") for block in blocks)
    assert any(block.type == "table" for block in blocks)
    table_block = next(block for block in blocks if block.type == "table")
    assert "| Name | Score |" in table_block.content


def test_vietnamese_text_preserved_through_loader_and_chunker(tmp_path: Path) -> None:
    path = tmp_path / "vi.txt"
    vi_text = "Tiếng Việt có dấu: Trường đại học, nghiên cứu, dữ liệu."
    path.write_text(vi_text, encoding="utf-8")

    docs = TextLoader().load(path, metadata={"lang": "vi"})
    chunks = Chunker(chunk_size=16, chunk_overlap=2).chunk_documents(docs)

    assert docs[0].content == vi_text
    assert any("Tiếng Việt có dấu" in chunk.content for chunk in chunks)
    assert all(chunk.metadata.get("language") == "auto" or chunk.metadata.get("language") == "vi" for chunk in chunks)
    assert all(chunk.metadata.get("block_type") == "text" for chunk in chunks)


def test_text_parser_splits_paragraphs_without_losing_utf8(tmp_path: Path) -> None:
    path = tmp_path / "mixed.txt"
    path.write_text("Đoạn một.\n\nĐoạn hai có bảng giả lập.", encoding="utf-8")

    blocks = TextParser().parse(path)

    assert len(blocks) == 2
    assert blocks[0].content == "Đoạn một."
    assert "Đoạn hai" in blocks[1].content
