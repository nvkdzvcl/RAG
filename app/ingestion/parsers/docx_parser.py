"""DOCX parser implementation."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers.base import BaseDocumentParser
from app.ingestion.parsers.utils import rows_to_markdown_table, split_paragraphs
from app.schemas.ingestion import DocumentBlock

try:
    from docx import Document
    from docx.document import Document as DocxDocument
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl
    from docx.table import Table
    from docx.text.paragraph import Paragraph
except ModuleNotFoundError:  # pragma: no cover - dependency validation tested elsewhere.
    Document = None  # type: ignore[assignment]
    DocxDocument = object  # type: ignore[assignment,misc]
    CT_P = object  # type: ignore[assignment,misc]
    CT_Tbl = object  # type: ignore[assignment,misc]
    Table = object  # type: ignore[assignment,misc]
    Paragraph = object  # type: ignore[assignment,misc]


class DocxParser(BaseDocumentParser):
    """Parse .docx files into text and table blocks."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".docx"

    @staticmethod
    def _iter_block_items(document: DocxDocument) -> list[Paragraph | Table]:
        parent = document.element.body
        items: list[Paragraph | Table] = []
        for child in parent.iterchildren():
            if isinstance(child, CT_P):
                items.append(Paragraph(child, document))
            elif isinstance(child, CT_Tbl):
                items.append(Table(child, document))
        return items

    def parse(self, path: Path) -> list[DocumentBlock]:
        if Document is None:
            raise RuntimeError("DOCX support requires the 'python-docx' package.")

        doc = Document(str(path))
        blocks: list[DocumentBlock] = []
        current_section: str | None = None

        for item in self._iter_block_items(doc):
            if isinstance(item, Paragraph):
                paragraph_text = (item.text or "").strip()
                if not paragraph_text:
                    continue
                style_name = (item.style.name or "").lower() if item.style else ""
                is_heading = style_name.startswith("heading")
                if is_heading:
                    current_section = paragraph_text

                for paragraph in split_paragraphs(paragraph_text):
                    blocks.append(
                        DocumentBlock(
                            type="text",
                            content=paragraph,
                            metadata={
                                "page": None,
                                "section": current_section,
                                "bbox": None,
                                "is_heading": is_heading,
                            },
                        )
                    )
                continue

            if isinstance(item, Table):
                rows = [[cell.text.strip() for cell in row.cells] for row in item.rows]
                table_text = rows_to_markdown_table(rows)
                if not table_text:
                    continue
                blocks.append(
                    DocumentBlock(
                        type="table",
                        content=table_text,
                        metadata={
                            "page": None,
                            "section": current_section,
                            "bbox": None,
                        },
                    )
                )

        return blocks
