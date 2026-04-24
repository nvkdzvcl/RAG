"""PDF parser implementation for mixed-content block extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.parsers.base import BaseDocumentParser
from app.ingestion.parsers.utils import rows_to_markdown_table, split_paragraphs
from app.schemas.ingestion import DocumentBlock

try:
    import pdfplumber
except ModuleNotFoundError:  # pragma: no cover - dependency validation tested elsewhere.
    pdfplumber = None  # type: ignore[assignment]


class PDFParser(BaseDocumentParser):
    """Parse PDF pages into text/table/image blocks."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    @staticmethod
    def _heading_candidate(paragraph: str) -> bool:
        stripped = paragraph.strip()
        if not stripped:
            return False
        words = stripped.split()
        if len(words) > 12:
            return False
        if stripped.endswith("."):
            return False
        return True

    @staticmethod
    def _image_bbox(image: dict[str, Any]) -> list[float] | None:
        x0 = image.get("x0")
        top = image.get("top")
        x1 = image.get("x1")
        bottom = image.get("bottom")
        coords = [x0, top, x1, bottom]
        if any(value is None for value in coords):
            return None
        return [float(value) for value in coords]

    def parse(self, path: Path) -> list[DocumentBlock]:
        if pdfplumber is None:
            raise RuntimeError("PDF parsing requires the 'pdfplumber' package.")

        blocks: list[DocumentBlock] = []
        current_section: str | None = None

        with pdfplumber.open(str(path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                paragraphs = split_paragraphs(page_text)

                for paragraph in paragraphs:
                    is_heading = self._heading_candidate(paragraph)
                    if is_heading:
                        current_section = paragraph

                    blocks.append(
                        DocumentBlock(
                            type="text",
                            content=paragraph,
                            metadata={
                                "page": page_idx,
                                "section": current_section,
                                "bbox": None,
                                "is_heading": is_heading,
                            },
                        )
                    )

                tables = page.extract_tables() or []
                for table in tables:
                    table_text = rows_to_markdown_table(table or [])
                    if not table_text:
                        continue
                    blocks.append(
                        DocumentBlock(
                            type="table",
                            content=table_text,
                            metadata={
                                "page": page_idx,
                                "section": current_section,
                                "bbox": None,
                            },
                        )
                    )

                images = page.images or []
                for image_idx, image in enumerate(images, start=1):
                    image_name = image.get("name") or f"image_{image_idx}"
                    blocks.append(
                        DocumentBlock(
                            type="image",
                            content=f"[image:{image_name}]",
                            metadata={
                                "page": page_idx,
                                "section": current_section,
                                "bbox": self._image_bbox(image),
                                "image_index": image_idx,
                            },
                        )
                    )

        return blocks
