"""PDF parser implementation for mixed-content block extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.ingestion.ocr import (
    configure_tesseract_cmd,
    is_tesseract_available,
    ocr_pdf_page_with_pymupdf,
)
from app.ingestion.parsers.base import BaseDocumentParser
from app.ingestion.parsers.utils import rows_to_markdown_table, split_paragraphs
from app.schemas.ingestion import DocumentBlock

try:
    import pdfplumber
except ModuleNotFoundError:  # pragma: no cover - dependency validation tested elsewhere.
    pdfplumber = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class PDFParser(BaseDocumentParser):
    """Parse PDF pages into text/table/image blocks."""

    def __init__(
        self,
        *,
        ocr_enabled: bool | None = None,
        ocr_language: str | None = None,
        ocr_min_text_chars: int | None = None,
        ocr_render_dpi: int | None = None,
        tesseract_cmd: str | None = None,
        ocr_confidence_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self.ocr_enabled = ocr_enabled if ocr_enabled is not None else settings.ocr_enabled
        self.ocr_language = ocr_language if ocr_language is not None else settings.ocr_language
        self.ocr_min_text_chars = (
            ocr_min_text_chars if ocr_min_text_chars is not None else settings.ocr_min_text_chars
        )
        self.ocr_render_dpi = ocr_render_dpi if ocr_render_dpi is not None else settings.ocr_render_dpi
        self.tesseract_cmd = tesseract_cmd if tesseract_cmd is not None else settings.tesseract_cmd
        self.ocr_confidence_threshold = (
            ocr_confidence_threshold
            if ocr_confidence_threshold is not None
            else settings.ocr_confidence_threshold
        )
        configure_tesseract_cmd(self.tesseract_cmd)

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

    def _extract_ocr_block(
        self,
        *,
        path: Path,
        page_index_zero_based: int,
        page_number: int,
        section: str | None,
    ) -> DocumentBlock | None:
        try:
            ocr_text = ocr_pdf_page_with_pymupdf(
                path,
                page_index=page_index_zero_based,
                dpi=self.ocr_render_dpi,
                lang=self.ocr_language,
                confidence_threshold=self.ocr_confidence_threshold,
            ).strip()
        except Exception:
            logger.warning(
                "OCR failed for PDF page; continuing without OCR block.",
                extra={"path": str(path), "page": page_number},
                exc_info=True,
            )
            return None

        if not ocr_text:
            return None

        return DocumentBlock(
            type="text",
            content=ocr_text,
            metadata={
                "page": page_number,
                "section": section,
                "bbox": None,
                "block_type": "ocr_text",
                "ocr": True,
                "ocr_language": self.ocr_language,
                "ocr_render_dpi": self.ocr_render_dpi,
                "ocr_confidence_threshold": self.ocr_confidence_threshold,
            },
        )

    def parse(self, path: Path) -> list[DocumentBlock]:
        if pdfplumber is None:
            raise RuntimeError("PDF parsing requires the 'pdfplumber' package.")

        blocks: list[DocumentBlock] = []
        current_section: str | None = None
        ocr_ready = False
        if self.ocr_enabled:
            ocr_ready = is_tesseract_available()
            if not ocr_ready:
                logger.warning(
                    "OCR is enabled but Tesseract is not available; PDF OCR fallback is skipped.",
                    extra={"path": str(path)},
                )

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

                page_text_chars = len(page_text.strip())
                should_try_ocr = (
                    ocr_ready
                    and page_text_chars < self.ocr_min_text_chars
                )
                if should_try_ocr:
                    ocr_block = self._extract_ocr_block(
                        path=path,
                        page_index_zero_based=page_idx - 1,
                        page_number=page_idx,
                        section=current_section,
                    )
                    if ocr_block is not None:
                        blocks.append(ocr_block)

        return blocks
