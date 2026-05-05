"""PDF parser implementation for mixed-content block extraction."""

from __future__ import annotations

import logging
from pathlib import Path
import re
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
except (
    ModuleNotFoundError
):  # pragma: no cover - dependency validation tested elsewhere.
    pdfplumber = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class PDFParser(BaseDocumentParser):
    """Parse PDF pages into text/table/image blocks."""

    _TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ỹĐđ0-9]+")

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
        self.ocr_enabled = (
            ocr_enabled if ocr_enabled is not None else settings.ocr_enabled
        )
        self.ocr_language = (
            ocr_language if ocr_language is not None else settings.ocr_language
        )
        self.ocr_min_text_chars = (
            ocr_min_text_chars
            if ocr_min_text_chars is not None
            else settings.ocr_min_text_chars
        )
        self.ocr_render_dpi = (
            ocr_render_dpi if ocr_render_dpi is not None else settings.ocr_render_dpi
        )
        self.tesseract_cmd = (
            tesseract_cmd if tesseract_cmd is not None else settings.tesseract_cmd
        )
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
        if x0 is None or top is None or x1 is None or bottom is None:
            return None
        return [float(x0), float(top), float(x1), float(bottom)]

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
                "language": "vi" if "vie" in self.ocr_language.lower() else "auto",
                "ocr_render_dpi": self.ocr_render_dpi,
                "ocr_confidence_threshold": self.ocr_confidence_threshold,
            },
        )

    @classmethod
    def _text_quality_signals(cls, text: str) -> dict[str, float | int]:
        stripped = text.strip()
        if not stripped:
            return {
                "chars": 0,
                "token_count": 0,
                "alnum_ratio": 0.0,
                "unique_token_ratio": 0.0,
                "replacement_chars": 0,
            }

        chars = len(stripped)
        tokens = cls._TOKEN_PATTERN.findall(stripped)
        token_count = len(tokens)
        unique_token_ratio = (
            len({token.casefold() for token in tokens}) / token_count
            if token_count > 0
            else 0.0
        )
        alnum_count = sum(1 for char in stripped if char.isalnum())
        replacement_chars = stripped.count("\uFFFD")
        return {
            "chars": chars,
            "token_count": token_count,
            "alnum_ratio": round(alnum_count / max(chars, 1), 4),
            "unique_token_ratio": round(unique_token_ratio, 4),
            "replacement_chars": replacement_chars,
        }

    def _should_try_ocr(
        self,
        *,
        page_text: str,
        image_count: int,
    ) -> tuple[bool, str, dict[str, float | int]]:
        signals = self._text_quality_signals(page_text)
        chars = int(signals["chars"])
        token_count = int(signals["token_count"])
        alnum_ratio = float(signals["alnum_ratio"])
        unique_token_ratio = float(signals["unique_token_ratio"])
        replacement_chars = int(signals["replacement_chars"])

        if chars < self.ocr_min_text_chars:
            return True, "below_min_text_chars", signals
        if replacement_chars > 0:
            return True, "replacement_chars_detected", signals
        if image_count > 0 and token_count <= max(6, self.ocr_min_text_chars // 30):
            return True, "few_tokens_with_images", signals
        if image_count > 0 and alnum_ratio < 0.45:
            return True, "low_alnum_ratio_with_images", signals
        if token_count >= 12 and unique_token_ratio < 0.25:
            return True, "high_repetition_suspected_noise", signals
        return False, "text_quality_ok", signals

    def parse(self, path: Path) -> list[DocumentBlock]:
        if pdfplumber is None:
            raise RuntimeError("PDF parsing requires the 'pdfplumber' package.")

        blocks: list[DocumentBlock] = []
        current_section: str | None = None
        ocr_ready = False
        logger.info(
            (
                "PDF upload OCR settings | path=%s | OCR_ENABLED=%s | OCR_LANGUAGE=%s "
                "| OCR_MIN_TEXT_CHARS=%s | OCR_RENDER_DPI=%s | OCR_CONFIDENCE_THRESHOLD=%s"
            ),
            str(path),
            self.ocr_enabled,
            self.ocr_language,
            self.ocr_min_text_chars,
            self.ocr_render_dpi,
            self.ocr_confidence_threshold,
        )
        if self.ocr_enabled:
            ocr_ready = is_tesseract_available()
            if not ocr_ready:
                logger.warning(
                    "OCR is enabled but Tesseract is not available; PDF OCR fallback is skipped.",
                    extra={"path": str(path)},
                )

        with pdfplumber.open(str(path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                    logger.warning(
                        "PDF text extraction failed for page; continuing with fallback signals.",
                        extra={"path": str(path), "page": page_idx},
                        exc_info=True,
                    )
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

                try:
                    tables = page.extract_tables() or []
                except Exception:
                    tables = []
                    logger.warning(
                        "PDF table extraction failed for page; continuing without table blocks.",
                        extra={"path": str(path), "page": page_idx},
                        exc_info=True,
                    )
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

                try:
                    images = page.images or []
                except Exception:
                    images = []
                    logger.warning(
                        "PDF image extraction failed for page; continuing without image blocks.",
                        extra={"path": str(path), "page": page_idx},
                        exc_info=True,
                    )
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
                ocr_reason = "ocr_disabled_or_unavailable"
                text_signals = self._text_quality_signals(page_text)
                if ocr_ready:
                    should_try_ocr, ocr_reason, text_signals = self._should_try_ocr(
                        page_text=page_text,
                        image_count=len(images),
                    )
                else:
                    should_try_ocr = False
                logger.info(
                    (
                        "PDF page extraction stats | path=%s | page=%s | extracted_text_length=%s "
                        "| ocr_threshold=%s | ocr_ran=%s | ocr_reason=%s | token_count=%s "
                        "| alnum_ratio=%s | unique_token_ratio=%s | replacement_chars=%s | image_count=%s"
                    ),
                    str(path),
                    page_idx,
                    page_text_chars,
                    self.ocr_min_text_chars,
                    should_try_ocr,
                    ocr_reason,
                    int(text_signals["token_count"]),
                    float(text_signals["alnum_ratio"]),
                    float(text_signals["unique_token_ratio"]),
                    int(text_signals["replacement_chars"]),
                    len(images),
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
                        preview = ocr_block.content[:300].replace("\n", " ").strip()
                        logger.info(
                            (
                                "PDF OCR output | path=%s | page=%s | ocr_ran=true "
                                "| ocr_text_length=%s | ocr_preview=%s"
                            ),
                            str(path),
                            page_idx,
                            len(ocr_block.content),
                            preview,
                        )
                    else:
                        logger.info(
                            "PDF OCR output | path=%s | page=%s | ocr_ran=true | ocr_text_length=0 | ocr_preview=",
                            str(path),
                            page_idx,
                        )
                else:
                    logger.info(
                        "PDF OCR output | path=%s | page=%s | ocr_ran=false | ocr_text_length=0 | ocr_preview=",
                        str(path),
                        page_idx,
                    )

        ocr_blocks = sum(
            1 for block in blocks if block.metadata.get("block_type") == "ocr_text"
        )
        logger.info(
            "PDF parse summary | path=%s | total_blocks=%s | ocr_blocks=%s",
            str(path),
            len(blocks),
            ocr_blocks,
        )

        return blocks
