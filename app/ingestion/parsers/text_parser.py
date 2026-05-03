"""Plain-text parser implementation."""

from __future__ import annotations

from pathlib import Path

from app.ingestion.parsers.base import BaseDocumentParser
from app.ingestion.parsers.utils import read_text_with_fallback, split_paragraphs
from app.schemas.ingestion import DocumentBlock


class TextParser(BaseDocumentParser):
    """Parse .txt files into paragraph-level text blocks."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

    def parse(self, path: Path) -> list[DocumentBlock]:
        raw_text = read_text_with_fallback(path)
        paragraphs = split_paragraphs(raw_text)

        if not paragraphs:
            return []

        return [
            DocumentBlock(
                type="text",
                content=paragraph,
                metadata={"page": None, "section": None, "bbox": None},
            )
            for paragraph in paragraphs
        ]
