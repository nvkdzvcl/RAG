"""Loader for markdown documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.base_loader import (
    BaseLoader,
    blocks_to_loaded_documents,
    build_doc_id,
)
from app.ingestion.parsers import MarkdownParser
from app.ingestion.parsers.utils import read_text_with_fallback
from app.schemas.ingestion import DocumentBlock, LoadedDocument


class MarkdownLoader(BaseLoader):
    """Load .md/.markdown files and split by heading sections."""

    def __init__(self, parser: MarkdownParser | None = None) -> None:
        self.parser = parser or MarkdownParser()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in {".md", ".markdown"}

    def load(
        self,
        path: Path,
        *,
        doc_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[LoadedDocument]:
        if not self.supports(path):
            raise ValueError(f"Unsupported markdown file: {path}")

        resolved_doc_id = doc_id or build_doc_id(path)
        resolved_title = title or path.stem

        try:
            blocks = self.parser.parse(path)
        except Exception:
            raw = read_text_with_fallback(path)
            blocks = [
                DocumentBlock(
                    type="text",
                    content=raw,
                    metadata={
                        "page": None,
                        "section": None,
                        "bbox": None,
                        "parser_fallback": True,
                    },
                )
            ]

        if not blocks:
            return []

        extracted_heading = next(
            (
                block.content.strip()
                for block in blocks
                if block.metadata.get("is_heading") and block.content.strip()
            ),
            None,
        )
        final_title = resolved_title
        if title is None and extracted_heading:
            final_title = extracted_heading

        return blocks_to_loaded_documents(
            blocks=blocks,
            file_path=path,
            doc_id=resolved_doc_id,
            title=final_title,
            metadata=metadata,
        )
