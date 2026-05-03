"""Loader for plain text documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.base_loader import (
    BaseLoader,
    blocks_to_loaded_documents,
    build_doc_id,
)
from app.ingestion.parsers import TextParser
from app.ingestion.parsers.utils import read_text_with_fallback
from app.schemas.ingestion import DocumentBlock, LoadedDocument


class TextLoader(BaseLoader):
    """Load .txt files as single logical documents."""

    def __init__(self, parser: TextParser | None = None) -> None:
        self.parser = parser or TextParser()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".txt"

    def load(
        self,
        path: Path,
        *,
        doc_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[LoadedDocument]:
        if not self.supports(path):
            raise ValueError(f"Unsupported text file: {path}")

        resolved_doc_id = doc_id or build_doc_id(path)
        resolved_title = title or path.stem

        try:
            blocks = self.parser.parse(path)
        except Exception:
            blocks = [
                DocumentBlock(
                    type="text",
                    content=read_text_with_fallback(path),
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

        return blocks_to_loaded_documents(
            blocks=blocks,
            file_path=path,
            doc_id=resolved_doc_id,
            title=resolved_title,
            metadata=metadata,
        )
