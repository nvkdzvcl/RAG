"""Loader for Word (.docx) documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.base_loader import BaseLoader, blocks_to_loaded_documents, build_doc_id
from app.ingestion.parsers import DocxParser
from app.schemas.ingestion import DocumentBlock, LoadedDocument


class DocxLoader(BaseLoader):
    """Load .docx files with structured text/table extraction."""

    def __init__(self, parser: DocxParser | None = None) -> None:
        self.parser = parser or DocxParser()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".docx"

    def load(
        self,
        path: Path,
        *,
        doc_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[LoadedDocument]:
        if not self.supports(path):
            raise ValueError(f"Unsupported DOCX file: {path}")

        resolved_doc_id = doc_id or build_doc_id(path)
        resolved_title = title or path.stem

        try:
            blocks = self.parser.parse(path)
        except Exception:
            blocks = [
                DocumentBlock(
                    type="text",
                    content=path.read_text(encoding="utf-8", errors="ignore"),
                    metadata={"page": None, "section": None, "bbox": None, "parser_fallback": True},
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
