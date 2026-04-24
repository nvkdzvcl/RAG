"""Loader for plain text documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.base_loader import BaseLoader, build_doc_id
from app.schemas.ingestion import LoadedDocument


class TextLoader(BaseLoader):
    """Load .txt files as single logical documents."""

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

        content = path.read_text(encoding="utf-8")
        base_metadata = dict(metadata or {})

        return [
            LoadedDocument(
                doc_id=doc_id or build_doc_id(path),
                source=str(path),
                title=title or path.stem,
                content=content,
                metadata=base_metadata,
            )
        ]
