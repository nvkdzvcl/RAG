"""Loader for PDF documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.base_loader import BaseLoader, build_doc_id
from app.schemas.ingestion import LoadedDocument

try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # pragma: no cover - exercised only when dependency is missing.
    PdfReader = None  # type: ignore[assignment]


class PdfLoader(BaseLoader):
    """Load .pdf files and emit one logical document per non-empty page."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() == ".pdf"

    def load(
        self,
        path: Path,
        *,
        doc_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[LoadedDocument]:
        if not self.supports(path):
            raise ValueError(f"Unsupported PDF file: {path}")
        if PdfReader is None:
            raise RuntimeError("PDF support requires the 'pypdf' package.")

        reader = PdfReader(str(path))
        base_metadata = dict(metadata or {})
        resolved_doc_id = doc_id or build_doc_id(path)
        resolved_title = title or path.stem

        documents: list[LoadedDocument] = []
        for page_index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue

            documents.append(
                LoadedDocument(
                    doc_id=resolved_doc_id,
                    source=str(path),
                    title=resolved_title,
                    section=None,
                    page=page_index,
                    content=text,
                    metadata=base_metadata,
                )
            )

        if not documents:
            raise ValueError(f"No extractable text found in PDF: {path}")
        return documents
