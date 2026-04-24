"""Loader for PDF documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.base_loader import BaseLoader, blocks_to_loaded_documents, build_doc_id
from app.ingestion.parsers import PDFParser
from app.schemas.ingestion import DocumentBlock, LoadedDocument

try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # pragma: no cover - exercised only when dependency is missing.
    PdfReader = None  # type: ignore[assignment]


class PdfLoader(BaseLoader):
    """Load .pdf files with parser-driven mixed-content extraction."""

    def __init__(self, parser: PDFParser | None = None) -> None:
        self.parser = parser or PDFParser()

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
        resolved_doc_id = doc_id or build_doc_id(path)
        resolved_title = title or path.stem

        try:
            blocks = self.parser.parse(path)
        except Exception:
            if PdfReader is None:
                raise RuntimeError("PDF parsing fallback requires the 'pypdf' package.")
            reader = PdfReader(str(path))
            blocks = []
            for page_index, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if not text:
                    continue
                blocks.append(
                    DocumentBlock(
                        type="text",
                        content=text,
                        metadata={
                            "page": page_index,
                            "section": None,
                            "bbox": None,
                            "parser_fallback": True,
                        },
                    )
                )

        if not blocks:
            raise ValueError(f"No extractable text found in PDF: {path}")

        return blocks_to_loaded_documents(
            blocks=blocks,
            file_path=path,
            doc_id=resolved_doc_id,
            title=resolved_title,
            metadata=metadata,
        )
