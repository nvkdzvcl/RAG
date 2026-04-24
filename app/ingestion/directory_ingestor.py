"""Directory-level ingestion pipeline for text, markdown, and PDF documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.base_loader import BaseLoader
from app.ingestion.markdown_loader import MarkdownLoader
from app.ingestion.pdf_loader import PdfLoader
from app.ingestion.text_loader import TextLoader
from app.schemas.ingestion import LoadedDocument


class DirectoryIngestor:
    """Ingest supported files from a directory tree into normalized documents."""

    def __init__(self, loaders: list[BaseLoader] | None = None) -> None:
        self.loaders = loaders or [MarkdownLoader(), TextLoader(), PdfLoader()]

    def _resolve_loader(self, path: Path) -> BaseLoader | None:
        for loader in self.loaders:
            if loader.supports(path):
                return loader
        return None

    def iter_supported_files(self, root_dir: Path | str) -> list[Path]:
        root = Path(root_dir)
        if not root.exists():
            raise FileNotFoundError(f"Corpus directory not found: {root}")
        if not root.is_dir():
            raise ValueError(f"Corpus path must be a directory: {root}")

        files = sorted(path for path in root.rglob("*") if path.is_file())
        return [path for path in files if self._resolve_loader(path) is not None]

    def ingest_directory(
        self,
        root_dir: Path | str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> list[LoadedDocument]:
        root = Path(root_dir)
        supported_files = self.iter_supported_files(root)
        if not supported_files:
            raise ValueError(f"No supported documents found in corpus directory: {root}")

        base_metadata = dict(metadata or {})
        loaded_documents: list[LoadedDocument] = []

        for file_path in supported_files:
            loader = self._resolve_loader(file_path)
            if loader is None:
                continue

            relative_path = str(file_path.relative_to(root))
            doc_metadata = dict(base_metadata)
            doc_metadata.update(
                {
                    "relative_path": relative_path,
                    "file_extension": file_path.suffix.lower(),
                }
            )

            loaded_documents.extend(loader.load(file_path, metadata=doc_metadata))

        if not loaded_documents:
            raise ValueError(f"No documents could be loaded from corpus directory: {root}")

        return loaded_documents
