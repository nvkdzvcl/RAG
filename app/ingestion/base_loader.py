"""Base loader contract and shared helper functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha1
from pathlib import Path
from typing import Any

from app.schemas.ingestion import DocumentBlock, LoadedDocument


class BaseLoader(ABC):
    """Abstract document loader."""

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Return whether this loader can parse the path."""

    @abstractmethod
    def load(
        self,
        path: Path,
        *,
        doc_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[LoadedDocument]:
        """Load path into one or more normalized documents."""


def build_doc_id(path: Path) -> str:
    """Build stable doc id from file path."""
    digest = sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    stem = path.stem.lower().replace(" ", "_")
    return f"{stem}_{digest}"


def blocks_to_loaded_documents(
    *,
    blocks: list[DocumentBlock],
    file_path: Path,
    doc_id: str,
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[LoadedDocument]:
    """Convert parser blocks to loader-normalized documents."""
    base_metadata = dict(metadata or {})
    resolved_title = title or file_path.stem

    loaded_documents: list[LoadedDocument] = []
    for block_index, block in enumerate(blocks):
        merged_metadata = dict(base_metadata)
        merged_metadata.update(dict(block.metadata))
        merged_metadata.update({"block_index": block_index})

        loaded_documents.append(
            LoadedDocument(
                doc_id=doc_id,
                source=str(file_path),
                title=resolved_title,
                section=block.metadata.get("section"),
                page=block.metadata.get("page"),
                content=block.content,
                block_type=block.type,
                language=str(block.metadata.get("language", "auto")),
                metadata=merged_metadata,
            )
        )

    return loaded_documents
