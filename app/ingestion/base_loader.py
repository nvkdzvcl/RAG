"""Base loader contract and shared helper functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from hashlib import sha1
from pathlib import Path
from typing import Any

from app.schemas.ingestion import LoadedDocument


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
