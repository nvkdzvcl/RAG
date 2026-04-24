"""Parser contracts for mixed-content document ingestion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.ingestion import DocumentBlock


class BaseDocumentParser(ABC):
    """Abstract parser interface for file-specific block extraction."""

    @abstractmethod
    def supports(self, path: Path) -> bool:
        """Return whether this parser can read the provided path."""

    @abstractmethod
    def parse(self, path: Path) -> list[DocumentBlock]:
        """Extract structured content blocks from a file."""
