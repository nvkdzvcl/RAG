"""Embedding provider interfaces for indexing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding backends."""

    name: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of document texts."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query text."""


class EmbeddingProvider(Protocol):
    """Structural typing contract for embedding backends."""

    name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of document texts."""

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query text."""
