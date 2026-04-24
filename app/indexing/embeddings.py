"""Embedding provider interfaces for indexing."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Contract for embedding backends."""

    name: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents."""

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a query text."""
