"""Vector index abstractions and in-memory implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.schemas.ingestion import DocumentChunk


class VectorIndex(ABC):
    """Abstract vector index contract."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of indexed chunks."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension of the index."""

    @property
    @abstractmethod
    def revision(self) -> int:
        """Monotonic revision incremented when index contents change."""

    @property
    @abstractmethod
    def chunks(self) -> list[DocumentChunk]:
        """Indexed chunks in internal order."""

    @abstractmethod
    def build(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        """Build index from chunks and pre-computed vectors."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize index payload for persistence."""

    @classmethod
    @abstractmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VectorIndex":
        """Restore index instance from persisted payload."""


class InMemoryVectorIndex(VectorIndex):
    """In-memory vector index storing chunk payload + embedding vectors."""

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []
        self._dimension: int = 0
        self._revision: int = 0

    @property
    def size(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)

    @property
    def vectors(self) -> list[list[float]]:
        return [list(vector) for vector in self._vectors]

    def build(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if not chunks:
            raise ValueError("Cannot build vector index from empty chunks")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")

        dimension = len(vectors[0])
        if dimension == 0:
            raise ValueError("vectors must have non-zero dimension")
        for vector in vectors:
            if len(vector) != dimension:
                raise ValueError("All vectors must share the same dimension")

        self._chunks = [chunk.model_copy(deep=True) for chunk in chunks]
        self._vectors = [list(vector) for vector in vectors]
        self._dimension = dimension
        self._revision += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self._dimension,
            "revision": self._revision,
            "entries": [
                {
                    "chunk": chunk.model_dump(),
                    "vector": vector,
                }
                for chunk, vector in zip(self._chunks, self._vectors)
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InMemoryVectorIndex":
        index = cls()
        entries = payload.get("entries", [])
        persisted_revision = payload.get("revision")

        chunks: list[DocumentChunk] = []
        vectors: list[list[float]] = []
        for entry in entries:
            chunks.append(DocumentChunk.model_validate(entry["chunk"]))
            vectors.append([float(value) for value in entry["vector"]])

        if chunks:
            index.build(chunks, vectors)
        if isinstance(persisted_revision, int) and persisted_revision >= 0:
            index._revision = max(index._revision, persisted_revision)
        return index
