"""FAISS-backed vector index implementation."""

from __future__ import annotations

import importlib
import os
from typing import Any

import numpy as np

from app.indexing.vector_index import VectorIndex
from app.schemas.ingestion import DocumentChunk


class FaissVectorIndex(VectorIndex):
    """Vector index backed by ``faiss.IndexFlatIP`` with L2-normalized vectors."""

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[list[float]] = []
        self._id_map: list[str] = []
        self._dimension: int = 0
        self._revision: int = 0
        self._faiss_index: Any | None = None

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

    @property
    def id_map(self) -> list[str]:
        return list(self._id_map)

    @staticmethod
    def _normalize_l2(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        non_zero = (norms[:, 0] > 0).astype(np.bool_)
        if np.any(non_zero):
            matrix[non_zero] = matrix[non_zero] / norms[non_zero]
        return matrix

    @staticmethod
    def _to_matrix(vectors: list[list[float]]) -> np.ndarray:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError("Vector index must contain a 2D matrix of vectors")
        return matrix

    @staticmethod
    def _load_faiss() -> Any:
        try:
            return importlib.import_module("faiss")
        except Exception as exc:
            backend = os.getenv("VECTOR_INDEX_BACKEND", "inmemory").strip().lower()
            if backend == "faiss":
                raise RuntimeError(
                    "VECTOR_INDEX_BACKEND=faiss but faiss is not installed. "
                    "Install faiss-cpu (or faiss-gpu) to use this backend."
                ) from exc
            raise RuntimeError(
                "FaissVectorIndex requires the 'faiss' package. "
                "Install faiss-cpu (or faiss-gpu), or switch to inmemory backend."
            ) from exc

    def _build_faiss_index(self) -> None:
        if not self._vectors:
            self._faiss_index = None
            return
        faiss = self._load_faiss()
        index = faiss.IndexFlatIP(self._dimension)
        matrix = self._to_matrix(self._vectors)
        index.add(matrix)
        self._faiss_index = index

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

        matrix = self._to_matrix(vectors)
        normalized = self._normalize_l2(matrix)

        self._chunks = [chunk.model_copy(deep=True) for chunk in chunks]
        self._vectors = normalized.astype(np.float64).tolist()
        self._id_map = [chunk.chunk_id for chunk in self._chunks]
        self._dimension = dimension
        self._revision += 1
        self._build_faiss_index()

    def search(
        self, query_vector: list[float], top_k: int = 5
    ) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        if self.size == 0:
            return []

        if self._faiss_index is None:
            self._build_faiss_index()
        if self._faiss_index is None:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self._dimension:
            raise ValueError("Vector dimension mismatch for search")

        query_matrix = query.reshape(1, -1)
        query_matrix = self._normalize_l2(query_matrix)

        k = min(top_k, self.size)
        scores, indices = self._faiss_index.search(query_matrix, k)

        ranked: list[tuple[int, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            resolved_idx = int(idx)
            if resolved_idx < 0:
                continue
            ranked.append((resolved_idx, float(score)))
        return ranked

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self._dimension,
            "revision": self._revision,
            "id_map": list(self._id_map),
            "entries": [
                {
                    "chunk": chunk.model_dump(),
                    "vector": vector,
                }
                for chunk, vector in zip(self._chunks, self._vectors)
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FaissVectorIndex":
        index = cls()
        persisted_revision = payload.get("revision")

        chunks: list[DocumentChunk] = []
        vectors: list[list[float]] = []
        entries = payload.get("entries")
        if isinstance(entries, list):
            for entry in entries:
                chunks.append(DocumentChunk.model_validate(entry["chunk"]))
                vectors.append([float(value) for value in entry["vector"]])
        else:
            raw_chunks = payload.get("chunks")
            raw_vectors = payload.get("vectors")
            if isinstance(raw_chunks, list) and isinstance(raw_vectors, list):
                chunks = [DocumentChunk.model_validate(item) for item in raw_chunks]
                vectors = [
                    [float(value) for value in vector]
                    for vector in raw_vectors
                    if isinstance(vector, list)
                ]
                if len(vectors) != len(raw_vectors):
                    raise ValueError("Invalid vector payload for FaissVectorIndex")

        if chunks:
            index.build(chunks, vectors)

        if isinstance(persisted_revision, int) and persisted_revision >= 0:
            index._revision = max(index._revision, persisted_revision)

        raw_id_map = payload.get("id_map", [])
        if (
            isinstance(raw_id_map, list)
            and len(raw_id_map) == len(index._chunks)
            and all(isinstance(item, str) for item in raw_id_map)
        ):
            index._id_map = list(raw_id_map)

        return index
