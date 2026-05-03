"""Unit tests for FAISS-backed vector index."""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from app.indexing.faiss_index import FaissVectorIndex
from app.schemas.ingestion import DocumentChunk


class _FakeIndexFlatIP:
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self._matrix = np.empty((0, self.dimension), dtype=np.float32)

    def add(self, matrix: np.ndarray) -> None:
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError("shape mismatch")
        if self._matrix.size == 0:
            self._matrix = matrix.copy()
            return
        self._matrix = np.vstack([self._matrix, matrix])

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        query_count = query.shape[0]
        scores = np.zeros((query_count, top_k), dtype=np.float32)
        indices = np.full((query_count, top_k), -1, dtype=np.int64)
        if self._matrix.size == 0:
            return scores, indices

        similarities = query @ self._matrix.T
        for row_idx in range(query_count):
            row = similarities[row_idx]
            ranked = np.lexsort((np.arange(row.shape[0]), -row))
            selected = ranked[:top_k]
            scores[row_idx, : selected.shape[0]] = row[selected]
            indices[row_idx, : selected.shape[0]] = selected
        return scores, indices


def _install_fake_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_module = types.SimpleNamespace(IndexFlatIP=_FakeIndexFlatIP)
    monkeypatch.setattr(
        "app.indexing.faiss_index.importlib.import_module",
        lambda name: fake_module if name == "faiss" else __import__(name),
    )


def _build_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="chunk_0",
            doc_id="doc_0",
            source="memory://doc_0",
            content="alpha chunk",
            metadata={"source_type": "unit_test"},
        ),
        DocumentChunk(
            chunk_id="chunk_1",
            doc_id="doc_1",
            source="memory://doc_1",
            content="beta chunk",
            metadata={"source_type": "unit_test"},
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            doc_id="doc_2",
            source="memory://doc_2",
            content="gamma chunk",
            metadata={"source_type": "unit_test"},
        ),
    ]


def test_faiss_vector_index_build_and_search_returns_expected_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_faiss(monkeypatch)
    index = FaissVectorIndex()
    chunks = _build_chunks()
    vectors = [
        [1.0, 0.0, 0.0],
        [0.8, 0.2, 0.0],
        [0.0, 1.0, 0.0],
    ]

    index.build(chunks, vectors)
    ranked = index.search([5.0, 0.0, 0.0], top_k=2)

    assert index.size == 3
    assert index.dimension == 3
    assert len(ranked) == 2

    first_idx, first_score = ranked[0]
    second_idx, second_score = ranked[1]

    assert index.chunks[first_idx].chunk_id == "chunk_0"
    assert index.chunks[second_idx].chunk_id == "chunk_1"
    assert first_score > second_score


def test_faiss_vector_index_to_dict_from_dict_roundtrip_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_faiss(monkeypatch)
    chunks = _build_chunks()
    vectors = [
        [0.1, 0.2, 0.3],
        [0.3, 0.2, 0.1],
        [0.4, 0.4, 0.1],
    ]

    index = FaissVectorIndex()
    index.build(chunks, vectors)
    payload = index.to_dict()

    restored = FaissVectorIndex.from_dict(payload)
    ranked = restored.search([0.4, 0.4, 0.1], top_k=1)

    assert "entries" in payload
    assert len(payload["entries"]) == 3
    assert payload["entries"][0]["chunk"]["chunk_id"] == "chunk_0"
    assert isinstance(payload["entries"][0]["vector"], list)
    assert payload["id_map"] == ["chunk_0", "chunk_1", "chunk_2"]
    assert restored.size == index.size
    assert restored.dimension == index.dimension
    assert restored.id_map == index.id_map
    assert [item.chunk_id for item in restored.chunks] == [
        item.chunk_id for item in index.chunks
    ]
    assert restored.chunks[0].metadata["source_type"] == "unit_test"
    assert ranked
    assert restored.chunks[ranked[0][0]].chunk_id == "chunk_2"


def test_faiss_vector_index_from_dict_supports_legacy_chunks_vectors_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_faiss(monkeypatch)
    chunks = _build_chunks()
    index = FaissVectorIndex()
    index.build(
        chunks,
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    payload = index.to_dict()

    legacy_payload = {
        "dimension": payload["dimension"],
        "revision": payload["revision"],
        "id_map": payload["id_map"],
        "chunks": [entry["chunk"] for entry in payload["entries"]],
        "vectors": [entry["vector"] for entry in payload["entries"]],
    }
    restored = FaissVectorIndex.from_dict(legacy_payload)
    ranked = restored.search([1.0, 0.0, 0.0], top_k=1)

    assert restored.size == 3
    assert ranked
    assert restored.chunks[ranked[0][0]].chunk_id == "chunk_0"


def test_faiss_vector_index_handles_empty_index_safely() -> None:
    index = FaissVectorIndex()
    assert index.search([1.0, 0.0, 0.0], top_k=5) == []

    payload = index.to_dict()
    restored = FaissVectorIndex.from_dict(payload)

    assert restored.size == 0
    assert restored.dimension == 0
    assert restored.search([1.0, 0.0, 0.0], top_k=5) == []


def test_faiss_vector_index_dimension_mismatch_handling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_faiss(monkeypatch)
    chunks = _build_chunks()

    with pytest.raises(ValueError, match="length mismatch"):
        FaissVectorIndex().build(chunks, [[1.0, 0.0]])

    with pytest.raises(ValueError, match="same dimension"):
        FaissVectorIndex().build(chunks[:2], [[1.0, 0.0], [1.0]])

    index = FaissVectorIndex()
    index.build(chunks[:2], [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="dimension mismatch"):
        index.search([1.0, 0.0, 0.0], top_k=1)


def test_faiss_missing_dependency_raises_clear_error_when_backend_is_faiss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _missing_faiss(name: str, *_args: Any, **_kwargs: Any) -> Any:
        if name == "faiss":
            raise ModuleNotFoundError("No module named 'faiss'")
        return __import__(name)

    monkeypatch.setenv("VECTOR_INDEX_BACKEND", "faiss")
    monkeypatch.setattr(
        "app.indexing.faiss_index.importlib.import_module",
        _missing_faiss,
    )

    with pytest.raises(RuntimeError, match="VECTOR_INDEX_BACKEND=faiss"):
        FaissVectorIndex().build(
            _build_chunks()[:1],
            [[1.0, 0.0, 0.0]],
        )
