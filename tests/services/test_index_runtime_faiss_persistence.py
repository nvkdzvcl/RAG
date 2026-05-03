"""FAISS runtime persistence tests for uploaded/seeded index artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["EMBEDDING_HASH_DIMENSION"] = "64"
os.environ["RERANKER_PROVIDER"] = "score_only"
os.environ["RERANKER_ENABLED"] = "false"
os.environ["LLM_PROVIDER"] = "stub"

from app.core.config import get_settings
from app.ingestion.base_loader import build_doc_id
from app.indexing import BaseEmbeddingProvider
from app.services.index_runtime import RuntimeIndexManager


class _CountingEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimension: int = 8) -> None:
        self.name = "counting-provider"
        self.dimension = dimension
        self.document_inputs: list[str] = []
        self.query_inputs: list[str] = []

    def _vectorize(self, text: str) -> list[float]:
        base = sum(ord(char) for char in text) % 997
        return [float((base + index) % 97) for index in range(self.dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_inputs.extend(texts)
        return [self._vectorize(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_inputs.append(text)
        return self._vectorize(text)


class _NoRebuildEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimension: int = 8) -> None:
        self.name = "counting-provider"
        self.dimension = dimension
        self.query_inputs: list[str] = []

    def _vectorize(self, text: str) -> list[float]:
        base = sum(ord(char) for char in text) % 997
        return [float((base + index) % 97) for index in range(self.dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        _ = texts
        raise AssertionError(
            "embed_documents should not be called when loading persisted FAISS uploaded index"
        )

    def embed_query(self, text: str) -> list[float]:
        self.query_inputs.append(text)
        return self._vectorize(text)


class _FakeIndexFlatIP:
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self._matrix = np.empty((0, self.dimension), dtype=np.float32)

    def add(self, matrix: np.ndarray) -> None:
        if matrix.ndim != 2 or matrix.shape[1] != self.dimension:
            raise ValueError("shape mismatch")
        if matrix.shape[0] == 0:
            return
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
    def _write_index(index: _FakeIndexFlatIP, path: str) -> None:
        payload = {
            "dimension": int(index.dimension),
            "matrix": index._matrix.tolist(),
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _read_index(path: str) -> _FakeIndexFlatIP:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        dimension = int(payload.get("dimension", 0))
        restored = _FakeIndexFlatIP(dimension)
        matrix = np.asarray(payload.get("matrix", []), dtype=np.float32)
        if matrix.size > 0:
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            restored.add(matrix)
        return restored

    fake_module = SimpleNamespace(
        IndexFlatIP=_FakeIndexFlatIP,
        write_index=_write_index,
        read_index=_read_index,
    )
    monkeypatch.setattr(
        "app.indexing.faiss_index.importlib.import_module",
        lambda name: fake_module if name == "faiss" else __import__(name),
    )
    monkeypatch.setattr(
        "app.indexing.persistence.importlib.import_module",
        lambda name: fake_module if name == "faiss" else __import__(name),
    )


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_faiss_uploaded_index_persistence_and_restart_search_consistency(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_faiss(monkeypatch)
    monkeypatch.setenv("VECTOR_INDEX_BACKEND", "faiss")

    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "indexes"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    uploaded_file = tmp_path / "uploaded-faiss.txt"
    uploaded_file.write_text(
        "uploaded-faiss-token-9291 appears only here.",
        encoding="utf-8",
    )
    doc_id = build_doc_id(uploaded_file)

    manager_first = RuntimeIndexManager(
        corpus_dir=corpus_dir,
        index_dir=index_dir,
        embedding_provider=_CountingEmbeddingProvider(dimension=8),
    )
    first_count = manager_first.activate_from_uploaded_files(
        [uploaded_file], active_document_ids={doc_id}
    )
    first_query_vector = manager_first.embedding_provider.embed_query(
        "uploaded-faiss-token-9291"
    )
    first_vector_index = manager_first._load_uploaded_vector_index()  # noqa: SLF001
    first_ranked = first_vector_index.search(first_query_vector, top_k=5)
    first_result_chunk_ids = [
        first_vector_index.chunks[idx].chunk_id for idx, _score in first_ranked
    ]

    manager_second = RuntimeIndexManager(
        corpus_dir=corpus_dir,
        index_dir=index_dir,
        embedding_provider=_NoRebuildEmbeddingProvider(dimension=8),
    )
    second_count = manager_second.activate_from_uploaded_files(
        [uploaded_file], active_document_ids={doc_id}
    )
    second_query_vector = manager_second.embedding_provider.embed_query(
        "uploaded-faiss-token-9291"
    )
    second_vector_index = manager_second._load_uploaded_vector_index()  # noqa: SLF001
    second_ranked = second_vector_index.search(second_query_vector, top_k=5)
    second_result_chunk_ids = [
        second_vector_index.chunks[idx].chunk_id for idx, _score in second_ranked
    ]

    assert first_count > 0
    assert second_count == first_count
    assert first_result_chunk_ids
    assert second_result_chunk_ids
    assert first_result_chunk_ids == second_result_chunk_ids

    settings = get_settings()
    uploaded_binary = index_dir / settings.faiss_uploaded_index_filename
    uploaded_metadata = index_dir / settings.faiss_uploaded_metadata_filename
    uploaded_manifest = index_dir / manager_first.UPLOADED_MANIFEST_FILENAME
    uploaded_bm25 = index_dir / manager_first.UPLOADED_BM25_FILENAME
    assert uploaded_binary.exists()
    assert uploaded_metadata.exists()
    assert uploaded_manifest.exists()
    assert uploaded_bm25.exists()


def test_faiss_seeded_index_persistence_writes_seeded_split_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_faiss(monkeypatch)
    monkeypatch.setenv("VECTOR_INDEX_BACKEND", "faiss")

    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "indexes"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "seeded-faiss.txt").write_text(
        "seeded-faiss-token-5512 is in seeded corpus.",
        encoding="utf-8",
    )

    manager = RuntimeIndexManager(
        corpus_dir=corpus_dir,
        index_dir=index_dir,
        embedding_provider=_CountingEmbeddingProvider(dimension=8),
    )
    chunk_count = manager.activate_from_seeded_corpus()

    settings = get_settings()
    assert chunk_count > 0
    assert (index_dir / settings.faiss_index_filename).exists()
    assert (index_dir / settings.faiss_metadata_filename).exists()
    assert (index_dir / settings.faiss_seeded_index_filename).exists()
    assert (index_dir / settings.faiss_seeded_metadata_filename).exists()
    assert (index_dir / manager.SEEDED_BM25_FILENAME).exists()


def test_faiss_uploaded_manifest_mismatch_triggers_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_faiss(monkeypatch)
    monkeypatch.setenv("VECTOR_INDEX_BACKEND", "faiss")

    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "indexes"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    uploaded_file = tmp_path / "manifest-faiss.txt"
    uploaded_file.write_text("manifest-faiss-token-8802", encoding="utf-8")
    doc_id = build_doc_id(uploaded_file)

    first_provider = _CountingEmbeddingProvider(dimension=8)
    manager_first = RuntimeIndexManager(
        corpus_dir=corpus_dir,
        index_dir=index_dir,
        embedding_provider=first_provider,
    )
    assert (
        manager_first.activate_from_uploaded_files(
            [uploaded_file], active_document_ids={doc_id}
        )
        > 0
    )

    manifest_path = index_dir / manager_first.UPLOADED_MANIFEST_FILENAME
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["fingerprint"] = "tampered-faiss-fingerprint"
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    second_provider = _CountingEmbeddingProvider(dimension=8)
    manager_second = RuntimeIndexManager(
        corpus_dir=corpus_dir,
        index_dir=index_dir,
        embedding_provider=second_provider,
    )
    rebuilt_count = manager_second.activate_from_uploaded_files(
        [uploaded_file], active_document_ids={doc_id}
    )

    manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rebuilt_count > 0
    assert second_provider.document_inputs
    assert manifest_after["fingerprint"] != "tampered-faiss-fingerprint"
