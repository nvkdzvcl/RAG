"""Local persistence for indexing artifacts."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from app.indexing.bm25_index import BM25Index
from app.indexing.vector_index import InMemoryVectorIndex, VectorIndex


class LocalIndexStore:
    """Store and load local vector/BM25 indexes as JSON files."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        return path

    def _read_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_faiss_module() -> Any:
        return importlib.import_module("faiss")

    def save_vector_index(
        self, index: VectorIndex, filename: str = "vector_index.json"
    ) -> Path:
        return self._write_json(self.base_dir / filename, index.to_dict())

    def load_vector_index(
        self, filename: str = "vector_index.json"
    ) -> InMemoryVectorIndex:
        payload = self._read_json(self.base_dir / filename)
        return InMemoryVectorIndex.from_dict(payload)

    def save_faiss_vector_index(
        self,
        index: VectorIndex,
        *,
        binary_filename: str,
        metadata_filename: str,
    ) -> tuple[Path, Path]:
        from app.indexing.faiss_index import FaissVectorIndex

        if not isinstance(index, FaissVectorIndex):
            raise TypeError("save_faiss_vector_index expects FaissVectorIndex input")

        faiss = self._load_faiss_module()
        binary_path = self.base_dir / binary_filename
        metadata_path = self.base_dir / metadata_filename
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss_index = getattr(index, "_faiss_index", None)
        if faiss_index is None:
            index._build_faiss_index()  # noqa: SLF001 - internal persistence hook
            faiss_index = getattr(index, "_faiss_index", None)
        if faiss_index is None:
            raise ValueError("Cannot persist empty/uninitialized FAISS index")

        faiss.write_index(faiss_index, str(binary_path))
        self._write_json(metadata_path, index.to_dict())
        return binary_path, metadata_path

    def load_faiss_vector_index(
        self,
        *,
        binary_filename: str,
        metadata_filename: str,
    ) -> VectorIndex:
        from app.indexing.faiss_index import FaissVectorIndex

        faiss = self._load_faiss_module()
        binary_path = self.base_dir / binary_filename
        metadata_path = self.base_dir / metadata_filename

        metadata_payload = self._read_json(metadata_path)
        index = FaissVectorIndex.from_dict(metadata_payload)
        restored_faiss_index = faiss.read_index(str(binary_path))
        setattr(index, "_faiss_index", restored_faiss_index)
        return index

    def save_bm25_index(
        self, index: BM25Index, filename: str = "bm25_index.json"
    ) -> Path:
        return self._write_json(self.base_dir / filename, index.to_dict())

    def load_bm25_index(self, filename: str = "bm25_index.json") -> BM25Index:
        payload = self._read_json(self.base_dir / filename)
        return BM25Index.from_dict(payload)
