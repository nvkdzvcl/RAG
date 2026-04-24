"""Local persistence for indexing artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from app.indexing.bm25_index import BM25Index
from app.indexing.vector_index import InMemoryVectorIndex, VectorIndex


class LocalIndexStore:
    """Store and load local vector/BM25 indexes as JSON files."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: dict) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return path

    def _read_json(self, path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    def save_vector_index(self, index: VectorIndex, filename: str = "vector_index.json") -> Path:
        return self._write_json(self.base_dir / filename, index.to_dict())

    def load_vector_index(self, filename: str = "vector_index.json") -> InMemoryVectorIndex:
        payload = self._read_json(self.base_dir / filename)
        return InMemoryVectorIndex.from_dict(payload)

    def save_bm25_index(self, index: BM25Index, filename: str = "bm25_index.json") -> Path:
        return self._write_json(self.base_dir / filename, index.to_dict())

    def load_bm25_index(self, filename: str = "bm25_index.json") -> BM25Index:
        payload = self._read_json(self.base_dir / filename)
        return BM25Index.from_dict(payload)
