"""Tests for runtime index embedding provider integration."""

from __future__ import annotations

from pathlib import Path

from app.indexing import BaseEmbeddingProvider
from app.services.index_runtime import RuntimeIndexManager


class _CountingEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dimension: int = 6) -> None:
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


def test_runtime_index_build_uses_embedding_provider_interface(tmp_path: Path) -> None:
    corpus_dir = tmp_path / "corpus"
    index_dir = tmp_path / "indexes"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "vi.txt").write_text(
        "Tài liệu tiếng Việt mô tả hệ thống truy hồi thông tin cho câu hỏi hỗn hợp Việt-Anh.",
        encoding="utf-8",
    )

    provider = _CountingEmbeddingProvider(dimension=6)
    manager = RuntimeIndexManager(
        corpus_dir=corpus_dir,
        index_dir=index_dir,
        embedding_provider=provider,
    )

    chunk_count = manager.activate_from_seeded_corpus()
    results = manager.get_retriever().retrieve("Truy hoi thong tin la gi?", top_k=3)

    assert chunk_count > 0
    assert manager.get_active_source() == "seeded"
    assert manager.get_active_chunk_count() == chunk_count
    assert provider.document_inputs
    assert provider.query_inputs
    assert results
    assert (index_dir / "vector_index.json").exists()
    assert (index_dir / "bm25_index.json").exists()


def test_runtime_index_manager_uses_configured_provider_factory(monkeypatch, tmp_path: Path) -> None:
    class _Settings:
        embedding_provider = "hash"
        embedding_model = "intfloat/multilingual-e5-base"
        embedding_device = "cpu"
        embedding_batch_size = 32
        embedding_normalize = False
        embedding_hash_dimension = 79

    captured: dict[str, object] = {}
    configured_provider = _CountingEmbeddingProvider(dimension=79)

    def _fake_factory(
        *,
        provider_name: str,
        model: str,
        device: str,
        batch_size: int,
        normalize: bool,
        fallback_hash_dimension: int,
    ) -> BaseEmbeddingProvider:
        captured.update(
            {
                "provider_name": provider_name,
                "model": model,
                "device": device,
                "batch_size": batch_size,
                "normalize": normalize,
                "fallback_hash_dimension": fallback_hash_dimension,
            }
        )
        return configured_provider

    monkeypatch.setattr("app.services.index_runtime.get_settings", lambda: _Settings())
    monkeypatch.setattr("app.services.index_runtime.create_embedding_provider", _fake_factory)

    manager = RuntimeIndexManager(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "indexes",
    )

    assert manager.embedding_provider is configured_provider
    assert captured == {
        "provider_name": "hash",
        "model": "intfloat/multilingual-e5-base",
        "device": "cpu",
        "batch_size": 32,
        "normalize": False,
        "fallback_hash_dimension": 79,
    }


def test_runtime_index_manager_uses_sentence_transformers_when_configured(monkeypatch, tmp_path: Path) -> None:
    class _Settings:
        embedding_provider = "sentence_transformers"
        embedding_model = "intfloat/multilingual-e5-base"
        embedding_device = "cpu"
        embedding_batch_size = 16
        embedding_normalize = True
        embedding_hash_dimension = 64

    captured: dict[str, object] = {}
    configured_provider = _CountingEmbeddingProvider(dimension=12)

    def _fake_factory(
        *,
        provider_name: str,
        model: str,
        device: str,
        batch_size: int,
        normalize: bool,
        fallback_hash_dimension: int,
    ) -> BaseEmbeddingProvider:
        captured.update(
            {
                "provider_name": provider_name,
                "model": model,
                "device": device,
                "batch_size": batch_size,
                "normalize": normalize,
                "fallback_hash_dimension": fallback_hash_dimension,
            }
        )
        return configured_provider

    monkeypatch.setattr("app.services.index_runtime.get_settings", lambda: _Settings())
    monkeypatch.setattr("app.services.index_runtime.create_embedding_provider", _fake_factory)

    manager = RuntimeIndexManager(
        corpus_dir=tmp_path / "corpus",
        index_dir=tmp_path / "indexes",
    )

    assert manager.embedding_provider is configured_provider
    assert captured["provider_name"] == "sentence_transformers"
    assert captured["model"] == "intfloat/multilingual-e5-base"
