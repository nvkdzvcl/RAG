"""Tests for embedding provider selection and fallback behavior."""

from __future__ import annotations

from app.indexing import BaseEmbeddingProvider, HashEmbeddingProvider
from app.indexing.providers.factory import create_embedding_provider


class _FakeSentenceTransformerProvider(BaseEmbeddingProvider):
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        batch_size: int,
        normalize: bool,
    ) -> None:
        self.name = "fake-st"
        self.dimension = 3
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        _ = text
        return [1.0, 0.0, 0.0]


def test_provider_selection_from_config(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.indexing.providers.factory.SentenceTransformerEmbeddingProvider",
        _FakeSentenceTransformerProvider,
    )

    provider = create_embedding_provider(
        provider_name="sentence_transformers",
        model="intfloat/multilingual-e5-base",
        device="cpu",
        batch_size=16,
        normalize=True,
        fallback_hash_dimension=32,
    )

    assert isinstance(provider, _FakeSentenceTransformerProvider)
    assert provider.model_name == "intfloat/multilingual-e5-base"
    assert provider.device == "cpu"
    assert provider.batch_size == 16
    assert provider.normalize is True


def test_sentence_transformer_fallback_to_hash_provider(monkeypatch) -> None:
    class _BrokenSentenceTransformerProvider:
        def __init__(self, **_: object) -> None:
            raise RuntimeError("Model init failed")

    monkeypatch.setattr(
        "app.indexing.providers.factory.SentenceTransformerEmbeddingProvider",
        _BrokenSentenceTransformerProvider,
    )

    provider = create_embedding_provider(
        provider_name="sentence_transformers",
        model="intfloat/multilingual-e5-base",
        device="cpu",
        batch_size=16,
        normalize=True,
        fallback_hash_dimension=77,
    )

    assert isinstance(provider, HashEmbeddingProvider)
    assert provider.dimension == 77


def test_unknown_provider_falls_back_to_hash_provider() -> None:
    provider = create_embedding_provider(
        provider_name="unknown-provider",
        model="unused",
        device="cpu",
        batch_size=4,
        normalize=False,
        fallback_hash_dimension=19,
    )

    assert isinstance(provider, HashEmbeddingProvider)
    assert provider.dimension == 19
