"""Tests for sentence-transformer embedding provider behavior."""

from __future__ import annotations

from app.indexing.providers.sentence_transformer_embedding import (
    SentenceTransformerEmbeddingProvider,
)


class _FakeSentenceTransformerModel:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls: list[dict[str, object]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "sentences": list(sentences),
                "batch_size": batch_size,
                "convert_to_numpy": convert_to_numpy,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
            }
        )
        return [[float(idx + 1)] * self.dimension for idx, _ in enumerate(sentences)]


def test_e5_prefix_behavior_for_documents_and_queries() -> None:
    model = _FakeSentenceTransformerModel(dimension=5)
    provider = SentenceTransformerEmbeddingProvider(
        model_name="intfloat/multilingual-e5-base",
        device="cpu",
        batch_size=16,
        normalize=True,
        model=model,
    )

    doc_vectors = provider.embed_documents(["Xin chao Viet Nam"])
    query_vector = provider.embed_query("Viet Nam nam o dau?")

    assert provider.dimension == 5
    assert len(doc_vectors) == 1
    assert len(doc_vectors[0]) == 5
    assert len(query_vector) == 5

    assert model.calls[0]["sentences"] == ["passage: Xin chao Viet Nam"]
    assert model.calls[1]["sentences"] == ["query: Viet Nam nam o dau?"]


def test_vietnamese_text_embedding_preserves_diacritics() -> None:
    model = _FakeSentenceTransformerModel(dimension=3)
    provider = SentenceTransformerEmbeddingProvider(
        model_name="intfloat/multilingual-e5-base",
        device="cpu",
        batch_size=8,
        normalize=True,
        model=model,
    )

    text = "Tiếng Việt có dấu: Trường đại học Bách Khoa Hà Nội"
    vectors = provider.embed_documents([text])

    assert len(vectors) == 1
    assert len(vectors[0]) == 3

    encoded_text = model.calls[0]["sentences"][0]
    assert isinstance(encoded_text, str)
    assert "Tiếng Việt có dấu" in encoded_text
    assert encoded_text.startswith("passage: ")
