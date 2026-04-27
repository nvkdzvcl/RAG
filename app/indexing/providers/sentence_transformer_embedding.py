"""Sentence-transformers embedding provider with multilingual E5 defaults."""

from __future__ import annotations

from typing import Any, Protocol, cast

from app.indexing.embeddings import BaseEmbeddingProvider


class _SentenceTransformerLike(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any:
        """Encode sentences into vectors."""

    def get_sentence_embedding_dimension(self) -> int | None:
        """Return embedding dimension when available."""


class SentenceTransformerEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider backed by sentence-transformers models."""

    def __init__(
        self,
        *,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
        model: _SentenceTransformerLike | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.name = "sentence-transformers"
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self._model = model or self._load_model(model_name=model_name, device=device)
        self.dimension = self._resolve_dimension()

    @staticmethod
    def _load_model(*, model_name: str, device: str) -> _SentenceTransformerLike:
        try:
            from sentence_transformers import SentenceTransformer
        except (
            ModuleNotFoundError
        ) as exc:  # pragma: no cover - tested via provider factory fallback.
            raise RuntimeError(
                "sentence-transformers is not installed. Install dependencies or use hash embeddings."
            ) from exc

        return cast(
            _SentenceTransformerLike, SentenceTransformer(model_name, device=device)
        )

    def _resolve_dimension(self) -> int:
        getter = getattr(self._model, "get_embedding_dimension", None)
        if getter is None:
            getter = getattr(self._model, "get_sentence_embedding_dimension", None)

        if callable(getter):
            dimension = getter()
            if dimension is not None and int(dimension) > 0:
                return int(dimension)

        probe_vectors = self._encode_batch([self._format_passage("dimension probe")])
        if not probe_vectors or not probe_vectors[0]:
            raise ValueError(
                "Could not infer embedding dimension from sentence-transformers model."
            )
        return len(probe_vectors[0])

    @staticmethod
    def _format_passage(text: str) -> str:
        return f"passage: {text}"

    @staticmethod
    def _format_query(text: str) -> str:
        return f"query: {text}"

    @staticmethod
    def _normalize_vectors(raw: Any) -> list[list[float]]:
        if hasattr(raw, "tolist"):
            raw = raw.tolist()

        if not isinstance(raw, list):
            raw = [list(row) for row in raw]

        if raw and raw[0] and isinstance(raw[0], (float, int)):
            raw = [raw]

        vectors: list[list[float]] = []
        for row in raw:
            vectors.append([float(value) for value in row])
        return vectors

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        raw_vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )
        return self._normalize_vectors(raw_vectors)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode_batch([self._format_passage(text) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        vectors = self._encode_batch([self._format_query(text)])
        return vectors[0] if vectors else []
