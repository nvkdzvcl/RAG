"""Reranker interfaces and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from collections.abc import Sequence
from typing import Any, Protocol, cast

from app.schemas.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Reranker interface for post-retrieval ordering."""

    name: str

    @abstractmethod
    def rerank(
        self, query: str, docs: list[RetrievalResult], top_k: int | None = None
    ) -> list[RetrievalResult]:
        """Return reranked docs with updated scores."""


class PassThroughReranker(BaseReranker):
    """No-op reranker that preserves original retrieval order."""

    name = "pass-through-reranker"

    def __init__(self, *, reason: str | None = None) -> None:
        self.reason = reason

    def rerank(
        self, query: str, docs: list[RetrievalResult], top_k: int | None = None
    ) -> list[RetrievalResult]:
        _ = query
        if not docs:
            return []
        limit = len(docs) if top_k is None else max(0, top_k)
        if limit == 0:
            return []

        reranked: list[RetrievalResult] = []
        for rank, doc in enumerate(docs[:limit], start=1):
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "pre_rerank_score": doc.score,
                    "pre_rerank_score_type": doc.score_type,
                    "pre_rerank_rank": doc.rank,
                }
            )
            if self.reason:
                metadata["rerank_fallback_reason"] = self.reason
            reranked.append(
                doc.model_copy(
                    update={
                        "metadata": metadata,
                        "rank": rank,
                    }
                )
            )
        return reranked


class ScoreOnlyReranker(BaseReranker):
    """Deterministic reranker that orders candidates by retrieval score."""

    name = "score-only-reranker"

    @staticmethod
    def _to_reranked(doc: RetrievalResult, rerank_score: float) -> RetrievalResult:
        metadata = dict(doc.metadata)
        metadata.update(
            {
                "pre_rerank_score": doc.score,
                "pre_rerank_score_type": doc.score_type,
                "pre_rerank_rank": doc.rank,
            }
        )
        return doc.model_copy(
            update={
                "metadata": metadata,
                "score": float(rerank_score),
                "score_type": "rerank",
                "rerank_score": float(rerank_score),
            }
        )

    def rerank(
        self, query: str, docs: list[RetrievalResult], top_k: int | None = None
    ) -> list[RetrievalResult]:
        _ = query
        if not docs:
            return []
        limit = len(docs) if top_k is None else max(0, top_k)
        if limit == 0:
            return []

        candidates = list(docs[:limit])
        candidates.sort(key=lambda item: item.score, reverse=True)

        reranked: list[RetrievalResult] = []
        for rank, doc in enumerate(candidates, start=1):
            updated = self._to_reranked(doc, rerank_score=doc.score)
            updated.rank = rank
            reranked.append(updated)
        return reranked


class _CrossEncoderLike(Protocol):
    def predict(
        self,
        sentences: Sequence[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> Any:
        """Return raw cross-encoder scores."""


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder reranker using sentence-transformers."""

    name = "cross-encoder-reranker"

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        batch_size: int = 8,
        model: _CrossEncoderLike | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = model or self._load_model(model_name=model_name, device=device)
        self._fallback = PassThroughReranker(reason="runtime_failure")

    @staticmethod
    def _load_model(*, model_name: str, device: str) -> _CrossEncoderLike:
        try:
            from sentence_transformers import CrossEncoder
        except (
            ModuleNotFoundError
        ) as exc:  # pragma: no cover - covered via factory fallback tests.
            raise RuntimeError(
                "sentence-transformers is not installed. Install dependencies or use score-only reranking."
            ) from exc
        return cast(_CrossEncoderLike, CrossEncoder(model_name, device=device))

    @staticmethod
    def _normalize_scores(raw_scores: Any) -> list[float]:
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        if isinstance(raw_scores, (int, float)):
            return [float(raw_scores)]

        normalized: list[float] = []
        for value in raw_scores:
            if isinstance(value, (int, float)):
                normalized.append(float(value))
                continue
            if isinstance(value, Sequence) and value:
                normalized.append(float(value[-1]))
                continue
            normalized.append(0.0)
        return normalized

    def rerank(
        self, query: str, docs: list[RetrievalResult], top_k: int | None = None
    ) -> list[RetrievalResult]:
        if not docs:
            return []
        limit = len(docs) if top_k is None else max(0, top_k)
        if limit == 0:
            return []
        candidates = list(docs[:limit])

        try:
            pairs = [(query, doc.content) for doc in candidates]
            raw_scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            scores = self._normalize_scores(raw_scores)
            if len(scores) != len(candidates):
                raise ValueError("Cross-encoder returned mismatched score count.")
        except Exception as exc:
            logger.warning(
                (
                    "Cross-encoder reranking failed at runtime. "
                    "Falling back to pass-through retrieval order."
                ),
                exc_info=exc,
            )
            return self._fallback.rerank(query, docs, top_k=top_k)

        scored: list[tuple[float, RetrievalResult]] = []
        for doc, rerank_score in zip(candidates, scores):
            updated = ScoreOnlyReranker._to_reranked(
                doc, rerank_score=float(rerank_score)
            )
            scored.append((float(rerank_score), updated))

        scored.sort(key=lambda row: row[0], reverse=True)

        ranked: list[RetrievalResult] = []
        for rank, (_, item) in enumerate(scored[:limit], start=1):
            item.rank = rank
            ranked.append(item)
        return ranked


CROSS_ENCODER_PROVIDER_NAMES = {"cross_encoder", "cross-encoder", "crossencoder"}
SCORE_ONLY_PROVIDER_NAMES = {"score_only", "score-only", "noop", "no_op", "none"}


def create_reranker(
    *,
    provider_name: str,
    model: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cpu",
    batch_size: int = 8,
) -> BaseReranker:
    """Create reranker with safe fallback to pass-through behavior."""
    normalized = provider_name.strip().lower() if provider_name else ""

    if normalized in CROSS_ENCODER_PROVIDER_NAMES:
        try:
            return CrossEncoderReranker(
                model_name=model,
                device=device,
                batch_size=batch_size,
            )
        except Exception as exc:
            logger.warning(
                (
                    "Failed to initialize cross-encoder reranker model '%s' on device '%s'. "
                    "Falling back to pass-through retrieval order."
                ),
                model,
                device,
                exc_info=exc,
            )
            return PassThroughReranker(reason="load_failure")

    if normalized in SCORE_ONLY_PROVIDER_NAMES:
        return ScoreOnlyReranker()

    logger.warning(
        "Unknown reranker provider '%s'. Falling back to score-only reranker.",
        provider_name,
    )
    return ScoreOnlyReranker()
