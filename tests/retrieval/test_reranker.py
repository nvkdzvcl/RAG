"""Tests for production reranker implementations and fallback behavior."""

from __future__ import annotations

from typing import Any

from app.retrieval import CrossEncoderReranker, ScoreOnlyReranker, create_reranker
from app.schemas.retrieval import RetrievalResult


class _FakeCrossEncoderModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[dict[str, Any]] = []

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
    ) -> list[float]:
        self.calls.append(
            {
                "sentences": list(sentences),
                "batch_size": batch_size,
                "show_progress_bar": show_progress_bar,
            }
        )
        return self.scores[: len(sentences)]


def _sample_docs() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            source="s1",
            content="alpha overview",
            score=0.6,
            score_type="hybrid",
            dense_score=0.7,
            sparse_score=0.4,
            rank=1,
        ),
        RetrievalResult(
            chunk_id="c2",
            doc_id="d2",
            source="s2",
            content="beta deep dive",
            score=0.5,
            score_type="hybrid",
            dense_score=0.4,
            sparse_score=0.6,
            rank=2,
        ),
        RetrievalResult(
            chunk_id="c3",
            doc_id="d3",
            source="s3",
            content="gamma details",
            score=0.4,
            score_type="hybrid",
            dense_score=0.3,
            sparse_score=0.5,
            rank=3,
        ),
    ]


def test_cross_encoder_reranker_changes_order_when_scores_differ() -> None:
    model = _FakeCrossEncoderModel(scores=[0.2, 0.95, 0.1])
    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cpu",
        batch_size=8,
        model=model,
    )

    reranked = reranker.rerank("beta", _sample_docs(), top_k=3)

    assert [item.chunk_id for item in reranked] == ["c2", "c1", "c3"]
    assert all(item.rerank_score is not None for item in reranked)
    assert reranked[0].metadata["pre_rerank_score"] == 0.5
    assert reranked[0].metadata["pre_rerank_score_type"] == "hybrid"


def test_cross_encoder_only_reranks_top_k_candidates() -> None:
    model = _FakeCrossEncoderModel(scores=[0.01, 0.99, 0.5])
    reranker = CrossEncoderReranker(
        model_name="BAAI/bge-reranker-v2-m3",
        device="cpu",
        batch_size=4,
        model=model,
    )

    reranked = reranker.rerank("query", _sample_docs(), top_k=2)

    assert len(reranked) == 2
    assert len(model.calls) == 1
    assert len(model.calls[0]["sentences"]) == 2
    assert reranked[0].chunk_id == "c2"


def test_score_only_reranker_fallback_factory(monkeypatch) -> None:
    class _BrokenCrossEncoderReranker:
        def __init__(self, **_: object) -> None:
            raise RuntimeError("cross encoder unavailable")

    monkeypatch.setattr(
        "app.retrieval.reranker.CrossEncoderReranker",
        _BrokenCrossEncoderReranker,
    )

    reranker = create_reranker(
        provider_name="cross_encoder",
        model="BAAI/bge-reranker-v2-m3",
        device="cpu",
        batch_size=8,
    )

    assert isinstance(reranker, ScoreOnlyReranker)
