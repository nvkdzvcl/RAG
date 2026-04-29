"""Deterministic tests for cascade reranking policy."""

from app.schemas.retrieval import RetrievalResult
from app.workflows.rerank_policy import choose_rerank_policy


def _candidates(scores: list[float]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=f"c_{idx}",
            doc_id=f"d_{idx}",
            source="seeded://policy",
            content=f"Candidate {idx}",
            score=score,
            score_type="hybrid",
            dense_score=score - 0.05,
            sparse_score=score - 0.1,
            rank=idx + 1,
        )
        for idx, score in enumerate(scores)
    ]


def test_rerank_policy_skips_cross_encoder_for_simple_extractive() -> None:
    policy = choose_rerank_policy(
        mode="standard",
        query_complexity="simple_extractive",
        candidates=_candidates([0.91, 0.83, 0.72, 0.66, 0.61]),
        cascade_enabled=True,
        simple_skip_cross_encoder=True,
        min_candidates_for_cross_encoder=4,
        score_gap_threshold=0.2,
        top_score_threshold=0.75,
        reranker_supports_cross_encoder=True,
    )

    assert policy.use_cross_encoder is False
    assert policy.reranker_used == "score_only"
    assert policy.reason == "simple_extractive_skip"


def test_rerank_policy_uses_cross_encoder_for_complex_ambiguous_scores() -> None:
    policy = choose_rerank_policy(
        mode="standard",
        query_complexity="complex",
        candidates=_candidates([0.56, 0.54, 0.53, 0.52, 0.51]),
        cascade_enabled=True,
        simple_skip_cross_encoder=True,
        min_candidates_for_cross_encoder=4,
        score_gap_threshold=0.2,
        top_score_threshold=0.75,
        reranker_supports_cross_encoder=True,
    )

    assert policy.use_cross_encoder is True
    assert policy.reranker_used == "cross_encoder"
    assert policy.reason == "ambiguous_scores_use_cross_encoder"


def test_rerank_policy_cascade_disabled_forces_legacy_cross_encoder_path() -> None:
    policy = choose_rerank_policy(
        mode="standard",
        query_complexity="simple_extractive",
        candidates=_candidates([0.88, 0.78, 0.69, 0.58, 0.49]),
        cascade_enabled=False,
        simple_skip_cross_encoder=True,
        min_candidates_for_cross_encoder=4,
        score_gap_threshold=0.2,
        top_score_threshold=0.75,
        reranker_supports_cross_encoder=True,
    )

    assert policy.use_cross_encoder is True
    assert policy.reason == "cascade_disabled_legacy"
