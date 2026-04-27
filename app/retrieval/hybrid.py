"""Hybrid retrieval by fusing dense and sparse rankings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.core.async_utils import run_coro_sync
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.schemas.retrieval import RetrievalResult


@dataclass
class FusionConfig:
    """Configuration for reciprocal-rank-fusion."""

    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    candidate_multiplier: int = 3
    min_candidates_per_retriever: int = 20


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    sparse_results: list[RetrievalResult],
    *,
    top_k: int,
    config: FusionConfig | None = None,
) -> list[RetrievalResult]:
    """Merge two ranked lists with weighted reciprocal-rank fusion."""
    if top_k <= 0:
        return []

    cfg = config or FusionConfig()

    fused: dict[str, RetrievalResult] = {}

    for rank, item in enumerate(dense_results, start=1):
        score = cfg.dense_weight / (cfg.rrf_k + rank)
        if item.chunk_id not in fused:
            fused[item.chunk_id] = item.model_copy(update={"score": 0.0, "score_type": "hybrid"})
        fused[item.chunk_id].score += score
        fused[item.chunk_id].dense_score = item.score

    for rank, item in enumerate(sparse_results, start=1):
        score = cfg.sparse_weight / (cfg.rrf_k + rank)
        if item.chunk_id not in fused:
            fused[item.chunk_id] = item.model_copy(update={"score": 0.0, "score_type": "hybrid"})
        fused[item.chunk_id].score += score
        fused[item.chunk_id].sparse_score = item.score

    ranked = sorted(fused.values(), key=lambda row: row.score, reverse=True)
    for idx, item in enumerate(ranked[:top_k], start=1):
        item.rank = idx
        item.score_type = "hybrid"

    return ranked[:top_k]


class HybridRetriever:
    """Hybrid retriever combining dense and sparse retrievers."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        *,
        fusion_config: FusionConfig | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_config = fusion_config or FusionConfig()

    def _resolve_candidate_k(self, top_k: int) -> int:
        multiplier = max(1, int(self.fusion_config.candidate_multiplier))
        minimum = max(1, int(self.fusion_config.min_candidates_per_retriever))
        return max(top_k, top_k * multiplier, minimum)

    async def retrieve_async(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []
        candidate_k = self._resolve_candidate_k(top_k)
        dense_task = asyncio.to_thread(
            self.dense_retriever.retrieve,
            query,
            candidate_k,
        )
        sparse_task = asyncio.to_thread(
            self.sparse_retriever.retrieve,
            query,
            candidate_k,
        )
        dense, sparse = await asyncio.gather(dense_task, sparse_task)
        return reciprocal_rank_fusion(
            dense,
            sparse,
            top_k=top_k,
            config=self.fusion_config,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Sync compatibility wrapper."""
        return run_coro_sync(self.retrieve_async(query, top_k=top_k))
