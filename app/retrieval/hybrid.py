"""Hybrid retrieval by fusing dense and sparse rankings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import time
from typing import Any

from app.core.async_utils import run_coro_sync
from app.core.cache import QueryCache, make_cache_key
from app.core.timing import normalize_timing_payload
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.schemas.retrieval import RetrievalBatch, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """Configuration for reciprocal-rank-fusion."""

    rrf_k: int = 60
    dense_weight: float = 0.5
    sparse_weight: float = 0.5
    candidate_multiplier: int = 3
    min_candidates_per_retriever: int = 20


@dataclass
class _FusionAccumulator:
    """Internal accumulator used while fusing ranked retriever lists."""

    representative: RetrievalResult
    best_rank: int
    score: float = 0.0
    dense_score: float | None = None
    sparse_score: float | None = None


def _dedupe_key(item: RetrievalResult) -> str:
    """Return a stable key for document-level deduplication."""
    doc_id = str(item.doc_id).strip()
    if doc_id:
        return f"doc:{doc_id}"
    return f"chunk:{item.chunk_id}"


def _deduplicate_ranked_results(
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Drop duplicate documents while preserving ranked-list order."""
    deduped: list[RetrievalResult] = []
    seen: set[str] = set()
    for item in results:
        key = _dedupe_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _debug_results(results: list[RetrievalResult]) -> list[dict[str, object]]:
    """Serialize retrieval results into concise debug-friendly payloads."""
    payload: list[dict[str, object]] = []
    for item in results:
        payload.append(
            {
                "rank": item.rank,
                "chunk_id": item.chunk_id,
                "doc_id": item.doc_id,
                "score": round(float(item.score), 6),
                "dense_score": None
                if item.dense_score is None
                else round(float(item.dense_score), 6),
                "sparse_score": None
                if item.sparse_score is None
                else round(float(item.sparse_score), 6),
                "score_type": item.score_type,
            }
        )
    return payload


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
    dense_unique = _deduplicate_ranked_results(dense_results)
    sparse_unique = _deduplicate_ranked_results(sparse_results)

    fused: dict[str, _FusionAccumulator] = {}

    for rank, item in enumerate(dense_unique, start=1):
        key = _dedupe_key(item)
        score = cfg.dense_weight / (cfg.rrf_k + rank)
        if key not in fused:
            fused[key] = _FusionAccumulator(
                representative=item,
                best_rank=rank,
            )
        accumulator = fused[key]
        accumulator.score += score
        accumulator.dense_score = item.score
        if rank < accumulator.best_rank:
            accumulator.representative = item
            accumulator.best_rank = rank

    for rank, item in enumerate(sparse_unique, start=1):
        key = _dedupe_key(item)
        score = cfg.sparse_weight / (cfg.rrf_k + rank)
        if key not in fused:
            fused[key] = _FusionAccumulator(
                representative=item,
                best_rank=rank,
            )
        accumulator = fused[key]
        accumulator.score += score
        accumulator.sparse_score = item.score
        if rank < accumulator.best_rank:
            accumulator.representative = item
            accumulator.best_rank = rank

    ranked = [
        accumulator.representative.model_copy(
            update={
                "score": accumulator.score,
                "score_type": "hybrid",
                "dense_score": accumulator.dense_score,
                "sparse_score": accumulator.sparse_score,
            }
        )
        for accumulator in fused.values()
    ]
    ranked.sort(key=lambda row: (-row.score, row.chunk_id))
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
        retrieval_cache: QueryCache | None = None,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.fusion_config = fusion_config or FusionConfig()
        self._retrieval_cache = retrieval_cache
        self._last_timing: dict[str, int] = {
            "retrieval_total_ms": 0,
            "dense_retrieve_ms": 0,
            "sparse_retrieve_ms": 0,
            "hybrid_merge_ms": 0,
        }
        self._last_cache_debug: dict[str, bool] = {
            "embedding_cache_hit": False,
            "retrieval_cache_hit": False,
        }

    def get_last_timing(self) -> dict[str, int]:
        """Return timing for the most recent retrieval call.

        This legacy diagnostic is shared by all calls on this retriever and is
        not request-safe under concurrency. New callers should use
        ``retrieve_with_timing``/``retrieve_with_timing_async``.
        """
        return dict(self._last_timing)

    def _resolve_candidate_k(self, top_k: int) -> int:
        multiplier = max(1, int(self.fusion_config.candidate_multiplier))
        minimum = max(1, int(self.fusion_config.min_candidates_per_retriever))
        return max(top_k, top_k * multiplier, minimum)

    @staticmethod
    def _clone_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
        return [item.model_copy(deep=True) for item in results]

    @staticmethod
    def _normalize_cached_results(raw: Any) -> list[RetrievalResult] | None:
        if not isinstance(raw, list):
            return None
        normalized: list[RetrievalResult] = []
        for item in raw:
            if isinstance(item, RetrievalResult):
                normalized.append(item.model_copy(deep=True))
                continue
            if isinstance(item, dict):
                try:
                    normalized.append(RetrievalResult.model_validate(item))
                except Exception:
                    return None
                continue
            return None
        return normalized

    def _cache_key(self, query: str, *, top_k: int) -> str:
        dense_revision = int(getattr(self.dense_retriever.vector_index, "revision", 0))
        return make_cache_key(
            query,
            top_k,
            self.fusion_config.rrf_k,
            self.fusion_config.dense_weight,
            self.fusion_config.sparse_weight,
            self.fusion_config.candidate_multiplier,
            self.fusion_config.min_candidates_per_retriever,
            dense_revision,
        )

    def get_last_cache_debug(self) -> dict[str, bool]:
        return dict(self._last_cache_debug)

    async def retrieve_with_timing_async(
        self, query: str, top_k: int = 5
    ) -> RetrievalBatch:
        if top_k <= 0:
            timing = {
                "retrieval_total_ms": 0,
                "dense_retrieve_ms": 0,
                "sparse_retrieve_ms": 0,
                "hybrid_merge_ms": 0,
            }
            self._last_timing = timing
            self._last_cache_debug = {
                "embedding_cache_hit": False,
                "retrieval_cache_hit": False,
            }
            return RetrievalBatch(
                results=[],
                timings_ms=timing,
                timing_breakdown_available=True,
                cache_debug=self._last_cache_debug,
            )

        total_started = time.perf_counter()

        # Check retrieval cache.
        cache_key: str | None = None
        if self._retrieval_cache is not None:
            cache_key = self._cache_key(query, top_k=top_k)
            hit, cached = self._retrieval_cache.get(cache_key)
            if hit:
                normalized_cached = self._normalize_cached_results(cached)
                if normalized_cached is None:
                    self._retrieval_cache.invalidate()
                else:
                    self._last_cache_debug = {
                        "embedding_cache_hit": False,
                        "retrieval_cache_hit": True,
                    }
                    timing = {
                        "retrieval_total_ms": int(
                            (time.perf_counter() - total_started) * 1000
                        ),
                        "dense_retrieve_ms": 0,
                        "sparse_retrieve_ms": 0,
                        "hybrid_merge_ms": 0,
                    }
                    self._last_timing = timing
                    return RetrievalBatch(
                        results=normalized_cached,
                        timings_ms=timing,
                        timing_breakdown_available=True,
                        cache_debug=self._last_cache_debug,
                    )
                timing = {
                    "retrieval_total_ms": int(
                        (time.perf_counter() - total_started) * 1000
                    ),
                    "dense_retrieve_ms": 0,
                    "sparse_retrieve_ms": 0,
                    "hybrid_merge_ms": 0,
                }
                self._last_timing = timing

        candidate_k = self._resolve_candidate_k(top_k)
        dense_ms = 0
        sparse_ms = 0

        def _timed_dense() -> tuple[list[RetrievalResult], int]:
            started = time.perf_counter()
            result = self.dense_retriever.retrieve(query, candidate_k)
            return result, int((time.perf_counter() - started) * 1000)

        def _timed_sparse() -> tuple[list[RetrievalResult], int]:
            started = time.perf_counter()
            result = self.sparse_retriever.retrieve(query, candidate_k)
            return result, int((time.perf_counter() - started) * 1000)

        dense_task = asyncio.to_thread(_timed_dense)
        sparse_task = asyncio.to_thread(_timed_sparse)
        dense_payload, sparse_payload = await asyncio.gather(dense_task, sparse_task)
        dense, dense_ms = dense_payload
        sparse, sparse_ms = sparse_payload
        logger.debug(
            "Hybrid dense results | query=%s | candidate_k=%s | results=%s",
            query,
            candidate_k,
            _debug_results(dense),
        )
        logger.debug(
            "Hybrid bm25 results | query=%s | candidate_k=%s | results=%s",
            query,
            candidate_k,
            _debug_results(sparse),
        )
        merge_started = time.perf_counter()
        merged = reciprocal_rank_fusion(
            dense,
            sparse,
            top_k=top_k,
            config=self.fusion_config,
        )
        merge_ms = int((time.perf_counter() - merge_started) * 1000)
        logger.debug(
            "Hybrid merged results | query=%s | top_k=%s | results=%s",
            query,
            top_k,
            _debug_results(merged),
        )
        timing = {
            "retrieval_total_ms": int((time.perf_counter() - total_started) * 1000),
            "dense_retrieve_ms": dense_ms,
            "sparse_retrieve_ms": sparse_ms,
            "hybrid_merge_ms": merge_ms,
        }
        self._last_timing = normalize_timing_payload(timing)
        dense_cache_getter = getattr(self.dense_retriever, "get_last_cache_debug", None)
        embedding_cache_hit = False
        if callable(dense_cache_getter):
            payload = dense_cache_getter()
            if isinstance(payload, dict):
                embedding_cache_hit = bool(payload.get("embedding_cache_hit", False))
        self._last_cache_debug = {
            "embedding_cache_hit": embedding_cache_hit,
            "retrieval_cache_hit": False,
        }

        # Store in retrieval cache.
        if self._retrieval_cache is not None and cache_key is not None:
            self._retrieval_cache.put(cache_key, self._clone_results(merged))

        return RetrievalBatch(
            results=merged,
            timings_ms=self._last_timing,
            timing_breakdown_available=True,
            cache_debug=self._last_cache_debug,
        )

    async def retrieve_async(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return (await self.retrieve_with_timing_async(query, top_k=top_k)).results

    def retrieve_with_timing(self, query: str, top_k: int = 5) -> RetrievalBatch:
        """Sync compatibility wrapper returning request-scoped timing."""
        return run_coro_sync(self.retrieve_with_timing_async(query, top_k=top_k))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Sync compatibility wrapper."""
        return self.retrieve_with_timing(query, top_k=top_k).results
