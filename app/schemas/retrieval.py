"""Schemas for retrieval and reranking outputs."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.ingestion import DocumentChunk


class RetrievalResult(BaseModel):
    """Unified result object used across retrievers and rerankers."""

    chunk_id: str
    doc_id: str
    source: str
    title: str | None = None
    section: str | None = None
    page: int | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    score: float
    score_type: Literal["dense", "sparse", "hybrid", "rerank"]

    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    rank: int | None = None

    @classmethod
    def from_chunk(
        cls,
        chunk: DocumentChunk,
        *,
        score: float,
        score_type: Literal["dense", "sparse", "hybrid", "rerank"],
        dense_score: float | None = None,
        sparse_score: float | None = None,
        rerank_score: float | None = None,
        rank: int | None = None,
    ) -> "RetrievalResult":
        """Build a retrieval result from a chunk and scoring fields."""
        return cls(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            source=chunk.source,
            title=chunk.title,
            section=chunk.section,
            page=chunk.page,
            content=chunk.content,
            metadata=dict(chunk.metadata),
            score=score,
            score_type=score_type,
            dense_score=dense_score,
            sparse_score=sparse_score,
            rerank_score=rerank_score,
            rank=rank,
        )


@dataclass(frozen=True)
class RetrievalBatch:
    """Request-scoped retrieval results plus diagnostics.

    This avoids requiring callers to read mutable retriever-level
    ``last_timing``/``last_filter_debug`` fields after a request, which is
    unsafe when a shared retriever handles concurrent calls.
    """

    results: list[RetrievalResult]
    timings_ms: dict[str, int] = dataclass_field(default_factory=dict)
    timing_breakdown_available: bool = False
    filter_debug: dict[str, Any] = dataclass_field(default_factory=dict)
