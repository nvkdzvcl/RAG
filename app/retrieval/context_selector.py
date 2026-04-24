"""Context selection utilities for generation-ready context windows."""

from __future__ import annotations

from app.schemas.retrieval import RetrievalResult


class ContextSelector:
    """Select top context chunks while enforcing lightweight bounds."""

    def __init__(self, max_chunks: int = 5, max_chars: int | None = 4000) -> None:
        self.max_chunks = max_chunks
        self.max_chars = max_chars

    def select(self, docs: list[RetrievalResult], top_k: int | None = None) -> list[RetrievalResult]:
        limit = top_k if top_k is not None else self.max_chunks
        if limit <= 0:
            return []

        selected: list[RetrievalResult] = []
        seen_ids: set[str] = set()
        total_chars = 0

        for doc in docs:
            if doc.chunk_id in seen_ids:
                continue

            candidate_chars = len(doc.content)
            if self.max_chars is not None and (total_chars + candidate_chars) > self.max_chars:
                continue

            selected.append(doc)
            seen_ids.add(doc.chunk_id)
            total_chars += candidate_chars

            if len(selected) >= limit:
                break

        return selected
