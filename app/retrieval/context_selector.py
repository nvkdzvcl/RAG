"""Context selection utilities for generation-ready context windows."""

from __future__ import annotations

import re

from app.schemas.retrieval import RetrievalResult

_SENTENCE_BOUNDARY = re.compile(r"[.!?。]\s+")


def _truncate_at_boundary(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars*, preferring a sentence or word boundary."""
    if len(text) <= max_chars:
        return text

    candidate = text[:max_chars]

    # Try to cut at the last sentence boundary.
    last_sentence = -1
    for match in _SENTENCE_BOUNDARY.finditer(candidate):
        last_sentence = match.end()
    if last_sentence > max_chars // 3:
        return candidate[:last_sentence].rstrip()

    # Fall back to the last whitespace boundary.
    last_space = candidate.rfind(" ")
    if last_space > max_chars // 3:
        return candidate[:last_space].rstrip()

    # No good boundary — hard cut.
    return candidate.rstrip()


class ContextSelector:
    """Select top context chunks while enforcing lightweight bounds.

    When a chunk would push the running total past *max_chars*, the selector
    truncates it at a sentence / word boundary instead of discarding it, as
    long as the remaining budget is at least *min_useful_chars*.
    """

    def __init__(
        self,
        max_chunks: int = 5,
        max_chars: int | None = 4000,
        min_useful_chars: int = 80,
    ) -> None:
        self.max_chunks = max_chunks
        self.max_chars = max_chars
        self.min_useful_chars = max(1, min_useful_chars)

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

            # --- chunk fits entirely ---
            if self.max_chars is None or (total_chars + candidate_chars) <= self.max_chars:
                selected.append(doc)
                seen_ids.add(doc.chunk_id)
                total_chars += candidate_chars
                if len(selected) >= limit:
                    break
                continue

            # --- budget exhausted: try to include a truncated version ---
            remaining = self.max_chars - total_chars
            if remaining < self.min_useful_chars:
                break  # not enough room for any useful content

            truncated_content = _truncate_at_boundary(doc.content, remaining)
            if len(truncated_content) < self.min_useful_chars:
                break

            truncated_metadata = dict(doc.metadata)
            truncated_metadata["truncated"] = True
            truncated_metadata["original_chars"] = candidate_chars

            truncated_doc = doc.model_copy(
                update={
                    "content": truncated_content,
                    "metadata": truncated_metadata,
                },
            )
            selected.append(truncated_doc)
            seen_ids.add(doc.chunk_id)
            total_chars += len(truncated_content)

            break  # budget consumed after truncation

        return selected
