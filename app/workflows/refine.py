"""Answer refinement step for advanced workflow."""

from __future__ import annotations

from app.schemas.retrieval import RetrievalResult
from app.schemas.workflow import CritiqueResult


class AnswerRefiner:
    """Refine draft answers using critique hints and context snippets."""

    def refine(
        self,
        query: str,
        draft_answer: str,
        critique: CritiqueResult,
        context: list[RetrievalResult],
    ) -> str:
        _ = query
        refined = draft_answer.strip()

        if critique.missing_aspects:
            refined += "\n\nAdditional coverage: " + ", ".join(critique.missing_aspects[:3]) + "."

        if context:
            lead_source = context[0]
            refined += (
                f"\n\nEvidence note: supported by {lead_source.title or lead_source.doc_id}"
                f" ({lead_source.chunk_id})."
            )

        return refined
