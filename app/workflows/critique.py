"""Critique module for advanced workflow decisions."""

from __future__ import annotations

import re

from app.schemas.retrieval import RetrievalResult
from app.schemas.workflow import CritiqueResult


class HeuristicCritic:
    """Heuristic critic producing structured critique output."""

    token_pattern = re.compile(r"\w+")

    def _terms(self, text: str) -> set[str]:
        return {token for token in self.token_pattern.findall(text.lower()) if len(token) > 2}

    def critique(
        self,
        query: str,
        draft_answer: str,
        context: list[RetrievalResult],
        *,
        loop_count: int,
        max_loops: int,
    ) -> CritiqueResult:
        query_terms = self._terms(query)
        answer_terms = self._terms(draft_answer)
        context_text = " ".join(item.content for item in context)
        context_terms = self._terms(context_text)

        force_retry = "force retry" in query.lower()
        force_abstain = "force abstain" in query.lower()

        enough_evidence = len(context) >= 1 and len(context_text.strip()) >= 40 and not force_abstain
        grounded = bool(answer_terms.intersection(context_terms)) and enough_evidence and not force_abstain

        missing_aspects = sorted(
            [
                term
                for term in query_terms
                if term in context_terms and term not in answer_terms
            ]
        )[:5]

        has_conflict = False
        if len({item.doc_id for item in context}) > 1:
            context_lower = context_text.lower()
            has_conflict = ("however" in context_lower and "therefore" in context_lower) or (
                "conflict" in context_lower
            )

        should_retry_retrieval = (
            force_retry
            or ((not enough_evidence or not grounded) and loop_count < max_loops)
        ) and not force_abstain

        should_refine_answer = bool(missing_aspects) and enough_evidence and grounded and not force_abstain

        better_queries: list[str] = []
        if should_retry_retrieval:
            better_queries.append(f"{query} supporting evidence")
            for aspect in missing_aspects[:2]:
                better_queries.append(f"{query} {aspect} details")

        if force_abstain:
            confidence = 0.0
            note = "Forced abstain requested by query signal."
        elif grounded and enough_evidence:
            confidence = 0.82 if not has_conflict else 0.65
            note = "Answer appears grounded in selected context."
        elif should_retry_retrieval:
            confidence = 0.35
            note = "Evidence is weak; retry retrieval recommended."
        else:
            confidence = 0.2
            note = "Evidence insufficient and no additional retries available."

        return CritiqueResult(
            grounded=grounded,
            enough_evidence=enough_evidence,
            has_conflict=has_conflict,
            missing_aspects=missing_aspects,
            should_retry_retrieval=should_retry_retrieval,
            should_refine_answer=should_refine_answer,
            better_queries=better_queries,
            confidence=confidence,
            note=note,
        )
