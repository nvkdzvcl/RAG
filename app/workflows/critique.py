"""Critique module for advanced workflow decisions."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.core.config import get_settings
from app.core.json_utils import parse_json_object
from app.core.prompting import PromptRepository
from app.generation.llm_client import LLMClient
from app.schemas.retrieval import RetrievalResult
from app.schemas.workflow import CritiqueResult

_CRITIQUE_PROMPT_FALLBACK = (
    "Critique the draft answer against selected context.\\n"
    "Return strict JSON only.\\n"
    "Schema:\\n"
    "{\\n"
    '  "grounded": bool,\\n'
    '  "enough_evidence": bool,\\n'
    '  "has_conflict": bool,\\n'
    '  "missing_aspects": [string],\\n'
    '  "should_retry_retrieval": bool,\\n'
    '  "should_refine_answer": bool,\\n'
    '  "better_queries": [string],\\n'
    '  "confidence": float,\\n'
    '  "note": string\\n'
    "}\\n"
    "Keep language compatible with the question (Vietnamese/English).\\n"
    "question: $question\\n"
    "draft_answer: $draft_answer\\n"
    "selected_context: $selected_context\\n"
    "loop_count: $loop_count / $max_loops"
)


class HeuristicCritic:
    """Heuristic critic producing structured critique output."""

    token_pattern = re.compile(r"\w+")

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        prompt_repository: PromptRepository | None = None,
        prompt_dir: str | Path | None = None,
        use_llm: bool = True,
    ) -> None:
        settings = get_settings()
        self.llm_client = llm_client
        self.use_llm = use_llm
        resolved_prompt_dir = prompt_dir or settings.prompt_dir
        self.prompt_repository = prompt_repository or PromptRepository(resolved_prompt_dir)

    def _terms(self, text: str) -> set[str]:
        return {token for token in self.token_pattern.findall(text.lower()) if len(token) > 2}

    @staticmethod
    def _to_bool(value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        return default

    @staticmethod
    def _to_list_of_strings(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result

    @staticmethod
    def _to_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _heuristic_critique(
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

    def _llm_critique(
        self,
        query: str,
        draft_answer: str,
        context: list[RetrievalResult],
        *,
        loop_count: int,
        max_loops: int,
        fallback: CritiqueResult,
    ) -> CritiqueResult | None:
        if not self.use_llm or self.llm_client is None:
            return None

        context_payload = [
            {
                "chunk_id": item.chunk_id,
                "doc_id": item.doc_id,
                "content": item.content,
            }
            for item in context
        ]

        prompt = self.prompt_repository.render(
            "critique.md",
            fallback=_CRITIQUE_PROMPT_FALLBACK,
            question=query,
            draft_answer=draft_answer,
            selected_context=json.dumps(context_payload, ensure_ascii=False),
            loop_count=loop_count,
            max_loops=max_loops,
        )

        try:
            raw = self.llm_client.complete(prompt)
        except Exception:
            return None

        payload = parse_json_object(raw)
        if payload is None:
            return None

        result_payload = {
            "grounded": self._to_bool(payload.get("grounded"), fallback.grounded),
            "enough_evidence": self._to_bool(payload.get("enough_evidence"), fallback.enough_evidence),
            "has_conflict": self._to_bool(payload.get("has_conflict"), fallback.has_conflict),
            "missing_aspects": self._to_list_of_strings(payload.get("missing_aspects")) or fallback.missing_aspects,
            "should_retry_retrieval": self._to_bool(
                payload.get("should_retry_retrieval"),
                fallback.should_retry_retrieval,
            ),
            "should_refine_answer": self._to_bool(
                payload.get("should_refine_answer"),
                fallback.should_refine_answer,
            ),
            "better_queries": self._to_list_of_strings(payload.get("better_queries")) or fallback.better_queries,
            "confidence": self._to_float(payload.get("confidence"), fallback.confidence),
            "note": str(payload.get("note") or fallback.note),
        }

        if loop_count >= max_loops:
            result_payload["should_retry_retrieval"] = False

        try:
            return CritiqueResult.model_validate(result_payload)
        except Exception:
            return None

    def critique(
        self,
        query: str,
        draft_answer: str,
        context: list[RetrievalResult],
        *,
        loop_count: int,
        max_loops: int,
    ) -> CritiqueResult:
        heuristic = self._heuristic_critique(
            query=query,
            draft_answer=draft_answer,
            context=context,
            loop_count=loop_count,
            max_loops=max_loops,
        )

        normalized_query = query.lower()
        if "force retry" in normalized_query or "force abstain" in normalized_query:
            return heuristic

        llm_result = self._llm_critique(
            query=query,
            draft_answer=draft_answer,
            context=context,
            loop_count=loop_count,
            max_loops=max_loops,
            fallback=heuristic,
        )
        return llm_result or heuristic
