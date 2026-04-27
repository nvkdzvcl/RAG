"""Critique module for advanced workflow decisions."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.core.async_utils import run_coro_sync
from app.core.config import get_settings
from app.core.json_utils import parse_json_object
from app.core.prompting import PromptRepository
from app.generation.llm_client import LLMClient, complete_with_model
from app.schemas.retrieval import RetrievalResult
from app.schemas.workflow import CritiqueResult
from app.workflows.shared import (
    build_chat_history_context,
    build_language_system_prompt,
    response_language_name,
)

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
    "Keep note/missing_aspects/better_queries in $response_language_name.\\n"
    "response_language: $response_language\\n"
    "chat_history: $chat_history\\n"
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
        self.memory_window = max(0, int(getattr(settings, "memory_window", 3)))
        self.max_tokens = max(1, int(getattr(settings, "llm_critique_max_tokens", 384)))
        resolved_prompt_dir = prompt_dir or settings.prompt_dir
        self.prompt_repository = prompt_repository or PromptRepository(
            resolved_prompt_dir
        )

    def _terms(self, text: str) -> set[str]:
        return {
            token
            for token in self.token_pattern.findall(text.lower())
            if len(token) > 2
        }

    @staticmethod
    def _overlap_ratio(base_terms: set[str], support_terms: set[str]) -> float:
        if not base_terms:
            return 0.0
        overlap = len(base_terms.intersection(support_terms))
        return overlap / max(len(base_terms), 1)

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
        query_context_overlap = self._overlap_ratio(query_terms, context_terms)
        answer_context_overlap = self._overlap_ratio(answer_terms, context_terms)

        force_retry = "force retry" in query.lower()
        force_abstain = "force abstain" in query.lower()

        missing_aspects = sorted(
            [
                term
                for term in query_terms
                if term in context_terms and term not in answer_terms
            ]
        )[:5]

        if force_abstain:
            critique_category = "no_evidence"
        elif not context or not context_text.strip() or query_context_overlap <= 0.0:
            critique_category = "no_evidence"
        elif answer_context_overlap < 0.02:
            critique_category = "hallucination"
        elif answer_context_overlap < 0.12:
            critique_category = "weak_evidence"
        elif missing_aspects:
            critique_category = "incomplete_answer"
        else:
            critique_category = "grounded"

        enough_evidence = critique_category != "no_evidence"
        grounded = critique_category in {"grounded", "incomplete_answer"}

        has_conflict = False
        if len({item.doc_id for item in context}) > 1:
            context_lower = context_text.lower()
            has_conflict = (
                "however" in context_lower and "therefore" in context_lower
            ) or ("conflict" in context_lower)

        should_retry_retrieval = (
            force_retry
            or (critique_category == "no_evidence" and loop_count < max_loops)
        ) and not force_abstain

        should_refine_answer = (
            critique_category in {"weak_evidence", "incomplete_answer", "hallucination"}
            and enough_evidence
            and not force_abstain
        )

        better_queries: list[str] = []
        if should_retry_retrieval:
            better_queries.append(f"{query} supporting evidence")
            for aspect in missing_aspects[:2]:
                better_queries.append(f"{query} {aspect} details")

        if force_abstain:
            confidence = 0.0
            note = "no_evidence: Forced abstain requested by query signal."
        elif critique_category == "grounded":
            confidence = 0.82 if not has_conflict else 0.65
            note = "grounded: Answer appears grounded in selected context."
        elif critique_category == "incomplete_answer":
            confidence = 0.62
            note = "incomplete_answer: Context is relevant but answer misses requested aspects."
        elif critique_category == "weak_evidence":
            confidence = 0.42
            note = "weak_evidence: Context exists but support for the current wording is weak."
        elif critique_category == "hallucination":
            confidence = 0.28
            note = "hallucination: Draft contains claims not well supported by selected context."
        elif should_retry_retrieval:
            confidence = 0.35
            note = (
                "no_evidence: Relevant support not found; retry retrieval recommended."
            )
        else:
            confidence = 0.2
            note = "no_evidence: Evidence insufficient and no additional retries available."

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

    async def _llm_critique(
        self,
        query: str,
        draft_answer: str,
        context: list[RetrievalResult],
        *,
        loop_count: int,
        max_loops: int,
        fallback: CritiqueResult,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
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
            chat_history=build_chat_history_context(
                chat_history,
                memory_window=self.memory_window,
            ),
            loop_count=loop_count,
            max_loops=max_loops,
            response_language=response_language,
            response_language_name=response_language_name(response_language),
        )

        try:
            raw = await complete_with_model(
                self.llm_client,
                prompt,
                system_prompt=build_language_system_prompt(response_language),
                model=model,
                max_tokens=self.max_tokens,
            )
        except Exception:
            return None

        payload = parse_json_object(raw)
        if payload is None:
            return None

        result_payload = {
            "grounded": self._to_bool(payload.get("grounded"), fallback.grounded),
            "enough_evidence": self._to_bool(
                payload.get("enough_evidence"), fallback.enough_evidence
            ),
            "has_conflict": self._to_bool(
                payload.get("has_conflict"), fallback.has_conflict
            ),
            "missing_aspects": self._to_list_of_strings(payload.get("missing_aspects"))
            or fallback.missing_aspects,
            "should_retry_retrieval": self._to_bool(
                payload.get("should_retry_retrieval"),
                fallback.should_retry_retrieval,
            ),
            "should_refine_answer": self._to_bool(
                payload.get("should_refine_answer"),
                fallback.should_refine_answer,
            ),
            "better_queries": self._to_list_of_strings(payload.get("better_queries"))
            or fallback.better_queries,
            "confidence": self._to_float(
                payload.get("confidence"), fallback.confidence
            ),
            "note": str(payload.get("note") or fallback.note),
        }

        if loop_count >= max_loops:
            result_payload["should_retry_retrieval"] = False

        try:
            return CritiqueResult.model_validate(result_payload)
        except Exception:
            return None

    async def critique_async(
        self,
        query: str,
        draft_answer: str,
        context: list[RetrievalResult],
        *,
        loop_count: int,
        max_loops: int,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
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

        llm_result = await self._llm_critique(
            query=query,
            draft_answer=draft_answer,
            context=context,
            loop_count=loop_count,
            max_loops=max_loops,
            fallback=heuristic,
            chat_history=chat_history,
            model=model,
            response_language=response_language,
        )
        return llm_result or heuristic

    def critique(
        self,
        query: str,
        draft_answer: str,
        context: list[RetrievalResult],
        *,
        loop_count: int,
        max_loops: int,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
    ) -> CritiqueResult:
        """Sync wrapper for legacy callers."""
        return run_coro_sync(
            self.critique_async(
                query=query,
                draft_answer=draft_answer,
                context=context,
                loop_count=loop_count,
                max_loops=max_loops,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        )
