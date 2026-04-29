"""Adaptive policy for reducing unnecessary Advanced-mode LLM calls."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from app.workflows.query_budget import QueryComplexity

Strictness = Literal["relaxed", "balanced", "strict"]

_HIGH_RISK_TERMS = (
    "legal advice",
    "attorney",
    "luật sư",
    "tu van phap ly",
    "tư vấn pháp lý",
    "medical advice",
    "chẩn đoán",
    "chan doan",
    "kê đơn",
    "ke don",
    "điều trị",
    "dieu tri",
    "đầu tư",
    "dau tu",
    "financial advice",
    "tax advice",
    "thuế",
    "thue",
    "compliance",
    "trách nhiệm pháp lý",
    "trach nhiem phap ly",
)
_AMBIGUOUS_INTENT_PATTERN = re.compile(
    r"\b(?:help|assist|gợi ý|goi y|khuyên|khuyen|nên làm gì|nen lam gi|advice)\b",
    flags=re.IGNORECASE,
)
_INTERPRETATION_PATTERN = re.compile(
    r"\b(?:interpret|analysis|analyze|đánh giá|danh gia|phân tích|phan tich)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class AdvancedPolicyInput:
    query_complexity: QueryComplexity | str
    retrieval_confidence: float | None
    citation_count: int
    grounding_lexical_score: float | None
    answer_length: int
    fast_path_used: bool
    user_selected_strictness: str | None = None


class AdvancedPolicy:
    """Adaptive decision helpers for advanced orchestration."""

    def __init__(
        self,
        *,
        adaptive_enabled: bool = True,
        force_llm_gate: bool = False,
        force_llm_critic: bool = False,
    ) -> None:
        self.adaptive_enabled = bool(adaptive_enabled)
        self.force_llm_gate = bool(force_llm_gate)
        self.force_llm_critic = bool(force_llm_critic)

    @staticmethod
    def normalize_complexity(value: str | None) -> QueryComplexity:
        raw = (value or "normal").strip().lower()
        if raw == "simple_extractive":
            return "simple_extractive"
        if raw == "complex":
            return "complex"
        return "normal"

    @staticmethod
    def _strictness(value: str | None) -> Strictness:
        raw = (value or "balanced").strip().lower()
        if raw in {"strict", "high"}:
            return "strict"
        if raw in {"relaxed", "low"}:
            return "relaxed"
        return "balanced"

    @staticmethod
    def _is_high_risk_query(query: str) -> bool:
        lowered = query.strip().lower()
        if any(term in lowered for term in _HIGH_RISK_TERMS):
            return True
        return bool(_INTERPRETATION_PATTERN.search(query))

    @staticmethod
    def _is_ambiguous_intent(query: str) -> bool:
        if not query.strip():
            return False
        token_count = len(query.strip().split())
        if token_count <= 2:
            return True
        return bool(_AMBIGUOUS_INTENT_PATTERN.search(query))

    def should_use_llm_gate(
        self,
        *,
        query: str,
        query_complexity: QueryComplexity | str,
        heuristic_reason: str,
        user_selected_strictness: str | None = None,
    ) -> bool:
        if heuristic_reason in {
            "empty_query",
            "forced_retrieval",
            "small_talk",
            "small_talk_short",
        }:
            return False
        if self.force_llm_gate:
            return True
        if not self.adaptive_enabled:
            return True

        strictness = self._strictness(user_selected_strictness)
        normalized_complexity = self.normalize_complexity(str(query_complexity))
        if strictness == "strict":
            return True
        if self._is_high_risk_query(query):
            return True
        if normalized_complexity == "complex":
            return True
        if self._is_ambiguous_intent(query):
            return True
        return False

    def should_use_llm_critic(
        self,
        *,
        query: str,
        signal: AdvancedPolicyInput,
    ) -> bool:
        if self.force_llm_critic:
            return True
        if not self.adaptive_enabled:
            return True

        strictness = self._strictness(signal.user_selected_strictness)
        complexity = self.normalize_complexity(signal.query_complexity)
        citation_count = max(0, int(signal.citation_count))
        grounding_score = (
            None
            if signal.grounding_lexical_score is None
            else float(signal.grounding_lexical_score)
        )
        retrieval_confidence = (
            None
            if signal.retrieval_confidence is None
            else float(signal.retrieval_confidence)
        )

        if strictness == "strict":
            return True
        if self._is_high_risk_query(query):
            return True
        if citation_count <= 0:
            return True
        if grounding_score is not None and grounding_score < 0.06:
            return True
        if complexity == "complex":
            return True
        if signal.answer_length >= 420:
            return True
        if retrieval_confidence is not None and retrieval_confidence < 0.35:
            return True
        if (
            signal.fast_path_used
            and citation_count > 0
            and (grounding_score or 0.0) >= 0.08
        ):
            return False
        return False

    def should_run_hallucination_refine(
        self,
        *,
        query: str,
        signal: AdvancedPolicyInput,
        hallucination_detected: bool,
    ) -> bool:
        if not hallucination_detected:
            return False
        if not self.adaptive_enabled:
            return True

        strictness = self._strictness(signal.user_selected_strictness)
        complexity = self.normalize_complexity(signal.query_complexity)
        citation_count = max(0, int(signal.citation_count))
        grounding_score = (
            None
            if signal.grounding_lexical_score is None
            else float(signal.grounding_lexical_score)
        )

        if strictness == "strict":
            return True
        if self._is_high_risk_query(query):
            return True
        if complexity == "complex":
            return True
        if citation_count <= 0:
            return True
        if grounding_score is not None and grounding_score < 0.035:
            return True
        return False
