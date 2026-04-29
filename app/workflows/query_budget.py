"""Deterministic query complexity and dynamic budget policy for standard mode."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

QueryComplexity = Literal["simple_extractive", "normal", "complex"]

_WHITESPACE = re.compile(r"\s+")
_ARTICLE_ENTITY = re.compile(
    r"\b(?:article|section|clause|điều|dieu|khoản|khoan|mục|muc)\s+\d+[a-z]?\b",
    flags=re.IGNORECASE,
)
_QUOTED_ENTITY = re.compile(r"\"[^\"]{2,}\"|'[^']{2,}'")
_UPPER_TOKEN = re.compile(r"\b[A-Z][A-Z0-9_-]{1,}\b")

_TITLE_PATTERNS = (
    re.compile(
        r"\b(?:title|name)\b.{0,40}\b(?:article|section|clause)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:article|section|clause)\s+\d+[a-z]?\b.{0,30}\b(?:title|name)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\btên\s+(?:của\s+)?(?:điều|mục|khoản)\s+\d+[a-z]?\b", re.IGNORECASE),
    re.compile(r"\bten\s+(?:cua\s+)?(?:dieu|muc|khoan)\s+\d+[a-z]?\b", re.IGNORECASE),
)
_DEFINITION_PATTERNS = (
    re.compile(r"\b(?:what is|define|definition of|meaning of)\b", re.IGNORECASE),
    re.compile(r"\b(?:là gì|la gi|định nghĩa|dinh nghia|nghĩa là gì|nghia la gi)\b"),
)
_NUMERIC_PATTERNS = (
    re.compile(
        r"\b(?:date|day|month|year|number|amount|threshold|deadline|limit|quota|"
        r"percent|percentage)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ngày|tháng|năm|số|mức|ngưỡng|hạn|hạn chót|hạn cuối|bao nhiêu|"
        r"bao nhieu|tỷ lệ|ti le)\b"
    ),
)
_EXTRACT_PATTERNS = (
    re.compile(r"\b(?:quote|quoted|exact text|verbatim|extract)\b", re.IGNORECASE),
    re.compile(r"\b(?:trích|trich|nguyên văn|nguyen van)\b"),
)

_COMPLEX_HINTS = (
    "compare",
    "contrast",
    "analyze",
    "analysis",
    "summarize",
    "summary",
    "pros and cons",
    "evaluate",
    "synthesize",
    "trade-off",
    "tradeoff",
    "so sánh",
    "so sanh",
    "phân biệt",
    "phan biet",
    "phân tích",
    "phan tich",
    "đánh giá",
    "danh gia",
    "tổng hợp",
    "tong hop",
    "ưu nhược điểm",
    "uu nhuoc diem",
)


@dataclass(frozen=True)
class QueryBudget:
    """Effective per-query retrieval and generation budget."""

    complexity: QueryComplexity
    dynamic_enabled: bool
    hybrid_top_k: int
    rerank_top_k: int
    context_top_k: int
    context_max_chars: int
    max_tokens: int

    def as_trace_payload(self) -> dict[str, int | bool | str]:
        return {
            "complexity": self.complexity,
            "dynamic_enabled": self.dynamic_enabled,
            "hybrid_top_k": self.hybrid_top_k,
            "rerank_top_k": self.rerank_top_k,
            "context_top_k": self.context_top_k,
            "context_max_chars": self.context_max_chars,
            "max_tokens": self.max_tokens,
        }


def _normalize_query(query: str) -> str:
    return _WHITESPACE.sub(" ", query.strip()).casefold()


def _has_clear_entity(raw_query: str, normalized_query: str) -> bool:
    if _ARTICLE_ENTITY.search(normalized_query):
        return True
    if _QUOTED_ENTITY.search(raw_query):
        return True
    return bool(_UPPER_TOKEN.search(raw_query))


def _is_simple_extractive(query: str, normalized_query: str) -> bool:
    if any(pattern.search(query) for pattern in _TITLE_PATTERNS):
        return True
    if any(pattern.search(query) for pattern in _DEFINITION_PATTERNS):
        return True
    if any(pattern.search(query) for pattern in _NUMERIC_PATTERNS):
        return True
    if any(pattern.search(query) for pattern in _EXTRACT_PATTERNS):
        return True

    token_count = len(normalized_query.split())
    return token_count <= 10 and _has_clear_entity(query, normalized_query)


def classify_query_complexity(query: str) -> QueryComplexity:
    """Classify query complexity with deterministic regex/keyword heuristics."""
    normalized = _normalize_query(query)
    if not normalized:
        return "normal"

    if any(hint in normalized for hint in _COMPLEX_HINTS):
        return "complex"
    if _is_simple_extractive(query, normalized):
        return "simple_extractive"
    return "normal"


def _positive_int(value: int) -> int:
    return max(1, int(value))


def choose_query_budget(
    query: str,
    *,
    dynamic_enabled: bool,
    base_hybrid_top_k: int,
    base_rerank_top_k: int,
    base_context_top_k: int,
    base_context_max_chars: int,
    base_llm_max_tokens: int,
    simple_max_tokens: int,
    normal_max_tokens: int,
    complex_max_tokens: int,
    simple_context_chars: int,
    normal_context_chars: int,
    retrieval_top_k_locked: bool = False,
) -> QueryBudget:
    """Return the effective budget for one query."""
    base_hybrid = _positive_int(base_hybrid_top_k)
    base_rerank = _positive_int(min(base_rerank_top_k, base_hybrid))
    base_context_k = _positive_int(base_context_top_k)
    base_context_chars = _positive_int(base_context_max_chars)
    base_tokens = _positive_int(base_llm_max_tokens)

    complexity = classify_query_complexity(query)
    if not dynamic_enabled:
        return QueryBudget(
            complexity=complexity,
            dynamic_enabled=False,
            hybrid_top_k=base_hybrid,
            rerank_top_k=base_rerank,
            context_top_k=base_context_k,
            context_max_chars=base_context_chars,
            max_tokens=base_tokens,
        )

    if complexity == "simple_extractive":
        target_hybrid = 3
        target_rerank = 3
        target_context_k = 2
        target_context_chars = _positive_int(simple_context_chars)
        target_tokens = _positive_int(simple_max_tokens)
    elif complexity == "normal":
        target_hybrid = 5
        target_rerank = 5
        target_context_k = 3
        target_context_chars = _positive_int(normal_context_chars)
        target_tokens = _positive_int(normal_max_tokens)
    else:
        target_hybrid = base_hybrid
        target_rerank = base_rerank
        target_context_k = base_context_k
        target_context_chars = base_context_chars
        target_tokens = _positive_int(complex_max_tokens)

    if retrieval_top_k_locked:
        hybrid_top_k = base_hybrid
        rerank_top_k = base_rerank
    else:
        hybrid_top_k = min(base_hybrid, _positive_int(target_hybrid))
        rerank_top_k = min(hybrid_top_k, base_rerank, _positive_int(target_rerank))

    context_top_k = min(base_context_k, _positive_int(target_context_k))
    return QueryBudget(
        complexity=complexity,
        dynamic_enabled=True,
        hybrid_top_k=_positive_int(hybrid_top_k),
        rerank_top_k=_positive_int(rerank_top_k),
        context_top_k=_positive_int(context_top_k),
        context_max_chars=_positive_int(target_context_chars),
        max_tokens=_positive_int(target_tokens),
    )
