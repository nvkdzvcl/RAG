"""Shared workflow helpers."""

from __future__ import annotations

import re

ResponseLanguage = str

_VI_DIACRITIC_PATTERN = re.compile(r"[àáạảãăắằặẳẵâấầậẩẫèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹđ]", re.IGNORECASE)
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ỹĐđ']+")

_VI_COMMON_WORDS = {
    "la",
    "là",
    "va",
    "và",
    "cua",
    "của",
    "cho",
    "trong",
    "tai",
    "tại",
    "voi",
    "với",
    "duoc",
    "được",
    "mot",
    "một",
    "nhung",
    "những",
    "the",
    "thế",
    "nao",
    "nào",
    "gi",
    "gì",
    "bao",
    "nhieu",
    "bao_nhieu",
    "vi",
    "vì",
    "sao",
    "hay",
    "hãy",
    "khong",
    "không",
    "toi",
    "tôi",
    "ban",
    "bạn",
}

_EN_COMMON_WORDS = {
    "the",
    "and",
    "is",
    "are",
    "to",
    "of",
    "in",
    "for",
    "with",
    "on",
    "that",
    "this",
    "it",
    "as",
    "be",
    "from",
    "or",
}


def normalize_query(query: str) -> str:
    """Normalize query text before retrieval or generation."""
    return query.strip()


def detect_response_language(query: str) -> ResponseLanguage:
    """Infer response language from query text (currently vi/en)."""
    normalized = normalize_query(query)
    if not normalized:
        return "en"

    lowered = normalized.lower()

    if _VI_DIACRITIC_PATTERN.search(lowered):
        return "vi"

    phrase_markers = (" la gi", " nhu the nao", " bao nhieu", " tai sao")
    if any(marker in f" {lowered}" for marker in phrase_markers):
        return "vi"

    words = _WORD_PATTERN.findall(lowered)
    vi_hits = sum(1 for word in words if word in _VI_COMMON_WORDS)
    if vi_hits >= 2:
        return "vi"

    return "en"


def response_language_name(response_language: ResponseLanguage) -> str:
    """Map language code to model-facing language name."""
    return "Vietnamese" if response_language == "vi" else "English"


def build_language_system_prompt(
    response_language: ResponseLanguage,
    *,
    require_json: bool = True,
) -> str:
    """Build strict language system instruction for model calls."""
    language_name = response_language_name(response_language)
    parts = [
        f"You must answer in {language_name}. Do not switch languages.",
        "Do not answer in Chinese unless the user explicitly asks in Chinese.",
    ]
    if require_json:
        parts.append("Return only valid JSON that matches the requested schema.")
    return " ".join(parts)


def is_language_mismatch(answer: str, expected_language: ResponseLanguage) -> bool:
    """Heuristic language mismatch detection for final answers."""
    text = answer.strip()
    if not text:
        return False

    if expected_language != "vi":
        return False

    if _CJK_PATTERN.search(text):
        return True

    if _VI_DIACRITIC_PATTERN.search(text):
        return False

    words = [word.lower() for word in _WORD_PATTERN.findall(text)]
    if not words:
        return False

    vi_hits = sum(1 for word in words if word in _VI_COMMON_WORDS)
    en_hits = sum(1 for word in words if word in _EN_COMMON_WORDS)
    if vi_hits == 0 and en_hits >= max(3, len(words) // 5):
        return True
    if vi_hits < 2 and len(words) >= 8:
        return True

    return False


def localized_insufficient_evidence(response_language: ResponseLanguage) -> str:
    """Localized insufficient-evidence message for final user responses."""
    if response_language == "vi":
        return "Không đủ bằng chứng để đưa ra câu trả lời có căn cứ."
    return "Insufficient evidence to provide a grounded answer."


def _normalized_terms(text: str) -> set[str]:
    terms = {token.lower() for token in _WORD_PATTERN.findall(text)}
    return {term for term in terms if len(term) >= 3}


def grounded_overlap_score(answer: str, context_chunks: list[str]) -> float:
    """Compute a lightweight groundedness proxy via token overlap ratio."""
    answer_terms = _normalized_terms(answer)
    if not answer_terms:
        return 0.0

    context_terms = _normalized_terms(" ".join(context_chunks))
    if not context_terms:
        return 0.0

    overlap_count = len(answer_terms.intersection(context_terms))
    return round(overlap_count / max(len(answer_terms), 1), 4)


def detect_hallucination(answer: str, context_chunks: list[str], *, status: str | None = None) -> bool:
    """Heuristic hallucination marker based on low overlap with selected context."""
    normalized_status = (status or "").strip().lower()
    if normalized_status == "insufficient_evidence":
        return False

    score = grounded_overlap_score(answer, context_chunks)
    # Answered/partial outputs with almost no lexical grounding are risky.
    return score < 0.08
