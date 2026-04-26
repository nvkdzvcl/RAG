"""Shared workflow helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

ResponseLanguage = str

_VI_DIACRITIC_PATTERN = re.compile(
    r"[àáạảãăắằặẳẵâấầậẩẫèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ỹĐđ0-9']+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_SANITIZE_PATTERN = re.compile(r"[^A-Za-zÀ-ỹĐđ0-9\s']")

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
    "nay",
    "này",
    "do",
    "neu",
    "nếu",
    "theo",
    "tren",
    "trên",
    "duoi",
    "dưới",
    "dang",
    "đang",
    "se",
    "sẽ",
    "da",
    "đã",
    "can",
    "cần",
    "them",
    "thêm",
    "tu",
    "từ",
    "den",
    "đến",
    "day",
    "đây",
    "kia",
    "đó",
    "ve",
    "về",
    "cung",
    "cũng",
    "len",
    "lên",
    "xuong",
    "xuống",
    "roi",
    "rồi",
    "chi",
    "chỉ",
    "rat",
    "rất",
    "co",
    "có",
    "thong",
    "thông",
    "tin",
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
    "an",
    "by",
    "at",
    "was",
    "were",
    "can",
    "could",
    "should",
    "would",
    "about",
    "into",
    "over",
    "under",
    "than",
    "then",
    "also",
    "more",
    "most",
    "very",
    "some",
    "any",
    "all",
    "each",
    "every",
    "such",
    "other",
    "their",
    "there",
    "here",
    "our",
    "your",
    "my",
}

_GENERIC_PHRASES = (
    "không đủ thông tin",
    "khó xác định",
    "cần thêm thông tin",
    "dựa trên ngữ cảnh hiện có",
    "theo ngữ cảnh hiện có",
    "thông tin liên quan là",
    "cannot determine",
    "can not determine",
    "based on the available context",
    "evidence is limited",
    "insufficient evidence",
)


@dataclass(frozen=True)
class GroundingAssessment:
    grounded_score: float
    grounding_reason: str
    hallucination_detected: bool


def _normalize_match_text(text: str) -> str:
    lowered = _WHITESPACE_PATTERN.sub(" ", text).strip().lower()
    return _SANITIZE_PATTERN.sub(" ", lowered)


def _meaningful_keywords(text: str) -> set[str]:
    normalized = _normalize_match_text(text)
    keywords: set[str] = set()
    for token in _WORD_PATTERN.findall(normalized):
        term = token.lower().strip()
        if not term:
            continue
        if term in _VI_COMMON_WORDS or term in _EN_COMMON_WORDS:
            continue
        if term.isdigit():
            if len(term) >= 2:
                keywords.add(term)
            continue
        if len(term) < 2:
            continue
        keywords.add(term)
    return keywords


def _char_ngram_precision(answer: str, context_chunks: list[str], n: int = 3) -> float:
    answer_text = _normalize_match_text(answer).replace(" ", "")
    context_text = _normalize_match_text(" ".join(context_chunks)).replace(" ", "")
    if len(answer_text) < n or len(context_text) < n:
        return 0.0

    answer_ngrams = {answer_text[idx : idx + n] for idx in range(len(answer_text) - n + 1)}
    context_ngrams = {context_text[idx : idx + n] for idx in range(len(context_text) - n + 1)}
    if not answer_ngrams or not context_ngrams:
        return 0.0

    overlap = len(answer_ngrams.intersection(context_ngrams))
    return round(overlap / max(len(answer_ngrams), 1), 4)


def _is_generic_answer(answer: str) -> bool:
    normalized = _normalize_match_text(answer)
    if not normalized:
        return True
    if any(phrase in normalized for phrase in _GENERIC_PHRASES):
        return True
    return len(_meaningful_keywords(answer)) <= 2


def normalize_query(query: str) -> str:
    """Normalize query text before retrieval or generation."""
    return query.strip()


def trim_chat_history(
    chat_history: list[dict[str, str]] | None,
    *,
    memory_window: int,
) -> list[dict[str, str]]:
    """Return the latest bounded conversation window for prompt context."""
    if not chat_history:
        return []
    if memory_window <= 0:
        return []

    # Memory window is measured in turns; one turn roughly equals user + assistant.
    max_messages = max(1, memory_window * 2)
    normalized: list[dict[str, str]] = []
    for item in chat_history[-max_messages:]:
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def build_chat_history_context(
    chat_history: list[dict[str, str]] | None,
    *,
    memory_window: int,
) -> str:
    """Render bounded chat history into concise prompt text."""
    window = trim_chat_history(chat_history, memory_window=memory_window)
    if not window:
        return "(empty)"

    lines: list[str] = []
    for idx, message in enumerate(window, start=1):
        role_label = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"{idx}. {role_label}: {message['content']}")
    return "\n".join(lines)


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
        return "Không đủ thông tin từ tài liệu để trả lời"
    return "Insufficient evidence to provide a grounded answer."


def _normalized_terms(text: str) -> set[str]:
    return _meaningful_keywords(text)


def grounded_overlap_score(answer: str, context_chunks: list[str]) -> float:
    """Compute a tolerant groundedness score using keyword and char overlap."""
    answer_terms = _normalized_terms(answer)
    context_terms = _normalized_terms(" ".join(context_chunks))

    keyword_precision = 0.0
    if answer_terms and context_terms:
        overlap_count = len(answer_terms.intersection(context_terms))
        keyword_precision = overlap_count / max(len(answer_terms), 1)
    char_precision = _char_ngram_precision(answer, context_chunks)

    if answer_terms:
        blended = (0.8 * keyword_precision) + (0.2 * char_precision)
    else:
        blended = char_precision
    return round(max(0.0, min(1.0, blended)), 4)


def assess_grounding(
    answer: str,
    context_chunks: list[str],
    *,
    citation_count: int = 0,
    has_selected_context: bool | None = None,
    status: str | None = None,
) -> GroundingAssessment:
    """Assess grounding and hallucination risk from answer/context signals."""
    normalized_status = (status or "").strip().lower()
    has_context = bool(context_chunks) if has_selected_context is None else bool(has_selected_context)

    score = grounded_overlap_score(answer, context_chunks)
    if normalized_status == "insufficient_evidence":
        return GroundingAssessment(
            grounded_score=score,
            grounding_reason="insufficient_evidence_status",
            hallucination_detected=False,
        )

    if not answer.strip():
        return GroundingAssessment(
            grounded_score=0.0,
            grounding_reason="empty_answer",
            hallucination_detected=has_context or citation_count > 0,
        )

    if not has_context:
        return GroundingAssessment(
            grounded_score=score,
            grounding_reason="no_selected_context",
            hallucination_detected=True,
        )

    generic_answer = _is_generic_answer(answer)
    if citation_count > 0:
        if score >= 0.12:
            return GroundingAssessment(score, "strong_grounding_with_citations", False)
        if score >= 0.02:
            return GroundingAssessment(score, "weak_grounding_with_citations", False)
        if score >= 0.005:
            return GroundingAssessment(score, "very_low_overlap_but_cited", False)
        return GroundingAssessment(score, "almost_no_overlap_even_with_citations", True)

    if generic_answer:
        return GroundingAssessment(
            grounded_score=score,
            grounding_reason="generic_answer_without_citations",
            hallucination_detected=True,
        )

    if score >= 0.12:
        return GroundingAssessment(score, "strong_grounding_no_citations", False)
    if score >= 0.04:
        return GroundingAssessment(score, "weak_grounding_no_citations", False)
    return GroundingAssessment(score, "low_overlap_without_citations", True)


def detect_hallucination(
    answer: str,
    context_chunks: list[str],
    *,
    status: str | None = None,
    citation_count: int = 0,
    has_selected_context: bool | None = None,
) -> bool:
    """Backward-compatible hallucination flag wrapper."""
    assessment = assess_grounding(
        answer,
        context_chunks,
        citation_count=citation_count,
        has_selected_context=has_selected_context,
        status=status,
    )
    return assessment.hallucination_detected
