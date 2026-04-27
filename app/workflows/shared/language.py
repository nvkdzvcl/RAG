"""Language detection, normalization, and mismatch helpers."""

from __future__ import annotations

import re
import unicodedata

ResponseLanguage = str

_VI_DIACRITIC_PATTERN = re.compile(
    r"[àáạảãăắằặẳẵâấầậẩẫèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹđ]",
    re.IGNORECASE,
)
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ỹĐđ0-9']+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_SANITIZE_PATTERN = re.compile(r"[^A-Za-zÀ-ỹĐđ0-9\s'_+]")

_VI_COMMON_WORDS = {
    "la", "là", "va", "và", "cua", "của", "cho", "trong", "tai", "tại",
    "voi", "với", "duoc", "được", "mot", "một", "nhung", "những", "the",
    "thế", "nao", "nào", "gi", "gì", "bao", "nhieu", "bao_nhieu", "vi",
    "vì", "sao", "hay", "hãy", "khong", "không", "toi", "tôi", "ban",
    "bạn", "nay", "này", "do", "neu", "nếu", "theo", "tren", "trên",
    "duoi", "dưới", "dang", "đang", "se", "sẽ", "da", "đã", "can",
    "cần", "them", "thêm", "tu", "từ", "den", "đến", "day", "đây",
    "kia", "đó", "ve", "về", "cung", "cũng", "len", "lên", "xuong",
    "xuống", "roi", "rồi", "chi", "chỉ", "rat", "rất", "co", "có",
    "thong", "thông", "tin",
}

_EN_COMMON_WORDS = {
    "the", "and", "is", "are", "to", "of", "in", "for", "with", "on",
    "that", "this", "it", "as", "be", "from", "or", "an", "by", "at",
    "was", "were", "can", "could", "should", "would", "about", "into",
    "over", "under", "than", "then", "also", "more", "most", "very",
    "some", "any", "all", "each", "every", "such", "other", "their",
    "there", "here", "our", "your", "my",
}


def normalize_query(query: str) -> str:
    """Normalize query text before retrieval or generation.

    Steps:
    1. Unicode NFC normalization (consistent Vietnamese diacritics).
    2. Collapse multiple whitespace / newlines into a single space.
    3. Sanitize with ``_SANITIZE_PATTERN`` (consistent with matching helpers).
    4. Re-collapse whitespace and trim.

    Semantic characters (Vietnamese, CJK, digits, ``_``, ``+``, ``'``) are
    preserved so compound tokens (``sinh_viên``, ``C++``) remain intact.
    """
    if not query:
        return ""
    text = unicodedata.normalize("NFC", query)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    text = _SANITIZE_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text)  # re-collapse after sanitization
    return text.strip()


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
