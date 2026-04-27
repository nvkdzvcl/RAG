"""Grounding assessment, hallucination detection, and semantic similarity."""

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from app.core.math_utils import cosine_similarity
from app.workflows.shared.language import _VI_COMMON_WORDS, _EN_COMMON_WORDS

# ---------------------------------------------------------------------------
# Internal text-processing patterns (shared with language module via re-import)
# ---------------------------------------------------------------------------

_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ỹĐđ0-9']+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_SANITIZE_PATTERN = re.compile(r"[^A-Za-zÀ-ỹĐđ0-9\s'_+]")


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Grounding semantic config
# ---------------------------------------------------------------------------

_GROUNDING_SEMANTIC_ENABLED = _env_bool("GROUNDING_SEMANTIC_ENABLED", True)
_GROUNDING_SEMANTIC_MODEL = (
    os.getenv("GROUNDING_SEMANTIC_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip()
    or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
_GROUNDING_SEMANTIC_DEVICE = os.getenv("GROUNDING_SEMANTIC_DEVICE", "cpu").strip() or "cpu"
_GROUNDING_SEMANTIC_LOCAL_FILES_ONLY = _env_bool("GROUNDING_SEMANTIC_LOCAL_FILES_ONLY", False)
_GROUNDING_SEMANTIC_MAX_CONTEXT_CHUNKS = _env_int("GROUNDING_SEMANTIC_MAX_CONTEXT_CHUNKS", 6, minimum=1)
_GROUNDING_SEMANTIC_MIN_SIMILARITY = max(
    0.0,
    min(1.0, _env_float("GROUNDING_SEMANTIC_MIN_SIMILARITY", 0.58)),
)
_GROUNDING_SEMANTIC_WEIGHT = max(
    0.0,
    min(1.0, _env_float("GROUNDING_SEMANTIC_WEIGHT", 0.35)),
)


class _SentenceTransformerLike(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any:
        """Encode sentences into vectors."""


_SEMANTIC_ENCODER: _SentenceTransformerLike | None = None
_SEMANTIC_ENCODER_INITIALIZED = False
_SEMANTIC_ENCODER_LOCK = threading.Lock()





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


# ---------------------------------------------------------------------------
# Text normalization & keyword extraction
# ---------------------------------------------------------------------------

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


def _normalized_terms(text: str) -> set[str]:
    return _meaningful_keywords(text)


# ---------------------------------------------------------------------------
# Semantic encoder (lazy singleton)
# ---------------------------------------------------------------------------

def _load_semantic_encoder() -> _SentenceTransformerLike | None:
    global _SEMANTIC_ENCODER_INITIALIZED, _SEMANTIC_ENCODER
    if not _GROUNDING_SEMANTIC_ENABLED:
        return None
    if _SEMANTIC_ENCODER_INITIALIZED:
        return _SEMANTIC_ENCODER

    with _SEMANTIC_ENCODER_LOCK:
        if _SEMANTIC_ENCODER_INITIALIZED:
            return _SEMANTIC_ENCODER
        _SEMANTIC_ENCODER_INITIALIZED = True
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            _SEMANTIC_ENCODER = None
            return None

        try:
            _SEMANTIC_ENCODER = SentenceTransformer(
                _GROUNDING_SEMANTIC_MODEL,
                device=_GROUNDING_SEMANTIC_DEVICE,
                local_files_only=_GROUNDING_SEMANTIC_LOCAL_FILES_ONLY,
            )
        except TypeError:
            try:
                _SEMANTIC_ENCODER = SentenceTransformer(
                    _GROUNDING_SEMANTIC_MODEL,
                    device=_GROUNDING_SEMANTIC_DEVICE,
                )
            except Exception:
                _SEMANTIC_ENCODER = None
        except Exception:
            _SEMANTIC_ENCODER = None
        return _SEMANTIC_ENCODER


def _normalize_vectors(raw_vectors: Any) -> list[list[float]]:
    if hasattr(raw_vectors, "tolist"):
        raw_vectors = raw_vectors.tolist()
    if not isinstance(raw_vectors, list):
        raw_vectors = [list(row) for row in raw_vectors]

    if raw_vectors and raw_vectors[0] and isinstance(raw_vectors[0], (float, int)):
        raw_vectors = [raw_vectors]

    vectors: list[list[float]] = []
    for row in raw_vectors:
        if not isinstance(row, list):
            continue
        vectors.append([float(value) for value in row])
    return vectors


def _semantic_context_similarity(answer: str, context_chunks: list[str]) -> float | None:
    if not _GROUNDING_SEMANTIC_ENABLED:
        return None

    normalized_answer = answer.strip()
    normalized_contexts = [chunk.strip() for chunk in context_chunks if chunk.strip()]
    if not normalized_answer or not normalized_contexts:
        return None

    encoder = _load_semantic_encoder()
    if encoder is None:
        return None

    normalized_contexts = normalized_contexts[:_GROUNDING_SEMANTIC_MAX_CONTEXT_CHUNKS]
    texts = [normalized_answer, *normalized_contexts]
    try:
        raw_vectors = encoder.encode(
            texts,
            batch_size=min(16, len(texts)),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception:
        return None

    vectors = _normalize_vectors(raw_vectors)
    if len(vectors) <= 1:
        return None
    answer_vector = vectors[0]
    context_vectors = vectors[1:]
    if not answer_vector or not context_vectors:
        return None

    best_similarity = max(
        (cosine_similarity(answer_vector, vector) for vector in context_vectors),
        default=0.0,
    )
    return round(max(0.0, min(1.0, best_similarity)), 4)


# ---------------------------------------------------------------------------
# Public scoring API
# ---------------------------------------------------------------------------

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


def grounded_score(answer: str, context_chunks: list[str]) -> float:
    """Compute groundedness score by blending overlap and optional semantic similarity."""
    overlap_score = grounded_overlap_score(answer, context_chunks)
    semantic_score = _semantic_context_similarity(answer, context_chunks)
    if semantic_score is None or semantic_score < _GROUNDING_SEMANTIC_MIN_SIMILARITY:
        return overlap_score

    blended = ((1.0 - _GROUNDING_SEMANTIC_WEIGHT) * overlap_score) + (_GROUNDING_SEMANTIC_WEIGHT * semantic_score)
    return round(max(overlap_score, min(1.0, blended)), 4)


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

    score = grounded_score(answer, context_chunks)
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
