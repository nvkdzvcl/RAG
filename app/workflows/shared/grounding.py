"""Grounding assessment, hallucination detection, and semantic similarity."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Literal, Protocol

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
    os.getenv(
        "GROUNDING_SEMANTIC_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ).strip()
    or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
_GROUNDING_SEMANTIC_DEVICE = (
    os.getenv("GROUNDING_SEMANTIC_DEVICE", "cpu").strip() or "cpu"
)
_GROUNDING_SEMANTIC_LOCAL_FILES_ONLY = _env_bool(
    "GROUNDING_SEMANTIC_LOCAL_FILES_ONLY", False
)
_GROUNDING_SEMANTIC_MAX_CONTEXT_CHUNKS = _env_int(
    "GROUNDING_SEMANTIC_MAX_CONTEXT_CHUNKS", 6, minimum=1
)
_GROUNDING_SEMANTIC_MIN_SIMILARITY = max(
    0.0,
    min(1.0, _env_float("GROUNDING_SEMANTIC_MIN_SIMILARITY", 0.58)),
)
_GROUNDING_SEMANTIC_WEIGHT = max(
    0.0,
    min(1.0, _env_float("GROUNDING_SEMANTIC_WEIGHT", 0.35)),
)
_GROUNDING_POLICY = (
    os.getenv("GROUNDING_POLICY", "adaptive").strip().lower() or "adaptive"
)
_GROUNDING_SEMANTIC_STANDARD_ENABLED = _env_bool(
    "GROUNDING_SEMANTIC_STANDARD_ENABLED", False
)
_GROUNDING_SEMANTIC_ADVANCED_ENABLED = _env_bool(
    "GROUNDING_SEMANTIC_ADVANCED_ENABLED", True
)
_GROUNDING_STANDARD_LONG_ANSWER_CHARS = _env_int(
    "GROUNDING_STANDARD_LONG_ANSWER_CHARS", 320, minimum=32
)
_GROUNDING_STANDARD_LOW_RETRIEVAL_CONFIDENCE = max(
    0.0,
    min(1.0, _env_float("GROUNDING_STANDARD_LOW_RETRIEVAL_CONFIDENCE", 0.35)),
)
_GROUNDING_STANDARD_LOW_LEXICAL_SCORE = max(
    0.0,
    min(1.0, _env_float("GROUNDING_STANDARD_LOW_LEXICAL_SCORE", 0.04)),
)
_GROUNDING_POLICY_VERSION = "adaptive-v1"
_GROUNDING_CACHE_MAX_SIZE = _env_int("GROUNDING_CACHE_MAX_SIZE", 512, minimum=1)


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
_GROUNDING_CACHE_LOCK = threading.Lock()


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


GroundingMode = Literal["standard", "advanced", "compare"]
GroundingQueryComplexity = Literal["simple", "normal", "complex"]


@dataclass(frozen=True)
class GroundingPolicy:
    mode: GroundingMode
    query_complexity: GroundingQueryComplexity | str
    generated_status: str | None
    answer_length: int
    citation_count: int
    retrieval_confidence: float | None
    fast_path_used: bool

    def normalized_complexity(self) -> GroundingQueryComplexity:
        raw = (self.query_complexity or "normal").strip().lower()
        if raw == "simple_extractive":
            return "simple"
        if raw in {"simple", "normal", "complex"}:
            return raw  # type: ignore[return-value]
        return "normal"

    def policy_version(
        self,
        *,
        status: str,
        has_context: bool,
        semantic_requested: bool,
        decision_reason: str,
    ) -> str:
        confidence = (
            "none"
            if self.retrieval_confidence is None
            else f"{max(0.0, min(1.0, float(self.retrieval_confidence))):.4f}"
        )
        return "|".join(
            (
                _GROUNDING_POLICY_VERSION,
                _GROUNDING_POLICY,
                f"mode={self.mode}",
                f"complexity={self.normalized_complexity()}",
                f"status={status or 'unknown'}",
                f"generated_status={self.generated_status or 'unknown'}",
                f"answer_len={max(0, int(self.answer_length))}",
                f"citations={max(0, int(self.citation_count))}",
                f"retrieval_confidence={confidence}",
                f"fast_path={int(bool(self.fast_path_used))}",
                f"has_context={int(bool(has_context))}",
                f"semantic_requested={int(bool(semantic_requested))}",
                f"decision={decision_reason}",
            )
        )


@dataclass(frozen=True)
class GroundingEvaluation:
    assessment: GroundingAssessment
    grounding_policy: str
    grounding_semantic_used: bool
    grounding_cache_hit: bool


@dataclass(frozen=True)
class _CachedGroundingResult:
    assessment: GroundingAssessment
    grounding_policy: str
    grounding_semantic_used: bool


_GROUNDING_RESULT_CACHE: dict[str, _CachedGroundingResult] = {}


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

    answer_ngrams = {
        answer_text[idx : idx + n] for idx in range(len(answer_text) - n + 1)
    }
    context_ngrams = {
        context_text[idx : idx + n] for idx in range(len(context_text) - n + 1)
    }
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


def _semantic_context_similarity(
    answer: str, context_chunks: list[str]
) -> float | None:
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
# Policy & cache helpers
# ---------------------------------------------------------------------------


def _clear_grounding_cache() -> None:
    """Testing hook: clear in-memory grounding result cache."""
    with _GROUNDING_CACHE_LOCK:
        _GROUNDING_RESULT_CACHE.clear()


def _default_policy(
    *,
    status: str,
    answer: str,
    citation_count: int,
) -> GroundingPolicy:
    return GroundingPolicy(
        mode="advanced",
        query_complexity="normal",
        generated_status=status or None,
        answer_length=len(answer.strip()),
        citation_count=max(0, int(citation_count)),
        retrieval_confidence=None,
        fast_path_used=False,
    )


def _mode_semantic_enabled(mode: GroundingMode) -> bool:
    if mode == "standard":
        return _GROUNDING_SEMANTIC_STANDARD_ENABLED
    if mode in {"advanced", "compare"}:
        return _GROUNDING_SEMANTIC_ADVANCED_ENABLED
    return False


def _should_use_semantic(
    policy: GroundingPolicy,
    *,
    lexical_score: float,
) -> tuple[bool, str]:
    mode = policy.mode
    complexity = policy.normalized_complexity()

    if not _GROUNDING_SEMANTIC_ENABLED:
        return False, "semantic_disabled_global"
    if policy.fast_path_used:
        return False, "fast_path_lexical_only"
    if _GROUNDING_POLICY in {"off", "lexical", "lexical_only"}:
        return False, "policy_lexical_only"
    if _GROUNDING_POLICY == "strict":
        if _mode_semantic_enabled(mode):
            return True, "strict_policy"
        return False, f"{mode}_semantic_disabled"

    if mode == "standard":
        if complexity == "simple":
            return False, "standard_simple_lexical_only"
        if not _mode_semantic_enabled(mode):
            return False, "standard_semantic_disabled"
        if complexity == "complex":
            return True, "standard_complex_semantic_allowed"

        risky_reasons: list[str] = []
        if policy.citation_count <= 0:
            risky_reasons.append("no_citations")
        if policy.answer_length >= _GROUNDING_STANDARD_LONG_ANSWER_CHARS:
            risky_reasons.append("long_answer")
        if (
            policy.retrieval_confidence is not None
            and float(policy.retrieval_confidence)
            < _GROUNDING_STANDARD_LOW_RETRIEVAL_CONFIDENCE
        ):
            risky_reasons.append("low_retrieval_confidence")
        if lexical_score < _GROUNDING_STANDARD_LOW_LEXICAL_SCORE:
            risky_reasons.append("low_lexical_score")

        if risky_reasons:
            return True, "standard_normal_risky:" + ",".join(risky_reasons)
        return False, "standard_normal_lexical_ok"

    if _mode_semantic_enabled(mode):
        return True, f"{mode}_semantic_allowed"
    return False, f"{mode}_semantic_disabled"


def _grounding_cache_key(
    answer: str,
    context_chunks: list[str],
    *,
    policy_version: str,
) -> str:
    normalized_contexts = [
        chunk.strip()
        for chunk in context_chunks
        if isinstance(chunk, str) and chunk.strip()
    ]
    payload = "\n".join(
        (
            answer.strip(),
            "\n".join(normalized_contexts),
            policy_version,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_get(cache_key: str) -> _CachedGroundingResult | None:
    with _GROUNDING_CACHE_LOCK:
        return _GROUNDING_RESULT_CACHE.get(cache_key)


def _cache_set(cache_key: str, value: _CachedGroundingResult) -> None:
    with _GROUNDING_CACHE_LOCK:
        _GROUNDING_RESULT_CACHE[cache_key] = value
        while len(_GROUNDING_RESULT_CACHE) > _GROUNDING_CACHE_MAX_SIZE:
            _GROUNDING_RESULT_CACHE.pop(next(iter(_GROUNDING_RESULT_CACHE)))


def _blended_grounded_score(
    answer: str,
    context_chunks: list[str],
    *,
    overlap_score: float,
    semantic_requested: bool,
) -> tuple[float, bool]:
    if not semantic_requested:
        return overlap_score, False

    semantic_score = _semantic_context_similarity(answer, context_chunks)
    if semantic_score is None or semantic_score < _GROUNDING_SEMANTIC_MIN_SIMILARITY:
        return overlap_score, False

    blended = ((1.0 - _GROUNDING_SEMANTIC_WEIGHT) * overlap_score) + (
        _GROUNDING_SEMANTIC_WEIGHT * semantic_score
    )
    final_score = round(max(overlap_score, min(1.0, blended)), 4)
    return final_score, True


def _classify_assessment(
    *,
    answer: str,
    score: float,
    citation_count: int,
    has_context: bool,
    normalized_status: str,
) -> GroundingAssessment:
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
    blended_score, _ = _blended_grounded_score(
        answer,
        context_chunks,
        overlap_score=overlap_score,
        semantic_requested=True,
    )
    return blended_score


def assess_grounding_with_policy(
    answer: str,
    context_chunks: list[str],
    *,
    citation_count: int = 0,
    has_selected_context: bool | None = None,
    status: str | None = None,
    policy: GroundingPolicy | None = None,
) -> GroundingEvaluation:
    """Assess grounding and return policy/caching diagnostics."""
    normalized_status = (status or "").strip().lower()
    has_context = (
        bool(context_chunks)
        if has_selected_context is None
        else bool(has_selected_context)
    )

    effective_policy = policy or _default_policy(
        status=normalized_status,
        answer=answer,
        citation_count=citation_count,
    )
    overlap_score = grounded_overlap_score(answer, context_chunks)
    semantic_requested, decision_reason = _should_use_semantic(
        effective_policy,
        lexical_score=overlap_score,
    )
    policy_label = (
        f"{_GROUNDING_POLICY}:{effective_policy.mode}:"
        f"{effective_policy.normalized_complexity()}:{decision_reason}"
    )
    cache_key = _grounding_cache_key(
        answer,
        context_chunks,
        policy_version=effective_policy.policy_version(
            status=normalized_status,
            has_context=has_context,
            semantic_requested=semantic_requested,
            decision_reason=decision_reason,
        ),
    )
    cached = _cache_get(cache_key)
    if cached is not None:
        return GroundingEvaluation(
            assessment=cached.assessment,
            grounding_policy=cached.grounding_policy,
            grounding_semantic_used=cached.grounding_semantic_used,
            grounding_cache_hit=True,
        )

    score, semantic_used = _blended_grounded_score(
        answer,
        context_chunks,
        overlap_score=overlap_score,
        semantic_requested=semantic_requested,
    )
    assessment = _classify_assessment(
        answer=answer,
        score=score,
        citation_count=max(0, int(citation_count)),
        has_context=has_context,
        normalized_status=normalized_status,
    )
    cached_value = _CachedGroundingResult(
        assessment=assessment,
        grounding_policy=policy_label,
        grounding_semantic_used=semantic_used,
    )
    _cache_set(cache_key, cached_value)
    return GroundingEvaluation(
        assessment=assessment,
        grounding_policy=policy_label,
        grounding_semantic_used=semantic_used,
        grounding_cache_hit=False,
    )


def assess_grounding(
    answer: str,
    context_chunks: list[str],
    *,
    citation_count: int = 0,
    has_selected_context: bool | None = None,
    status: str | None = None,
) -> GroundingAssessment:
    """Assess grounding and hallucination risk from answer/context signals."""
    return assess_grounding_with_policy(
        answer,
        context_chunks,
        citation_count=citation_count,
        has_selected_context=has_selected_context,
        status=status,
        policy=None,
    ).assessment


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
