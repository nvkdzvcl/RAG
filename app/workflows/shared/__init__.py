"""Shared workflow helpers — backward-compatible re-export layer.

This package was split from the original monolithic ``shared.py`` into
focused sub-modules:

* ``language`` — query normalization, language detection, mismatch
* ``grounding`` — grounding scoring, hallucination detection
* ``history`` — chat-history trimming and rendering

All public symbols are re-exported here so that existing
``from app.workflows.shared import ...`` statements keep working.
"""

from __future__ import annotations

# -- language ---------------------------------------------------------------
from app.workflows.shared.language import (
    ResponseLanguage,
    build_language_system_prompt,
    detect_response_language,
    is_language_mismatch,
    localized_insufficient_evidence,
    normalize_query,
    response_language_name,
)

# -- grounding --------------------------------------------------------------
from app.workflows.shared.grounding import (
    GroundingAssessment,
    assess_grounding,
    detect_hallucination,
    grounded_overlap_score,
    grounded_score,
)

# -- history ----------------------------------------------------------------
from app.workflows.shared.history import (
    build_chat_history_context,
    trim_chat_history,
)

__all__ = [
    # language
    "ResponseLanguage",
    "build_language_system_prompt",
    "detect_response_language",
    "is_language_mismatch",
    "localized_insufficient_evidence",
    "normalize_query",
    "response_language_name",
    # grounding
    "GroundingAssessment",
    "assess_grounding",
    "detect_hallucination",
    "grounded_overlap_score",
    "grounded_score",
    # history
    "build_chat_history_context",
    "trim_chat_history",
]
