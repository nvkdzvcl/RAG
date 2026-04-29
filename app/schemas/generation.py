"""Schemas for baseline answer generation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import Citation


GenerationStatus = Literal["answered", "partial", "insufficient_evidence"]


class ParsedAnswer(BaseModel):
    """Structured answer payload parsed from model output."""

    answer: str
    confidence: float | None = None
    status: GenerationStatus = "answered"


class GeneratedAnswer(BaseModel):
    """Final generation payload used by workflows/API mapping layers."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float | None = None
    status: GenerationStatus = "answered"
    stop_reason: str | None = None
    raw_output: str | None = None
    llm_fallback_used: bool = False
    llm_cache_hit: bool = False
