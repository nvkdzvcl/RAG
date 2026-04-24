"""Typed schemas for evaluation inputs and outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import Mode


class EvalExample(BaseModel):
    """One golden evaluation item."""

    id: str
    question: str
    expected_behavior: Literal["answer", "abstain", "partial"]
    reference_answer: str | None = None
    gold_sources: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None


class EvalCaseResult(BaseModel):
    """Evaluation result for one question/mode pair."""

    example_id: str
    mode: Mode
    valid_schema: bool
    behavior_match: bool
    status: str | None = None
    error: str | None = None


class EvalReport(BaseModel):
    """Summary report for an evaluation run."""

    dataset_size: int
    mode_runs: int
    schema_valid_rate: float
    behavior_match_rate: float
    invalid_payloads: int
    results: list[EvalCaseResult] = Field(default_factory=list)
