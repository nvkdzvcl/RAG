"""Timing consistency tests across workflow modes."""

from __future__ import annotations

from typing import Any

from app.schemas.api import AdvancedQueryResponse, CompareQueryResponse, StandardQueryResponse
from app.schemas.common import Mode
from app.workflows.runner import WorkflowRunner


def _timing_summary(trace: list[dict[str, Any]]) -> dict[str, Any]:
    return next(step for step in trace if step.get("step") == "timing_summary")


def _assert_non_negative_ms(trace: list[dict[str, Any]]) -> None:
    for step in trace:
        for key, value in step.items():
            if not key.endswith("_ms"):
                continue
            if isinstance(value, (int, float)):
                assert value >= 0, f"negative timing detected for {key}: {value}"


def _assert_common_timing_shape(summary: dict[str, Any]) -> None:
    for key in ("total_ms", "llm_generate_ms", "retrieval_total_ms"):
        assert key in summary
        assert isinstance(summary[key], int)
        assert summary[key] >= 0
    assert isinstance(summary.get("timing_breakdown_available"), bool)


def test_timing_shape_standard_advanced_compare() -> None:
    runner = WorkflowRunner()

    standard = runner.run("timing consistency standard", mode=Mode.STANDARD)
    assert isinstance(standard, StandardQueryResponse)
    standard_summary = _timing_summary(standard.trace)
    _assert_common_timing_shape(standard_summary)
    assert isinstance(standard_summary.get("llm_call_count_estimate"), int)
    _assert_non_negative_ms(standard.trace)
    assert standard.trace and standard.trace[-1].get("step") == "completed"

    advanced = runner.run("timing consistency advanced", mode=Mode.ADVANCED)
    assert isinstance(advanced, AdvancedQueryResponse)
    advanced_summary = _timing_summary(advanced.trace)
    _assert_common_timing_shape(advanced_summary)
    for key in ("standard_pipeline_ms", "critique_ms", "retrieval_gate_ms", "refine_ms"):
        assert key in advanced_summary
        assert isinstance(advanced_summary[key], int)
        assert advanced_summary[key] >= 0
    assert isinstance(advanced_summary.get("llm_call_count_estimate"), int)
    _assert_non_negative_ms(advanced.trace)
    assert advanced.trace and advanced.trace[-1].get("step") == "completed"

    compare = runner.run("timing consistency compare", mode=Mode.COMPARE)
    assert isinstance(compare, CompareQueryResponse)
    standard_branch_summary = _timing_summary(compare.standard.trace)
    advanced_branch_summary = _timing_summary(compare.advanced.trace)
    _assert_common_timing_shape(standard_branch_summary)
    _assert_common_timing_shape(advanced_branch_summary)
    assert isinstance(advanced_branch_summary.get("standard_pipeline_ms"), int)
    assert isinstance(advanced_branch_summary.get("critique_ms"), int)
    assert isinstance(advanced_branch_summary.get("retrieval_gate_ms"), int)
    assert isinstance(advanced_branch_summary.get("refine_ms"), int)
    _assert_non_negative_ms(compare.standard.trace)
    _assert_non_negative_ms(compare.advanced.trace)
    assert compare.standard.trace and compare.standard.trace[-1].get("step") == "completed"
    assert compare.advanced.trace and compare.advanced.trace[-1].get("step") == "completed"
