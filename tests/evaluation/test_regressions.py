"""Regression checks for three-mode API output contracts."""

import json
from pathlib import Path

from app.evaluation.runner import EvaluationRunner, stub_predictor, workflow_predictor
from app.schemas.api import (
    AdvancedQueryResponse,
    CompareQueryResponse,
    StandardQueryResponse,
    validate_query_response,
)
from app.schemas.common import Mode


def _load_regression_payloads() -> dict:
    fixture = Path("tests/fixtures/regression_outputs.json")
    return json.loads(fixture.read_text(encoding="utf-8"))


def test_standard_regression_payload_shape() -> None:
    payload = _load_regression_payloads()["standard"]
    parsed = validate_query_response(payload)
    assert isinstance(parsed, StandardQueryResponse)
    assert parsed.mode == "standard"


def test_advanced_regression_payload_shape() -> None:
    payload = _load_regression_payloads()["advanced"]
    parsed = validate_query_response(payload)
    assert isinstance(parsed, AdvancedQueryResponse)
    assert parsed.loop_count == 1


def test_compare_regression_payload_shape() -> None:
    payload = _load_regression_payloads()["compare"]
    parsed = validate_query_response(payload)
    assert isinstance(parsed, CompareQueryResponse)
    assert parsed.standard.mode == "standard"
    assert parsed.advanced.mode == "advanced"
    assert parsed.comparison.confidence_delta == 0.18


def test_evaluation_runner_with_stub_predictor() -> None:
    runner = EvaluationRunner(
        dataset_path=Path("data/eval/golden.jsonl"),
        predictor=stub_predictor,
    )
    report = runner.run()

    assert report.dataset_size >= 6
    assert report.mode_runs == report.dataset_size * 3
    assert report.invalid_payloads == 0


def test_workflow_predictor_standard_regression_payload_shape() -> None:
    payload = workflow_predictor("What does standard mode do?", Mode.STANDARD)
    parsed = validate_query_response(payload)

    assert isinstance(parsed, StandardQueryResponse)
    assert parsed.mode == "standard"
    assert parsed.answer


def test_workflow_predictor_advanced_regression_payload_shape() -> None:
    payload = workflow_predictor("How does advanced mode improve reliability?", Mode.ADVANCED)
    parsed = validate_query_response(payload)

    assert isinstance(parsed, AdvancedQueryResponse)
    assert parsed.mode == "advanced"
    assert parsed.trace is not None


def test_workflow_predictor_compare_regression_payload_shape() -> None:
    payload = workflow_predictor("Compare standard and advanced mode", Mode.COMPARE)
    parsed = validate_query_response(payload)

    assert isinstance(parsed, CompareQueryResponse)
    assert parsed.mode == "compare"
    assert parsed.standard.mode == "standard"
    assert parsed.advanced.mode == "advanced"


def test_evaluation_runner_with_workflow_predictor_schema_valid() -> None:
    runner = EvaluationRunner(
        dataset_path=Path("data/eval/golden.jsonl"),
        predictor=workflow_predictor,
    )
    report = runner.run(modes=[Mode.STANDARD, Mode.ADVANCED, Mode.COMPARE])

    assert report.dataset_size >= 6
    assert report.mode_runs == report.dataset_size * 3
    assert report.invalid_payloads == 0
    assert report.schema_valid_rate == 1.0
