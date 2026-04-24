"""Evaluation runner for schema and behavior regressions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.evaluation.dataset import load_eval_dataset
from app.evaluation.schemas import EvalCaseResult, EvalReport
from app.schemas.api import CompareQueryResponse, validate_query_response
from app.schemas.common import Mode
from app.workflows.runner import WorkflowRunner

Predictor = Callable[[str, Mode], dict[str, Any]]


def _status_matches(expected_behavior: str, status: str | None) -> bool:
    normalized = (status or "answered").lower()
    if expected_behavior == "abstain":
        return normalized in {"abstained", "insufficient_evidence"}
    if expected_behavior == "partial":
        return normalized in {"partial", "answered"}
    return normalized in {"answered", "partial"}


def stub_predictor(query: str, mode: Mode) -> dict[str, Any]:
    """Deterministic predictor used before workflow implementation is complete."""
    standard = {
        "mode": "standard",
        "answer": f"Stub standard answer for: {query}",
        "citations": [
            {
                "chunk_id": "chunk_stub_001",
                "doc_id": "doc_stub",
                "source": "docs/stub.md",
                "title": "Stub Document",
            }
        ],
        "confidence": 0.6,
        "status": "answered",
        "stop_reason": "stub",
        "latency_ms": 120,
    }

    advanced = {
        "mode": "advanced",
        "answer": f"Stub advanced answer for: {query}",
        "citations": [
            {
                "chunk_id": "chunk_stub_001",
                "doc_id": "doc_stub",
                "source": "docs/stub.md",
                "title": "Stub Document",
            }
        ],
        "confidence": 0.72,
        "status": "answered",
        "stop_reason": "stub",
        "latency_ms": 220,
        "loop_count": 1,
        "trace": [{"step": "critique", "decision": "stop"}],
    }

    if mode == Mode.STANDARD:
        return standard
    if mode == Mode.ADVANCED:
        return advanced
    if mode == Mode.COMPARE:
        return {
            "mode": "compare",
            "standard": standard,
            "advanced": advanced,
            "comparison": {
                "confidence_delta": 0.12,
                "latency_delta_ms": 100,
                "citation_delta": 0,
                "note": "Advanced has higher confidence in stub mode.",
            },
        }
    raise ValueError(f"Unsupported mode: {mode}")


def workflow_predictor(query: str, mode: Mode) -> dict[str, Any]:
    """Predict using the real workflow runner."""
    runner = WorkflowRunner()
    response = runner.run(query=query, mode=mode, chat_history=None)
    if hasattr(response, "model_dump"):
        return response.model_dump()  # type: ignore[no-any-return]
    return dict(response)


class EvaluationRunner:
    """Runs evaluation examples across one or more modes."""

    def __init__(self, dataset_path: Path, predictor: Predictor) -> None:
        self.dataset_path = dataset_path
        self.predictor = predictor

    def run(self, modes: list[Mode] | None = None) -> EvalReport:
        selected_modes = modes or [Mode.STANDARD, Mode.ADVANCED, Mode.COMPARE]
        dataset = load_eval_dataset(self.dataset_path)

        results: list[EvalCaseResult] = []
        invalid_payloads = 0
        behavior_matches = 0

        for example in dataset:
            for mode in selected_modes:
                try:
                    payload = self.predictor(example.question, mode)
                    parsed = validate_query_response(payload)
                    valid_schema = True
                    status = None
                    if isinstance(parsed, CompareQueryResponse):
                        status = "answered"
                    else:
                        status = parsed.status
                    behavior_match = _status_matches(example.expected_behavior, status)
                except (ValidationError, ValueError, TypeError, NotImplementedError) as exc:
                    valid_schema = False
                    behavior_match = False
                    status = None
                    invalid_payloads += 1
                    results.append(
                        EvalCaseResult(
                            example_id=example.id,
                            mode=mode,
                            valid_schema=False,
                            behavior_match=False,
                            error=str(exc),
                        )
                    )
                    continue

                if behavior_match:
                    behavior_matches += 1

                results.append(
                    EvalCaseResult(
                        example_id=example.id,
                        mode=mode,
                        valid_schema=valid_schema,
                        behavior_match=behavior_match,
                        status=status,
                    )
                )

        mode_runs = len(dataset) * len(selected_modes)
        valid_runs = sum(1 for item in results if item.valid_schema)

        return EvalReport(
            dataset_size=len(dataset),
            mode_runs=mode_runs,
            schema_valid_rate=(valid_runs / mode_runs) if mode_runs else 0.0,
            behavior_match_rate=(behavior_matches / mode_runs) if mode_runs else 0.0,
            invalid_payloads=invalid_payloads,
            results=results,
        )


def run_evaluation(dataset_path: Path, predictor: Predictor, modes: list[Mode] | None = None) -> EvalReport:
    """Convenience function for running an evaluation sweep."""
    return EvaluationRunner(dataset_path=dataset_path, predictor=predictor).run(modes=modes)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Self-RAG evaluation checks.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/golden.jsonl"),
        help="Path to JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--predictor",
        choices=["stub", "workflow"],
        default="stub",
        help="Predictor backend used for evaluation.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[mode.value for mode in Mode],
        default=[mode.value for mode in Mode],
        help="Modes to evaluate (default: standard advanced compare).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = _parse_args()
    predictor = stub_predictor if args.predictor == "stub" else workflow_predictor
    selected_modes = [Mode(mode) for mode in args.modes]
    report = run_evaluation(
        dataset_path=args.dataset,
        predictor=predictor,
        modes=selected_modes,
    )
    print(json.dumps(report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
