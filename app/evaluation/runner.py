"""Practical evaluation runner for standard, advanced, and compare modes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from app.evaluation.dataset import load_eval_dataset
from app.evaluation.metrics import compute_metrics, extract_trace_fields
from app.evaluation.reporting import build_comparative_summary, write_report_artifacts
from app.evaluation.schemas import CompareEvalOutput, EvalExample, EvalReport, ModeEvalOutput
from app.schemas.api import CompareQueryResponse, QueryResponse, validate_query_response
from app.schemas.common import Citation, Mode
from app.workflows.runner import WorkflowRunner


class ModePredictor(Protocol):
    """Callable predictor contract for one query/mode pair."""

    def __call__(self, query: str, mode: Mode) -> dict[str, Any]:
        """Return query response payload as dict."""


class WorkflowPredictor:
    """Predictor backed by real workflow execution."""

    def __init__(self, workflow_runner: WorkflowRunner | None = None) -> None:
        self.workflow_runner = workflow_runner or WorkflowRunner()

    def __call__(self, query: str, mode: Mode) -> dict[str, Any]:
        response = self.workflow_runner.run(query=query, mode=mode, chat_history=None)
        if hasattr(response, "model_dump"):
            return response.model_dump()  # type: ignore[no-any-return]
        return dict(response)


class StubPredictor:
    """Deterministic offline predictor for fast local eval plumbing checks."""

    _source = "stub/golden"

    def _standard_payload(self, query: str) -> dict[str, Any]:
        return {
            "mode": "standard",
            "answer": f"Stub standard answer for: {query}",
            "citations": [{"chunk_id": "stub_c1", "doc_id": "stub_d1", "source": self._source}],
            "confidence": 0.55,
            "status": "answered",
            "stop_reason": "stub_generated",
            "latency_ms": 12,
            "trace": [
                {"step": "retrieve", "count": 2, "chunk_ids": ["stub_c1", "stub_c2"]},
                {
                    "step": "rerank",
                    "count": 2,
                    "docs": [
                        {"chunk_id": "stub_c1", "rerank_score": 0.8, "score": 0.8},
                        {"chunk_id": "stub_c2", "rerank_score": 0.3, "score": 0.3},
                    ],
                },
                {
                    "step": "context_select",
                    "count": 1,
                    "chunk_ids": ["stub_c1"],
                    "docs": [
                        {
                            "chunk_id": "stub_c1",
                            "doc_id": "stub_d1",
                            "content": "Stub context for grounded answer generation.",
                        }
                    ],
                },
            ],
        }

    def _advanced_payload(self, query: str) -> dict[str, Any]:
        retry_loop = 2 if "retry" in query.lower() else 1
        return {
            "mode": "advanced",
            "answer": f"Stub advanced answer for: {query}",
            "citations": [{"chunk_id": "stub_c1", "doc_id": "stub_d1", "source": self._source}],
            "confidence": 0.72,
            "status": "answered",
            "stop_reason": "critique_pass",
            "latency_ms": 24,
            "loop_count": retry_loop,
            "trace": [
                {"step": "retrieval_gate", "need_retrieval": True, "reason": "stub_default"},
                {
                    "step": "loop",
                    "loop": retry_loop,
                    "query": query,
                    "retrieved_count": 2,
                    "reranked_count": 2,
                    "reranked_docs": [
                        {"chunk_id": "stub_c1", "rerank_score": 0.91},
                        {"chunk_id": "stub_c2", "rerank_score": 0.2},
                    ],
                    "selected_count": 1,
                    "selected_context_docs": [
                        {
                            "chunk_id": "stub_c1",
                            "doc_id": "stub_d1",
                            "content": "Stub advanced context for critique and refinement.",
                        }
                    ],
                },
            ],
        }

    def __call__(self, query: str, mode: Mode) -> dict[str, Any]:
        if mode == Mode.STANDARD:
            return self._standard_payload(query)
        if mode == Mode.ADVANCED:
            return self._advanced_payload(query)
        if mode == Mode.COMPARE:
            standard = self._standard_payload(query)
            advanced = self._advanced_payload(query)
            return {
                "mode": "compare",
                "standard": standard,
                "advanced": advanced,
                "comparison": {
                    "confidence_delta": advanced["confidence"] - standard["confidence"],
                    "latency_delta_ms": advanced["latency_ms"] - standard["latency_ms"],
                    "citation_delta": len(advanced["citations"]) - len(standard["citations"]),
                    "note": "stub compare",
                },
            }
        raise ValueError(f"Unsupported mode: {mode}")


def create_predictor(name: str) -> ModePredictor:
    """Resolve predictor strategy from CLI/runtime configuration."""
    normalized = name.strip().lower()
    if normalized == "workflow":
        return WorkflowPredictor()
    if normalized == "stub":
        return StubPredictor()
    raise ValueError(f"Unsupported predictor '{name}'.")


def _collect_mode_eval_output(
    *,
    example: EvalExample,
    mode: Mode,
    response: QueryResponse,
    run_source: str = "direct",
) -> ModeEvalOutput:
    if mode not in {Mode.STANDARD, Mode.ADVANCED}:
        raise ValueError("Mode output collection only supports standard/advanced.")

    answer = getattr(response, "answer", "") if response is not None else ""
    citations = list(getattr(response, "citations", []))
    confidence = getattr(response, "confidence", None)
    status = getattr(response, "status", None)
    loop_count = getattr(response, "loop_count", None)
    stop_reason = getattr(response, "stop_reason", None)
    latency_ms = getattr(response, "latency_ms", None)
    trace = list(getattr(response, "trace", []))

    trace_fields = extract_trace_fields(trace)
    metrics = compute_metrics(
        expected_behavior=example.expected_behavior,
        answer=answer,
        citations=citations,
        confidence=confidence,
        status=status,
        loop_count=loop_count,
        stop_reason=stop_reason,
        latency_ms=latency_ms,
        trace_fields=trace_fields,
        reference_answer=example.reference_answer,
        gold_sources=example.gold_sources,
    )

    return ModeEvalOutput(
        example_id=example.id,
        mode=mode,
        question=example.question,
        category=example.category,
        expected_behavior=example.expected_behavior,
        answer=answer,
        citations=[Citation.model_validate(citation) for citation in citations],
        confidence=confidence,
        status=status,
        retrieved_chunk_ids=trace_fields.retrieved_chunk_ids,
        rerank_scores=trace_fields.rerank_scores,
        loop_count=loop_count,
        stop_reason=stop_reason,
        latency_ms=latency_ms,
        retrieved_count=trace_fields.retrieved_count,
        selected_context_count=trace_fields.selected_context_count,
        metrics=metrics,
        trace=trace,
        run_source=run_source,  # type: ignore[arg-type]
    )


def _error_mode_output(example: EvalExample, mode: Mode, error: str, run_source: str = "direct") -> ModeEvalOutput:
    trace_fields = extract_trace_fields([])
    metrics = compute_metrics(
        expected_behavior=example.expected_behavior,
        answer="",
        citations=[],
        confidence=None,
        status="error",
        loop_count=None,
        stop_reason="evaluation_error",
        latency_ms=None,
        trace_fields=trace_fields,
        reference_answer=example.reference_answer,
        gold_sources=example.gold_sources,
    )
    return ModeEvalOutput(
        example_id=example.id,
        mode=mode,
        question=example.question,
        category=example.category,
        expected_behavior=example.expected_behavior,
        answer="",
        citations=[],
        confidence=None,
        status="error",
        retrieved_chunk_ids=[],
        rerank_scores={},
        loop_count=None,
        stop_reason="evaluation_error",
        latency_ms=None,
        retrieved_count=0,
        selected_context_count=0,
        metrics=metrics,
        trace=[],
        run_source=run_source,  # type: ignore[arg-type]
        error=error,
    )


class EvaluationRunner:
    """Runs repeatable mode evaluations and writes report artifacts."""

    def __init__(
        self,
        *,
        dataset_path: Path,
        output_dir: Path = Path("data/eval/results"),
        predictor: ModePredictor | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.predictor = predictor or WorkflowPredictor()

    def _run_mode(self, example: EvalExample, mode: Mode) -> dict[str, Any]:
        payload = self.predictor(example.question, mode)
        parsed = validate_query_response(payload)
        return {"payload": payload, "parsed": parsed}

    def run(self, modes: list[Mode] | None = None) -> EvalReport:
        selected_modes = modes or [Mode.STANDARD, Mode.ADVANCED]
        dataset = load_eval_dataset(self.dataset_path)

        mode_outputs: list[ModeEvalOutput] = []
        compare_outputs: list[CompareEvalOutput] = []

        for example in dataset:
            for mode in selected_modes:
                try:
                    result = self._run_mode(example, mode)
                    parsed = result["parsed"]
                except Exception as exc:
                    if mode == Mode.COMPARE:
                        compare_outputs.append(
                            CompareEvalOutput(
                                example_id=example.id,
                                question=example.question,
                                category=example.category,
                                expected_behavior=example.expected_behavior,
                                standard=_error_mode_output(
                                    example,
                                    Mode.STANDARD,
                                    error=f"Compare mode failed: {exc}",
                                    run_source="compare_branch",
                                ),
                                advanced=_error_mode_output(
                                    example,
                                    Mode.ADVANCED,
                                    error=f"Compare mode failed: {exc}",
                                    run_source="compare_branch",
                                ),
                                comparison={"error": str(exc)},
                            )
                        )
                    elif mode in {Mode.STANDARD, Mode.ADVANCED}:
                        mode_outputs.append(_error_mode_output(example, mode, error=str(exc)))
                    continue

                if mode in {Mode.STANDARD, Mode.ADVANCED}:
                    mode_outputs.append(
                        _collect_mode_eval_output(
                            example=example,
                            mode=mode,
                            response=parsed,
                            run_source="direct",
                        )
                    )
                    continue

                if mode == Mode.COMPARE and isinstance(parsed, CompareQueryResponse):
                    standard_eval = _collect_mode_eval_output(
                        example=example,
                        mode=Mode.STANDARD,
                        response=parsed.standard,
                        run_source="compare_branch",
                    )
                    advanced_eval = _collect_mode_eval_output(
                        example=example,
                        mode=Mode.ADVANCED,
                        response=parsed.advanced,
                        run_source="compare_branch",
                    )
                    compare_outputs.append(
                        CompareEvalOutput(
                            example_id=example.id,
                            question=example.question,
                            category=example.category,
                            expected_behavior=example.expected_behavior,
                            standard=standard_eval,
                            advanced=advanced_eval,
                            comparison=parsed.comparison.model_dump(),
                        )
                    )

        summary = build_comparative_summary(mode_outputs, compare_outputs)
        report = EvalReport(
            dataset_path=str(self.dataset_path),
            generated_at=datetime.now(timezone.utc),
            modes=selected_modes,
            dataset_size=len(dataset),
            output_count=len(mode_outputs) + len(compare_outputs),
            standard_advanced_summary=summary,
            mode_outputs=mode_outputs,
            compare_outputs=compare_outputs,
            artifacts={},
        )
        return write_report_artifacts(report, self.output_dir)


def _parse_modes(raw_modes: list[str]) -> list[Mode]:
    return [Mode(value) for value in raw_modes]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options for evaluation runs."""
    parser = argparse.ArgumentParser(description="Run practical evaluation for Self-RAG modes.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/golden_dataset.jsonl"),
        help="Path to JSONL evaluation dataset.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[mode.value for mode in Mode],
        default=[Mode.STANDARD.value, Mode.ADVANCED.value],
        help="Modes to evaluate. Add 'compare' to evaluate compare mode as well.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval/results"),
        help="Directory for JSON/Markdown/CSV outputs.",
    )
    parser.add_argument(
        "--predictor",
        choices=["workflow", "stub"],
        default="workflow",
        help="Prediction backend: real workflows or deterministic stub payloads.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for `python -m app.evaluation.runner`."""
    args = parse_args(argv)
    runner = EvaluationRunner(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        predictor=create_predictor(args.predictor),
    )
    report = runner.run(modes=_parse_modes(args.modes))
    print("Evaluation complete.")
    print(f"- JSON: {report.artifacts.get('results_json', '')}")
    print(f"- Markdown: {report.artifacts.get('report_md', '')}")
    print(f"- CSV: {report.artifacts.get('summary_csv', '')}")


if __name__ == "__main__":
    main()
