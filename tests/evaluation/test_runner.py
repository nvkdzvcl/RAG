"""Evaluation runner tests with mocked workflow responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.evaluation.runner import EvaluationRunner, create_predictor, parse_args
from app.schemas.common import Mode


def _write_dataset(path: Path) -> None:
    rows = [
        {
            "id": "eval_mock_001",
            "question": "How does advanced mode work?",
            "expected_behavior": "answer",
            "reference_answer": "Advanced mode can retry retrieval.",
            "gold_sources": ["docs/MODES.md"],
            "category": "multi_hop",
            "notes": "mock",
        },
        {
            "id": "eval_mock_002",
            "question": "force retry retrieval loop for diagnostics",
            "expected_behavior": "retry",
            "reference_answer": None,
            "gold_sources": [],
            "category": "vietnamese",
            "notes": "mock",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mock_predictor(query: str, mode: Mode) -> dict[str, Any]:
    if mode == Mode.STANDARD:
        return {
            "mode": "standard",
            "answer": f"Standard answer: {query}",
            "citations": [
                {"chunk_id": "c1", "doc_id": "d1", "source": "docs/MODES.md"}
            ],
            "confidence": 0.55,
            "status": "answered",
            "stop_reason": "generated",
            "latency_ms": 100,
            "trace": [
                {"step": "retrieve", "count": 3, "chunk_ids": ["c1", "c2", "c3"]},
                {
                    "step": "rerank",
                    "count": 2,
                    "docs": [
                        {"chunk_id": "c1", "rerank_score": 0.88, "score": 0.88},
                        {"chunk_id": "c2", "rerank_score": 0.42, "score": 0.42},
                    ],
                },
                {
                    "step": "context_select",
                    "count": 2,
                    "chunk_ids": ["c1", "c2"],
                    "docs": [
                        {
                            "chunk_id": "c1",
                            "doc_id": "d1",
                            "content": "Advanced mode has retry steps.",
                        },
                        {
                            "chunk_id": "c2",
                            "doc_id": "d2",
                            "content": "Standard mode is faster.",
                        },
                    ],
                },
            ],
        }
    if mode == Mode.ADVANCED:
        return {
            "mode": "advanced",
            "answer": f"Advanced answer: {query}",
            "citations": [
                {"chunk_id": "c1", "doc_id": "d1", "source": "docs/MODES.md"}
            ],
            "confidence": 0.77,
            "status": "answered",
            "stop_reason": "critique_pass",
            "latency_ms": 180,
            "loop_count": 2 if "force retry" in query else 1,
            "trace": [
                {
                    "step": "retrieval_gate",
                    "need_retrieval": True,
                    "reason": "default_retrieval",
                },
                {
                    "step": "loop",
                    "loop": 1,
                    "query": query,
                    "retrieved_count": 3,
                    "reranked_count": 2,
                    "reranked_docs": [
                        {"chunk_id": "c1", "rerank_score": 0.93},
                        {"chunk_id": "c2", "rerank_score": 0.39},
                    ],
                    "selected_count": 1,
                    "selected_context_docs": [
                        {
                            "chunk_id": "c1",
                            "doc_id": "d1",
                            "content": "Retry helps reliability.",
                        },
                    ],
                },
            ],
        }
    if mode == Mode.COMPARE:
        standard = _mock_predictor(query, Mode.STANDARD)
        advanced = _mock_predictor(query, Mode.ADVANCED)
        return {
            "mode": "compare",
            "standard": standard,
            "advanced": advanced,
            "comparison": {
                "confidence_delta": (advanced["confidence"] - standard["confidence"]),
                "latency_delta_ms": (advanced["latency_ms"] - standard["latency_ms"]),
                "citation_delta": 0,
                "note": "mock compare",
            },
        }
    raise ValueError(f"Unsupported mode: {mode}")


def test_evaluation_runner_with_mocked_workflows(tmp_path: Path) -> None:
    dataset_path = tmp_path / "golden_dataset.jsonl"
    output_dir = tmp_path / "results"
    _write_dataset(dataset_path)

    runner = EvaluationRunner(
        dataset_path=dataset_path,
        output_dir=output_dir,
        predictor=_mock_predictor,
    )
    report = runner.run(modes=[Mode.STANDARD, Mode.ADVANCED, Mode.COMPARE])

    assert report.dataset_size == 2
    assert len(report.mode_outputs) == 4
    assert len(report.compare_outputs) == 2
    assert report.standard_advanced_summary.paired_count == 2
    assert report.standard_advanced_summary.avg_latency_delta_ms == 80.0
    assert report.artifacts["report_md"].endswith("report.md")
    assert Path(report.artifacts["results_json"]).exists()
    assert Path(report.artifacts["report_md"]).exists()
    assert Path(report.artifacts["summary_csv"]).exists()

    first_standard = next(
        item for item in report.mode_outputs if item.mode == Mode.STANDARD
    )
    assert "c1" in first_standard.retrieved_chunk_ids
    assert first_standard.rerank_scores["c1"] == 0.88


def test_stub_predictor_runs_without_workflow_stack(tmp_path: Path) -> None:
    dataset_path = tmp_path / "golden_dataset.jsonl"
    output_dir = tmp_path / "results_stub"
    _write_dataset(dataset_path)

    runner = EvaluationRunner(
        dataset_path=dataset_path,
        output_dir=output_dir,
        predictor=create_predictor("stub"),
    )
    report = runner.run(modes=[Mode.STANDARD, Mode.ADVANCED, Mode.COMPARE])

    assert report.dataset_size == 2
    assert len(report.mode_outputs) == 4
    assert len(report.compare_outputs) == 2
    assert Path(report.artifacts["results_json"]).exists()
    assert Path(report.artifacts["report_md"]).exists()


def test_parse_args_accepts_predictor_flag() -> None:
    args = parse_args(
        [
            "--dataset",
            "data/eval/golden.jsonl",
            "--predictor",
            "stub",
            "--modes",
            "standard",
            "advanced",
        ]
    )
    assert args.predictor == "stub"
    assert str(args.dataset).endswith("data/eval/golden.jsonl")
