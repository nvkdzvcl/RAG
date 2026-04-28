"""Tests for evaluation summary and report artifact generation."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.evaluation.reporting import (
    build_comparative_summary,
    report_to_markdown,
    write_report_artifacts,
)
from app.evaluation.schemas import (
    ComparativeSummary,
    EvalMetrics,
    EvalReport,
    ModeEvalOutput,
    RetrievalModeMetrics,
)
from app.schemas.common import Mode


def _build_mode_output(
    example_id: str, mode: Mode, latency_ms: int, confidence: float, retry: bool
) -> ModeEvalOutput:
    return ModeEvalOutput(
        example_id=example_id,
        mode=mode,
        question="q",
        category="simple",
        expected_behavior="answer",
        answer="answer",
        citations=[],
        confidence=confidence,
        status="answered",
        retrieved_chunk_ids=["c1"],
        rerank_scores={"c1": 0.5},
        loop_count=2 if retry else 1,
        stop_reason="ok",
        latency_ms=latency_ms,
        retrieved_count=1,
        selected_context_count=1,
        metrics=EvalMetrics(
            citation_count=0,
            has_citations=False,
            abstain_match=True,
            retry_used=retry,
            latency_ms=latency_ms,
            confidence=confidence,
            retrieved_count=1,
            selected_context_count=1,
            answer_non_empty=True,
            answer_contains_reference_keywords=None,
            cited_gold_source_overlap=None,
            groundedness_proxy=0.2,
            groundedness_proxy_note="proxy",
            retrieval_by_mode={
                "dense": RetrievalModeMetrics(hit=True, mrr=0.5, ndcg=0.5),
                "bm25": RetrievalModeMetrics(hit=True, mrr=0.5, ndcg=0.5),
                "hybrid": RetrievalModeMetrics(hit=True, mrr=0.5, ndcg=0.5),
                "hybrid_rerank": RetrievalModeMetrics(hit=True, mrr=1.0, ndcg=1.0),
            },
        ),
        trace=[],
        run_source="direct",
    )


def test_build_comparative_summary_and_markdown() -> None:
    mode_outputs = [
        _build_mode_output(
            "e1", Mode.STANDARD, latency_ms=100, confidence=0.5, retry=False
        ),
        _build_mode_output(
            "e1", Mode.ADVANCED, latency_ms=180, confidence=0.7, retry=True
        ),
    ]
    summary = build_comparative_summary(mode_outputs, compare_outputs=[])

    assert isinstance(summary, ComparativeSummary)
    assert summary.paired_count == 1
    assert summary.avg_latency_delta_ms == 80.0
    assert summary.avg_confidence_delta == pytest.approx(0.2)
    assert summary.advanced_retry_rate == 1.0

    report = EvalReport(
        dataset_path="data/eval/golden_dataset.jsonl",
        generated_at=datetime.now(timezone.utc),
        modes=[Mode.STANDARD, Mode.ADVANCED],
        dataset_size=1,
        output_count=2,
        standard_advanced_summary=summary,
        mode_outputs=mode_outputs,
        compare_outputs=[],
        artifacts={},
    )

    markdown = report_to_markdown(report)
    assert "Evaluation Report" in markdown
    assert "Standard vs Advanced" in markdown
    assert "Per-Category Summary" in markdown


def test_report_includes_hybrid_rerank_mode_metrics() -> None:
    mode_outputs = [
        _build_mode_output(
            "e1", Mode.STANDARD, latency_ms=100, confidence=0.5, retry=False
        ),
        _build_mode_output(
            "e1", Mode.ADVANCED, latency_ms=180, confidence=0.7, retry=True
        ),
    ]
    summary = build_comparative_summary(mode_outputs, compare_outputs=[])

    assert "hybrid_rerank" in summary.retrieval_by_mode
    hybrid_rerank = summary.retrieval_by_mode["hybrid_rerank"]
    assert hybrid_rerank.count == 2
    assert hybrid_rerank.hit_rate == 1.0
    assert hybrid_rerank.avg_mrr == 1.0
    assert hybrid_rerank.avg_ndcg == 1.0
    markdown = report_to_markdown(
        EvalReport(
            dataset_path="data/eval/golden_dataset.jsonl",
            generated_at=datetime.now(timezone.utc),
            modes=[Mode.STANDARD, Mode.ADVANCED],
            dataset_size=1,
            output_count=2,
            standard_advanced_summary=summary,
            mode_outputs=mode_outputs,
            compare_outputs=[],
            artifacts={},
        )
    )
    assert "hybrid_rerank" in markdown


def test_write_report_artifacts(tmp_path: Path) -> None:
    summary = ComparativeSummary(
        paired_count=0,
        avg_latency_delta_ms=None,
        avg_confidence_delta=None,
        advanced_retry_rate=0.0,
        abstain_rate_by_mode={},
        citation_rate_by_mode={},
        per_category=[],
    )
    report = EvalReport(
        dataset_path="data/eval/golden_dataset.jsonl",
        generated_at=datetime.now(timezone.utc),
        modes=[Mode.STANDARD],
        dataset_size=0,
        output_count=0,
        standard_advanced_summary=summary,
        mode_outputs=[],
        compare_outputs=[],
        artifacts={},
    )

    updated = write_report_artifacts(report, tmp_path / "results")
    assert Path(updated.artifacts["results_json"]).exists()
    assert Path(updated.artifacts["report_md"]).exists()
    assert Path(updated.artifacts["summary_csv"]).exists()
