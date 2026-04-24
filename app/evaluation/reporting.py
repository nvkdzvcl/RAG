"""Report aggregation and serialization helpers for evaluation runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.evaluation.schemas import (
    CategorySummary,
    ComparativeSummary,
    CompareEvalOutput,
    EvalReport,
    ModeEvalOutput,
)
from app.schemas.common import Mode


def _avg(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _flatten_outputs_for_mode_summary(
    mode_outputs: list[ModeEvalOutput],
    compare_outputs: list[CompareEvalOutput],
) -> list[ModeEvalOutput]:
    flattened: list[ModeEvalOutput] = list(mode_outputs)

    mode_ids = {(item.example_id, item.mode.value) for item in flattened if item.run_source == "direct"}
    for compare in compare_outputs:
        std_id = (compare.standard.example_id, compare.standard.mode.value)
        adv_id = (compare.advanced.example_id, compare.advanced.mode.value)

        if std_id not in mode_ids:
            flattened.append(compare.standard)
        if adv_id not in mode_ids:
            flattened.append(compare.advanced)
    return flattened


def build_comparative_summary(
    mode_outputs: list[ModeEvalOutput],
    compare_outputs: list[CompareEvalOutput],
) -> ComparativeSummary:
    """Build aggregate comparison report for standard vs advanced."""
    flattened = _flatten_outputs_for_mode_summary(mode_outputs, compare_outputs)
    standard_rows = [item for item in flattened if item.mode == Mode.STANDARD]
    advanced_rows = [item for item in flattened if item.mode == Mode.ADVANCED]

    standard_by_id = {item.example_id: item for item in standard_rows}
    advanced_by_id = {item.example_id: item for item in advanced_rows}
    paired_ids = sorted(set(standard_by_id).intersection(advanced_by_id))

    latency_deltas: list[float] = []
    confidence_deltas: list[float] = []
    for example_id in paired_ids:
        std = standard_by_id[example_id]
        adv = advanced_by_id[example_id]
        if std.latency_ms is not None and adv.latency_ms is not None:
            latency_deltas.append(float(adv.latency_ms - std.latency_ms))
        if std.confidence is not None and adv.confidence is not None:
            confidence_deltas.append(float(adv.confidence - std.confidence))

    def _rate(rows: list[ModeEvalOutput], predicate) -> float:
        if not rows:
            return 0.0
        return sum(1 for row in rows if predicate(row)) / len(rows)

    abstain_rate_by_mode = {
        "standard": _rate(standard_rows, lambda row: row.status in {"insufficient_evidence", "abstained"}),
        "advanced": _rate(advanced_rows, lambda row: row.status in {"insufficient_evidence", "abstained"}),
    }
    citation_rate_by_mode = {
        "standard": _rate(standard_rows, lambda row: row.metrics.has_citations),
        "advanced": _rate(advanced_rows, lambda row: row.metrics.has_citations),
    }
    advanced_retry_rate = _rate(advanced_rows, lambda row: row.metrics.retry_used)

    category_rows: list[CategorySummary] = []
    rows_by_mode = {
        Mode.STANDARD: standard_rows,
        Mode.ADVANCED: advanced_rows,
    }
    for mode, rows in rows_by_mode.items():
        categories = sorted({item.category for item in rows})
        for category in categories:
            subset = [item for item in rows if item.category == category]
            latencies = [float(item.latency_ms) for item in subset if item.latency_ms is not None]
            confidences = [float(item.confidence) for item in subset if item.confidence is not None]
            category_rows.append(
                CategorySummary(
                    mode=mode,
                    category=category,
                    count=len(subset),
                    avg_latency_ms=_avg(latencies),
                    avg_confidence=_avg(confidences),
                    citation_rate=_rate(subset, lambda row: row.metrics.has_citations),
                    abstain_rate=_rate(subset, lambda row: row.status in {"insufficient_evidence", "abstained"}),
                    retry_rate=_rate(subset, lambda row: row.metrics.retry_used),
                )
            )

    return ComparativeSummary(
        paired_count=len(paired_ids),
        avg_latency_delta_ms=_avg(latency_deltas),
        avg_confidence_delta=_avg(confidence_deltas),
        advanced_retry_rate=advanced_retry_rate,
        abstain_rate_by_mode=abstain_rate_by_mode,
        citation_rate_by_mode=citation_rate_by_mode,
        per_category=category_rows,
    )


def report_to_markdown(report: EvalReport) -> str:
    """Render eval report into markdown for contributor-friendly review."""
    summary = report.standard_advanced_summary
    lines = [
        "# Evaluation Report",
        "",
        f"- Generated at: `{report.generated_at.isoformat()}`",
        f"- Dataset: `{report.dataset_path}`",
        f"- Modes: `{', '.join(mode.value for mode in report.modes)}`",
        f"- Dataset size: `{report.dataset_size}`",
        f"- Mode outputs: `{report.output_count}`",
        "",
        "## Standard vs Advanced",
        "",
        f"- Paired examples: `{summary.paired_count}`",
        f"- Avg latency delta (advanced - standard, ms): `{summary.avg_latency_delta_ms}`",
        f"- Avg confidence delta (advanced - standard): `{summary.avg_confidence_delta}`",
        f"- Advanced retry rate: `{summary.advanced_retry_rate:.3f}`",
        "",
        "## Rates",
        "",
        f"- Abstain rate (standard): `{summary.abstain_rate_by_mode.get('standard', 0.0):.3f}`",
        f"- Abstain rate (advanced): `{summary.abstain_rate_by_mode.get('advanced', 0.0):.3f}`",
        f"- Citation rate (standard): `{summary.citation_rate_by_mode.get('standard', 0.0):.3f}`",
        f"- Citation rate (advanced): `{summary.citation_rate_by_mode.get('advanced', 0.0):.3f}`",
        "",
        "## Per-Category Summary",
        "",
        "| mode | category | count | avg_latency_ms | avg_confidence | citation_rate | abstain_rate | retry_rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in summary.per_category:
        lines.append(
            "| "
            f"{row.mode.value} | {row.category} | {row.count} | "
            f"{row.avg_latency_ms} | {row.avg_confidence} | "
            f"{row.citation_rate:.3f} | {row.abstain_rate:.3f} | {row.retry_rate:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `groundedness_proxy` is a lexical-overlap proxy only and not a perfect groundedness metric.",
        ]
    )
    return "\n".join(lines)


def write_csv_summary(path: Path, mode_outputs: list[ModeEvalOutput]) -> None:
    """Write per-case output rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "example_id",
                "mode",
                "category",
                "expected_behavior",
                "status",
                "latency_ms",
                "confidence",
                "citation_count",
                "has_citations",
                "abstain_match",
                "retry_used",
                "retrieved_count",
                "selected_context_count",
                "answer_non_empty",
                "answer_contains_reference_keywords",
                "cited_gold_source_overlap",
                "groundedness_proxy",
            ]
        )
        for item in mode_outputs:
            writer.writerow(
                [
                    item.example_id,
                    item.mode.value,
                    item.category,
                    item.expected_behavior,
                    item.status,
                    item.latency_ms,
                    item.confidence,
                    item.metrics.citation_count,
                    item.metrics.has_citations,
                    item.metrics.abstain_match,
                    item.metrics.retry_used,
                    item.metrics.retrieved_count,
                    item.metrics.selected_context_count,
                    item.metrics.answer_non_empty,
                    item.metrics.answer_contains_reference_keywords,
                    item.metrics.cited_gold_source_overlap,
                    item.metrics.groundedness_proxy,
                ]
            )


def write_report_artifacts(report: EvalReport, output_dir: Path) -> EvalReport:
    """Persist report JSON/Markdown/CSV artifacts and return updated report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results_json = output_dir / "results.json"
    report_md = output_dir / "report.md"
    summary_csv = output_dir / "summary.csv"

    markdown = report_to_markdown(report)
    report_md.write_text(markdown, encoding="utf-8")
    write_csv_summary(summary_csv, report.mode_outputs)
    updated = report.model_copy(
        update={
            "artifacts": {
                "results_json": str(results_json),
                "report_md": str(report_md),
                "summary_csv": str(summary_csv),
            }
        }
    )
    results_json.write_text(json.dumps(updated.model_dump(mode="json"), indent=2), encoding="utf-8")
    return updated
