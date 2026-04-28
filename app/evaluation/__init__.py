"""Evaluation helpers for dataset loading and typed reports."""

from app.evaluation.dataset import load_eval_dataset
from app.evaluation.metrics import compute_metrics, extract_trace_fields
from app.evaluation.schemas import (
    CompareEvalOutput,
    EvalExample,
    EvalMetrics,
    EvalReport,
    ModeEvalOutput,
)


def __getattr__(name: str):
    if name == "EvaluationRunner":
        from app.evaluation.runner import EvaluationRunner

        return EvaluationRunner
    raise AttributeError(f"module 'app.evaluation' has no attribute '{name}'")


__all__ = [
    "CompareEvalOutput",
    "EvaluationRunner",
    "EvalExample",
    "EvalMetrics",
    "EvalReport",
    "ModeEvalOutput",
    "compute_metrics",
    "extract_trace_fields",
    "load_eval_dataset",
]
