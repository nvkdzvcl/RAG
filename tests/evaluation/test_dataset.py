"""Evaluation dataset shape tests."""

from pathlib import Path

from app.evaluation.dataset import load_eval_dataset


def test_golden_dataset_loads() -> None:
    dataset = load_eval_dataset(Path("data/eval/golden.jsonl"))
    assert len(dataset) >= 6


def test_golden_dataset_contains_required_categories() -> None:
    dataset = load_eval_dataset(Path("data/eval/golden.jsonl"))
    tags = {tag for example in dataset for tag in example.tags}
    assert "factual" in tags
    assert "multi_hop" in tags
    assert "ambiguous" in tags
    assert "insufficient_evidence" in tags
    assert "conflicting_sources" in tags
