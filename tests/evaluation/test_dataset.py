"""Evaluation dataset tests for golden_dataset format."""

from pathlib import Path

from app.evaluation.dataset import load_eval_dataset


def test_golden_dataset_loads() -> None:
    dataset = load_eval_dataset(Path("data/eval/golden_dataset.jsonl"))
    assert len(dataset) >= 8


def test_golden_dataset_contains_required_categories() -> None:
    dataset = load_eval_dataset(Path("data/eval/golden_dataset.jsonl"))
    categories = {example.category for example in dataset}
    assert "simple" in categories
    assert "multi_hop" in categories
    assert "ambiguous" in categories
    assert "insufficient_context" in categories
    assert "conflicting_sources" in categories
    assert "vietnamese" in categories


def test_golden_dataset_has_vietnamese_examples() -> None:
    dataset = load_eval_dataset(Path("data/eval/golden_dataset.jsonl"))
    vietnamese = [example for example in dataset if example.category == "vietnamese"]
    assert len(vietnamese) >= 2


def test_alternate_golden_jsonl_loads_with_category_field() -> None:
    dataset = load_eval_dataset(Path("data/eval/golden.jsonl"))
    assert len(dataset) >= 8
    assert all(hasattr(example, "category") for example in dataset)
    vietnamese = [example for example in dataset if example.category == "vietnamese"]
    assert len(vietnamese) >= 2
