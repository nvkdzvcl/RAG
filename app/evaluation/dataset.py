"""Helpers for loading evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.schemas import EvalExample


def load_eval_dataset(path: Path) -> list[EvalExample]:
    """Load a JSONL golden dataset file into typed examples."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    examples: list[EvalExample] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_no, raw_line in enumerate(infile, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                examples.append(EvalExample.model_validate(payload))
            except (
                Exception
            ) as exc:  # pragma: no cover - exercised via tests for invalid paths
                raise ValueError(f"Invalid dataset row {line_no}: {exc}") from exc

    if not examples:
        raise ValueError(f"Dataset contains no examples: {path}")
    return examples
