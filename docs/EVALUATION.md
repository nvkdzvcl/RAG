# EVALUATION.md

## Purpose

Define a repeatable evaluation process for:

- `standard` mode
- `advanced` mode
- `compare` mode

## Dataset Format

Golden dataset path:

- `data/eval/golden_dataset.jsonl`

Each JSONL row contains:

- `id` (required)
- `question` (required)
- `expected_behavior` (required): `answer`, `abstain`, or `retry`
- `reference_answer` (optional)
- `gold_sources` (optional)
- `category` (required): `simple`, `multi_hop`, `ambiguous`, `insufficient_context`, `conflicting_sources`, `vietnamese`
- `notes` (optional)

## Required Coverage Categories

- simple
- multi_hop
- ambiguous
- insufficient_context
- conflicting_sources
- vietnamese

## Runner

CLI module:

- `python scripts/run_eval.py`

Options:

- `--dataset`: dataset path (default `data/eval/golden_dataset.jsonl`)
- `--modes standard advanced compare`
- `--output-dir data/eval/results`

Example:

```bash
python scripts/run_eval.py --dataset data/eval/golden_dataset.jsonl --modes standard advanced compare
```

## Current Regression Checks

Automated tests validate:

- dataset load/shape
- metric computation
- report generation
- evaluation runner with mocked workflows

## Metrics Reported

- citation count and citation rate
- abstain match and abstain rate
- retry usage and advanced retry rate
- confidence and latency summaries
- retrieved/selected context counts
- heuristic proxies:
  - answer non-empty
  - reference keyword overlap
  - gold source overlap
  - groundedness proxy via lexical overlap with selected context
