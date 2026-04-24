# EVALUATION.md

## Purpose

Define a repeatable evaluation process for:

- `standard` mode
- `advanced` mode
- `compare` mode

## Dataset Format

Golden dataset path:

- `data/eval/golden.jsonl`

Each JSONL row contains:

- `id` (required)
- `question` (required)
- `expected_behavior` (required): `answer`, `abstain`, or `partial`
- `reference_answer` (optional)
- `gold_sources` (optional)
- `tags` (optional)
- `notes` (optional)

## Required Coverage Categories

- factual
- multi_hop
- ambiguous
- insufficient_evidence
- conflicting_sources
- follow_up

## Runner

CLI module:

- `python3 -m app.evaluation.runner`

Options:

- `--dataset`: dataset path (default `data/eval/golden.jsonl`)
- `--predictor stub|workflow`
- `--modes standard advanced compare`

`stub` predictor validates contract-level behavior before full workflow implementation.
`workflow` predictor runs actual workflow code when available.

Example:

```bash
python3 -m app.evaluation.runner --predictor workflow --modes standard advanced compare
```

## Current Regression Checks

Automated tests validate:

- dataset load/shape
- schema compliance for `standard`, `advanced`, `compare` payloads (fixture + workflow predictor)
- compare-mode branch structure

## Metrics Reported

- schema valid rate
- expected behavior match rate
- invalid payload count
- per-case per-mode status
