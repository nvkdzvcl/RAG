# Evaluation Dataset Format

The practical golden evaluation dataset is stored as JSONL in [`golden_dataset.jsonl`](./golden_dataset.jsonl).
A compatible alias dataset is also kept in [`golden.jsonl`](./golden.jsonl).

Each line must include:

- `id`: unique string identifier
- `question`: user question text
- `expected_behavior`: one of `answer`, `abstain`, `retry`
- `category`: one of:
  - `simple`
  - `multi_hop`
  - `ambiguous`
  - `insufficient_context`
  - `conflicting_sources`
  - `vietnamese`

Optional fields:

- `reference_answer`: target answer for qualitative checks
- `gold_sources`: expected source/chunk identifiers
- `notes`: evaluator notes

Example row:

```json
{
  "id": "eval_001",
  "question": "Chế độ compare dùng để làm gì?",
  "expected_behavior": "answer",
  "reference_answer": "Compare mode chạy standard và advanced để đối chiếu.",
  "gold_sources": ["docs/MODES.md"],
  "category": "vietnamese",
  "notes": "Vietnamese factual sample."
}
```

Run evaluation:

```bash
python scripts/run_eval.py --dataset data/eval/golden_dataset.jsonl --modes standard advanced
python scripts/run_eval.py --dataset data/eval/golden_dataset.jsonl --modes standard advanced compare
python -m app.evaluation.runner --dataset data/eval/golden.jsonl --predictor stub
```
