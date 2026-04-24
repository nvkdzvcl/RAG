# Evaluation Dataset Format

The golden evaluation dataset is stored as JSONL in [`golden.jsonl`](./golden.jsonl).

Each line must include:

- `id`: unique string identifier
- `question`: user question text
- `expected_behavior`: one of `answer`, `abstain`, `partial`

Optional fields:

- `reference_answer`: target answer for qualitative checks
- `gold_sources`: expected source/chunk identifiers
- `tags`: category labels such as `factual`, `multi_hop`, `ambiguous`
- `notes`: evaluator notes

Example row:

```json
{
  "id": "q001",
  "question": "What is Self-RAG?",
  "expected_behavior": "answer",
  "reference_answer": "Self-RAG combines retrieval and self-critique.",
  "gold_sources": ["architecture#chunk_1"],
  "tags": ["factual"],
  "notes": "Simple factual definition."
}
```

Recommended category tags for the dataset:

- `factual`
- `multi_hop`
- `ambiguous`
- `insufficient_evidence`
- `conflicting_sources`
- `follow_up`

Run evaluation:

```bash
python3 -m app.evaluation.runner --dataset data/eval/golden.jsonl --predictor stub
python3 -m app.evaluation.runner --dataset data/eval/golden.jsonl --predictor workflow
python3 -m app.evaluation.runner --predictor workflow --modes standard advanced compare
```
