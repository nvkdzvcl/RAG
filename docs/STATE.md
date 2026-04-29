# Project State

Last updated: 2026-04-29 (Asia/Bangkok)

## Repository snapshot
- Branch: `main`
- HEAD: `5218fa63` (`reranker cascade: skip cross-encoder CPU khi không cần`)
- Working tree: clean.

## Current status
- Standard mode, Advanced mode, Compare mode run end-to-end.
- Retrieval timing is request-scoped where supported (`RetrievalBatch`) and avoids shared timing races where possible.
- Dynamic query budget for Standard mode is implemented (`simple_extractive` / `normal` / `complex`).
- Adaptive grounding is implemented with policy, lazy semantic usage, and grounding cache.
- Cascade reranking is implemented to reduce cross-encoder CPU overhead for cheap/high-confidence cases.
- Response schema compatibility is preserved (`latency_ms` and existing fields intact).

## Recently completed

### 1) Dynamic query budget
- Deterministic query complexity classifier (no LLM).
- Per-profile budget for retrieval/rerank/context/max_tokens.
- Standard pipeline uses selected budget for:
  - retrieval top_k
  - rerank top_k
  - context limits
  - generation max_tokens
- Key files:
  - `app/workflows/query_budget.py`
  - `app/workflows/standard.py`
  - `tests/workflows/test_query_budget.py`

### 2) Adaptive grounding
- Grounding policy inputs:
  - mode (`standard` / `advanced` / `compare`)
  - query complexity
  - status
  - answer length
  - citation count
  - retrieval confidence (if available)
  - fast-path flag
- Standard path is lexical-first; semantic is policy-controlled and lazy-loaded.
- Grounding cache key includes answer + selected context text + policy version.
- Added trace fields:
  - `grounding_policy`
  - `grounding_semantic_used`
  - `grounding_cache_hit`
  - `grounding_ms`
- Key files:
  - `app/workflows/shared/grounding.py`
  - `app/workflows/standard.py`
  - `app/workflows/advanced_pipeline.py`
  - `tests/workflows/test_grounding_helpers.py`

### 3) Cascade reranking
- Added deterministic `RerankPolicy`.
- Policy behavior:
  - `simple_extractive` -> skip cross-encoder
  - few candidates -> skip cross-encoder
  - high top score + clear gap -> skip cross-encoder
  - ambiguous normal/complex -> allow cross-encoder
  - advanced mode keeps quality path available
- Uses existing `ScoreOnlyReranker` for cheap path.
- Added trace fields:
  - `rerank_policy`
  - `reranker_used` (`score_only` / `cross_encoder` / `skipped`)
  - `rerank_ms`
- Key files:
  - `app/workflows/rerank_policy.py`
  - `app/workflows/standard.py`
  - `tests/workflows/test_rerank_policy.py`
  - `tests/workflows/test_standard_workflow.py`

## Environment/config additions
- Grounding:
  - `GROUNDING_POLICY=adaptive`
  - `GROUNDING_SEMANTIC_STANDARD_ENABLED=false`
  - `GROUNDING_SEMANTIC_ADVANCED_ENABLED=true`
- Rerank cascade:
  - `RERANK_CASCADE_ENABLED=true`
  - `RERANK_SIMPLE_SKIP_CROSS_ENCODER=true`
  - `RERANK_MIN_CANDIDATES_FOR_CROSS_ENCODER=4`
  - `RERANK_SCORE_GAP_THRESHOLD=0.2`

## Validation status (latest)
- `make test-fast`: pass (`255 passed, 8 deselected`)
- `mypy app tests`: pass (`Success: no issues found in 131 source files`)
- `.venv/bin/ruff check app/ tests/`: pass
- `.venv/bin/python -m ruff format --check app/ tests/`: pass (`131 files already formatted`)

## Notes for next session
- Keep preferring `retrieve_with_timing` / `retrieve_with_timing_async` in retrievers to avoid shared timing diagnostics.
- If quality/latency tradeoff needs tuning, adjust:
  - `RERANK_SCORE_GAP_THRESHOLD`
  - `RERANK_MIN_CANDIDATES_FOR_CROSS_ENCODER`
  - normal/complex dynamic budgets
- Optional next step: expose grounding/rerank policy details in frontend trace panels.
