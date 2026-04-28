# Project State

Last updated: 2026-04-29 (Asia/Bangkok)

## Repository snapshot
- Branch: `vankhanh`
- HEAD: `7347c63b` (`timing`)
- Working tree: timing instrumentation fixes in backend/workflow/tests/docs.

## Current status
- Standard mode, Advanced mode, Compare mode are running end-to-end.
- Per-step latency instrumentation is implemented across the RAG query pipeline.
- Retrieval timing now prefers request-scoped timing payloads over shared `get_last_timing`.
- Response schema remains backward compatible (`latency_ms` and existing fields still intact).

## Latency instrumentation completed

### Standard mode
- Added trace/metadata timings:
  - `normalize_query_ms`
  - `retrieval_total_ms`
  - `dense_retrieve_ms` (0 when unavailable)
  - `sparse_retrieve_ms` (0 when unavailable)
  - `hybrid_merge_ms` (0 when unavailable)
  - `breakdown_available` / `retrieval_timing_breakdown_available`
  - `rerank_ms`
  - `context_select_ms`
  - `llm_generate_ms`
  - `grounding_ms`
  - `total_ms`

### Advanced mode
- Added stage timings:
  - `retrieval_gate_ms`
  - `query_rewrite_ms`
  - `standard_pipeline_ms`
  - `critique_ms`
  - `refine_ms` (when executed)
  - `refine_stage_ms`
  - `language_guard_ms` (when executed)
  - `hallucination_guard_ms` (when executed)
  - `final_grounding_ms`
  - `total_ms`
- Added `timing_summary` trace step including these keys.

### Compare mode
- Added compare branch timings:
  - `standard_branch_ms`
  - `advanced_branch_ms`
  - `compare_total_ms`
- Each branch trace gets a `compare_timing` step.

## Streaming status
- Standard retrieval/generation events include `timings_ms`.
- Advanced stage events include stage timing and final timing bundle.
- Compare mode emits `compare_timing` event.
- SSE final response behavior is unchanged and still working.

## Key files touched for timing
- `app/core/timing.py`
- `app/retrieval/hybrid.py`
- `app/schemas/retrieval.py`
- `app/services/index_runtime.py`
- `app/workflows/standard.py`
- `app/workflows/advanced_pipeline.py`
- `app/workflows/advanced.py`
- `app/workflows/compare.py`
- `tests/workflows/test_standard_workflow.py`
- `tests/workflows/test_advanced_workflow.py`
- `tests/workflows/test_compare_workflow.py`

## Test validation (latest)
- `make test-fast`: pass (`232 passed, 8 deselected`)
- Targeted workflow + stream tests also pass.
- Timing fix targeted tests: pass (`5 passed`)
- Related retrieval/index/workflow timing tests: pass (`63 passed, 1 deselected`)

## Notes for next session
- If retriever internals are not exposed, sub-step dense/sparse/merge timing is reported as `0` with `breakdown_available=false`.
- Per-retriever `get_last_timing` is a legacy shared diagnostic and is unsafe under concurrent requests. Prefer `retrieve_with_timing` / `retrieve_with_timing_async`, which return request-scoped `RetrievalBatch` timing.
- Next optional step: surface these timing metrics in frontend trace/diagnostics panels.
