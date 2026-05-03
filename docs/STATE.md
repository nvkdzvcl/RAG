# Project State

Last updated: 2026-05-03 (Asia/Bangkok)

## Repository snapshot
- Branch: `main`
- HEAD: `40d325be`
- Working tree: clean.

## Current status
- Standard mode, Advanced mode, Compare mode remain available end-to-end.
- Vector index backend is now selectable via config:
  - `VECTOR_INDEX_BACKEND=inmemory` (default, unchanged behavior)
  - `VECTOR_INDEX_BACKEND=faiss`
- FAISS integration is implemented behind the existing vector-index contract and factory selection.
- Existing uploaded/seeded split and uploaded manifest stale-check mechanism are preserved.
- InMemory persistence flow remains unchanged.

## Recently completed (indexing/retrieval track)

### 1) Configurable vector backend + FAISS paths
- Added backend selector and safe fallback to `inmemory` for unsupported values.
- Added FAISS filenames resolved under existing `INDEX_DIR`.
- Updated `.env.example` with backend and FAISS filename variables.
- Key files:
  - `app/core/config.py`
  - `.env.example`

### 2) FAISS vector index class with same public contract
- Added `FaissVectorIndex` with contract-compatible methods:
  - `build(...)`
  - `search(...)`
  - `to_dict(...)`
  - `from_dict(...)`
  - required properties (`size`, `dimension`, `revision`, `chunks`, `vectors`)
- Uses `faiss.IndexFlatIP` and L2 normalization for cosine-like similarity.
- Keeps metadata/ID mapping JSON-serializable; FAISS binary stores vectors.
- Handles empty index and dimension mismatch paths safely.
- Key file:
  - `app/indexing/faiss_index.py`

### 3) Backend factory wiring
- Added vector index factory and wired runtime index build path to it.
- Lazy import of FAISS kept isolated to selected backend path.
- Key files:
  - `app/indexing/vector_factory.py`
  - `app/services/index_runtime.py`

### 4) FAISS persistence integrated with uploaded/seeded + manifest stale-check
- Added FAISS persistence methods in `LocalIndexStore`:
  - save/load FAISS binary index
  - save/load JSON metadata payload (`entries` + optional `id_map` etc.)
- Runtime now persists vector artifacts per source while preserving current manifest workflow:
  - uploaded: uploaded FAISS artifacts + uploaded BM25 + uploaded manifest
  - seeded: seeded FAISS artifacts + seeded BM25
  - generic artifacts still maintained where runtime expects them
- Uploaded stale manifest mismatch still triggers stale cleanup + rebuild.
- No new `documents.json` introduced; document registry ownership unchanged.
- Key files:
  - `app/indexing/persistence.py`
  - `app/services/index_runtime.py`

### 5) Tests added/updated
- Added backend selection tests.
- Added FAISS vector index contract tests.
- Added FAISS runtime persistence tests for:
  - uploaded index persistence
  - seeded index persistence
  - stale manifest rebuild trigger
  - restart/reload search consistency
- Key files:
  - `tests/indexing/test_vector_factory.py`
  - `tests/indexing/test_faiss_index.py`
  - `tests/services/test_index_runtime_faiss_persistence.py`
  - `tests/services/test_index_runtime_embeddings.py`

## Validation status (latest)
- `DATA_DIR=/tmp/rag-test-data CORPUS_DIR=/tmp/rag-test-corpus INDEX_DIR=/tmp/rag-test-indexes OCR_ENABLED=false EMBEDDING_PROVIDER=hash EMBEDDING_HASH_DIMENSION=64 RERANKER_PROVIDER=score_only RERANKER_ENABLED=false LLM_PROVIDER=stub .venv/bin/pytest -q tests/indexing/test_vector_factory.py tests/indexing/test_faiss_index.py tests/services/test_index_runtime_faiss_persistence.py`
  - Result: `13 passed`
- `DATA_DIR=/tmp/rag-test-data CORPUS_DIR=/tmp/rag-test-corpus INDEX_DIR=/tmp/rag-test-indexes OCR_ENABLED=false EMBEDDING_PROVIDER=hash EMBEDDING_HASH_DIMENSION=64 RERANKER_PROVIDER=score_only RERANKER_ENABLED=false LLM_PROVIDER=stub .venv/bin/pytest -q tests/services/test_index_runtime_embeddings.py::test_runtime_index_manager_build_path_uses_vector_index_factory`
  - Result: `1 passed`
- `.venv/bin/ruff check app/indexing/persistence.py app/services/index_runtime.py tests/services/test_index_runtime_faiss_persistence.py`
  - Result: `All checks passed`

## Notes for next session
- Continue using `.venv/bin/pytest` for consistent dependency versions.
- If tests look slow on local machine, ensure heavy runtime env is disabled during test runs (especially OCR).
- Next logical step: wire FAISS persistence behavior through broader integration suite (`make test-fast` / `make test-integration`) on your target environment.
