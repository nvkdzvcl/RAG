# RUNBOOK.md

## Local Backend Setup

1. Create and activate a virtual environment:
   `python3 -m venv .venv && source .venv/bin/activate`
2. Install dependencies:
   `pip install -r requirements-dev.txt`
3. Copy environment template:
   `cp .env.example .env`
4. Start backend:
   `uvicorn app.main:app --reload`

## Local Frontend Setup

1. Enter frontend folder:
   `cd frontend`
2. Install dependencies:
   `npm install`
3. Copy frontend env template:
   `cp .env.example .env`
4. Start dev server:
   `npm run dev`

## Typical Developer Flow

1. upload one or more documents with `POST /api/v1/documents/upload`
2. verify processing state via:
   - `GET /api/v1/documents`
   - `GET /api/v1/documents/{document_id}/status`
3. run queries across all 3 modes (`standard`, `advanced`, `compare`)
4. run evaluation set:
   `python scripts/run_eval.py --dataset data/eval/golden_dataset.jsonl --modes standard advanced compare`
5. run tests:
   `pytest`

## Embedding Configuration

Recommended multilingual/Vietnamese embedding setup:

- `EMBEDDING_PROVIDER=sentence_transformers`
- `EMBEDDING_MODEL=intfloat/multilingual-e5-base`
- `EMBEDDING_DEVICE=cpu`
- `EMBEDDING_BATCH_SIZE=16`
- `EMBEDDING_NORMALIZE=true`

Fallback behavior:

- if `sentence-transformers` is not installed, or model initialization fails, the app logs a warning and automatically falls back to `HashEmbeddingProvider`
- API startup continues (no hard crash), so local development is still possible

CPU performance expectations:

- first run may spend noticeable time loading/downloading model assets
- indexing throughput on CPU is lower than hash embeddings
- retrieval quality for Vietnamese and mixed Vietnamese-English documents is significantly better than hash fallback

## Reranker Configuration

Recommended production-style reranker setup:

- `RERANKER_PROVIDER=cross_encoder`
- `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`
- `RERANKER_DEVICE=cpu`
- `RERANKER_BATCH_SIZE=8`
- `RERANKER_TOP_N=6`

Reranker role:

- reranks top retrieved candidates using query-document pair scoring
- improves context quality before answer generation

Latency trade-off:

- cross-encoder reranking is slower than score-only ordering
- keep `RERANKER_TOP_N` small (for example `4-8`) on local CPU machines with 16GB RAM

Fallback behavior:

- if cross-encoder model initialization fails, backend logs warning and uses score-only reranker
- standard, advanced, and compare query modes continue to run

## Before Opening a Pull Request

- run tests and ensure green
- run evaluation at least once (`stub` or `workflow` predictor)
- update docs when architecture or behavior changed
- keep changes scoped and avoid unrelated refactors
- ensure no secrets are committed

## Debug Priority Order

If output quality or behavior is unexpected, check in this order:

1. schema request/response mismatches
2. ingestion metadata and chunking output
3. index persistence/load state
4. dense/sparse/hybrid retrieval output
5. reranker ordering and context selection
6. generator output parsing and insufficient-evidence handling
7. advanced workflow state transitions (gate/rewrite/critique/retry/refine/abstain)
8. frontend data mapping/rendering assumptions

## Document Processing Statuses

The backend exposes these status values for uploads:

- `uploaded`
- `splitting`
- `embedding`
- `indexing`
- `ready`
- `failed`

When one or more uploaded documents are `ready`, query workflows use uploaded indexes.
If none are `ready`, workflows fall back to the seeded corpus indexes.

## Mixed-Content Ingestion Notes

- Supported file types: `.pdf`, `.docx`, `.txt`, `.md`, `.markdown`
- Parser layer extracts structured blocks: `text`, `table`, `image`
- Tables are preserved as table blocks and chunked without splitting
- OCR is currently not enabled; images are tracked as metadata/image placeholder blocks
