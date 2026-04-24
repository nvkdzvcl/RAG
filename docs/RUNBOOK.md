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
   `python3 -m app.evaluation.runner --predictor workflow`
5. run tests:
   `pytest`

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
