# Self-RAG (Open-Source, 3 Modes)

Modular Self-RAG application with:

- `standard` mode: baseline RAG pipeline
- `advanced` mode: practical Self-RAG loop (gate/rewrite/critique/retry/refine/abstain)
- `compare` mode: run standard + advanced and return side-by-side outputs

## Stack

- Backend: Python, FastAPI, Pydantic
- Frontend: React, Vite, Tailwind CSS, shadcn/ui-compatible structure
- Evaluation: golden dataset + CLI runner + regression tests

## Repository Layout

```text
app/
  api/
  core/
  evaluation/
  generation/
  indexing/
  ingestion/
  retrieval/
  schemas/
  services/
  workflows/
data/
  eval/
docs/
frontend/
prompts/
tests/
```

## Backend: Install and Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Backend default URL: `http://127.0.0.1:8000`

Default document corpus directory: `docs/` (configurable via `CORPUS_DIR`).
Uploaded documents are stored under `data/raw/` (or `<DATA_DIR>/raw`) and processed into retrieval indexes.
Runtime retrieval behavior:
- if at least one uploaded document is in `ready` state, query workflows use uploaded indexes
- otherwise, workflows fall back to the seeded corpus (`CORPUS_DIR`)

Vector and BM25 indexes are persisted in `INDEX_DIR` (default `data/indexes/`).

## API Endpoints

- `GET /api/v1/health`
- `POST /api/v1/query`
- `POST /api/v1/documents/upload`
- `GET /api/v1/documents`
- `GET /api/v1/documents/{document_id}/status`

Example query request:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does advanced mode differ from standard mode?",
    "mode": "advanced",
    "chat_history": []
  }'
```

Supported `mode` values: `standard`, `advanced`, `compare`.

Example document upload:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload \
  -F "file=@./sample.md"
```

Supported upload types: `pdf`, `txt`, `md`, `markdown`.

## Frontend: Install and Run

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Frontend default URL: `http://127.0.0.1:5173`

By default, frontend requests go to `VITE_API_BASE_URL=/api/v1` and Vite proxies `/api/*` to `http://localhost:8000`.

## Evaluation

Golden dataset: `data/eval/golden.jsonl`

Run evaluation (stub predictor):

```bash
python3 -m app.evaluation.runner --dataset data/eval/golden.jsonl --predictor stub
```

Run evaluation (real workflows):

```bash
python3 -m app.evaluation.runner --dataset data/eval/golden.jsonl --predictor workflow
```

Run evaluation for selected modes only:

```bash
python3 -m app.evaluation.runner --predictor workflow --modes standard advanced compare
```

## Tests

```bash
pytest
```

## Notes

- `requirements.txt` and `requirements-dev.txt` are the canonical dependency files.
- `requirement.txt` is intentionally not used to avoid installation ambiguity.
