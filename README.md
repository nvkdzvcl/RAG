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

## Embedding Backend (Vietnamese + Multilingual)

Default embedding backend is sentence-transformers with:

- provider: `sentence_transformers`
- model: `intfloat/multilingual-e5-base`
- device: `cpu`
- batch size: `16`
- normalized vectors: `true`

E5 formatting is applied internally:

- document chunks -> `passage: {text}`
- user queries -> `query: {text}`

If sentence-transformers is unavailable or model loading fails, the backend logs a warning and falls back to deterministic `hash-embedding` so API startup still succeeds.

CPU expectation (rough guidance):

- first startup can be slower while loading/downloading model artifacts
- retrieval quality is improved for Vietnamese and mixed Vietnamese-English text compared with hash embeddings
- indexing latency is higher than hash mode, especially for large uploads

## Reranker Backend (Post-Retrieval)

Default reranker backend is a sentence-transformers cross-encoder:

- provider: `cross_encoder`
- model: `BAAI/bge-reranker-v2-m3`
- device: `cpu`
- batch size: `8`
- reranked candidate limit: `6`

Reranker role:

- takes `query + retrieved candidate chunks`
- scores each query-chunk pair
- sorts by rerank score to improve final context quality

Trade-off:

- better context quality and grounding robustness
- higher per-query latency than score-only fallback, especially on CPU

Fallback:

- if cross-encoder model loading fails, backend logs warning and falls back to score-only reranking
- API startup and query workflows continue without crashing

## Ingestion Parsing Strategy

The ingestion pipeline now uses a parser abstraction layer:

- `BaseDocumentParser`
- `PDFParser` (`pdfplumber`)
- `DocxParser` (`python-docx`)
- `TextParser`
- `MarkdownParser`

Parsers emit structured blocks (`text`, `table`, `image`) with metadata (`page`, `section`, `bbox`).
Chunking is structure-aware and preserves block metadata (`block_type`, `language`, `section`, `page`).
Table blocks are kept intact (not split across chunks).

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

Supported upload types: `pdf`, `docx`, `txt`, `md`, `markdown`.

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

Golden dataset: `data/eval/golden_dataset.jsonl`

Run evaluation (standard + advanced):

```bash
python scripts/run_eval.py --dataset data/eval/golden_dataset.jsonl --modes standard advanced
```

Run evaluation (including compare mode):

```bash
python scripts/run_eval.py --dataset data/eval/golden_dataset.jsonl --modes standard advanced compare
```

Generated artifacts:

- `data/eval/results/results.json`
- `data/eval/results/report.md`
- `data/eval/results/summary.csv`

How to add evaluation samples:

1. Append one JSON line to `data/eval/golden_dataset.jsonl`.
2. Include required fields: `id`, `question`, `expected_behavior`, `category`.
3. Optionally include: `reference_answer`, `gold_sources`, `notes`.
4. Keep category coverage balanced, including Vietnamese samples.

How to interpret results:

- Compare `avg_latency_delta_ms` and `avg_confidence_delta` for standard vs advanced.
- Check `advanced_retry_rate`, `abstain_rate`, and `citation_rate` in `report.md`.
- Treat `groundedness_proxy` as a heuristic indicator, not a definitive factuality score.

## Tests

```bash
pytest
```

## Notes

- `requirements.txt` and `requirements-dev.txt` are the canonical dependency files.
- `requirement.txt` is intentionally not used to avoid installation ambiguity.
- OCR for images is not enabled yet; image blocks are currently stored as metadata placeholders.
