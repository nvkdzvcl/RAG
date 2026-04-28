# Project State

<<<<<<< HEAD
## Current features
- RAG standard
- Self-RAG advanced
- Compare mode
- OCR (partial)
- Hybrid search
- reranker

## Known issues
- advanced too strict
- grounding threshold too high
- stale index after delete
- OCR not effective yet

## Goal
stabilize system before demo
=======
Last updated: 2026-04-26

## Current status
- Standard mode, Advanced mode, Compare mode are working end-to-end.
- Document lifecycle APIs support upload/list/status/delete-one/delete-all.
- OCR pipeline for scanned PDF is integrated into parse -> chunk -> runtime index -> retrieval.
- Grounding/hallucination heuristic is tuned to be less aggressive for Vietnamese paraphrases.
- Compare scoring prioritizes citations + groundedness + safety signals over raw confidence.
- Settings modal explains chunking/retrieval tradeoffs and remains usable on small screens.

## Regression validation (latest pass)
- `python -m compileall app`: pass (run in `.venv`)
- `pytest -m "not slow and not e2e"`: pass (`146 passed, 7 deselected`)
- `cd frontend && npm run build`: pass

## Scenario coverage
- Delete all documents clears registry/raw files/runtime and falls back to seeded retrieval.
- Delete one document removes it from list/status/query citations and keeps remaining uploaded docs retrievable.
- OCR upload returns debug metadata:
  - `total_blocks`
  - `text_blocks`
  - `table_blocks`
  - `image_blocks`
  - `ocr_blocks`
  - `total_chunks`
  - `ocr_chunks`
- OCR mocked path produces `ocr_text` blocks/chunks and those chunks are indexed/retrievable.
- Advanced workflow does not reject valid context too aggressively in tested Vietnamese cases.
- Compare workflow prefers grounded/cited branch and handles weak/tie cases.

## Remaining limitations
- No dedicated frontend e2e test yet for modal usability; currently validated by component structure + build.
- One non-blocking warning remains in tests (`FutureWarning` from sentence-transformers embedding dimension getter rename).
- OCR requires system Tesseract with Vietnamese language pack and re-upload after enabling OCR.

## Next stabilization focus
- Keep regression suite green while avoiding backend/frontend behavior drift.
>>>>>>> main
