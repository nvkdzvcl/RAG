# ARCHITECTURE.md

## Overview

This project is a modular open-source **three-mode RAG system**:

- **Standard Mode**: baseline RAG
- **Advanced Mode**: practical Self-RAG Level 2
- **Compare Mode**: side-by-side execution of standard and advanced

Architecture separates shared infrastructure from workflow orchestration.

## High-Level Layers

1. ingestion
2. indexing
3. retrieval
4. generation
5. critique (advanced)
6. workflows (standard/advanced/compare)
7. API
8. frontend
9. evaluation

## Shared vs Mode-Specific

Shared:

- schemas
- loaders/chunking
- indexes/retrieval/reranking
- generation and citation contracts
- config/logging
- evaluation tooling

Mode-specific:

- control flow
- retry/abstain behavior
- comparison aggregation

## Compare Mode Contract

Compare responses return:

- `standard` branch
- `advanced` branch
- `comparison` summary

This avoids duplicate implementations by reusing existing standard and advanced workflows.

## Current Implementation Status

Implemented end-to-end MVP layers:

- project scaffolding, config, structured logging
- typed schemas for ingestion/retrieval/generation/workflow/API contracts
- ingestion layer (text/markdown loaders, cleaner, chunker, metadata preservation)
- indexing layer (embedding interface, vector index abstraction, BM25 index, local persistence)
- retrieval layer (dense, sparse, hybrid fusion, reranker hook, context selector)
- generation baseline (LLM client abstraction, grounded structured output, citations, insufficient-evidence handling)
- workflows:
  - standard (`retrieve -> rerank -> select -> generate -> cite`)
  - advanced (retrieval gate, rewrite, critique, retry/refine/abstain with bounded loop)
  - compare (runs standard + advanced and aggregates summary)
- backend API (`/api/v1/health`, `/api/v1/query`)
- frontend integration with mode selection and mode-specific rendering panels
- evaluation dataset + runner + regression checks for standard/advanced/compare payloads

Known MVP gaps:

- retrieval/generation are deterministic baseline implementations intended for local development and testing
- prompt files exist but business logic still uses lightweight heuristic behavior in several modules
- production hardening (auth, rate limiting, streaming responses, deployment concerns) is intentionally out of scope
