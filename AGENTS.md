# AGENTS.md

## Project Overview

Build an open-source **Self-RAG Level 2** application with **three modes**:

1. **standard mode** = normal RAG
2. **advanced mode** = practical Self-RAG
3. **compare mode** = run both standard and advanced workflows for comparison

This is a **software engineering project**, not a paper-level reproduction of trained Self-RAG.
The goal is to build a clean, modular, testable, extensible system.

---

## Product Goals

The system must:

- ingest documents
- build searchable indexes
- answer questions with citations
- support three query modes:
  - standard mode
  - advanced mode
  - compare mode
- expose a backend API
- provide a reasonably polished frontend
- support local development and open-source collaboration
- remain easy to understand and extend

---

## Three Modes

### Standard Mode
Pipeline:

query -> retrieve -> rerank -> select context -> generate -> return

Characteristics:
- faster
- cheaper
- baseline mode
- always retrieval-based

### Advanced Mode
Pipeline:

query -> retrieval gate -> query rewrite -> retrieve -> rerank -> generate draft -> critique -> retry/refine/abstain -> return

Characteristics:
- slower
- more expensive
- more reliable
- can retry retrieval
- can abstain if evidence is insufficient

### Compare Mode
Pipeline:

query -> run standard + advanced -> aggregate outputs -> return

Characteristics:
- runs both pipelines for the same query
- enables direct side-by-side comparison
- returns both branches and a comparison summary

### Rule
All modes must share the same core infrastructure:
- ingestion
- indexing
- retrieval
- reranking
- generation
- citations
- logging
- evaluation
- API schemas

Only the workflow/orchestration differs.

---

## Non-Goals for MVP

Do not build these in the first implementation:

- model fine-tuning
- paper-level reflection token training
- graph RAG
- multimodal RAG
- distributed deployment
- auth system
- complex admin dashboard
- websocket streaming unless explicitly requested later

---

## Required Engineering Principles

### 1. Modular Design
Keep modules small and single-purpose.

### 2. Typed Python
Use typed Python throughout the backend.

### 3. Structured Schemas
Use Pydantic models or typed dataclasses for all major payloads.

### 4. Separation of Concerns
Do not mix:
- retrieval logic
- workflow orchestration
- prompt content
- API route handling
- UI logic

### 5. Testability
Every major module must be easy to test independently.

### 6. Minimal Coupling
Do not create giant pipeline files with many responsibilities.

---

## Project Build Order

When implementing this repository, always work in this order:

1. project scaffolding
2. config and logging
3. schemas
4. ingestion
5. indexing
6. retrieval
7. reranking
8. generation baseline
9. standard workflow
10. advanced workflow
11. evaluation
12. API
13. frontend
14. docs cleanup

Do not jump straight to frontend or advanced workflow before the backend baseline is stable.

---

## Required Backend Capabilities

### Ingestion
Implement:
- text loader
- markdown loader
- PDF loader later if needed
- cleaning
- chunking
- metadata preservation

Each chunk must preserve:
- chunk_id
- doc_id
- source
- title
- section if available
- page if available
- content
- metadata

### Indexing
Implement:
- embeddings interface
- vector index
- BM25 index
- local persistence

### Retrieval
Implement:
- dense retriever
- sparse retriever
- hybrid retrieval
- score fusion
- reranker
- context selector

### Generation
Implement:
- grounded answer generation
- citation generation
- structured answer output
- insufficient evidence handling

### Advanced Self-RAG Features
Implement:
- retrieval gate
- query rewriting
- critique step
- retry retrieval
- refine answer
- abstain path
- max loop limit

---

## Workflow State

Maintain one explicit shared workflow state object.

It must include at least:

- mode
- user_query
- normalized_query
- chat_history
- need_retrieval
- rewritten_queries
- retrieved_docs
- reranked_docs
- selected_context
- draft_answer
- final_answer
- citations
- critique
- confidence
- loop_count
- stop_reason

Do not pass around ad-hoc loose dictionaries unless unavoidable.

---

## Prompt Management

All prompts must live in `/prompts`.

Do not hardcode long prompts inside business logic.

Minimum prompt files:

- standard_answer.md
- retrieval_gate.md
- query_rewrite.md
- advanced_answer.md
- critique.md
- refine.md

---

## Logging Rules

Every workflow run must log:

- original query
- mode
- retrieval decision
- rewritten queries
- retrieved chunk IDs
- rerank scores
- selected chunk IDs
- draft answer
- critique result
- retry decision
- final answer
- stop reason
- timing information
- token usage if available

Logs must be structured and readable.

---

## Reliability Rules

- Never present unsupported facts as grounded facts
- If evidence is insufficient, abstain or return a clearly marked partial answer
- If sources conflict, surface the conflict explicitly
- Never allow infinite retry loops
- Keep advanced mode bounded and predictable

---

## Frontend Scope

The frontend should be reasonably polished but not over-engineered.

Required UI pages/components:
- main chat page
- mode selector
- answer panel
- citations panel
- retrieved sources panel
- advanced trace panel
- comparison-ready layout

Frontend stack target:
- React
- Vite
- Tailwind CSS
- shadcn/ui
- lucide-react
- recharts

Do not spend time on excessive animation before the backend is stable.

---

## Deliverables by Phase

### Phase 0
- repo scaffolding
- configs
- schemas
- docs skeleton

### Phase 1
- ingestion
- indexing
- persistence
- tests

### Phase 2
- hybrid retrieval
- reranker
- context selection
- tests

### Phase 3
- generation baseline
- citations
- standard workflow
- tests

### Phase 4
- retrieval gate
- query rewrite
- critique
- retry/refine/abstain
- advanced workflow
- tests

### Phase 5
- eval dataset
- regression checks
- API routes

### Phase 6
- frontend integration
- mode switching
- source/trace panels
- docs polish

---

## Required Interfaces

### Retriever Interface
Must support:
- retrieve_dense(query, top_k)
- retrieve_sparse(query, top_k)
- retrieve_hybrid(query, top_k)

### Reranker Interface
Must support:
- rerank(query, docs)

### Generator Interface
Must support:
- generate_answer(query, context, mode)

### Critic Interface
Must support:
- critique(query, draft_answer, context)

### Workflow Runner Interface
Must support:
- run(query, mode, chat_history=None)

---

## Critique Output Schema

The critique step must return structured output with:

- grounded: bool
- enough_evidence: bool
- has_conflict: bool
- missing_aspects: list[str]
- should_retry_retrieval: bool
- should_refine_answer: bool
- better_queries: list[str]
- confidence: float
- note: str

If critique parsing fails:
- log raw output
- fail safely
- do not crash the system

---

## Stop Conditions

Stop when any of the following is true:

- answer is grounded and evidence is sufficient
- no better query can be formed
- retrieval returns no useful evidence
- max loop count is reached
- critic recommends abstaining

Default max loop count for advanced mode:
- 2

---

## Testing Requirements

Add tests continuously.

Minimum coverage targets:

- loader output shape
- chunk metadata preservation
- chunk ID generation
- index build behavior
- hybrid retrieval output
- reranker ordering
- citation formatting
- critique schema validity
- retry loop limit
- abstain behavior
- standard mode run path
- advanced mode run path

### Test Execution Strategy

Use these commands by default during development:

- normal prompt/dev loop:
  - `pytest -m "not slow and not e2e"`
  - or `make test-fast`
- backend logic only:
  - `pytest tests/schemas tests/retrieval tests/generation -m "not slow"`
- integration checks:
  - `make test-integration`
- full validation before push/release:
  - `pytest`
  - or `make test-full`

Execution guidance for agents/contributors:

- small frontend/UI-only changes:
  - run `cd frontend && npm run build`
- backend unit-level changes:
  - run `make test-fast`
- retrieval/indexing/reranker changes:
  - run relevant integration tests (or `make test-integration`) in addition to fast tests
- release/stabilization:
  - run `make test-full`

---

## Evaluation Requirements

Create a small golden dataset with:
- simple factual questions
- multi-hop questions
- ambiguous questions
- insufficient-evidence questions
- conflicting-source questions

Track at least:
- retrieval quality proxy
- answer relevance
- groundedness
- citation correctness
- abstain correctness
- retry success rate

---

## Open-Source Requirements

This is an open-source project.

Always maintain:
- readable file structure
- installation docs
- requirements files
- .env.example
- contributor-friendly documentation
- minimal surprises for new users

---

## What to Avoid

Avoid:
- giant all-in-one files
- overusing globals
- hidden prompt strings in code
- hardcoded provider-specific assumptions across the repo
- building UI first
- endless refactors without tests
- similarity-only confidence scoring
- regex-heavy brittle parsing when schema validation is possible

---

## Definition of Done for MVP

The MVP is done only if:

- documents can be ingested
- vector and BM25 indexes can be built
- hybrid retrieval works
- reranking works
- standard mode works end-to-end
- advanced mode works end-to-end
- answers include citations
- advanced mode can retry once or twice
- advanced mode can abstain
- tests pass
- docs are understandable
- frontend can query all three modes

---

## Agent Instructions

When making changes:
- keep changes localized
- update tests
- update docs when architecture changes
- summarize what was built
- summarize assumptions
- summarize next recommended step

When unsure:
- choose the smallest safe assumption
- document it
- do not over-engineer
