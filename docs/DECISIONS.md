# DECISIONS.md

This document records important architectural and technical decisions.

Each decision includes:
- context
- decision
- alternatives considered
- reasoning
- consequences

The goal is to preserve the rationale behind the project so future contributors do not need to guess why certain choices were made.

---

## DECISION 001 — Build Three Modes in One System

### Context
The project needs a baseline RAG flow, an advanced Self-RAG flow, and a direct comparison flow.

### Decision
Build one shared system with three workflows:
- standard mode
- advanced mode
- compare mode

### Alternatives
- build only advanced mode
- build separate codebases by mode
- build standard mode first and redesign later

### Reasoning
A shared architecture reduces duplication and makes comparison easier.

### Consequences
Pros:
- easier benchmarking
- cleaner demos
- less duplicated code

Cons:
- requires better initial architecture discipline

---

## DECISION 002 — Use Practical Self-RAG Instead of Paper-Level Training

### Context
The original Self-RAG paper uses model-level training and reflection-style behavior.

### Decision
Implement a system-level Self-RAG workflow with:
- retrieval gate
- critique
- retry
- refine
- abstain

### Alternatives
- fully trained Self-RAG reproduction
- standard RAG only

### Reasoning
This is more realistic for an open-source academic software project.

### Consequences
Pros:
- feasible to implement
- easier to debug
- easier to demonstrate

Cons:
- not a full paper reproduction

---

## DECISION 003 — Use Hybrid Retrieval

### Context
Dense-only retrieval can miss keyword-heavy or exact-match queries.

### Decision
Use hybrid retrieval:
- dense retrieval
- BM25 retrieval
- fusion

### Alternatives
- dense-only retrieval
- sparse-only retrieval

### Reasoning
Hybrid retrieval improves recall across different question types.

### Consequences
Pros:
- better retrieval robustness
- better support for technical/domain terms

Cons:
- more components to maintain

---

## DECISION 004 — Add a Reranker

### Context
Initial retrieval results can contain noisy or weakly relevant chunks.

### Decision
Apply reranking after retrieval.

### Alternatives
- no reranker
- basic score sorting only

### Reasoning
Better-ranked context improves generation quality and reduces noise.

### Consequences
Pros:
- stronger context quality
- better downstream answers

Cons:
- extra latency
- extra dependency

---

## DECISION 005 — Limit Generation Context

### Context
Passing too many chunks can reduce answer quality and increase cost.

### Decision
Use only top selected chunks after reranking for generation.

### Alternatives
- pass all retrieved chunks
- dynamic unlimited context

### Reasoning
Focused context tends to produce better grounded answers.

### Consequences
Pros:
- lower token cost
- cleaner answers

Cons:
- risk of excluding useful evidence if selection is poor

---

## DECISION 006 — Separate Standard and Advanced Workflows

### Context
Standard mode and advanced mode share components but differ in control flow.

### Decision
Implement separate workflow modules:
- standard.py
- advanced.py
- shared.py
- router.py

### Alternatives
- one giant workflow file with many flags
- duplicate separate pipelines

### Reasoning
This keeps orchestration readable and modular.

### Consequences
Pros:
- easier maintenance
- easier debugging
- cleaner testing

Cons:
- slightly more upfront structure

---

## DECISION 007 — Add Retrieval Gate in Advanced Mode

### Context
Not every query truly requires retrieval.

### Decision
Add a retrieval-gate step before retrieval in advanced mode.

### Alternatives
- always retrieve
- rule-only retrieval decision

### Reasoning
This more closely matches practical Self-RAG behavior and reduces unnecessary work.

### Consequences
Pros:
- lower cost in some cases
- more flexible system behavior

Cons:
- risk of incorrect no-retrieval decisions

---

## DECISION 008 — Use Structured Critique Output

### Context
Free-form critique text is hard to parse and debug.

### Decision
Require critique output to follow a structured schema.

### Alternatives
- free-text critique
- rule-only critique

### Reasoning
Structured output is easier to validate and consume programmatically.

### Consequences
Pros:
- more reliable workflow logic
- easier debugging
- cleaner logs

Cons:
- requires careful prompt design

---

## DECISION 009 — Limit Advanced Retry Loop

### Context
Unbounded loops cause cost spikes and unpredictable latency.

### Decision
Set default maximum retry loop count to 2.

### Alternatives
- unlimited retries
- no retry

### Reasoning
Most gains come from one or two retries, not endless looping.

### Consequences
Pros:
- bounded complexity
- predictable runtime

Cons:
- some hard cases may remain unresolved

---

## DECISION 010 — Allow Abstaining

### Context
The system should not hallucinate when evidence is insufficient.

### Decision
Allow the system to abstain or return a clearly marked insufficient-evidence response.

### Alternatives
- always answer
- silently guess

### Reasoning
Trustworthiness is more important than forced completeness.

### Consequences
Pros:
- stronger reliability
- better user trust

Cons:
- some users may prefer a speculative answer

---

## DECISION 011 — Use Shared State Schema Across Modes

### Context
All workflows need consistent internal state handling, especially for logging and frontend trace display.

### Decision
Use one shared workflow state schema, with some fields unused in standard mode.

### Alternatives
- fully separate state models
- untyped dictionaries

### Reasoning
Shared state improves consistency and reduces serialization complexity.

### Consequences
Pros:
- simpler API integration
- simpler logging
- simpler trace rendering

Cons:
- standard mode may carry a few unused fields

---

## DECISION 012 — Store Prompts in Separate Files

### Context
Prompt strings embedded directly in code become hard to maintain.

### Decision
Store prompts under `/prompts`.

### Alternatives
- inline prompt strings
- hidden constant blocks in code

### Reasoning
Separate prompt files are easier to version, compare, and improve.

### Consequences
Pros:
- cleaner code
- easier prompt iteration
- easier contributor onboarding

Cons:
- one more directory to maintain

---

## DECISION 013 — Build a Frontend Early but Keep It Moderate

### Context
The project needs a demo-friendly interface, but frontend work must not dominate backend work.

### Decision
Build a reasonably polished frontend after backend baseline stability.

### Alternatives
- no frontend
- very heavy frontend first

### Reasoning
A moderate frontend supports demos and usability without derailing the backend.

### Consequences
Pros:
- better presentation
- easier evaluation and demonstration

Cons:
- additional coordination between frontend and backend

---

## DECISION 014 — Use React + Vite + Tailwind + shadcn/ui

### Context
The project needs a frontend that is attractive, practical, and fast to build.

### Decision
Use:
- React
- Vite
- Tailwind CSS
- shadcn/ui
- lucide-react
- recharts

### Alternatives
- plain React without design system
- Next.js
- heavier UI frameworks

### Reasoning
This stack is fast to develop, visually strong, and suitable for open-source work.

### Consequences
Pros:
- attractive UI quickly
- modern developer experience
- component flexibility

Cons:
- requires frontend setup overhead

---

## DECISION 015 — Keep Installation Simple with requirements.txt

### Context
Open-source contributors should be able to install backend dependencies quickly.

### Decision
Provide:
- requirements.txt
- requirements-dev.txt
- .env.example

Do not keep a separate `requirement.txt` shim because it creates ambiguity for contributors.

### Alternatives
- pyproject only
- undocumented manual install

### Reasoning
This lowers setup friction for contributors and evaluators.

### Consequences
Pros:
- easier onboarding
- easier local setup

Cons:
- dependencies must be maintained carefully

---

## DECISION 016 — Prioritize Readability Over Premature Optimization

### Context
The project is an academic software project and open-source reference.

### Decision
Prefer readable modular code over aggressive optimization in the first version.

### Alternatives
- optimize early
- compress multiple concerns into fewer files

### Reasoning
Maintainability and clarity are essential for learning and collaboration.

### Consequences
Pros:
- easier contributor understanding
- easier grading and demonstration

Cons:
- some performance optimizations may be deferred

## DECISION 017 — Add Compare Mode

### Context
The project needs a clear way to compare baseline RAG and Advanced Self-RAG behavior.

### Decision
Add a compare mode that runs standard and advanced workflows side by side for the same query.

### Alternatives
- compare manually outside the system
- keep only standard and advanced modes

### Reasoning
This improves demo quality, evaluation visibility, and academic value.

### Consequences
Pros:
- easier benchmarking
- better demos
- clearer explanation of trade-offs

Cons:
- additional response schema complexity
- frontend becomes slightly more complex
