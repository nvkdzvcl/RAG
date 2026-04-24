# MODES.md

## Overview

This project supports three query modes:

- **standard**
- **advanced**
- **compare**

The design keeps shared infrastructure reusable while isolating orchestration logic.

## Standard Mode

Pipeline:

`query -> retrieve -> rerank -> select context -> generate -> cite -> return`

Characteristics:

- always retrieval-based
- no critique loop
- lower latency/cost baseline

## Advanced Mode

Pipeline:

`query -> retrieval gate -> query rewrite -> retrieve -> rerank -> generate draft -> critique -> retry/refine/abstain -> return`

Characteristics:

- bounded self-check loop
- can retry retrieval
- can abstain on weak evidence
- higher latency/cost for reliability

## Compare Mode

Pipeline:

`query -> run standard + advanced -> aggregate outputs -> compute summary -> return`

Characteristics:

- executes standard and advanced independently
- returns side-by-side results
- includes comparison summary fields (confidence/latency deltas)

## Shared Components

All modes share:

- loaders/chunking/indexes
- retrievers/reranker/context selector
- generation and citation contracts
- configuration/logging/schemas
- evaluation dataset format

Only orchestration differs.
