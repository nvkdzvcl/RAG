"""Schemas for ingestion inputs and chunk outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LoadedDocument(BaseModel):
    """Normalized document object produced by loaders."""

    doc_id: str
    source: str
    title: str | None = None
    section: str | None = None
    page: int | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Chunk object consumed by downstream indexing/retrieval."""

    chunk_id: str
    doc_id: str
    source: str
    title: str | None = None
    section: str | None = None
    page: int | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
