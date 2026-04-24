"""Schemas for ingestion inputs and chunk outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


BlockType = Literal["text", "table", "image"]


class DocumentBlock(BaseModel):
    """Structured parser output block for mixed-content documents."""

    type: BlockType
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadedDocument(BaseModel):
    """Normalized document object produced by loaders."""

    doc_id: str
    source: str
    title: str | None = None
    section: str | None = None
    page: int | None = None
    content: str
    block_type: BlockType = "text"
    language: str = "auto"
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
    block_type: BlockType = "text"
    language: str = "auto"
    metadata: dict[str, Any] = Field(default_factory=dict)
