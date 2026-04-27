"""Schemas for persisted uploaded-index manifest and fingerprint validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UploadedIndexFileEntry(BaseModel):
    """Fingerprint entry for one ready uploaded file used to build runtime indexes."""

    doc_id: str
    stored_path: str
    size_bytes: int = Field(ge=0)
    modified_ns: int = Field(ge=0)


class UploadedIndexManifest(BaseModel):
    """Manifest persisted alongside uploaded indexes to prevent stale index reuse."""

    schema_version: Literal[1] = 1
    source: Literal["uploaded"] = "uploaded"
    chunk_size: int
    chunk_overlap: int
    embedding_provider: str
    embedding_dimension: int
    active_doc_ids: list[str] = Field(default_factory=list)
    files: list[UploadedIndexFileEntry] = Field(default_factory=list)
    fingerprint: str
