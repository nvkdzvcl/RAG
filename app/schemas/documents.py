"""Schemas for document upload, delete, settings, and processing APIs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DocumentProcessingStatus(str, Enum):
    """Processing statuses for uploaded documents."""

    UPLOADED = "uploaded"
    SPLITTING = "splitting"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class StoredDocumentRecord(BaseModel):
    """Persistent document metadata record."""

    document_id: str
    filename: str
    original_filename: str | None = None
    file_type: str | None = None
    stored_path: str
    status: DocumentProcessingStatus
    chunk_count: int | None = None
    created_at: datetime
    updated_at: datetime
    message: str | None = None

    @classmethod
    def create(
        cls,
        *,
        document_id: str,
        filename: str,
        original_filename: str | None = None,
        file_type: str | None = None,
        stored_path: str,
        status: DocumentProcessingStatus = DocumentProcessingStatus.UPLOADED,
    ) -> "StoredDocumentRecord":
        now = datetime.now(timezone.utc)
        resolved_original_filename = original_filename or filename
        resolved_file_type = (
            file_type or Path(filename).suffix.lower().lstrip(".") or None
        )
        return cls(
            document_id=document_id,
            filename=filename,
            original_filename=resolved_original_filename,
            file_type=resolved_file_type,
            stored_path=stored_path,
            status=status,
            chunk_count=None,
            created_at=now,
            updated_at=now,
            message=None,
        )

    def with_status(
        self,
        status: DocumentProcessingStatus,
        *,
        chunk_count: int | None = None,
        message: str | None = None,
    ) -> "StoredDocumentRecord":
        return self.model_copy(
            update={
                "status": status,
                "chunk_count": chunk_count
                if chunk_count is not None
                else self.chunk_count,
                "message": message,
                "updated_at": datetime.now(timezone.utc),
            }
        )


class DocumentResponse(BaseModel):
    """API shape for document listing/status/upload responses."""

    document_id: str
    id: str
    filename: str
    original_filename: str | None = None
    file_type: str | None = None
    status: DocumentProcessingStatus
    stage: DocumentProcessingStatus
    chunk_count: int | None = None
    total_blocks: int | None = None
    text_blocks: int | None = None
    table_blocks: int | None = None
    image_blocks: int | None = None
    ocr_blocks: int | None = None
    total_chunks: int | None = None
    ocr_chunks: int | None = None
    created_at: datetime
    uploaded_at: datetime
    message: str | None = None

    @classmethod
    def from_record(
        cls,
        record: StoredDocumentRecord,
        *,
        debug_stats: dict[str, int] | None = None,
    ) -> "DocumentResponse":
        stats = debug_stats or {}
        resolved_file_type = (
            record.file_type or Path(record.filename).suffix.lower().lstrip(".") or None
        )
        resolved_original_filename = record.original_filename or record.filename
        return cls(
            document_id=record.document_id,
            id=record.document_id,
            filename=record.filename,
            original_filename=resolved_original_filename,
            file_type=resolved_file_type,
            status=record.status,
            stage=record.status,
            chunk_count=record.chunk_count,
            total_blocks=stats.get("total_blocks"),
            text_blocks=stats.get("text_blocks"),
            table_blocks=stats.get("table_blocks"),
            image_blocks=stats.get("image_blocks"),
            ocr_blocks=stats.get("ocr_blocks"),
            total_chunks=stats.get("total_chunks"),
            ocr_chunks=stats.get("ocr_chunks"),
            created_at=record.created_at,
            uploaded_at=record.created_at,
            message=record.message,
        )


class DocumentListResponse(BaseModel):
    """API payload for document collection response."""

    documents: list[DocumentResponse] = Field(default_factory=list)


class DeleteAllDocumentsResponse(BaseModel):
    """API payload for deleting all uploaded documents."""

    status: Literal["deleted"] = "deleted"
    deleted_documents: int = 0
    deleted_files: int = 0


class DeleteDocumentResponse(BaseModel):
    """API payload for deleting one uploaded document."""

    status: Literal["deleted"] = "deleted"
    document_id: str
    remaining_documents: int = 0
    deleted_files: int = 0


class ChunkSettingsRequest(BaseModel):
    """Request payload for chunk strategy updates and reindexing."""

    chunk_size: int = Field(ge=100)
    chunk_overlap: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkSettingsRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self


ChunkingMode = Literal["small", "medium", "large", "custom"]
ChunkConfigMode = Literal["preset", "custom"]
RetrievalMode = Literal["low", "balanced", "high", "custom"]
RetrievalConfigMode = Literal["preset", "custom"]


class ChunkingSettingsRequest(BaseModel):
    """Request payload for chunk mode + optional custom settings."""

    mode: ChunkingMode
    chunk_size: int | None = None
    chunk_overlap: int | None = None

    @model_validator(mode="after")
    def validate_mode_payload(self) -> "ChunkingSettingsRequest":
        if self.mode != "custom":
            return self

        if self.chunk_size is None or self.chunk_overlap is None:
            raise ValueError(
                "chunk_size and chunk_overlap are required when mode=custom."
            )
        if self.chunk_size < 100 or self.chunk_size > 4000:
            raise ValueError(
                "chunk_size must be between 100 and 4000 when mode=custom."
            )
        if self.chunk_overlap < 0 or self.chunk_overlap > 1000:
            raise ValueError(
                "chunk_overlap must be between 0 and 1000 when mode=custom."
            )
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        return self


class ReindexDocumentsResponse(BaseModel):
    """API payload returned after chunk-setting reindex."""

    status: Literal["reindexed"] = "reindexed"
    chunk_size: int
    chunk_overlap: int
    reindexed_documents: int = 0
    active_chunks: int = 0


class ChunkingSettingsResponse(BaseModel):
    """API payload returned after applying chunking mode/settings."""

    status: Literal["reindexed"] = "reindexed"
    mode: ChunkingMode
    chunk_mode: ChunkConfigMode
    chunk_size: int
    chunk_overlap: int
    reindexed_documents: int = 0
    active_chunks: int = 0


class RetrievalSettingsRequest(BaseModel):
    """Request payload for retrieval mode + optional custom top_k."""

    mode: RetrievalMode
    top_k: int | None = None

    @model_validator(mode="after")
    def validate_mode_payload(self) -> "RetrievalSettingsRequest":
        if self.mode != "custom":
            return self

        if self.top_k is None:
            raise ValueError("top_k is required when mode=custom.")
        if self.top_k < 1 or self.top_k > 20:
            raise ValueError("top_k must be between 1 and 20 when mode=custom.")
        return self


class RetrievalSettingsResponse(BaseModel):
    """API payload returned after applying retrieval mode/settings."""

    status: Literal["updated"] = "updated"
    mode: RetrievalMode
    retrieval_mode: RetrievalConfigMode
    top_k: int
    rerank_top_n: int
    context_top_k: int
