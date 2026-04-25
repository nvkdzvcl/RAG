"""Schemas for document upload, delete, and processing APIs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


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
        stored_path: str,
        status: DocumentProcessingStatus = DocumentProcessingStatus.UPLOADED,
    ) -> "StoredDocumentRecord":
        now = datetime.now(timezone.utc)
        return cls(
            document_id=document_id,
            filename=filename,
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
                "chunk_count": chunk_count if chunk_count is not None else self.chunk_count,
                "message": message,
                "updated_at": datetime.now(timezone.utc),
            }
        )


class DocumentResponse(BaseModel):
    """API shape for document listing/status/upload responses."""

    document_id: str
    id: str
    filename: str
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
    message: str | None = None

    @classmethod
    def from_record(
        cls,
        record: StoredDocumentRecord,
        *,
        debug_stats: dict[str, int] | None = None,
    ) -> "DocumentResponse":
        stats = debug_stats or {}
        return cls(
            document_id=record.document_id,
            id=record.document_id,
            filename=record.filename,
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
