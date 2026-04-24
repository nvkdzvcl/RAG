"""Schemas for document upload and processing APIs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

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
    created_at: datetime
    message: str | None = None

    @classmethod
    def from_record(cls, record: StoredDocumentRecord) -> "DocumentResponse":
        return cls(
            document_id=record.document_id,
            id=record.document_id,
            filename=record.filename,
            status=record.status,
            stage=record.status,
            chunk_count=record.chunk_count,
            created_at=record.created_at,
            message=record.message,
        )


class DocumentListResponse(BaseModel):
    """API payload for document collection response."""

    documents: list[DocumentResponse] = Field(default_factory=list)
