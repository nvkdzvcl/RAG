"""Document upload and status routes."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.schemas.documents import (
    ChunkSettingsRequest,
    DeleteAllDocumentsResponse,
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentResponse,
    ReindexDocumentsResponse,
)
from app.services.document_service import DocumentService
from app.services.runtime import get_document_service

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED
)
async def upload_document(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Upload a document and process it through ingestion/indexing."""
    try:
        return await asyncio.to_thread(document_service.upload_document, file)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document_compat(
    file: UploadFile = File(...),
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Backward-compatible upload route."""
    return await upload_document(file=file, document_service=document_service)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentListResponse:
    """List uploaded documents and their processing status."""
    return await asyncio.to_thread(document_service.list_documents)


@router.post("/reindex", response_model=ReindexDocumentsResponse)
async def reindex_documents(
    payload: ChunkSettingsRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> ReindexDocumentsResponse:
    """Reindex ready uploaded documents using new chunk strategy."""
    try:
        return await asyncio.to_thread(document_service.update_chunk_settings, payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc


@router.get("/{document_id}/status", response_model=DocumentResponse)
async def get_document_status(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Return processing status for one document."""
    try:
        return await asyncio.to_thread(
            document_service.get_document_status, document_id
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document_status_compat(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DocumentResponse:
    """Backward-compatible status route."""
    return await get_document_status(
        document_id=document_id, document_service=document_service
    )


@router.delete("", response_model=DeleteAllDocumentsResponse)
async def delete_all_documents(
    document_service: DocumentService = Depends(get_document_service),
) -> DeleteAllDocumentsResponse:
    """Delete all uploaded documents, raw files, and uploaded runtime indexes."""
    return await asyncio.to_thread(document_service.delete_all_documents)


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
async def delete_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service),
) -> DeleteDocumentResponse:
    """Delete one uploaded document and rebuild runtime indexes from remaining ready documents."""
    try:
        return await asyncio.to_thread(document_service.delete_document, document_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
