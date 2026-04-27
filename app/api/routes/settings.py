"""Settings routes for runtime chunking configuration."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.documents import (
    ChunkingSettingsRequest,
    ChunkingSettingsResponse,
    RetrievalSettingsRequest,
    RetrievalSettingsResponse,
)
from app.services.document_service import DocumentService
from app.services.query_service import QueryService
from app.services.runtime import get_document_service, get_query_service

router = APIRouter(prefix="/settings", tags=["settings"])


@router.post("/chunking", response_model=ChunkingSettingsResponse)
async def update_chunking_settings(
    payload: ChunkingSettingsRequest,
    document_service: DocumentService = Depends(get_document_service),
) -> ChunkingSettingsResponse:
    """Apply preset/custom chunking mode and trigger uploaded-document reindex."""
    try:
        return await asyncio.to_thread(document_service.apply_chunking_settings, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/retrieval", response_model=RetrievalSettingsResponse)
async def update_retrieval_settings(
    payload: RetrievalSettingsRequest,
    query_service: QueryService = Depends(get_query_service),
) -> RetrievalSettingsResponse:
    """Apply preset/custom retrieval top_k without changing indexing artifacts."""
    try:
        return await asyncio.to_thread(query_service.update_retrieval_settings, payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
