"""Application runtime service wiring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import Request

from app.core.config import Settings
from app.services.document_service import DocumentService
from app.services.index_runtime import RuntimeIndexManager
from app.services.query_service import QueryService
from app.workflows.runner import WorkflowRunner


@dataclass
class AppServices:
    """Container for request-independent service singletons."""

    index_manager: RuntimeIndexManager
    query_service: QueryService
    document_service: DocumentService


def build_app_services(settings: Settings) -> AppServices:
    """Build all shared runtime services for one FastAPI app instance."""
    data_dir = Path(settings.data_dir)
    raw_dir = data_dir / "raw"

    index_manager = RuntimeIndexManager(
        corpus_dir=settings.corpus_dir,
        index_dir=settings.index_dir,
    )
    document_service = DocumentService(
        data_dir=data_dir,
        raw_dir=raw_dir,
        index_manager=index_manager,
    )
    query_service = QueryService(
        runner=WorkflowRunner(index_manager=index_manager),
    )

    return AppServices(
        index_manager=index_manager,
        query_service=query_service,
        document_service=document_service,
    )


def get_app_services(request: Request) -> AppServices:
    """Retrieve shared services attached to the current app instance."""
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise RuntimeError("Application services are not initialized.")
    return services


def get_query_service(request: Request) -> QueryService:
    """FastAPI dependency provider for QueryService."""
    return get_app_services(request).query_service


def get_document_service(request: Request) -> DocumentService:
    """FastAPI dependency provider for DocumentService."""
    return get_app_services(request).document_service
