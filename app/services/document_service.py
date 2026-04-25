"""Service for document upload, processing status, and runtime index refresh."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import UploadFile

from app.core.config import get_settings
from app.ingestion import BaseLoader, Chunker, DocxLoader, MarkdownLoader, PdfLoader, TextCleaner, TextLoader
from app.ingestion.base_loader import build_doc_id
from app.schemas.documents import (
    DeleteAllDocumentsResponse,
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentProcessingStatus,
    DocumentResponse,
    StoredDocumentRecord,
)
from app.schemas.ingestion import DocumentChunk, LoadedDocument
from app.services.index_runtime import RuntimeIndexManager

logger = logging.getLogger(__name__)


@dataclass
class UploadDebugStats:
    total_blocks: int = 0
    text_blocks: int = 0
    table_blocks: int = 0
    image_blocks: int = 0
    ocr_blocks: int = 0
    total_chunks: int = 0
    ocr_chunks: int = 0
    indexed_ocr_chunks: int = 0

    def response_payload(self) -> dict[str, int]:
        return {
            "total_blocks": self.total_blocks,
            "text_blocks": self.text_blocks,
            "table_blocks": self.table_blocks,
            "image_blocks": self.image_blocks,
            "ocr_blocks": self.ocr_blocks,
            "total_chunks": self.total_chunks,
        }


class DocumentService:
    """Manage uploaded documents and keep runtime retrieval indexes in sync."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}

    def __init__(
        self,
        *,
        data_dir: Path | str,
        raw_dir: Path | str,
        index_manager: RuntimeIndexManager,
        chunk_size: int = 320,
        chunk_overlap: int = 40,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.raw_dir = Path(raw_dir)
        self.registry_path = self.data_dir / "document_registry.json"
        self.index_manager = index_manager
        self.cleaner = TextCleaner()
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.loaders: list[BaseLoader] = [MarkdownLoader(), TextLoader(), PdfLoader(), DocxLoader()]
        self._lock = RLock()

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self._records = self._load_records()
        self._refresh_runtime_indexes()

    def _resolve_loader(self, path: Path) -> BaseLoader | None:
        for loader in self.loaders:
            if loader.supports(path):
                return loader
        return None

    def _load_records(self) -> dict[str, StoredDocumentRecord]:
        if not self.registry_path.exists():
            return {}

        try:
            payload = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Invalid document registry JSON. Starting with an empty registry.")
            return {}

        records: dict[str, StoredDocumentRecord] = {}
        for raw_record in payload.get("documents", []):
            record = StoredDocumentRecord.model_validate(raw_record)
            if record.status == DocumentProcessingStatus.READY and not Path(record.stored_path).exists():
                record = record.with_status(
                    DocumentProcessingStatus.FAILED,
                    message="Stored file not found on disk.",
                )
            records[record.document_id] = record
        return records

    def _persist_records(self) -> None:
        ordered = sorted(self._records.values(), key=lambda item: item.created_at, reverse=True)
        payload = {"documents": [record.model_dump(mode="json") for record in ordered]}
        self.registry_path.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _ready_paths(self) -> list[Path]:
        with self._lock:
            paths: list[Path] = []
            for record in self._records.values():
                if record.status != DocumentProcessingStatus.READY:
                    continue
                path = Path(record.stored_path)
                if path.exists() and path.is_file():
                    paths.append(path)
            return sorted(paths)

    def _refresh_runtime_indexes(self) -> None:
        ready_paths = self._ready_paths()
        try:
            self.index_manager.refresh(ready_paths)
        except Exception:  # pragma: no cover - defensive fallback path.
            logger.exception("Failed to refresh indexes from uploaded docs, falling back to seeded corpus.")
            self.index_manager.activate_from_seeded_corpus()

    @staticmethod
    def _is_within_directory(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    def _safe_delete_uploaded_file(self, file_path: Path) -> bool:
        if not self._is_within_directory(file_path, self.raw_dir):
            logger.warning(
                "Skipped deleting file outside configured raw_dir",
                extra={"path": str(file_path), "raw_dir": str(self.raw_dir)},
            )
            return False

        if not file_path.exists() or not file_path.is_file():
            return False

        try:
            file_path.unlink()
            return True
        except OSError:
            logger.exception("Failed to delete uploaded file", extra={"path": str(file_path)})
            return False

    def _rebuild_after_deletion(self) -> None:
        ready_paths = self._ready_paths()
        if ready_paths:
            self.index_manager.refresh(ready_paths)
            return

        self.index_manager.clear_uploaded_indexes()
        self.index_manager.activate_from_seeded_corpus()

    def _upsert_record(self, record: StoredDocumentRecord) -> None:
        with self._lock:
            self._records[record.document_id] = record
            self._persist_records()

    def _update_status(
        self,
        document_id: str,
        status: DocumentProcessingStatus,
        *,
        chunk_count: int | None = None,
        message: str | None = None,
    ) -> StoredDocumentRecord:
        with self._lock:
            record = self._records[document_id]
            updated = record.with_status(status, chunk_count=chunk_count, message=message)
            self._records[document_id] = updated
            self._persist_records()
            return updated

    def _load_file(self, file_path: Path) -> list[LoadedDocument]:
        loader = self._resolve_loader(file_path)
        if loader is None:
            raise ValueError(f"Unsupported file type: {file_path.suffix.lower()}")
        if file_path.suffix.lower() == ".pdf":
            settings = get_settings()
            logger.info(
                (
                    "Upload OCR config | file=%s | OCR_ENABLED=%s | OCR_LANGUAGE=%s "
                    "| OCR_MIN_TEXT_CHARS=%s | OCR_RENDER_DPI=%s | OCR_CONFIDENCE_THRESHOLD=%s"
                ),
                str(file_path),
                settings.ocr_enabled,
                settings.ocr_language,
                settings.ocr_min_text_chars,
                settings.ocr_render_dpi,
                settings.ocr_confidence_threshold,
            )
        return loader.load(
            file_path,
            metadata={
                "source_collection": "uploaded",
                "relative_path": file_path.name,
                "file_extension": file_path.suffix.lower(),
            },
        )

    def _chunk_file(self, file_path: Path) -> tuple[list[DocumentChunk], UploadDebugStats]:
        loaded = self._load_file(file_path)
        debug_stats = UploadDebugStats(
            total_blocks=len(loaded),
            text_blocks=sum(1 for doc in loaded if doc.block_type == "text"),
            table_blocks=sum(1 for doc in loaded if doc.block_type == "table"),
            image_blocks=sum(1 for doc in loaded if doc.block_type == "image"),
            ocr_blocks=sum(
                1
                for doc in loaded
                if doc.metadata.get("block_type") == "ocr_text" or bool(doc.metadata.get("ocr"))
            ),
        )
        cleaned = self.cleaner.clean_documents(loaded)
        chunks = self.chunker.chunk_documents(cleaned)
        debug_stats.total_chunks = len(chunks)
        debug_stats.ocr_chunks = sum(
            1
            for chunk in chunks
            if chunk.metadata.get("block_type") == "ocr_text" or bool(chunk.metadata.get("ocr"))
        )
        logger.info(
            (
                "Upload block/chunk stats | file=%s | total_blocks=%s | text_blocks=%s "
                "| table_blocks=%s | image_blocks=%s | ocr_blocks=%s | total_chunks=%s | ocr_chunks=%s"
            ),
            str(file_path),
            debug_stats.total_blocks,
            debug_stats.text_blocks,
            debug_stats.table_blocks,
            debug_stats.image_blocks,
            debug_stats.ocr_blocks,
            debug_stats.total_chunks,
            debug_stats.ocr_chunks,
        )
        return chunks, debug_stats

    def _process_uploaded_document(self, record: StoredDocumentRecord) -> tuple[StoredDocumentRecord, UploadDebugStats]:
        file_path = Path(record.stored_path)
        self._update_status(record.document_id, DocumentProcessingStatus.SPLITTING)

        chunks, debug_stats = self._chunk_file(file_path)
        if not chunks:
            raise ValueError(f"No chunks produced from uploaded document: {record.filename}")
        per_document_chunk_count = len(chunks)

        candidate_paths = self._ready_paths()
        if file_path not in candidate_paths:
            candidate_paths.append(file_path)
            candidate_paths = sorted(candidate_paths)

        self._update_status(record.document_id, DocumentProcessingStatus.EMBEDDING)
        self._update_status(record.document_id, DocumentProcessingStatus.INDEXING)
        self.index_manager.activate_from_uploaded_files(candidate_paths)
        activation_stats = self.index_manager.get_last_activation_stats()
        indexed_ocr_chunks = activation_stats.get("ocr_chunks")
        debug_stats.indexed_ocr_chunks = int(indexed_ocr_chunks) if isinstance(indexed_ocr_chunks, int) else 0
        logger.info(
            (
                "Upload indexing stats | file=%s | indexed_total_chunks=%s | indexed_ocr_chunks=%s"
            ),
            str(file_path),
            activation_stats.get("chunk_count", 0),
            debug_stats.indexed_ocr_chunks,
        )

        ready_record = self._update_status(
            record.document_id,
            DocumentProcessingStatus.READY,
            chunk_count=per_document_chunk_count,
            message="Document is indexed and ready for retrieval.",
        )
        return ready_record, debug_stats

    def upload_document(self, file: UploadFile) -> DocumentResponse:
        if not file.filename:
            raise ValueError("Uploaded file must include a filename.")

        safe_filename = Path(file.filename).name
        extension = Path(safe_filename).suffix.lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. Supported types: PDF, DOCX, TXT, Markdown."
            )

        unique_name = f"{uuid4().hex[:10]}_{safe_filename}"
        stored_path = self.raw_dir / unique_name
        stored_path.write_bytes(file.file.read())

        document_id = build_doc_id(stored_path)
        created = StoredDocumentRecord.create(
            document_id=document_id,
            filename=safe_filename,
            stored_path=str(stored_path),
            status=DocumentProcessingStatus.UPLOADED,
        )
        self._upsert_record(created)

        try:
            ready, debug_stats = self._process_uploaded_document(created)
            return DocumentResponse.from_record(
                ready,
                debug_stats=debug_stats.response_payload(),
            )
        except Exception as exc:
            logger.exception("Failed processing uploaded document", extra={"document_id": document_id})
            failed = self._update_status(
                document_id,
                DocumentProcessingStatus.FAILED,
                message=str(exc),
            )
            return DocumentResponse.from_record(failed)

    def list_documents(self) -> DocumentListResponse:
        with self._lock:
            records = sorted(self._records.values(), key=lambda item: item.created_at, reverse=True)
        return DocumentListResponse(documents=[DocumentResponse.from_record(record) for record in records])

    def get_document_status(self, document_id: str) -> DocumentResponse:
        with self._lock:
            record = self._records.get(document_id)
        if record is None:
            raise KeyError(f"Document not found: {document_id}")
        return DocumentResponse.from_record(record)

    def delete_all_documents(self) -> DeleteAllDocumentsResponse:
        with self._lock:
            deleted_documents = len(self._records)
            stored_paths = [Path(record.stored_path) for record in self._records.values()]
            self._records.clear()
            self._persist_records()

        deleted_files = 0
        for file_path in stored_paths:
            if self._safe_delete_uploaded_file(file_path):
                deleted_files += 1

        # Also clean stray files under raw_dir, still bounded to raw_dir.
        for file_path in sorted(self.raw_dir.rglob("*")):
            if not file_path.is_file():
                continue
            if self._safe_delete_uploaded_file(file_path):
                deleted_files += 1

        self.index_manager.clear_uploaded_indexes()
        self.index_manager.activate_from_seeded_corpus()

        return DeleteAllDocumentsResponse(
            status="deleted",
            deleted_documents=deleted_documents,
            deleted_files=deleted_files,
        )

    def delete_document(self, document_id: str) -> DeleteDocumentResponse:
        with self._lock:
            record = self._records.get(document_id)
            if record is None:
                raise KeyError(f"Document not found: {document_id}")
            self._records.pop(document_id)
            self._persist_records()
            remaining_documents = len(self._records)

        deleted_files = 1 if self._safe_delete_uploaded_file(Path(record.stored_path)) else 0
        self._rebuild_after_deletion()

        return DeleteDocumentResponse(
            status="deleted",
            document_id=document_id,
            remaining_documents=remaining_documents,
            deleted_files=deleted_files,
        )
