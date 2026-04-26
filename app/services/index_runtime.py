"""Runtime index management shared by workflows and document ingestion."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from app.core.config import get_settings
from app.indexing import BaseEmbeddingProvider, IndexBuilder, LocalIndexStore, create_embedding_provider
from app.ingestion.base_loader import build_doc_id
from app.ingestion import BaseLoader, Chunker, DocxLoader, MarkdownLoader, PdfLoader, TextCleaner, TextLoader
from app.retrieval import DenseRetriever, HybridRetriever, SparseRetriever
from app.schemas.index_manifest import UploadedIndexFileEntry, UploadedIndexManifest
from app.schemas.ingestion import DocumentChunk, LoadedDocument
from app.schemas.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryMetadataFilters:
    """Optional query-time metadata filters for uploaded retrieval results."""

    doc_ids: tuple[str, ...] | None = None
    filenames: tuple[str, ...] | None = None
    file_types: tuple[str, ...] | None = None
    uploaded_after: datetime | None = None
    uploaded_before: datetime | None = None
    include_ocr: bool | None = None

    @staticmethod
    def _normalize_string_list(values: Any) -> tuple[str, ...] | None:
        if not isinstance(values, list):
            return None
        normalized = sorted({str(item).strip() for item in values if str(item).strip()})
        return tuple(normalized) if normalized else None

    @staticmethod
    def _coerce_datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            resolved = value
        elif isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            if candidate.endswith("Z"):
                candidate = candidate[:-1] + "+00:00"
            try:
                resolved = datetime.fromisoformat(candidate)
            except ValueError:
                return None
        else:
            return None

        if resolved.tzinfo is None:
            return resolved.replace(tzinfo=timezone.utc)
        return resolved.astimezone(timezone.utc)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "QueryMetadataFilters":
        if not payload:
            return cls()

        raw_doc_ids = cls._normalize_string_list(payload.get("doc_ids"))
        raw_filenames = cls._normalize_string_list(payload.get("filenames"))
        raw_file_types = cls._normalize_string_list(payload.get("file_types"))

        normalized_filenames = (
            tuple(sorted({Path(name).name.lower() for name in raw_filenames}))
            if raw_filenames
            else None
        )
        normalized_file_types = (
            tuple(
                sorted(
                    {
                        file_type.strip().lower().lstrip(".")
                        for file_type in raw_file_types
                        if file_type.strip()
                    }
                )
            )
            if raw_file_types
            else None
        )
        if normalized_file_types == tuple():
            normalized_file_types = None

        include_ocr = payload.get("include_ocr")
        resolved_include_ocr = include_ocr if isinstance(include_ocr, bool) else None

        return cls(
            doc_ids=raw_doc_ids,
            filenames=normalized_filenames,
            file_types=normalized_file_types,
            uploaded_after=cls._coerce_datetime(payload.get("uploaded_after")),
            uploaded_before=cls._coerce_datetime(payload.get("uploaded_before")),
            include_ocr=resolved_include_ocr,
        )

    def has_any_filter(self) -> bool:
        return any(
            [
                bool(self.doc_ids),
                bool(self.filenames),
                bool(self.file_types),
                self.uploaded_after is not None,
                self.uploaded_before is not None,
                self.include_ocr is not None,
            ]
        )

    def requires_uploaded_scope(self) -> bool:
        # Requirement-8 filters are applied to uploaded document metadata.
        return self.has_any_filter()

    def as_debug_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.doc_ids:
            payload["doc_ids"] = list(self.doc_ids)
        if self.filenames:
            payload["filenames"] = list(self.filenames)
        if self.file_types:
            payload["file_types"] = list(self.file_types)
        if self.uploaded_after is not None:
            payload["uploaded_after"] = self.uploaded_after.isoformat()
        if self.uploaded_before is not None:
            payload["uploaded_before"] = self.uploaded_before.isoformat()
        if self.include_ocr is not None:
            payload["include_ocr"] = self.include_ocr
        return payload

    def _matches_uploaded_time(self, metadata: dict[str, Any]) -> bool:
        if self.uploaded_after is None and self.uploaded_before is None:
            return True
        timestamp = self._coerce_datetime(metadata.get("uploaded_at") or metadata.get("created_at"))
        if timestamp is None:
            return False
        if self.uploaded_after is not None and timestamp < self.uploaded_after:
            return False
        if self.uploaded_before is not None and timestamp > self.uploaded_before:
            return False
        return True

    @staticmethod
    def _normalize_source_filename(source: str | None) -> str:
        if not source:
            return ""
        candidate = str(source).strip()
        if not candidate:
            return ""
        if "://" in candidate:
            candidate = candidate.split("://", maxsplit=1)[1]
        return Path(candidate).name.strip().lower()

    @staticmethod
    def _resolve_filename(metadata: dict[str, Any], source: str | None) -> str:
        filename = str(metadata.get("file_name") or metadata.get("filename") or "").strip().lower()
        if filename:
            return Path(filename).name
        return QueryMetadataFilters._normalize_source_filename(source)

    @staticmethod
    def _resolve_file_type(metadata: dict[str, Any], source: str | None) -> str:
        file_type = str(metadata.get("file_type") or metadata.get("file_extension") or "").strip().lower()
        if file_type.startswith("."):
            file_type = file_type[1:]
        if file_type:
            return file_type
        filename = QueryMetadataFilters._resolve_filename(metadata, source)
        if not filename:
            return ""
        return Path(filename).suffix.lower().lstrip(".")

    def matches(self, result: RetrievalResult) -> bool:
        metadata = dict(result.metadata)
        source = str(result.source or "").strip()

        if self.doc_ids and result.doc_id not in self.doc_ids:
            return False

        if self.filenames:
            filename = self._resolve_filename(metadata, source)
            if filename not in self.filenames:
                return False

        if self.file_types:
            file_type = self._resolve_file_type(metadata, source)
            if file_type not in self.file_types:
                return False

        if self.include_ocr is not None:
            is_ocr = bool(metadata.get("ocr")) or str(metadata.get("block_type", "")).strip().lower() == "ocr_text"
            if is_ocr != self.include_ocr:
                return False

        if not self._matches_uploaded_time(metadata):
            return False

        return True


class EmptyRetriever:
    """Fallback retriever used when no indexable chunks are available."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        _ = query
        _ = top_k
        return []


class _ResultFilteringRetriever:
    """Retriever wrapper that enforces allowed uploaded document IDs at query time."""

    def __init__(
        self,
        retriever: HybridRetriever | EmptyRetriever,
        *,
        allowed_doc_ids: set[str] | None = None,
        active_source: str = "none",
        query_filters: QueryMetadataFilters | None = None,
    ) -> None:
        self._retriever = retriever
        self._allowed_doc_ids = set(allowed_doc_ids or set())
        self._active_source = active_source
        self._query_filters = query_filters or QueryMetadataFilters()
        self._last_filter_debug: dict[str, Any] = {
            "applied_filters": self._query_filters.as_debug_payload(),
            "candidate_count_before_filter": 0,
            "candidate_count_after_filter": 0,
        }

    def get_last_filter_debug(self) -> dict[str, Any]:
        return dict(self._last_filter_debug)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        applied_filters = self._query_filters.as_debug_payload()
        if top_k <= 0:
            self._last_filter_debug = {
                "applied_filters": applied_filters,
                "candidate_count_before_filter": 0,
                "candidate_count_after_filter": 0,
            }
            return []

        if self._query_filters.requires_uploaded_scope() and self._active_source != "uploaded":
            # Explicit uploaded-metadata filters must not fall back to unrelated seeded corpus.
            self._last_filter_debug = {
                "applied_filters": applied_filters,
                "candidate_count_before_filter": 0,
                "candidate_count_after_filter": 0,
                "filtered_source": self._active_source,
            }
            return []

        candidate_k = top_k
        if self._allowed_doc_ids or self._query_filters.has_any_filter():
            # Ask for a wider candidate set so post-filter rows still have enough candidates.
            candidate_k = max(top_k * 8, top_k)

        results = self._retriever.retrieve(query, top_k=candidate_k)
        filtered = list(results)
        if self._allowed_doc_ids:
            # Keep stale-index safety first: only active uploaded doc IDs are allowed.
            filtered = [item for item in filtered if item.doc_id in self._allowed_doc_ids]
        if self._query_filters.has_any_filter():
            filtered = [item for item in filtered if self._query_filters.matches(item)]

        self._last_filter_debug = {
            "applied_filters": applied_filters,
            "candidate_count_before_filter": len(results),
            "candidate_count_after_filter": len(filtered),
        }
        return filtered[:top_k]


class RuntimeIndexManager:
    """Build and swap active retrieval indexes at runtime."""

    UPLOADED_VECTOR_FILENAME = "uploaded_vector_index.json"
    UPLOADED_BM25_FILENAME = "uploaded_bm25_index.json"
    UPLOADED_MANIFEST_FILENAME = "uploaded_index_manifest.json"
    SEEDED_VECTOR_FILENAME = "seeded_vector_index.json"
    SEEDED_BM25_FILENAME = "seeded_bm25_index.json"

    def __init__(
        self,
        *,
        corpus_dir: Path | str,
        index_dir: Path | str,
        chunk_size: int = 320,
        chunk_overlap: int = 40,
        embedding_provider: BaseEmbeddingProvider | None = None,
        embedding_provider_name: str | None = None,
        embedding_model: str | None = None,
        embedding_device: str | None = None,
        embedding_batch_size: int | None = None,
        embedding_normalize: bool | None = None,
        embedding_dimension: int | None = None,
    ) -> None:
        settings = get_settings()
        resolved_provider_name = embedding_provider_name if embedding_provider_name is not None else settings.embedding_provider
        resolved_model = embedding_model if embedding_model is not None else settings.embedding_model
        resolved_device = embedding_device if embedding_device is not None else settings.embedding_device
        resolved_batch_size = embedding_batch_size if embedding_batch_size is not None else settings.embedding_batch_size
        resolved_normalize = embedding_normalize if embedding_normalize is not None else settings.embedding_normalize
        resolved_dimension = embedding_dimension if embedding_dimension is not None else settings.embedding_hash_dimension

        self.corpus_dir = Path(corpus_dir)
        self.index_dir = Path(index_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_provider = embedding_provider or create_embedding_provider(
            provider_name=resolved_provider_name,
            model=resolved_model,
            device=resolved_device,
            batch_size=resolved_batch_size,
            normalize=resolved_normalize,
            fallback_hash_dimension=resolved_dimension,
        )
        self.index_store = LocalIndexStore(self.index_dir)
        self.cleaner = TextCleaner()
        self.chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.loaders: list[BaseLoader] = [MarkdownLoader(), TextLoader(), PdfLoader(), DocxLoader()]

        self._lock = RLock()
        self._retriever: HybridRetriever | EmptyRetriever = EmptyRetriever()
        self._active_source = "none"
        self._active_chunk_count = 0
        self._active_uploaded_doc_ids: set[str] = set()
        self._last_activation_stats: dict[str, int | str] = {
            "chunk_count": 0,
            "ocr_chunks": 0,
            "source": "none",
        }

        logger.info(
            "Initialized runtime embedding provider",
            extra={
                "embedding_provider": self.embedding_provider.name,
                "embedding_dimension": self.embedding_provider.dimension,
            },
        )

    def _resolve_loader(self, path: Path) -> BaseLoader | None:
        for loader in self.loaders:
            if loader.supports(path):
                return loader
        return None

    @staticmethod
    def _display_filename(path: Path, *, source_label: str) -> str:
        filename = path.name
        if source_label != "uploaded":
            return filename

        prefix, sep, remainder = filename.partition("_")
        if (
            sep
            and remainder
            and len(prefix) == 10
            and all(char in "0123456789abcdef" for char in prefix.lower())
        ):
            return remainder
        return filename

    def _ingest_files(self, paths: list[Path], *, source_label: str) -> list[LoadedDocument]:
        loaded: list[LoadedDocument] = []
        for path in sorted(paths):
            if not path.exists() or not path.is_file():
                continue
            loader = self._resolve_loader(path)
            if loader is None:
                continue

            extension = path.suffix.lower()
            stat = path.stat()
            created_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            doc_id = build_doc_id(path)
            display_filename = self._display_filename(path, source_label=source_label)
            metadata = {
                "source_collection": source_label,
                "relative_path": path.name,
                "filename": display_filename,
                "file_name": display_filename,
                "file_extension": extension,
                "file_type": extension.lstrip("."),
                "doc_id": doc_id,
                "created_at": created_at,
            }
            if source_label == "uploaded":
                metadata["uploaded_at"] = created_at
            loaded.extend(loader.load(path, metadata=metadata))
        return loaded

    def _build_chunks(self, paths: list[Path], *, source_label: str) -> list[DocumentChunk]:
        loaded = self._ingest_files(paths, source_label=source_label)
        if not loaded:
            logger.info(
                "Runtime ingestion stats | source=%s | loaded_blocks=0 | ocr_blocks=0 | total_chunks=0 | ocr_chunks=0",
                source_label,
            )
            return []

        ocr_blocks = sum(
            1
            for doc in loaded
            if doc.metadata.get("block_type") == "ocr_text" or bool(doc.metadata.get("ocr"))
        )
        cleaned = self.cleaner.clean_documents(loaded)
        chunks = self.chunker.chunk_documents(cleaned)
        ocr_chunks = sum(
            1
            for chunk in chunks
            if chunk.metadata.get("block_type") == "ocr_text" or bool(chunk.metadata.get("ocr"))
        )
        logger.info(
            (
                "Runtime ingestion stats | source=%s | loaded_blocks=%s | ocr_blocks=%s "
                "| total_chunks=%s | ocr_chunks=%s"
            ),
            source_label,
            len(loaded),
            ocr_blocks,
            len(chunks),
            ocr_chunks,
        )
        return chunks

    @staticmethod
    def _stable_hash(payload: dict[str, Any]) -> str:
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _build_uploaded_manifest(
        self,
        uploaded_files: list[Path],
        *,
        active_document_ids: set[str],
    ) -> UploadedIndexManifest:
        entries: list[UploadedIndexFileEntry] = []
        expected_doc_ids = set(active_document_ids)

        for path in sorted(uploaded_files):
            if not path.exists() or not path.is_file():
                continue
            doc_id = build_doc_id(path)
            if expected_doc_ids and doc_id not in expected_doc_ids:
                continue
            stat = path.stat()
            entries.append(
                UploadedIndexFileEntry(
                    doc_id=doc_id,
                    stored_path=str(path.resolve()),
                    size_bytes=int(stat.st_size),
                    modified_ns=int(stat.st_mtime_ns),
                )
            )

        if not expected_doc_ids:
            expected_doc_ids = {entry.doc_id for entry in entries}

        payload_without_fingerprint: dict[str, Any] = {
            "schema_version": 1,
            "source": "uploaded",
            "chunk_size": int(self.chunk_size),
            "chunk_overlap": int(self.chunk_overlap),
            "embedding_provider": self.embedding_provider.name,
            "embedding_dimension": int(self.embedding_provider.dimension),
            "active_doc_ids": sorted(expected_doc_ids),
            "files": [entry.model_dump(mode="json") for entry in entries],
        }
        fingerprint = self._stable_hash(payload_without_fingerprint)
        return UploadedIndexManifest.model_validate(
            {
                **payload_without_fingerprint,
                "fingerprint": fingerprint,
            }
        )

    def _manifest_path(self) -> Path:
        return self.index_dir / self.UPLOADED_MANIFEST_FILENAME

    def _save_uploaded_manifest(self, manifest: UploadedIndexManifest) -> None:
        payload = manifest.model_dump(mode="json")
        self._manifest_path().write_text(
            json.dumps(payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _load_uploaded_manifest(self) -> UploadedIndexManifest | None:
        path = self._manifest_path()
        if not path.exists() or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return UploadedIndexManifest.model_validate(payload)
        except Exception:
            logger.warning("Invalid uploaded index manifest; forcing uploaded index rebuild.")
            return None

    def _is_uploaded_manifest_match(
        self,
        *,
        expected: UploadedIndexManifest,
        stored: UploadedIndexManifest,
    ) -> bool:
        stored_without_fingerprint = stored.model_dump(mode="json")
        stored_without_fingerprint.pop("fingerprint", None)
        stored_fingerprint = self._stable_hash(stored_without_fingerprint)
        if stored.fingerprint != stored_fingerprint:
            return False
        return stored_fingerprint == expected.fingerprint

    def _remove_uploaded_persisted_artifacts(self, *, include_legacy_generic: bool) -> int:
        deleted_files = 0
        index_root = self.index_dir.resolve()
        target_filenames = [
            self.UPLOADED_VECTOR_FILENAME,
            self.UPLOADED_BM25_FILENAME,
            self.UPLOADED_MANIFEST_FILENAME,
        ]
        if include_legacy_generic:
            target_filenames.extend(["vector_index.json", "bm25_index.json"])

        for filename in target_filenames:
            candidate = (self.index_dir / filename).resolve()
            try:
                candidate.relative_to(index_root)
            except ValueError:
                logger.warning("Skipped deleting index file outside index_dir", extra={"path": str(candidate)})
                continue

            if candidate.exists() and candidate.is_file():
                try:
                    candidate.unlink()
                    deleted_files += 1
                except OSError:
                    logger.exception("Failed to delete persisted index file", extra={"path": str(candidate)})
        return deleted_files

    def _load_uploaded_indexes_from_disk(
        self,
        *,
        expected_manifest: UploadedIndexManifest,
    ) -> int | None:
        stored_manifest = self._load_uploaded_manifest()
        if stored_manifest is None:
            return None
        if not self._is_uploaded_manifest_match(expected=expected_manifest, stored=stored_manifest):
            logger.info("Uploaded index manifest mismatch; stale uploaded indexes will be rebuilt.")
            self._remove_uploaded_persisted_artifacts(include_legacy_generic=False)
            return None

        try:
            vector_index = self.index_store.load_vector_index(filename=self.UPLOADED_VECTOR_FILENAME)
            bm25_index = self.index_store.load_bm25_index(filename=self.UPLOADED_BM25_FILENAME)
        except Exception:
            logger.warning("Failed loading persisted uploaded indexes; rebuilding from ready documents.")
            self._remove_uploaded_persisted_artifacts(include_legacy_generic=False)
            return None

        ocr_chunk_count = sum(
            1
            for chunk in vector_index.chunks
            if chunk.metadata.get("block_type") == "ocr_text" or bool(chunk.metadata.get("ocr"))
        )
        chunk_count = len(vector_index.chunks)
        dense = DenseRetriever(vector_index, self.embedding_provider)
        sparse = SparseRetriever(bm25_index)
        hybrid = HybridRetriever(dense, sparse)

        with self._lock:
            self._retriever = hybrid
            self._active_source = "uploaded"
            self._active_chunk_count = chunk_count
            self._active_uploaded_doc_ids = set(expected_manifest.active_doc_ids)
            self._last_activation_stats = {
                "chunk_count": chunk_count,
                "ocr_chunks": ocr_chunk_count,
                "source": "uploaded",
            }

        logger.info(
            "Loaded uploaded runtime indexes from persisted artifacts | chunk_count=%s | ocr_chunks=%s",
            chunk_count,
            ocr_chunk_count,
        )
        return chunk_count

    def _activate_chunks(
        self,
        chunks: list[DocumentChunk],
        *,
        source: str,
        active_uploaded_doc_ids: set[str] | None = None,
        uploaded_files: list[Path] | None = None,
    ) -> int:
        resolved_uploaded_ids = set(active_uploaded_doc_ids or set())
        if source == "uploaded" and not resolved_uploaded_ids:
            resolved_uploaded_ids = {chunk.doc_id for chunk in chunks}

        if not chunks:
            with self._lock:
                self._retriever = EmptyRetriever()
                self._active_source = source
                self._active_chunk_count = 0
                self._active_uploaded_doc_ids = resolved_uploaded_ids if source == "uploaded" else set()
                self._last_activation_stats = {
                    "chunk_count": 0,
                    "ocr_chunks": 0,
                    "source": source,
                }
            return 0

        ocr_chunk_count = sum(
            1
            for chunk in chunks
            if chunk.metadata.get("block_type") == "ocr_text" or bool(chunk.metadata.get("ocr"))
        )
        built = IndexBuilder(embedding_provider=self.embedding_provider).build(chunks)
        dense = DenseRetriever(built.vector_index, self.embedding_provider)
        sparse = SparseRetriever(built.bm25_index)
        hybrid = HybridRetriever(dense, sparse)

        self.index_store.save_vector_index(built.vector_index)
        self.index_store.save_bm25_index(built.bm25_index)
        if source == "uploaded":
            self.index_store.save_vector_index(
                built.vector_index,
                filename=self.UPLOADED_VECTOR_FILENAME,
            )
            self.index_store.save_bm25_index(
                built.bm25_index,
                filename=self.UPLOADED_BM25_FILENAME,
            )
            manifest = self._build_uploaded_manifest(
                uploaded_files or [],
                active_document_ids=resolved_uploaded_ids,
            )
            self._save_uploaded_manifest(manifest)
        elif source == "seeded":
            self.index_store.save_vector_index(
                built.vector_index,
                filename=self.SEEDED_VECTOR_FILENAME,
            )
            self.index_store.save_bm25_index(
                built.bm25_index,
                filename=self.SEEDED_BM25_FILENAME,
            )

        with self._lock:
            self._retriever = hybrid
            self._active_source = source
            self._active_chunk_count = built.chunk_count
            self._active_uploaded_doc_ids = resolved_uploaded_ids if source == "uploaded" else set()
            self._last_activation_stats = {
                "chunk_count": built.chunk_count,
                "ocr_chunks": ocr_chunk_count,
                "source": source,
            }

        logger.info(
            (
                "Activated runtime indexes | source=%s | chunk_count=%s | indexed_ocr_chunks=%s "
                "| embedding_provider=%s"
            ),
            source,
            built.chunk_count,
            ocr_chunk_count,
            built.embedding_provider,
        )
        return built.chunk_count

    def activate_from_uploaded_files(
        self,
        uploaded_files: list[Path],
        *,
        active_document_ids: set[str] | None = None,
    ) -> int:
        expected_doc_ids = set(active_document_ids or set())
        if not expected_doc_ids:
            expected_doc_ids = {
                build_doc_id(path)
                for path in uploaded_files
                if path.exists() and path.is_file()
            }
        expected_manifest = self._build_uploaded_manifest(
            uploaded_files,
            active_document_ids=expected_doc_ids,
        )
        loaded_chunk_count = self._load_uploaded_indexes_from_disk(
            expected_manifest=expected_manifest,
        )
        if loaded_chunk_count is not None:
            return loaded_chunk_count

        chunks = self._build_chunks(uploaded_files, source_label="uploaded")
        if expected_doc_ids:
            chunks = [chunk for chunk in chunks if chunk.doc_id in expected_doc_ids]
        return self._activate_chunks(
            chunks,
            source="uploaded",
            active_uploaded_doc_ids=expected_doc_ids,
            uploaded_files=uploaded_files,
        )

    def activate_from_seeded_corpus(self) -> int:
        if not self.corpus_dir.exists() or not self.corpus_dir.is_dir():
            return self._activate_chunks([], source="seeded")

        candidate_files = [path for path in self.corpus_dir.rglob("*") if path.is_file()]
        chunks = self._build_chunks(candidate_files, source_label="seeded")
        return self._activate_chunks(chunks, source="seeded")

    def refresh(
        self,
        uploaded_files: list[Path],
        *,
        active_document_ids: set[str] | None = None,
    ) -> int:
        if uploaded_files:
            return self.activate_from_uploaded_files(
                uploaded_files,
                active_document_ids=active_document_ids,
            )
        return self.activate_from_seeded_corpus()

    def get_retriever(
        self,
        *,
        query_filters: dict[str, Any] | None = None,
    ) -> HybridRetriever | EmptyRetriever | _ResultFilteringRetriever:
        with self._lock:
            retriever = self._retriever
            active_source = self._active_source
            uploaded_doc_ids = set(self._active_uploaded_doc_ids)
        normalized_filters = QueryMetadataFilters.from_payload(query_filters)
        if active_source == "uploaded" or normalized_filters.has_any_filter():
            return _ResultFilteringRetriever(
                retriever,
                allowed_doc_ids=uploaded_doc_ids if active_source == "uploaded" else set(),
                active_source=active_source,
                query_filters=normalized_filters,
            )
        return retriever

    def get_active_source(self) -> str:
        with self._lock:
            return self._active_source

    def get_active_chunk_count(self) -> int:
        with self._lock:
            return self._active_chunk_count

    def get_active_uploaded_document_ids(self) -> set[str]:
        with self._lock:
            return set(self._active_uploaded_doc_ids)

    def get_last_activation_stats(self) -> dict[str, int | str]:
        with self._lock:
            return dict(self._last_activation_stats)

    def set_chunking(self, *, chunk_size: int, chunk_overlap: int) -> None:
        """Update runtime chunking strategy for subsequent ingest/reindex calls."""
        normalized_chunk_size = max(100, int(chunk_size))
        normalized_chunk_overlap = max(0, int(chunk_overlap))
        if normalized_chunk_overlap >= normalized_chunk_size:
            normalized_chunk_overlap = max(0, normalized_chunk_size // 5)

        with self._lock:
            self.chunk_size = normalized_chunk_size
            self.chunk_overlap = normalized_chunk_overlap
            self.chunker = Chunker(
                chunk_size=normalized_chunk_size,
                chunk_overlap=normalized_chunk_overlap,
            )

    def clear_uploaded_indexes(self) -> int:
        """Clear active uploaded indexes and remove persisted local index artifacts.

        Returns:
            Number of persisted index files deleted from index_dir.
        """
        with self._lock:
            previous_source = self._active_source
            if previous_source != "seeded":
                self._retriever = EmptyRetriever()
                self._active_source = "none"
                self._active_chunk_count = 0
                self._last_activation_stats = {
                    "chunk_count": 0,
                    "ocr_chunks": 0,
                    "source": "none",
                }
            self._active_uploaded_doc_ids = set()
        return self._remove_uploaded_persisted_artifacts(
            include_legacy_generic=(previous_source != "seeded"),
        )
