"""Runtime index management shared by workflows and document ingestion."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock

from app.core.config import get_settings
from app.indexing import BaseEmbeddingProvider, IndexBuilder, LocalIndexStore, create_embedding_provider
from app.ingestion.base_loader import build_doc_id
from app.ingestion import BaseLoader, Chunker, DocxLoader, MarkdownLoader, PdfLoader, TextCleaner, TextLoader
from app.retrieval import DenseRetriever, HybridRetriever, SparseRetriever
from app.schemas.ingestion import DocumentChunk, LoadedDocument
from app.schemas.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._retriever = retriever
        self._allowed_doc_ids = set(allowed_doc_ids or set())

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            return []

        candidate_k = top_k
        if self._allowed_doc_ids:
            # Ask for a wider candidate set so filtering by active doc IDs still returns enough rows.
            candidate_k = max(top_k * 4, top_k)

        results = self._retriever.retrieve(query, top_k=candidate_k)
        if not self._allowed_doc_ids:
            return results[:top_k]

        filtered = [item for item in results if item.doc_id in self._allowed_doc_ids]
        return filtered[:top_k]


class RuntimeIndexManager:
    """Build and swap active retrieval indexes at runtime."""

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

    def _ingest_files(self, paths: list[Path], *, source_label: str) -> list[LoadedDocument]:
        loaded: list[LoadedDocument] = []
        for path in sorted(paths):
            if not path.exists() or not path.is_file():
                continue
            loader = self._resolve_loader(path)
            if loader is None:
                continue

            metadata = {
                "source_collection": source_label,
                "relative_path": path.name,
                "file_extension": path.suffix.lower(),
            }
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

    def _activate_chunks(
        self,
        chunks: list[DocumentChunk],
        *,
        source: str,
        active_uploaded_doc_ids: set[str] | None = None,
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
            self.index_store.save_vector_index(built.vector_index, filename="uploaded_vector_index.json")
            self.index_store.save_bm25_index(built.bm25_index, filename="uploaded_bm25_index.json")
        elif source == "seeded":
            self.index_store.save_vector_index(built.vector_index, filename="seeded_vector_index.json")
            self.index_store.save_bm25_index(built.bm25_index, filename="seeded_bm25_index.json")

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
        chunks = self._build_chunks(uploaded_files, source_label="uploaded")
        expected_doc_ids = set(active_document_ids or set())
        if not expected_doc_ids:
            expected_doc_ids = {
                build_doc_id(path)
                for path in uploaded_files
                if path.exists() and path.is_file()
            }
        if expected_doc_ids:
            chunks = [chunk for chunk in chunks if chunk.doc_id in expected_doc_ids]
        return self._activate_chunks(
            chunks,
            source="uploaded",
            active_uploaded_doc_ids=expected_doc_ids,
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

    def get_retriever(self) -> HybridRetriever | EmptyRetriever | _ResultFilteringRetriever:
        with self._lock:
            retriever = self._retriever
            active_source = self._active_source
            uploaded_doc_ids = set(self._active_uploaded_doc_ids)
        if active_source == "uploaded":
            return _ResultFilteringRetriever(
                retriever,
                allowed_doc_ids=uploaded_doc_ids,
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

        deleted_files = 0
        index_root = self.index_dir.resolve()
        target_filenames = [
            "uploaded_vector_index.json",
            "uploaded_bm25_index.json",
        ]
        # Legacy uploaded artifacts were persisted under generic names.
        if previous_source != "seeded":
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
