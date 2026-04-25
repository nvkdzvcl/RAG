"""Standard workflow implementation for baseline RAG."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.generation import BaselineGenerator, LLMClient, create_llm_client_from_settings
from app.indexing.bm25_index import BM25Index
from app.indexing.vector_index import VectorIndex
from app.indexing import BaseEmbeddingProvider, IndexBuilder, LocalIndexStore, create_embedding_provider
from app.ingestion import Chunker, DirectoryIngestor, TextCleaner
from app.retrieval import BaseReranker, ContextSelector, DenseRetriever, HybridRetriever, SparseRetriever, create_reranker
from app.schemas.api import StandardQueryResponse
from app.schemas.common import Mode
from app.schemas.generation import GeneratedAnswer
from app.schemas.ingestion import DocumentChunk
from app.schemas.retrieval import RetrievalResult
from app.services.index_runtime import EmptyRetriever, RuntimeIndexManager
from app.workflows.shared import normalize_query


@dataclass
class StandardPipelineResult:
    """Intermediate artifacts from one standard retrieval-generation pass."""

    normalized_query: str
    retrieved: list[RetrievalResult]
    reranked: list[RetrievalResult]
    selected_context: list[RetrievalResult]
    generated: GeneratedAnswer


class StandardWorkflow:
    """Baseline RAG workflow: retrieve -> rerank -> select context -> generate."""

    def __init__(
        self,
        *,
        hybrid_top_k: int = 8,
        rerank_top_k: int | None = None,
        context_top_k: int = 4,
        context_max_chars: int = 4000,
        corpus_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
        chunk_size: int = 320,
        chunk_overlap: int = 40,
        persist_indexes: bool = False,
        index_manager: RuntimeIndexManager | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        reranker: BaseReranker | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        settings = get_settings()
        self.hybrid_top_k = hybrid_top_k
        self.rerank_top_k = rerank_top_k if rerank_top_k is not None else settings.reranker_top_n
        self.context_top_k = context_top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.corpus_dir = Path(corpus_dir or settings.corpus_dir)
        self.index_dir = Path(index_dir or settings.index_dir)
        self.persist_indexes = persist_indexes
        self.index_manager = index_manager

        self._fallback_retriever: HybridRetriever | EmptyRetriever | None = None

        if self.index_manager is None:
            resolved_provider = embedding_provider or create_embedding_provider(
                provider_name=settings.embedding_provider,
                model=settings.embedding_model,
                device=settings.embedding_device,
                batch_size=settings.embedding_batch_size,
                normalize=settings.embedding_normalize,
                fallback_hash_dimension=settings.embedding_hash_dimension,
            )
            chunks = self._build_chunks_from_corpus()

            built = IndexBuilder(embedding_provider=resolved_provider).build(chunks)
            if self.persist_indexes:
                self._save_indexes(built.vector_index, built.bm25_index)

            dense = DenseRetriever(built.vector_index, resolved_provider)
            sparse = SparseRetriever(built.bm25_index)
            self._fallback_retriever = HybridRetriever(dense, sparse)

        self.reranker = reranker or create_reranker(
            provider_name=settings.reranker_provider,
            model=settings.reranker_model,
            device=settings.reranker_device,
            batch_size=settings.reranker_batch_size,
        )
        self.context_selector = ContextSelector(max_chunks=context_top_k, max_chars=context_max_chars)
        resolved_llm_client = llm_client or create_llm_client_from_settings(settings)
        self.generator = BaselineGenerator(
            llm_client=resolved_llm_client,
            prompt_dir=settings.prompt_dir,
        )
        self.llm_client = self.generator.llm_client

    def _build_chunks_from_corpus(self) -> list[DocumentChunk]:
        ingestor = DirectoryIngestor()
        loaded = ingestor.ingest_directory(
            self.corpus_dir,
            metadata={
                "corpus_dir": str(self.corpus_dir),
            },
        )
        cleaner = TextCleaner()
        chunker = Chunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = chunker.chunk_documents(cleaner.clean_documents(loaded))
        if not chunks:
            raise ValueError(f"No chunks were produced from corpus directory: {self.corpus_dir}")
        return chunks

    def _save_indexes(self, vector_index: VectorIndex, bm25_index: BM25Index) -> None:
        store = LocalIndexStore(self.index_dir)
        store.save_vector_index(vector_index)
        store.save_bm25_index(bm25_index)

    def _get_retriever(self) -> HybridRetriever | EmptyRetriever:
        if self.index_manager is not None:
            return self.index_manager.get_retriever()
        if self._fallback_retriever is None:
            return EmptyRetriever()
        return self._fallback_retriever

    def _get_index_source(self) -> str:
        if self.index_manager is None:
            return "seeded"
        return self.index_manager.get_active_source()

    def run_pipeline(self, query: str, mode: Mode = Mode.STANDARD) -> StandardPipelineResult:
        """Run one retrieval-generation pass and return intermediate artifacts."""
        normalized_query = normalize_query(query)
        retrieval_top_k = max(self.hybrid_top_k, self.rerank_top_k)
        retrieved = self._get_retriever().retrieve(normalized_query, top_k=retrieval_top_k)
        reranked = self.reranker.rerank(normalized_query, retrieved, top_k=self.rerank_top_k)
        selected_context = self.context_selector.select(reranked, top_k=self.context_top_k)
        generated = self.generator.generate_answer(
            query=normalized_query,
            context=selected_context,
            mode=mode,
        )

        return StandardPipelineResult(
            normalized_query=normalized_query,
            retrieved=retrieved,
            reranked=reranked,
            selected_context=selected_context,
            generated=generated,
        )

    def run(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
    ) -> StandardQueryResponse:
        start_time = time.perf_counter()
        _ = chat_history

        pipeline = self.run_pipeline(query=query, mode=Mode.STANDARD)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        trace = [
            {
                "step": "retrieve",
                "query": pipeline.normalized_query,
                "index_source": self._get_index_source(),
                "count": len(pipeline.retrieved),
                "chunk_ids": [item.chunk_id for item in pipeline.retrieved],
            },
            {
                "step": "rerank",
                "provider": getattr(self.reranker, "name", self.reranker.__class__.__name__),
                "count": len(pipeline.reranked),
                "chunk_ids": [item.chunk_id for item in pipeline.reranked],
                "docs": [
                    {
                        "chunk_id": item.chunk_id,
                        "doc_id": item.doc_id,
                        "rank": item.rank,
                        "rerank_score": item.rerank_score,
                        "score": item.score,
                        "dense_score": item.dense_score,
                        "sparse_score": item.sparse_score,
                    }
                    for item in pipeline.reranked
                ],
            },
            {
                "step": "context_select",
                "count": len(pipeline.selected_context),
                "chunk_ids": [item.chunk_id for item in pipeline.selected_context],
                "docs": [
                    {
                        "chunk_id": item.chunk_id,
                        "doc_id": item.doc_id,
                        "content": item.content,
                    }
                    for item in pipeline.selected_context
                ],
            },
            {
                "step": "generate",
                "status": pipeline.generated.status,
                "stop_reason": pipeline.generated.stop_reason,
            },
        ]

        return StandardQueryResponse(
            mode="standard",
            answer=pipeline.generated.answer,
            citations=pipeline.generated.citations,
            confidence=pipeline.generated.confidence,
            stop_reason=pipeline.generated.stop_reason,
            status=pipeline.generated.status,
            latency_ms=elapsed_ms,
            trace=trace,
        )
