"""Standard workflow implementation for baseline RAG."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.generation import BaselineGenerator, StubLLMClient
from app.indexing.bm25_index import BM25Index
from app.indexing.vector_index import VectorIndex
from app.indexing import HashEmbeddingProvider, IndexBuilder, LocalIndexStore
from app.ingestion import Chunker, DirectoryIngestor, TextCleaner
from app.retrieval import ContextSelector, DenseRetriever, HybridRetriever, KeywordOverlapReranker, SparseRetriever
from app.schemas.api import StandardQueryResponse
from app.schemas.common import Mode
from app.schemas.generation import GeneratedAnswer
from app.schemas.ingestion import DocumentChunk
from app.schemas.retrieval import RetrievalResult
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
        rerank_top_k: int = 5,
        context_top_k: int = 4,
        context_max_chars: int = 4000,
        corpus_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
        chunk_size: int = 320,
        chunk_overlap: int = 40,
        persist_indexes: bool = False,
    ) -> None:
        settings = get_settings()
        self.hybrid_top_k = hybrid_top_k
        self.rerank_top_k = rerank_top_k
        self.context_top_k = context_top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.corpus_dir = Path(corpus_dir or settings.corpus_dir)
        self.index_dir = Path(index_dir or settings.index_dir)
        self.persist_indexes = persist_indexes

        embedding_provider = HashEmbeddingProvider(dimension=64)
        chunks = self._build_chunks_from_corpus()

        built = IndexBuilder(embedding_provider=embedding_provider).build(chunks)
        if self.persist_indexes:
            self._save_indexes(built.vector_index, built.bm25_index)

        dense = DenseRetriever(built.vector_index, embedding_provider)
        sparse = SparseRetriever(built.bm25_index)

        self.retriever = HybridRetriever(dense, sparse)
        self.reranker = KeywordOverlapReranker()
        self.context_selector = ContextSelector(max_chunks=context_top_k, max_chars=context_max_chars)
        self.generator = BaselineGenerator(llm_client=StubLLMClient())

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

    def run_pipeline(self, query: str, mode: Mode = Mode.STANDARD) -> StandardPipelineResult:
        """Run one retrieval-generation pass and return intermediate artifacts."""
        normalized_query = normalize_query(query)
        retrieved = self.retriever.retrieve(normalized_query, top_k=self.hybrid_top_k)
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
                "count": len(pipeline.retrieved),
                "chunk_ids": [item.chunk_id for item in pipeline.retrieved],
            },
            {
                "step": "rerank",
                "count": len(pipeline.reranked),
                "chunk_ids": [item.chunk_id for item in pipeline.reranked],
            },
            {
                "step": "context_select",
                "count": len(pipeline.selected_context),
                "chunk_ids": [item.chunk_id for item in pipeline.selected_context],
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
