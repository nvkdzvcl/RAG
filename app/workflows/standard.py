"""Standard workflow implementation for baseline RAG."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.generation import BaselineGenerator, StubLLMClient
from app.indexing import HashEmbeddingProvider, IndexBuilder
from app.ingestion.chunker import Chunker
from app.ingestion.cleaner import TextCleaner
from app.retrieval import ContextSelector, DenseRetriever, HybridRetriever, KeywordOverlapReranker, SparseRetriever
from app.schemas.api import StandardQueryResponse
from app.schemas.common import Mode
from app.schemas.generation import GeneratedAnswer
from app.schemas.ingestion import LoadedDocument
from app.schemas.retrieval import RetrievalResult
from app.workflows.shared import normalize_query

_DEFAULT_STANDARD_CORPUS: list[LoadedDocument] = [
    LoadedDocument(
        doc_id="std_doc_001",
        source="memory://standard_overview",
        title="Standard Workflow Overview",
        section="Pipeline",
        page=None,
        content=(
            "Standard mode follows a baseline RAG pipeline: retrieve, rerank, "
            "select context, generate grounded answer, and attach citations."
        ),
        metadata={"seed": True},
    ),
    LoadedDocument(
        doc_id="std_doc_002",
        source="memory://hybrid_retrieval",
        title="Hybrid Retrieval",
        section="Retrieval",
        page=None,
        content=(
            "Hybrid retrieval combines dense semantic matching and sparse BM25 matching "
            "to improve recall across query types."
        ),
        metadata={"seed": True},
    ),
    LoadedDocument(
        doc_id="std_doc_003",
        source="memory://grounded_generation",
        title="Grounded Generation",
        section="Generation",
        page=None,
        content=(
            "Grounded generation should rely on selected context and return citations "
            "for each supporting chunk."
        ),
        metadata={"seed": True},
    ),
]


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
    ) -> None:
        self.hybrid_top_k = hybrid_top_k
        self.rerank_top_k = rerank_top_k
        self.context_top_k = context_top_k

        embedding_provider = HashEmbeddingProvider(dimension=64)
        cleaner = TextCleaner()
        chunker = Chunker(chunk_size=320, chunk_overlap=40)
        chunks = chunker.chunk_documents(cleaner.clean_documents(_DEFAULT_STANDARD_CORPUS))

        built = IndexBuilder(embedding_provider=embedding_provider).build(chunks)
        dense = DenseRetriever(built.vector_index, embedding_provider)
        sparse = SparseRetriever(built.bm25_index)

        self.retriever = HybridRetriever(dense, sparse)
        self.reranker = KeywordOverlapReranker()
        self.context_selector = ContextSelector(max_chunks=context_top_k, max_chars=context_max_chars)
        self.generator = BaselineGenerator(llm_client=StubLLMClient())

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
