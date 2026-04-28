"""Standard workflow implementation for baseline RAG."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.async_utils import run_coro_sync
from app.core.cache import create_cache_group_from_settings
from app.core.config import get_settings
from app.core.json_utils import parse_json_object
from app.generation import (
    BaselineGenerator,
    LLMClient,
    close_llm_client,
    create_llm_client_from_settings,
)
from app.generation.llm_client import complete_with_model
from app.indexing.bm25_index import BM25Index
from app.indexing.vector_index import VectorIndex
from app.indexing import (
    BaseEmbeddingProvider,
    IndexBuilder,
    LocalIndexStore,
    create_embedding_provider,
)
from app.ingestion import Chunker, DirectoryIngestor, TextCleaner
from app.retrieval import (
    BaseReranker,
    ContextSelector,
    DenseRetriever,
    HybridRetriever,
    SparseRetriever,
    create_reranker,
)
from app.schemas.api import StandardQueryResponse
from app.schemas.common import Mode
from app.schemas.generation import GeneratedAnswer
from app.schemas.ingestion import DocumentChunk
from app.schemas.retrieval import RetrievalResult
from app.services.index_runtime import EmptyRetriever, RuntimeIndexManager
from app.workflows.shared import (
    assess_grounding,
    build_chat_history_context,
    build_language_system_prompt,
    detect_response_language,
    is_language_mismatch,
    localized_insufficient_evidence,
    normalize_query,
    response_language_name,
    trim_chat_history,
)
from app.workflows.streaming import StreamEventHandler, emit_stream_event


logger = logging.getLogger(__name__)


@dataclass
class StandardPipelineResult:
    """Intermediate artifacts from one standard retrieval-generation pass."""

    normalized_query: str
    retrieved: list[RetrievalResult]
    reranked: list[RetrievalResult]
    selected_context: list[RetrievalResult]
    generated: GeneratedAnswer
    retrieval_debug: dict[str, Any] = field(default_factory=dict)


class StandardWorkflow:
    """Baseline RAG workflow: retrieve -> rerank -> select context -> generate."""

    _LANGUAGE_REWRITE_PROMPT = (
        "Rewrite the answer into $response_language_name while keeping the same grounded meaning.\n"
        "Do not add new facts.\n"
        "Do not use Chinese unless explicitly requested.\n"
        'Return strict JSON only: {"answer": "string"}\n'
        "response_language: $response_language\n"
        "question: $question\n"
        "answer:\n$draft_answer"
    )

    def __init__(
        self,
        *,
        hybrid_top_k: int | None = None,
        rerank_top_k: int | None = None,
        context_top_k: int = 4,
        context_max_chars: int = 4000,
        corpus_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        persist_indexes: bool = False,
        index_manager: RuntimeIndexManager | None = None,
        embedding_provider: BaseEmbeddingProvider | None = None,
        reranker: BaseReranker | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        settings = get_settings()
        configured_top_k = (
            hybrid_top_k
            if hybrid_top_k is not None
            else int(getattr(settings, "retrieval_top_k", 8))
        )
        self.hybrid_top_k = max(1, int(configured_top_k))
        configured_rerank = (
            rerank_top_k if rerank_top_k is not None else int(settings.reranker_top_n)
        )
        self.configured_rerank_top_n = max(1, int(configured_rerank))
        self.rerank_top_k = min(self.configured_rerank_top_n, self.hybrid_top_k)
        self.context_top_k = context_top_k
        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else int(getattr(settings, "chunk_size", 320))
        )
        self.chunk_overlap = (
            chunk_overlap
            if chunk_overlap is not None
            else int(getattr(settings, "chunk_overlap", 40))
        )
        self.memory_window = max(0, int(getattr(settings, "memory_window", 3)))
        self.corpus_dir = Path(corpus_dir or settings.corpus_dir)
        self.index_dir = Path(index_dir or settings.index_dir)
        self.persist_indexes = persist_indexes
        self.index_manager = index_manager
        self.caches = create_cache_group_from_settings(settings)

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

            dense = DenseRetriever(
                built.vector_index,
                resolved_provider,
                embedding_cache=self.caches.embedding,
            )
            sparse = SparseRetriever(built.bm25_index)
            self._fallback_retriever = HybridRetriever(
                dense,
                sparse,
                retrieval_cache=self.caches.retrieval,
            )

        self.reranker = reranker or create_reranker(
            provider_name=settings.reranker_provider,
            model=settings.reranker_model,
            device=settings.reranker_device,
            batch_size=settings.reranker_batch_size,
        )
        self.context_selector = ContextSelector(
            max_chunks=context_top_k, max_chars=context_max_chars
        )
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
            raise ValueError(
                f"No chunks were produced from corpus directory: {self.corpus_dir}"
            )
        return chunks

    def _save_indexes(self, vector_index: VectorIndex, bm25_index: BM25Index) -> None:
        store = LocalIndexStore(self.index_dir)
        store.save_vector_index(vector_index)
        store.save_bm25_index(bm25_index)

    def _get_retriever(
        self,
        *,
        query_filters: dict[str, Any] | None = None,
    ) -> HybridRetriever | EmptyRetriever:
        if self.index_manager is not None:
            getter = getattr(self.index_manager, "get_retriever")
            if query_filters is None:
                return getter()
            try:
                return getter(query_filters=query_filters)
            except TypeError:
                # Backward-compatible with tests/fakes exposing legacy get_retriever() signature.
                return getter()
        if self._fallback_retriever is None:
            return EmptyRetriever()
        return self._fallback_retriever

    def _get_index_source(self) -> str:
        if self.index_manager is None:
            return "seeded"
        return self.index_manager.get_active_source()

    def _sync_chunk_settings_from_runtime(self) -> None:
        """Keep workflow trace chunk settings aligned with runtime reindex settings."""
        if self.index_manager is None:
            return

        runtime_chunk_size = getattr(self.index_manager, "chunk_size", None)
        runtime_chunk_overlap = getattr(self.index_manager, "chunk_overlap", None)

        if isinstance(runtime_chunk_size, int) and runtime_chunk_size > 0:
            self.chunk_size = runtime_chunk_size
        if isinstance(runtime_chunk_overlap, int) and runtime_chunk_overlap >= 0:
            self.chunk_overlap = runtime_chunk_overlap

    def set_retrieval_top_k(self, top_k: int) -> None:
        """Update retrieval top_k and keep rerank top_n bounded."""
        self.hybrid_top_k = max(1, int(top_k))
        self.rerank_top_k = min(self.configured_rerank_top_n, self.hybrid_top_k)

    def get_retrieval_top_k(self) -> int:
        """Return effective retrieval top_k."""
        return self.hybrid_top_k

    def get_rerank_top_n(self) -> int:
        """Return effective rerank top_n (always <= retrieval top_k)."""
        return self.rerank_top_k

    async def run_pipeline(
        self,
        query: str,
        mode: Mode = Mode.STANDARD,
        model: str | None = None,
        response_language: str = "en",
        chat_history: list[dict[str, str]] | None = None,
        query_filters: dict[str, Any] | None = None,
        event_handler: StreamEventHandler | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> StandardPipelineResult:
        """Run one retrieval-generation pass and return intermediate artifacts."""
        self._sync_chunk_settings_from_runtime()
        context_payload = dict(event_context or {})
        normalized_query = normalize_query(query)
        retrieval_top_k = max(1, self.hybrid_top_k)
        retriever = self._get_retriever(query_filters=query_filters)
        retrieve_async = getattr(retriever, "retrieve_async", None)
        if callable(retrieve_async):
            retrieved = await retrieve_async(normalized_query, top_k=retrieval_top_k)
        else:
            retrieved = await asyncio.to_thread(
                retriever.retrieve,
                normalized_query,
                retrieval_top_k,
            )
        retrieval_debug: dict[str, Any] = {}
        debug_getter = getattr(retriever, "get_last_filter_debug", None)
        if callable(debug_getter):
            debug_payload = debug_getter()
            if isinstance(debug_payload, dict):
                retrieval_debug = debug_payload
        reranked = await asyncio.to_thread(
            self.reranker.rerank,
            normalized_query,
            retrieved,
            self.rerank_top_k,
        )
        selected_context = await asyncio.to_thread(
            self.context_selector.select,
            reranked,
            self.context_top_k,
        )
        await emit_stream_event(
            event_handler,
            {
                "type": "retrieval",
                "mode": mode.value,
                "query": normalized_query,
                "retrieved_count": len(retrieved),
                "reranked_count": len(reranked),
                "selected_count": len(selected_context),
                "chunk_ids": [item.chunk_id for item in selected_context],
                **context_payload,
            },
        )
        if not selected_context:
            generated = GeneratedAnswer(
                answer=localized_insufficient_evidence(response_language),
                citations=[],
                confidence=0.0,
                status="insufficient_evidence",
                stop_reason="no_context",
                llm_fallback_used=False,
            )
            await emit_stream_event(
                event_handler,
                {
                    "type": "generation",
                    "mode": mode.value,
                    "phase": "skipped",
                    "reason": "no_context",
                    **context_payload,
                },
            )
        else:
            history_context = build_chat_history_context(
                chat_history,
                memory_window=self.memory_window,
            )
            await emit_stream_event(
                event_handler,
                {
                    "type": "generation",
                    "mode": mode.value,
                    "phase": "started",
                    "context_count": len(selected_context),
                    **context_payload,
                },
            )

            async def _on_llm_delta(delta: str) -> None:
                await emit_stream_event(
                    event_handler,
                    {
                        "type": "generation_delta",
                        "mode": mode.value,
                        "delta": delta,
                        **context_payload,
                    },
                )

            generated = await self.generator.generate_answer_async(
                query=normalized_query,
                context=selected_context,
                mode=mode,
                model=model,
                response_language=response_language,
                chat_history_context=history_context,
                on_llm_delta=_on_llm_delta if event_handler is not None else None,
            )
            await emit_stream_event(
                event_handler,
                {
                    "type": "generation",
                    "mode": mode.value,
                    "phase": "completed",
                    "status": generated.status,
                    "stop_reason": generated.stop_reason,
                    "answer": generated.answer,
                    "citation_count": len(generated.citations),
                    **context_payload,
                },
            )

        logger.info(
            (
                "Standard retrieval config | top_k=%s | rerank_top_n=%s | final_context_size=%s"
            ),
            retrieval_top_k,
            self.rerank_top_k,
            len(selected_context),
        )

        return StandardPipelineResult(
            normalized_query=normalized_query,
            retrieved=retrieved,
            reranked=reranked,
            selected_context=selected_context,
            generated=generated,
            retrieval_debug=retrieval_debug,
        )

    async def run_async(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
        query_filters: dict[str, Any] | None = None,
        event_handler: StreamEventHandler | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> StandardQueryResponse:
        start_time = time.perf_counter()
        normalized_history = trim_chat_history(
            chat_history,
            memory_window=self.memory_window,
        )

        resolved_language = response_language or detect_response_language(query)
        pipeline = await self.run_pipeline(
            query=query,
            mode=Mode.STANDARD,
            model=model,
            response_language=resolved_language,
            chat_history=normalized_history,
            query_filters=query_filters,
            event_handler=event_handler,
            event_context=event_context,
        )
        final_answer = pipeline.generated.answer
        language_mismatch = is_language_mismatch(final_answer, resolved_language)
        stop_reason = pipeline.generated.stop_reason
        context_texts = [
            item.content for item in pipeline.selected_context if item.content.strip()
        ]

        if language_mismatch and pipeline.generated.status != "insufficient_evidence":
            rewritten = await self._rewrite_answer_language(
                query=pipeline.normalized_query,
                answer=final_answer,
                response_language=resolved_language,
                model=model,
            )
            if rewritten:
                final_answer = rewritten
                language_mismatch = is_language_mismatch(
                    final_answer, resolved_language
                )
                if not language_mismatch:
                    stop_reason = "language_refined"

        final_citations = list(pipeline.generated.citations)
        citation_count = len(final_citations)
        grounding = assess_grounding(
            final_answer,
            context_texts,
            citation_count=citation_count,
            has_selected_context=bool(pipeline.selected_context),
            status=pipeline.generated.status,
        )
        grounded_score = grounding.grounded_score
        grounding_reason = grounding.grounding_reason
        hallucination_detected = grounding.hallucination_detected
        llm_fallback_used = bool(pipeline.generated.llm_fallback_used)

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        trace = [
            {
                "step": "retrieve",
                "query": pipeline.normalized_query,
                "response_language": resolved_language,
                "memory_window": self.memory_window,
                "memory_messages": len(normalized_history),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "top_k": self.hybrid_top_k,
                "rerank_top_n": self.rerank_top_k,
                "index_source": self._get_index_source(),
                "count": len(pipeline.retrieved),
                "chunk_ids": [item.chunk_id for item in pipeline.retrieved],
                "docs": [
                    {
                        "chunk_id": item.chunk_id,
                        "doc_id": item.doc_id,
                        "source": item.source,
                        "title": item.title,
                        "section": item.section,
                        "file_name": item.metadata.get("file_name")
                        or item.metadata.get("filename"),
                        "file_type": item.metadata.get("file_type"),
                        "uploaded_at": item.metadata.get("uploaded_at"),
                        "created_at": item.metadata.get("created_at"),
                        "page": item.page,
                        "rank": item.rank,
                        "block_type": item.metadata.get("block_type"),
                        "ocr": bool(item.metadata.get("ocr")),
                        "score": item.score,
                        "dense_score": item.dense_score,
                        "sparse_score": item.sparse_score,
                    }
                    for item in pipeline.retrieved
                ],
                "applied_filters": pipeline.retrieval_debug.get("applied_filters", {}),
                "candidate_count_before_filter": pipeline.retrieval_debug.get(
                    "candidate_count_before_filter",
                    len(pipeline.retrieved),
                ),
                "candidate_count_after_filter": pipeline.retrieval_debug.get(
                    "candidate_count_after_filter",
                    len(pipeline.retrieved),
                ),
            },
            {
                "step": "rerank",
                "provider": getattr(
                    self.reranker, "name", self.reranker.__class__.__name__
                ),
                "top_k": self.hybrid_top_k,
                "rerank_top_n": self.rerank_top_k,
                "count": len(pipeline.reranked),
                "chunk_ids": [item.chunk_id for item in pipeline.reranked],
                "docs": [
                    {
                        "chunk_id": item.chunk_id,
                        "doc_id": item.doc_id,
                        "source": item.source,
                        "title": item.title,
                        "section": item.section,
                        "file_name": item.metadata.get("file_name")
                        or item.metadata.get("filename"),
                        "file_type": item.metadata.get("file_type"),
                        "uploaded_at": item.metadata.get("uploaded_at"),
                        "created_at": item.metadata.get("created_at"),
                        "page": item.page,
                        "rank": item.rank,
                        "block_type": item.metadata.get("block_type"),
                        "ocr": bool(item.metadata.get("ocr")),
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
                "final_context_size": len(pipeline.selected_context),
                "chunk_ids": [item.chunk_id for item in pipeline.selected_context],
                "docs": [
                    {
                        "chunk_id": item.chunk_id,
                        "doc_id": item.doc_id,
                        "source": item.source,
                        "title": item.title,
                        "section": item.section,
                        "file_name": item.metadata.get("file_name")
                        or item.metadata.get("filename"),
                        "file_type": item.metadata.get("file_type"),
                        "uploaded_at": item.metadata.get("uploaded_at"),
                        "created_at": item.metadata.get("created_at"),
                        "page": item.page,
                        "block_type": item.metadata.get("block_type"),
                        "ocr": bool(item.metadata.get("ocr")),
                        "content": item.content,
                    }
                    for item in pipeline.selected_context
                ],
            },
            {
                "step": "generate",
                "status": pipeline.generated.status,
                "stop_reason": stop_reason,
                "response_language": resolved_language,
                "language_mismatch": language_mismatch,
                "grounded_score": grounded_score,
                "grounding_reason": grounding_reason,
                "hallucination_detected": hallucination_detected,
                "llm_fallback_used": llm_fallback_used,
            },
        ]

        if pipeline.generated.status != "insufficient_evidence":
            trace.append(
                {
                    "step": "language_guard",
                    "response_language": resolved_language,
                    "language_mismatch": language_mismatch,
                }
            )
            trace.append(
                {
                    "step": "grounding_check",
                    "grounded_score": grounded_score,
                    "grounding_reason": grounding_reason,
                    "hallucination_detected": hallucination_detected,
                    "citation_count": citation_count,
                    "llm_fallback_used": llm_fallback_used,
                }
            )

        return StandardQueryResponse(
            mode="standard",
            answer=final_answer,
            citations=final_citations,
            confidence=pipeline.generated.confidence,
            stop_reason=stop_reason,
            status=pipeline.generated.status,
            latency_ms=elapsed_ms,
            response_language=resolved_language,
            language_mismatch=language_mismatch,
            grounded_score=grounded_score,
            grounding_reason=grounding_reason,
            citation_count=citation_count,
            hallucination_detected=hallucination_detected,
            llm_fallback_used=llm_fallback_used,
            trace=trace,
        )

    def run(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
        query_filters: dict[str, Any] | None = None,
        event_handler: StreamEventHandler | None = None,
        event_context: dict[str, Any] | None = None,
    ) -> StandardQueryResponse:
        """Sync wrapper for CLI/tests."""
        return run_coro_sync(
            self.run_async(
                query=query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
                query_filters=query_filters,
                event_handler=event_handler,
                event_context=event_context,
            )
        )

    async def _rewrite_answer_language(
        self,
        *,
        query: str,
        answer: str,
        response_language: str,
        model: str | None = None,
    ) -> str | None:
        prompt = self.generator.prompt_repository.render(
            "refine.md",
            fallback=self._LANGUAGE_REWRITE_PROMPT,
            question=query,
            draft_answer=answer,
            critique=(
                '{"note":"language_mismatch_detected","should_refine_answer":true,'
                '"grounded":true,"enough_evidence":true,"has_conflict":false,'
                '"missing_aspects":[],"should_retry_retrieval":false,'
                '"better_queries":[],"confidence":0.0}'
            ),
            selected_context="(keep original grounded meaning; language-only rewrite)",
            response_language=response_language,
            response_language_name=response_language_name(response_language),
        )
        try:
            raw = await complete_with_model(
                self.llm_client,
                prompt,
                system_prompt=build_language_system_prompt(response_language),
                model=model,
            )
        except Exception:
            return None

        payload = parse_json_object(raw)
        if payload and isinstance(payload.get("answer"), str):
            rewritten = payload["answer"].strip()
            if rewritten:
                return rewritten
        if payload and isinstance(payload.get("refined_answer"), str):
            rewritten = payload["refined_answer"].strip()
            if rewritten:
                return rewritten

        fallback = raw.strip()
        if fallback and not fallback.startswith("{"):
            return fallback
        return None

    async def aclose(self) -> None:
        """Release async resources held by the workflow."""
        await close_llm_client(self.llm_client)
