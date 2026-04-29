"""Tests for standard workflow end-to-end path."""

import asyncio

from app.core.cache import QueryCache
from app.schemas.retrieval import RetrievalBatch, RetrievalResult
from app.schemas.api import StandardQueryResponse, validate_query_response
from app.schemas.common import Mode
from app.retrieval import BaseReranker, ScoreOnlyReranker
from app.services import QueryService
from app.workflows.runner import WorkflowRunner
from app.workflows.standard import StandardWorkflow


def test_standard_workflow_run_path() -> None:
    workflow = StandardWorkflow()

    response = workflow.run(
        query="How does standard mode perform retrieval?", chat_history=None
    )

    assert isinstance(response, StandardQueryResponse)
    assert response.mode == "standard"
    assert response.answer
    assert isinstance(response.citations, list)
    assert response.status in {"answered", "partial", "insufficient_evidence"}


def test_standard_workflow_applies_dynamic_budget_for_simple_query() -> None:
    class _RecordingRetriever:
        def __init__(self) -> None:
            self.top_k_calls: list[int] = []

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            self.top_k_calls.append(top_k)
            return [
                RetrievalResult(
                    chunk_id=f"simple_budget_{idx}",
                    doc_id=f"simple_budget_doc_{idx}",
                    source="seeded://simple-budget",
                    content=("Nội dung mẫu " * 120) + str(idx),
                    score=0.9 - (idx * 0.01),
                    score_type="hybrid",
                    rank=idx + 1,
                )
                for idx in range(5)
            ]

    class _FakeIndexManager:
        def __init__(self) -> None:
            self.retriever = _RecordingRetriever()

        def get_retriever(self) -> _RecordingRetriever:
            return self.retriever

        def get_active_source(self) -> str:
            return "seeded"

    class _RecordingLLM:
        def __init__(self) -> None:
            self.max_tokens: list[int | None] = []

        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            self.max_tokens.append(max_tokens)
            return '{"answer":"Simple budget answer.","confidence":0.82,"status":"answered"}'

    index_manager = _FakeIndexManager()
    llm = _RecordingLLM()
    workflow = StandardWorkflow(
        index_manager=index_manager,
        llm_client=llm,
        reranker=ScoreOnlyReranker(),
    )

    response = workflow.run(query="Mức phạt tối đa là bao nhiêu?")
    retrieve_step = response.trace[0]
    context_step = response.trace[2]

    assert retrieve_step["query_complexity"] == "simple_extractive"
    assert retrieve_step["top_k"] == 3
    assert retrieve_step["rerank_top_n"] == 3
    assert retrieve_step["query_budget"]["context_top_k"] == 2
    assert context_step["context_top_k"] == 2
    assert context_step["final_context_size"] <= 2
    assert llm.max_tokens and llm.max_tokens[-1] == workflow.simple_max_tokens
    assert (
        index_manager.retriever.top_k_calls
        and index_manager.retriever.top_k_calls[-1] == 3
    )


def test_standard_workflow_keeps_locked_retrieval_top_k_with_dynamic_budget() -> None:
    class _RecordingRetriever:
        def __init__(self) -> None:
            self.top_k_calls: list[int] = []

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            self.top_k_calls.append(top_k)
            return [
                RetrievalResult(
                    chunk_id="locked_budget_001",
                    doc_id="locked_budget_doc_001",
                    source="seeded://locked-budget",
                    content="Khung thông tin về retrieval custom.",
                    score=0.9,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def __init__(self) -> None:
            self.retriever = _RecordingRetriever()

        def get_retriever(self) -> _RecordingRetriever:
            return self.retriever

        def get_active_source(self) -> str:
            return "seeded"

    class _StubLLM:
        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            _ = max_tokens
            return '{"answer":"Locked retrieval answer.","confidence":0.8,"status":"answered"}'

    index_manager = _FakeIndexManager()
    workflow = StandardWorkflow(
        index_manager=index_manager,
        llm_client=_StubLLM(),
        reranker=ScoreOnlyReranker(),
    )
    workflow.set_retrieval_top_k(4)

    response = workflow.run(query="Retrieval custom token là gì?")
    retrieve_step = response.trace[0]

    assert retrieve_step["query_complexity"] == "simple_extractive"
    assert retrieve_step["top_k"] == 4
    assert retrieve_step["rerank_top_n"] == 4
    assert (
        index_manager.retriever.top_k_calls
        and index_manager.retriever.top_k_calls[-1] == 4
    )


def test_standard_workflow_can_disable_dynamic_budget(monkeypatch) -> None:
    class _Settings:
        corpus_dir = "docs"
        index_dir = "data/indexes"
        retrieval_top_k = 8
        reranker_enabled = True
        reranker_provider = "score_only"
        reranker_model = "stub-reranker"
        reranker_device = "cpu"
        reranker_batch_size = 4
        reranker_top_n = 6
        prompt_dir = "prompts"
        llm_provider = "stub"
        llm_model = "qwen2.5:3b"
        llm_api_base = "http://localhost:11434/v1"
        llm_api_key = "ollama"
        llm_temperature = 0.2
        llm_max_tokens = 512
        llm_timeout_seconds = 10
        rag_dynamic_budget_enabled = False
        rag_simple_max_tokens = 384
        rag_normal_max_tokens = 768
        rag_complex_max_tokens = 1536
        rag_simple_context_chars = 1600
        rag_normal_context_chars = 3000

    class _RecordingRetriever:
        def __init__(self) -> None:
            self.top_k_calls: list[int] = []

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            self.top_k_calls.append(top_k)
            return [
                RetrievalResult(
                    chunk_id="disable_budget_001",
                    doc_id="disable_budget_doc_001",
                    source="seeded://disable-budget",
                    content=("Context " * 80),
                    score=0.91,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def __init__(self) -> None:
            self.retriever = _RecordingRetriever()

        def get_retriever(self) -> _RecordingRetriever:
            return self.retriever

        def get_active_source(self) -> str:
            return "seeded"

    class _RecordingLLM:
        def __init__(self) -> None:
            self.max_tokens: list[int | None] = []

        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            self.max_tokens.append(max_tokens)
            return '{"answer":"Disabled dynamic budget answer.","confidence":0.8,"status":"answered"}'

    monkeypatch.setattr("app.workflows.standard.get_settings", lambda: _Settings())
    index_manager = _FakeIndexManager()
    llm = _RecordingLLM()
    workflow = StandardWorkflow(
        index_manager=index_manager,
        llm_client=llm,
        reranker=ScoreOnlyReranker(),
    )

    response = workflow.run(query="Mức phạt tối đa là bao nhiêu?")
    retrieve_step = response.trace[0]

    assert retrieve_step["query_budget"]["dynamic_enabled"] is False
    assert retrieve_step["top_k"] == 8
    assert retrieve_step["rerank_top_n"] == 6
    assert llm.max_tokens and llm.max_tokens[-1] == 512
    assert (
        index_manager.retriever.top_k_calls
        and index_manager.retriever.top_k_calls[-1] == 8
    )


def test_standard_simple_query_skips_cross_encoder_via_cascade_policy() -> None:
    class _Retriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            return [
                RetrievalResult(
                    chunk_id=f"cascade_simple_{idx}",
                    doc_id=f"cascade_simple_doc_{idx}",
                    source="seeded://cascade-simple",
                    content=f"Điều {idx + 1}. Nội dung căn cứ pháp lý mẫu {idx}.",
                    score=0.91 - (idx * 0.08),
                    score_type="hybrid",
                    rank=idx + 1,
                )
                for idx in range(top_k)
            ]

    class _FakeIndexManager:
        def __init__(self) -> None:
            self.retriever = _Retriever()

        def get_retriever(self) -> _Retriever:
            return self.retriever

        def get_active_source(self) -> str:
            return "seeded"

    class _SpyCrossEncoderReranker(BaseReranker):
        name = "cross-encoder-reranker"

        def __init__(self) -> None:
            self.calls = 0

        def rerank(
            self,
            query: str,
            docs: list[RetrievalResult],
            top_k: int | None = None,
        ) -> list[RetrievalResult]:
            _ = query
            self.calls += 1
            limit = len(docs) if top_k is None else max(0, top_k)
            return list(docs[:limit])

    class _StubLLM:
        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            _ = max_tokens
            return '{"answer":"Simple cascade answer.","confidence":0.83,"status":"answered"}'

    reranker = _SpyCrossEncoderReranker()
    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_StubLLM(),
        reranker=reranker,
    )
    workflow.set_retrieval_top_k(5)

    response = workflow.run(query="Mức phạt tối đa là bao nhiêu?")
    rerank_step = response.trace[1]
    context_step = response.trace[2]
    selected_chunk_ids = {
        doc["chunk_id"] for doc in context_step.get("docs", []) if "chunk_id" in doc
    }

    assert reranker.calls == 0
    assert rerank_step["reranker_used"] == "score_only"
    assert rerank_step["rerank_policy"]["reason"] == "simple_extractive_skip"
    assert response.citations
    assert {item.chunk_id for item in response.citations}.issubset(selected_chunk_ids)


def test_standard_complex_query_can_use_cross_encoder_with_cascade_policy() -> None:
    class _Retriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            scores = [0.56, 0.55, 0.54, 0.53, 0.52]
            return [
                RetrievalResult(
                    chunk_id=f"cascade_complex_{idx}",
                    doc_id=f"cascade_complex_doc_{idx}",
                    source="seeded://cascade-complex",
                    content=f"Nội dung so sánh phương án {idx}.",
                    score=scores[idx],
                    score_type="hybrid",
                    rank=idx + 1,
                )
                for idx in range(min(top_k, len(scores)))
            ]

    class _FakeIndexManager:
        def __init__(self) -> None:
            self.retriever = _Retriever()

        def get_retriever(self) -> _Retriever:
            return self.retriever

        def get_active_source(self) -> str:
            return "seeded"

    class _SpyCrossEncoderReranker(BaseReranker):
        name = "cross-encoder-reranker"

        def __init__(self) -> None:
            self.calls = 0

        def rerank(
            self,
            query: str,
            docs: list[RetrievalResult],
            top_k: int | None = None,
        ) -> list[RetrievalResult]:
            _ = query
            self.calls += 1
            limit = len(docs) if top_k is None else max(0, top_k)
            ranked = list(docs[:limit])
            ranked.reverse()
            for rank, item in enumerate(ranked, start=1):
                item.rank = rank
            return ranked

    class _StubLLM:
        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            _ = max_tokens
            return '{"answer":"Complex cascade answer.","confidence":0.81,"status":"answered"}'

    reranker = _SpyCrossEncoderReranker()
    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_StubLLM(),
        reranker=reranker,
    )

    response = workflow.run(query="Phân tích ưu nhược điểm của hai phương án.")
    rerank_step = response.trace[1]

    assert reranker.calls == 1
    assert rerank_step["reranker_used"] == "cross_encoder"
    assert rerank_step["rerank_policy"]["reason"] in {
        "ambiguous_scores_use_cross_encoder",
        "advanced_strict_allow_cross_encoder",
    }


def test_standard_can_force_legacy_cross_encoder_by_disabling_cascade() -> None:
    class _Retriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            return [
                RetrievalResult(
                    chunk_id=f"cascade_legacy_{idx}",
                    doc_id=f"cascade_legacy_doc_{idx}",
                    source="seeded://cascade-legacy",
                    content=f"Nội dung legacy {idx}.",
                    score=0.85 - (idx * 0.05),
                    score_type="hybrid",
                    rank=idx + 1,
                )
                for idx in range(top_k)
            ]

    class _FakeIndexManager:
        def __init__(self) -> None:
            self.retriever = _Retriever()

        def get_retriever(self) -> _Retriever:
            return self.retriever

        def get_active_source(self) -> str:
            return "seeded"

    class _SpyCrossEncoderReranker(BaseReranker):
        name = "cross-encoder-reranker"

        def __init__(self) -> None:
            self.calls = 0

        def rerank(
            self,
            query: str,
            docs: list[RetrievalResult],
            top_k: int | None = None,
        ) -> list[RetrievalResult]:
            _ = query
            self.calls += 1
            limit = len(docs) if top_k is None else max(0, top_k)
            return list(docs[:limit])

    class _StubLLM:
        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            _ = max_tokens
            return '{"answer":"Legacy cascade answer.","confidence":0.79,"status":"answered"}'

    reranker = _SpyCrossEncoderReranker()
    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_StubLLM(),
        reranker=reranker,
    )
    workflow.rerank_cascade_enabled = False
    workflow.set_retrieval_top_k(5)

    response = workflow.run(query="Mức phạt tối đa là bao nhiêu?")
    rerank_step = response.trace[1]

    assert reranker.calls == 1
    assert rerank_step["reranker_used"] == "cross_encoder"
    assert rerank_step["rerank_policy"]["reason"] == "cascade_disabled_legacy"


def test_standard_workflow_rerank_cache_returns_stable_order() -> None:
    class _Retriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            return [
                RetrievalResult(
                    chunk_id=f"rerank_cache_{idx}",
                    doc_id=f"rerank_cache_doc_{idx}",
                    source="seeded://rerank-cache",
                    content=f"Rerank cache context {idx}",
                    score=0.9 - (idx * 0.01),
                    score_type="hybrid",
                    rank=idx + 1,
                )
                for idx in range(top_k)
            ]

    class _FakeIndexManager:
        def __init__(self) -> None:
            self.retriever = _Retriever()

        def get_retriever(self) -> _Retriever:
            return self.retriever

        def get_active_source(self) -> str:
            return "seeded"

    class _CountingReranker(BaseReranker):
        name = "cross-encoder-reranker"
        model_name = "mock-reranker-v1"
        device = "cpu"
        batch_size = 8

        def __init__(self) -> None:
            self.calls = 0

        def rerank(
            self,
            query: str,
            docs: list[RetrievalResult],
            top_k: int | None = None,
        ) -> list[RetrievalResult]:
            _ = query
            self.calls += 1
            limit = len(docs) if top_k is None else max(0, top_k)
            ranked = list(reversed(docs[:limit]))
            for rank, item in enumerate(ranked, start=1):
                item.rank = rank
            return ranked

    class _StubLLM:
        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            _ = max_tokens
            return (
                '{"answer":"Rerank cache answer.","confidence":0.8,"status":"answered"}'
            )

    reranker = _CountingReranker()
    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_StubLLM(),
        reranker=reranker,
    )
    workflow.caches.rerank = QueryCache(maxsize=16, enabled=True)
    workflow.set_retrieval_top_k(4)

    first = workflow.run("test rerank cache")
    second = workflow.run("test rerank cache")

    first_rerank = first.trace[1]
    second_rerank = second.trace[1]
    assert reranker.calls == 1
    assert first_rerank["rerank_cache_hit"] is False
    assert second_rerank["rerank_cache_hit"] is True
    assert first_rerank["chunk_ids"] == second_rerank["chunk_ids"]


def test_standard_workflow_trace_contains_timing_metrics() -> None:
    class _TimingRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="timing_std_001",
                    doc_id="timing_doc_001",
                    source="seeded://timing",
                    content="Timing metadata should appear in trace.",
                    score=0.91,
                    score_type="hybrid",
                    rank=1,
                )
            ]

        def get_last_timing(self) -> dict[str, int]:
            return {
                "retrieval_total_ms": 6,
                "dense_retrieve_ms": 3,
                "sparse_retrieve_ms": 2,
                "hybrid_merge_ms": 1,
            }

    class _FakeIndexManager:
        def get_retriever(self) -> _TimingRetriever:
            return _TimingRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    class _StubLLM:
        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            return '{"answer":"Timing answer.","confidence":0.8,"status":"answered"}'

    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(), llm_client=_StubLLM()
    )
    response = workflow.run(query="timing test")

    retrieve_step = response.trace[0]
    assert isinstance(retrieve_step["normalize_query_ms"], int)
    assert isinstance(retrieve_step["retrieval_total_ms"], int)
    assert isinstance(retrieve_step["dense_retrieve_ms"], int)
    assert isinstance(retrieve_step["sparse_retrieve_ms"], int)
    assert isinstance(retrieve_step["hybrid_merge_ms"], int)
    assert retrieve_step["breakdown_available"] is True
    assert retrieve_step["retrieval_timing_breakdown_available"] is True

    timing_summary = next(
        step for step in response.trace if step.get("step") == "timing_summary"
    )
    for key in (
        "normalize_query_ms",
        "retrieval_total_ms",
        "dense_retrieve_ms",
        "sparse_retrieve_ms",
        "hybrid_merge_ms",
        "rerank_ms",
        "context_select_ms",
        "llm_generate_ms",
        "grounding_ms",
        "total_ms",
    ):
        assert key in timing_summary
        assert isinstance(timing_summary[key], int)
        assert timing_summary[key] >= 0
    assert timing_summary["breakdown_available"] is True
    generate_step = next(
        step for step in response.trace if step.get("step") == "generate"
    )
    assert isinstance(generate_step.get("grounding_policy"), str)
    assert isinstance(generate_step.get("grounding_semantic_used"), bool)
    assert isinstance(generate_step.get("grounding_cache_hit"), bool)
    assert isinstance(generate_step.get("grounding_ms"), int)


def test_standard_workflow_missing_retrieval_breakdown_uses_zero_timings() -> None:
    class _UntimedRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="untimed_std_001",
                    doc_id="untimed_doc_001",
                    source="seeded://untimed",
                    content="Untimed retriever still returns useful context.",
                    score=0.91,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def get_retriever(self) -> _UntimedRetriever:
            return _UntimedRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    class _StubLLM:
        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            return '{"answer":"Untimed answer.","confidence":0.8,"status":"answered"}'

    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_StubLLM(),
        reranker=ScoreOnlyReranker(),
    )

    response = workflow.run(query="untimed retrieval")
    retrieve_step = response.trace[0]
    timing_summary = next(
        step for step in response.trace if step.get("step") == "timing_summary"
    )

    assert retrieve_step["dense_retrieve_ms"] == 0
    assert retrieve_step["sparse_retrieve_ms"] == 0
    assert retrieve_step["hybrid_merge_ms"] == 0
    assert retrieve_step["breakdown_available"] is False
    assert timing_summary["breakdown_available"] is False


def test_standard_workflow_uses_request_scoped_timing_for_concurrent_runs() -> None:
    class _RequestScopedTimingRetriever:
        async def retrieve_with_timing_async(
            self, query: str, top_k: int = 5
        ) -> RetrievalBatch:
            _ = top_k
            is_slow = "slow" in query
            await asyncio.sleep(0.02 if is_slow else 0.01)
            marker = 41 if is_slow else 7
            return RetrievalBatch(
                results=[
                    RetrievalResult(
                        chunk_id=f"scoped_{marker}",
                        doc_id=f"scoped_doc_{marker}",
                        source="seeded://scoped-timing",
                        content=f"Request-scoped timing context {marker}.",
                        score=0.9,
                        score_type="hybrid",
                        rank=1,
                    )
                ],
                timings_ms={
                    "retrieval_total_ms": marker + 3,
                    "dense_retrieve_ms": marker,
                    "sparse_retrieve_ms": marker + 1,
                    "hybrid_merge_ms": marker + 2,
                },
                timing_breakdown_available=True,
            )

        async def retrieve_async(
            self, query: str, top_k: int = 5
        ) -> list[RetrievalResult]:
            return (await self.retrieve_with_timing_async(query, top_k=top_k)).results

        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            raise AssertionError("workflow should use request-scoped async timing")

        def get_last_timing(self) -> dict[str, int]:
            return {
                "retrieval_total_ms": 999,
                "dense_retrieve_ms": 999,
                "sparse_retrieve_ms": 999,
                "hybrid_merge_ms": 999,
            }

    class _FakeIndexManager:
        def __init__(self) -> None:
            self.retriever = _RequestScopedTimingRetriever()

        def get_retriever(self) -> _RequestScopedTimingRetriever:
            return self.retriever

        def get_active_source(self) -> str:
            return "seeded"

    class _StubLLM:
        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            return '{"answer":"Scoped timing answer.","confidence":0.8,"status":"answered"}'

    async def _run_pair() -> tuple[StandardQueryResponse, StandardQueryResponse]:
        workflow = StandardWorkflow(
            index_manager=_FakeIndexManager(),
            llm_client=_StubLLM(),
            reranker=ScoreOnlyReranker(),
        )
        slow, fast = await asyncio.gather(
            workflow.run_async(query="slow request scoped timing"),
            workflow.run_async(query="fast request scoped timing"),
        )
        return slow, fast

    slow_response, fast_response = asyncio.run(_run_pair())
    slow_retrieve = slow_response.trace[0]
    fast_retrieve = fast_response.trace[0]

    assert slow_retrieve["dense_retrieve_ms"] == 41
    assert fast_retrieve["dense_retrieve_ms"] == 7
    assert slow_retrieve["breakdown_available"] is True
    assert fast_retrieve["breakdown_available"] is True


def test_standard_workflow_uses_ingested_files_instead_of_memory_corpus() -> None:
    workflow = StandardWorkflow()

    response = workflow.run(query="What modes are supported?", chat_history=None)

    assert response.trace[0]["step"] == "retrieve"
    assert response.trace[0]["count"] > 0
    assert all(
        not citation.source.startswith("memory://") for citation in response.citations
    )


def test_standard_workflow_extractive_fast_path_skips_llm() -> None:
    class _Retriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="extractive_001",
                    doc_id="extractive_doc_001",
                    source="seeded://extractive",
                    content=(
                        "Điều 5. Nguyên tắc xử lý\n"
                        "Tổ chức, cá nhân phải tuân thủ quy định pháp luật hiện hành.\n"
                        "Việc xử lý phải bảo đảm khách quan, công khai và đúng thẩm quyền.\n"
                        "Điều 6. Trách nhiệm thi hành"
                    ),
                    score=0.92,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def get_retriever(self) -> _Retriever:
            return _Retriever()

        def get_active_source(self) -> str:
            return "seeded"

    class _FailIfCalledLLM:
        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            max_tokens: int | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            _ = max_tokens
            raise AssertionError(
                "LLM should not be called when extractive fast path is used."
            )

    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_FailIfCalledLLM(),
        reranker=ScoreOnlyReranker(),
    )
    response = workflow.run(query="Điều 5 quy định gì?", response_language="vi")

    generate_step = next(
        step for step in response.trace if step.get("step") == "generate"
    )

    assert response.status == "answered"
    assert response.stop_reason == "heuristic_extractive_article"
    assert "Điều 5 quy định:" in response.answer
    assert generate_step["fast_path_attempted"] is True
    assert generate_step["fast_path_used"] is True
    assert generate_step["fast_path_reason"] == "article_content_match"


def test_standard_runner_route_and_contract() -> None:
    runner = WorkflowRunner()

    response = runner.run(query="What is hybrid retrieval?", mode=Mode.STANDARD)
    parsed = validate_query_response(response.model_dump())

    assert isinstance(parsed, StandardQueryResponse)
    assert parsed.mode == "standard"


def test_query_service_standard_mode() -> None:
    service = QueryService()

    response = service.run(query="Explain citation grounding", mode=Mode.STANDARD)

    assert response.mode == "standard"
    assert response.answer


def test_standard_workflow_uses_configured_llm_client_factory(monkeypatch) -> None:
    class _MockConfiguredLLMClient:
        def complete(self, prompt: str, system_prompt: str | None = None) -> str:
            _ = prompt
            _ = system_prompt
            return '{"answer":"Configured LLM answer.","confidence":0.9,"status":"answered"}'

    class _Settings:
        corpus_dir = "docs"
        index_dir = "data/indexes"
        reranker_provider = "score_only"
        reranker_model = "stub-reranker"
        reranker_device = "cpu"
        reranker_batch_size = 4
        reranker_top_n = 3
        prompt_dir = "prompts"
        llm_provider = "openai_compatible"
        llm_model = "qwen2.5:3b"
        llm_api_base = "http://localhost:11434/v1"
        llm_api_key = "ollama"
        llm_temperature = 0.2
        llm_max_tokens = 512
        llm_timeout_seconds = 10

    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="c-std-001",
                    doc_id="d-std-001",
                    source="seeded://doc",
                    content="This context proves the answer can be grounded.",
                    score=0.9,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def get_retriever(self) -> _FakeRetriever:
            return _FakeRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    captured: dict[str, object] = {}

    def _fake_create_llm_client_from_settings(settings: object):
        captured["provider"] = getattr(settings, "llm_provider")
        captured["model"] = getattr(settings, "llm_model")
        captured["api_base"] = getattr(settings, "llm_api_base")
        return _MockConfiguredLLMClient()

    monkeypatch.setattr("app.workflows.standard.get_settings", lambda: _Settings())
    monkeypatch.setattr(
        "app.workflows.standard.create_llm_client_from_settings",
        _fake_create_llm_client_from_settings,
    )

    workflow = StandardWorkflow(index_manager=_FakeIndexManager())
    response = workflow.run(query="Use configured llm", chat_history=None)

    assert response.answer == "Configured LLM answer."
    assert type(workflow.llm_client).__name__ == "_MockConfiguredLLMClient"
    assert workflow.llm_client is workflow.generator.llm_client
    assert captured == {
        "provider": "openai_compatible",
        "model": "qwen2.5:3b",
        "api_base": "http://localhost:11434/v1",
    }


def test_standard_workflow_passes_model_override_to_llm_client() -> None:
    class _RecordingLLMClient:
        def __init__(self) -> None:
            self.models: list[str | None] = []

        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            self.models.append(model)
            return '{"answer":"Model override answer.","confidence":0.88,"status":"answered"}'

    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="c-model-001",
                    doc_id="d-model-001",
                    source="seeded://model",
                    content="Context for per-request model override checks.",
                    score=0.91,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def get_retriever(self) -> _FakeRetriever:
            return _FakeRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    llm_client = _RecordingLLMClient()
    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(), llm_client=llm_client
    )

    response = workflow.run(query="Use qwen2.5:7b", model="qwen2.5:7b")

    assert response.answer == "Model override answer."
    assert llm_client.models
    assert set(llm_client.models) == {"qwen2.5:7b"}


def test_standard_workflow_includes_recent_chat_history_for_follow_up() -> None:
    class _RecordingLLMClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
        ) -> str:
            _ = system_prompt
            _ = model
            self.prompts.append(prompt)
            return '{"answer":"Điều 3 quy định nghĩa vụ tuân thủ.","confidence":0.84,"status":"answered"}'

    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="c-memory-001",
                    doc_id="d-memory-001",
                    source="seeded://memory",
                    content="Điều 2 quy định phạm vi. Điều 3 quy định nghĩa vụ tuân thủ.",
                    score=0.92,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def get_retriever(self) -> _FakeRetriever:
            return _FakeRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    llm = _RecordingLLMClient()
    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=llm,
    )
    history = [
        {"role": "user", "content": "Điều 2 là gì?"},
        {"role": "assistant", "content": "Điều 2 quy định phạm vi áp dụng."},
        {"role": "user", "content": "còn điều 3 thì sao"},
    ]

    response = workflow.run(query="còn điều 3 thì sao", chat_history=history)

    assert response.answer
    assert llm.prompts
    prompt = llm.prompts[-1]
    assert "Chat history (latest turns):" in prompt
    assert "Điều 2 là gì?" in prompt
    assert "còn điều 3 thì sao" in prompt
