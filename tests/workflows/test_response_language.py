"""Tests for response-language consistency across workflows."""

from __future__ import annotations

from typing import cast

from app.generation import BaselineGenerator
from app.retrieval import ScoreOnlyReranker
from app.schemas.api import AdvancedQueryResponse, StandardQueryResponse
from app.schemas.common import Mode
from app.schemas.retrieval import RetrievalResult
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.compare import CompareWorkflow
from app.workflows.shared import detect_response_language
from app.workflows.standard import StandardWorkflow


def _sample_context() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk_lang_001",
            doc_id="doc_lang_001",
            source="seeded://lang",
            content="Self-RAG cần truy xuất tài liệu để tạo câu trả lời có căn cứ.",
            score=0.91,
            score_type="hybrid",
            rank=1,
        )
    ]


class _FakeRetriever:
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        _ = query
        _ = top_k
        return _sample_context()


class _FakeIndexManager:
    def get_retriever(self) -> _FakeRetriever:
        return _FakeRetriever()

    def get_active_source(self) -> str:
        return "seeded"


def test_detect_response_language_vi() -> None:
    assert detect_response_language("Self-RAG là gì?") == "vi"
    assert (
        detect_response_language("Co the giai thich self rag nhu the nao khong?")
        == "vi"
    )


def test_detect_response_language_en() -> None:
    assert detect_response_language("What is Self-RAG?") == "en"


def test_standard_prompt_includes_vietnamese_language_instruction() -> None:
    class _RecordingLLMClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.system_prompts: list[str | None] = []

        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
        ) -> str:
            _ = model
            self.prompts.append(prompt)
            self.system_prompts.append(system_prompt)
            return (
                '{"answer":"Đây là câu trả lời.","confidence":0.83,"status":"answered"}'
            )

    llm = _RecordingLLMClient()
    generator = BaselineGenerator(llm_client=llm)
    result = generator.generate_answer(
        query="Self-RAG là gì?",
        context=_sample_context(),
        mode=Mode.STANDARD,
        response_language="vi",
    )

    assert result.answer
    assert llm.prompts
    assert "Response language: `vi` (`Vietnamese`)" in llm.prompts[-1]
    assert llm.system_prompts[-1] is not None
    assert (
        "You must answer in Vietnamese. Do not switch languages."
        in llm.system_prompts[-1]
    )


def test_advanced_prompt_includes_vietnamese_language_instruction() -> None:
    class _RecordingLLMClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.system_prompts: list[str | None] = []

        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
        ) -> str:
            _ = model
            self.prompts.append(prompt)
            self.system_prompts.append(system_prompt)
            if "query_rewrite" in prompt.lower():
                return '{"rewrites":["Tự RAG là gì?"]}'
            return '{"answer":"Đây là câu trả lời nâng cao.","confidence":0.81,"status":"answered"}'

    standard = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_RecordingLLMClient(),
        reranker=ScoreOnlyReranker(),
    )
    advanced = AdvancedWorkflow(standard_workflow=standard, max_loops=1)
    response = advanced.run(
        query="force retrieval self-rag là gì?",
        response_language="vi",
    )

    assert response.response_language == "vi"
    recording_llm = cast(_RecordingLLMClient, standard.llm_client)
    prompts = recording_llm.prompts
    assert any("Mode: `advanced`" in prompt for prompt in prompts)
    assert any("Response language: `vi` (`Vietnamese`)" in prompt for prompt in prompts)
    assert any(
        prompt and "You must answer in Vietnamese. Do not switch languages." in prompt
        for prompt in recording_llm.system_prompts
    )


def test_compare_passes_same_response_language_to_both_branches() -> None:
    class _RecordingStandard:
        def __init__(self) -> None:
            self.received_response_language: str | None = None

        def run(
            self,
            query: str,
            chat_history: list[dict[str, str]] | None = None,
            model: str | None = None,
            response_language: str | None = None,
        ) -> StandardQueryResponse:
            _ = query
            _ = chat_history
            _ = model
            self.received_response_language = response_language
            return StandardQueryResponse(
                mode="standard",
                answer="Câu trả lời chuẩn",
                citations=[],
                confidence=0.6,
                status="answered",
                latency_ms=11,
                response_language=response_language or "en",
                language_mismatch=False,
                trace=[],
            )

    class _RecordingAdvanced:
        def __init__(self) -> None:
            self.received_response_language: str | None = None

        def run(
            self,
            query: str,
            chat_history: list[dict[str, str]] | None = None,
            model: str | None = None,
            response_language: str | None = None,
        ) -> AdvancedQueryResponse:
            _ = query
            _ = chat_history
            _ = model
            self.received_response_language = response_language
            return AdvancedQueryResponse(
                mode="advanced",
                answer="Câu trả lời nâng cao",
                citations=[],
                confidence=0.7,
                status="answered",
                latency_ms=13,
                loop_count=1,
                response_language=response_language or "en",
                language_mismatch=False,
                trace=[],
            )

    standard = _RecordingStandard()
    advanced = _RecordingAdvanced()
    compare = CompareWorkflow(
        standard_workflow=standard,  # type: ignore[arg-type]
        advanced_workflow=advanced,  # type: ignore[arg-type]
    )

    response = compare.run(query="Self-RAG là gì?")

    assert standard.received_response_language == "vi"
    assert advanced.received_response_language == "vi"
    assert response.standard.response_language == "vi"
    assert response.advanced.response_language == "vi"


def test_standard_sets_language_mismatch_for_chinese_output_on_vietnamese_query() -> (
    None
):
    class _AlwaysChineseLLM:
        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            return '{"answer":"这是中文回答。","confidence":0.9,"status":"answered"}'

    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_AlwaysChineseLLM(),
        reranker=ScoreOnlyReranker(),
    )

    response = workflow.run(query="Self-RAG là gì?")

    assert response.response_language == "vi"
    assert response.language_mismatch is True


def test_standard_response_language_defaults_to_english_for_english_query() -> None:
    class _EnglishLLM:
        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            return '{"answer":"Self-RAG uses retrieval and critique.","confidence":0.8,"status":"answered"}'

    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_EnglishLLM(),
        reranker=ScoreOnlyReranker(),
    )

    response = workflow.run(query="What is Self-RAG?")

    assert response.response_language == "en"
    assert response.language_mismatch is False


def test_language_guard_rewrite_failure_keeps_workflow_stable() -> None:
    class _ChineseThenFailRewriteLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            self.calls += 1
            if self.calls == 1:
                return (
                    '{"answer":"这是中文回答。","confidence":0.82,"status":"answered"}'
                )
            raise RuntimeError("rewrite failed")

    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_ChineseThenFailRewriteLLM(),
        reranker=ScoreOnlyReranker(),
    )

    response = workflow.run(query="Self-RAG là gì?")

    assert response.status == "answered"
    assert response.answer
    assert response.language_mismatch is True


def test_language_guard_rewrite_uses_llm_rewrite_max_tokens(monkeypatch) -> None:
    class _Settings:
        retrieval_top_k = 8
        reranker_top_k = 6
        reranker_top_n = 6
        reranker_enabled = True
        chunk_size = 320
        chunk_overlap = 40
        memory_window = 3
        corpus_dir = "docs"
        index_dir = "data/indexes"
        prompt_dir = "prompts"
        llm_max_tokens = 1024
        llm_rewrite_max_tokens = 123
        rag_dynamic_budget_enabled = False
        rag_simple_max_tokens = 384
        rag_normal_max_tokens = 768
        rag_complex_max_tokens = 1536
        rag_simple_context_chars = 1600
        rag_normal_context_chars = 3000

    class _ChineseThenVietnameseRewriteLLM:
        def __init__(self) -> None:
            self.calls = 0
            self.max_tokens_seen: list[int | None] = []

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
            self.calls += 1
            self.max_tokens_seen.append(max_tokens)
            if self.calls == 1:
                return (
                    '{"answer":"这是中文回答。","confidence":0.82,"status":"answered"}'
                )
            return '{"answer":"Đây là câu trả lời đã được viết lại.","confidence":0.8,"status":"answered"}'

    monkeypatch.setattr("app.workflows.standard.get_settings", lambda: _Settings())
    llm = _ChineseThenVietnameseRewriteLLM()
    workflow = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=llm,
        reranker=ScoreOnlyReranker(),
    )

    response = workflow.run(query="Self-RAG là gì?")

    assert response.answer.startswith("Đây là câu trả lời")
    assert response.language_mismatch is False
    assert len(llm.max_tokens_seen) >= 2
    assert llm.max_tokens_seen[0] == 1024
    assert llm.max_tokens_seen[1] == 123
