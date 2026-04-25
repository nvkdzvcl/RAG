"""Compare workflow tests."""

from app.schemas.api import CompareQueryResponse, validate_query_response
from app.schemas.common import Mode
from app.schemas.retrieval import RetrievalResult
from app.retrieval import ScoreOnlyReranker
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.compare import CompareWorkflow
from app.workflows.runner import WorkflowRunner
from app.workflows.standard import StandardWorkflow


def test_compare_mode_response_contains_both_branches() -> None:
    runner = WorkflowRunner()

    response = runner.run(query="Compare standard and advanced reliability", mode=Mode.COMPARE)

    assert isinstance(response, CompareQueryResponse)
    assert response.mode == "compare"
    assert response.standard.mode == "standard"
    assert response.advanced.mode == "advanced"
    assert response.standard.answer
    assert response.advanced.answer


def test_compare_mode_schema_contract() -> None:
    runner = WorkflowRunner()
    response = runner.run(query="What is compare mode?", mode=Mode.COMPARE)

    parsed = validate_query_response(response.model_dump())
    assert isinstance(parsed, CompareQueryResponse)
    assert parsed.standard.mode == "standard"
    assert parsed.advanced.mode == "advanced"
    assert parsed.comparison.citation_delta == (
        len(parsed.advanced.citations) - len(parsed.standard.citations)
    )


def test_compare_workflow_uses_injected_qwen_backed_branches() -> None:
    class _MockQwenClient:
        def complete(self, prompt: str, system_prompt: str | None = None) -> str:
            _ = prompt
            _ = system_prompt
            return '{"answer":"Qwen DI answer.","confidence":0.81,"status":"answered"}'

    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="c_cmp_001",
                    doc_id="d_cmp_001",
                    source="seeded://cmp",
                    content="Compare mode uses standard and advanced branches.",
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

    standard = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=_MockQwenClient(),
        reranker=ScoreOnlyReranker(),
    )
    advanced = AdvancedWorkflow(standard_workflow=standard, max_loops=1)
    compare = CompareWorkflow(
        standard_workflow=standard,
        advanced_workflow=advanced,
    )

    response = compare.run(query="compare via di", chat_history=None)

    assert response.standard.answer == "Qwen DI answer."
    assert response.advanced.answer == "Qwen DI answer."


def test_compare_workflow_preserves_model_override_for_both_branches() -> None:
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
            return '{"answer":"Shared override answer.","confidence":0.73,"status":"answered"}'

    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="c_cmp_model_001",
                    doc_id="d_cmp_model_001",
                    source="seeded://cmp-model",
                    content="Compare model override test context.",
                    score=0.86,
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
    standard = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=llm_client,
        reranker=ScoreOnlyReranker(),
    )
    advanced = AdvancedWorkflow(standard_workflow=standard, max_loops=1)
    compare = CompareWorkflow(
        standard_workflow=standard,
        advanced_workflow=advanced,
    )

    response = compare.run(query="compare model override", model="qwen3.5:9b")

    assert response.standard.answer == "Shared override answer."
    assert response.advanced.answer
    assert len(llm_client.models) >= 2
    assert set(llm_client.models) == {"qwen3.5:9b"}


def test_compare_prefers_grounded_answer_over_higher_confidence() -> None:
    class _GroundedStandard:
        def run(
            self,
            query: str,
            chat_history: list[dict[str, str]] | None = None,
            model: str | None = None,
            response_language: str | None = None,
        ):
            _ = query
            _ = chat_history
            _ = model
            return validate_query_response(
                {
                    "mode": "standard",
                    "answer": "Grounded standard answer",
                    "citations": [
                        {
                            "chunk_id": "cmp_ground_001",
                            "doc_id": "cmp_doc_001",
                            "source": "seeded://cmp",
                        }
                    ],
                    "confidence": 0.55,
                    "status": "answered",
                    "response_language": response_language or "en",
                    "grounded_score": 0.82,
                    "citation_count": 1,
                    "hallucination_detected": False,
                    "trace": [],
                }
            )

    class _UngroundedAdvanced:
        def run(
            self,
            query: str,
            chat_history: list[dict[str, str]] | None = None,
            model: str | None = None,
            response_language: str | None = None,
        ):
            _ = query
            _ = chat_history
            _ = model
            return validate_query_response(
                {
                    "mode": "advanced",
                    "answer": "Ungrounded advanced answer",
                    "citations": [],
                    "confidence": 0.96,
                    "status": "answered",
                    "response_language": response_language or "en",
                    "grounded_score": 0.04,
                    "citation_count": 0,
                    "hallucination_detected": True,
                    "trace": [],
                }
            )

    compare = CompareWorkflow(
        standard_workflow=_GroundedStandard(),  # type: ignore[arg-type]
        advanced_workflow=_UngroundedAdvanced(),  # type: ignore[arg-type]
    )

    response = compare.run(query="compare grounding quality", response_language="en")

    assert response.comparison.preferred_mode == "standard"
    assert response.comparison.grounded_score_delta is not None
    assert response.comparison.grounded_score_delta < 0
    assert response.comparison.note == "Standard is more reliable"


def test_compare_summary_rule_messages_for_vietnamese() -> None:
    def _mode_payload(
        *,
        mode: str,
        citations: list[dict[str, str]],
        confidence: float,
        grounded_score: float,
        hallucination_detected: bool = False,
        language_mismatch: bool = False,
        llm_fallback_used: bool = False,
    ):
        return validate_query_response(
            {
                "mode": mode,
                "answer": "placeholder",
                "citations": citations,
                "confidence": confidence,
                "status": "answered",
                "response_language": "vi",
                "grounded_score": grounded_score,
                "citation_count": len(citations),
                "hallucination_detected": hallucination_detected,
                "language_mismatch": language_mismatch,
                "llm_fallback_used": llm_fallback_used,
                "trace": [],
            }
        )

    class _StandardStrong:
        def run(self, query: str, chat_history=None, model=None, response_language=None):
            _ = query
            _ = chat_history
            _ = model
            _ = response_language
            return _mode_payload(
                mode="standard",
                citations=[{"chunk_id": "s1", "doc_id": "d1", "source": "seeded://s1"}],
                confidence=0.45,
                grounded_score=0.72,
            )

    class _AdvancedNoCitations:
        def run(self, query: str, chat_history=None, model=None, response_language=None):
            _ = query
            _ = chat_history
            _ = model
            _ = response_language
            return _mode_payload(
                mode="advanced",
                citations=[],
                confidence=0.95,
                grounded_score=0.05,
                hallucination_detected=True,
            )

    compare = CompareWorkflow(
        standard_workflow=_StandardStrong(),  # type: ignore[arg-type]
        advanced_workflow=_AdvancedNoCitations(),  # type: ignore[arg-type]
    )
    response = compare.run(query="so sanh", response_language="vi")
    assert response.comparison.preferred_mode == "standard"
    assert response.comparison.note == "Chuẩn đáng tin cậy hơn"

    class _BothWeakStandard:
        def run(self, query: str, chat_history=None, model=None, response_language=None):
            _ = query
            _ = chat_history
            _ = model
            _ = response_language
            return _mode_payload(
                mode="standard",
                citations=[],
                confidence=0.82,
                grounded_score=0.03,
                llm_fallback_used=True,
            )

    class _BothWeakAdvanced:
        def run(self, query: str, chat_history=None, model=None, response_language=None):
            _ = query
            _ = chat_history
            _ = model
            _ = response_language
            return _mode_payload(
                mode="advanced",
                citations=[],
                confidence=0.88,
                grounded_score=0.04,
                language_mismatch=True,
            )

    compare = CompareWorkflow(
        standard_workflow=_BothWeakStandard(),  # type: ignore[arg-type]
        advanced_workflow=_BothWeakAdvanced(),  # type: ignore[arg-type]
    )
    response = compare.run(query="so sanh", response_language="vi")
    assert response.comparison.preferred_mode == "review"
    assert response.comparison.note == "Cả hai cần kiểm tra lại"

    class _AdvancedStrong:
        def run(self, query: str, chat_history=None, model=None, response_language=None):
            _ = query
            _ = chat_history
            _ = model
            _ = response_language
            return _mode_payload(
                mode="advanced",
                citations=[{"chunk_id": "a1", "doc_id": "d2", "source": "seeded://a1"}],
                confidence=0.7,
                grounded_score=0.74,
            )

    compare = CompareWorkflow(
        standard_workflow=_BothWeakStandard(),  # type: ignore[arg-type]
        advanced_workflow=_AdvancedStrong(),  # type: ignore[arg-type]
    )
    response = compare.run(query="so sanh", response_language="vi")
    assert response.comparison.preferred_mode == "advanced"
    assert response.comparison.note == "Nâng cao đáng tin cậy hơn"
