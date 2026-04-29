"""Compare workflow tests."""

from app.schemas.api import CompareQueryResponse, validate_query_response
from app.schemas.common import Mode
from app.core.config import get_settings
from app.generation import StubLLMClient
from app.schemas.retrieval import RetrievalResult
from app.retrieval import ScoreOnlyReranker
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.compare import CompareWorkflow
from app.workflows.runner import WorkflowRunner
from app.workflows.standard import StandardWorkflow


def test_compare_mode_response_contains_both_branches() -> None:
    runner = WorkflowRunner()

    response = runner.run(
        query="Compare standard and advanced reliability", mode=Mode.COMPARE
    )

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


def test_compare_reuses_standard_pipeline_once_by_default() -> None:
    class _AlwaysRetrieveGate:
        async def decide_async(
            self,
            query: str,
            chat_history: list[dict[str, str]] | None = None,
            *,
            model: str | None = None,
            response_language: str = "en",
            allow_llm: bool = True,
        ) -> tuple[bool, str]:
            _ = query
            _ = chat_history
            _ = model
            _ = response_language
            _ = allow_llm
            return True, "always_retrieve"

    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="cmp_reuse_001",
                    doc_id="cmp_reuse_doc_001",
                    source="seeded://cmp-reuse",
                    content="Compare reuse should avoid duplicate standard pipeline work.",
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

    class _CountingStandardWorkflow(StandardWorkflow):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.pipeline_calls = 0

        async def run_pipeline(self, *args, **kwargs):
            self.pipeline_calls += 1
            return await super().run_pipeline(*args, **kwargs)

    standard = _CountingStandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=StubLLMClient(
            responder=lambda prompt, system, model=None: (
                '{"answer":"Reuse standard answer.","confidence":0.72,"status":"answered"}'
            )
        ),
        reranker=ScoreOnlyReranker(),
    )
    advanced = AdvancedWorkflow(
        standard_workflow=standard,
        max_loops=1,
        retrieval_gate=_AlwaysRetrieveGate(),  # type: ignore[arg-type]
    )
    compare = CompareWorkflow(
        standard_workflow=standard,
        advanced_workflow=advanced,
    )

    response = compare.run(
        query="force retrieval compare reuse default path",
        response_language="en",
    )

    assert standard.pipeline_calls == 1
    loop_step = next(
        step for step in response.advanced.trace if step.get("step") == "loop"
    )
    assert loop_step.get("reused_standard_result") is True
    assert loop_step.get("standard_reuse_saved_steps") == [
        "retrieve",
        "rerank",
        "generate",
    ]
    parsed = validate_query_response(response.model_dump())
    assert isinstance(parsed, CompareQueryResponse)


def test_compare_parallel_opt_in_keeps_legacy_duplicate_pipeline_path(
    monkeypatch,
) -> None:
    monkeypatch.setenv("COMPARE_PARALLEL_ENABLED", "true")
    get_settings.cache_clear()

    class _AlwaysRetrieveGate:
        async def decide_async(
            self,
            query: str,
            chat_history: list[dict[str, str]] | None = None,
            *,
            model: str | None = None,
            response_language: str = "en",
            allow_llm: bool = True,
        ) -> tuple[bool, str]:
            _ = query
            _ = chat_history
            _ = model
            _ = response_language
            _ = allow_llm
            return True, "always_retrieve"

    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="cmp_parallel_001",
                    doc_id="cmp_parallel_doc_001",
                    source="seeded://cmp-parallel",
                    content="Parallel opt-in keeps old compare behavior.",
                    score=0.88,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def get_retriever(self) -> _FakeRetriever:
            return _FakeRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    class _CountingStandardWorkflow(StandardWorkflow):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.pipeline_calls = 0

        async def run_pipeline(self, *args, **kwargs):
            self.pipeline_calls += 1
            return await super().run_pipeline(*args, **kwargs)

    try:
        standard = _CountingStandardWorkflow(
            index_manager=_FakeIndexManager(),
            llm_client=StubLLMClient(
                responder=lambda prompt, system, model=None: (
                    '{"answer":"Parallel compare answer.","confidence":0.71,"status":"answered"}'
                )
            ),
            reranker=ScoreOnlyReranker(),
        )
        advanced = AdvancedWorkflow(
            standard_workflow=standard,
            max_loops=1,
            retrieval_gate=_AlwaysRetrieveGate(),  # type: ignore[arg-type]
        )
        compare = CompareWorkflow(
            standard_workflow=standard,
            advanced_workflow=advanced,
        )

        response = compare.run(
            query="force retrieval compare parallel opt-in",
            response_language="en",
        )

        assert standard.pipeline_calls == 2
        loop_step = next(
            step for step in response.advanced.trace if step.get("step") == "loop"
        )
        assert loop_step.get("reused_standard_result") is False
    finally:
        monkeypatch.delenv("COMPARE_PARALLEL_ENABLED", raising=False)
        get_settings.cache_clear()


def test_compare_mode_trace_contains_branch_timing_metrics() -> None:
    class _StandardBranch:
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
                    "answer": "standard branch",
                    "citations": [],
                    "confidence": 0.5,
                    "status": "answered",
                    "latency_ms": 11,
                    "response_language": response_language or "en",
                    "grounded_score": 0.3,
                    "citation_count": 0,
                    "hallucination_detected": False,
                    "trace": [{"step": "timing_summary", "total_ms": 11}],
                }
            )

    class _AdvancedBranch:
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
                    "answer": "advanced branch",
                    "citations": [],
                    "confidence": 0.6,
                    "status": "answered",
                    "latency_ms": 13,
                    "response_language": response_language or "en",
                    "grounded_score": 0.35,
                    "citation_count": 0,
                    "hallucination_detected": False,
                    "trace": [{"step": "timing_summary", "total_ms": 13}],
                }
            )

    compare = CompareWorkflow(
        standard_workflow=_StandardBranch(),  # type: ignore[arg-type]
        advanced_workflow=_AdvancedBranch(),  # type: ignore[arg-type]
    )
    response = compare.run(query="compare timing")

    standard_timing = next(
        step for step in response.standard.trace if step.get("step") == "compare_timing"
    )
    advanced_timing = next(
        step for step in response.advanced.trace if step.get("step") == "compare_timing"
    )
    standard_summary = next(
        step for step in response.standard.trace if step.get("step") == "timing_summary"
    )
    advanced_summary = next(
        step for step in response.advanced.trace if step.get("step") == "timing_summary"
    )
    assert isinstance(standard_timing["standard_branch_ms"], int)
    assert isinstance(standard_timing["compare_total_ms"], int)
    assert isinstance(advanced_timing["advanced_branch_ms"], int)
    assert isinstance(advanced_timing["compare_total_ms"], int)
    for summary in (standard_summary, advanced_summary):
        assert isinstance(summary["total_ms"], int)
        assert isinstance(summary["llm_generate_ms"], int)
        assert isinstance(summary["retrieval_total_ms"], int)
        assert summary["total_ms"] >= 0
        assert summary["llm_generate_ms"] >= 0
        assert summary["retrieval_total_ms"] >= 0
    assert response.standard.trace[-1]["step"] == "completed"
    assert response.advanced.trace[-1]["step"] == "completed"


def test_compare_workflow_uses_injected_qwen_backed_branches() -> None:
    class _MockQwenClient:
        def complete(self, prompt: str, system_prompt: str | None = None) -> str:
            _ = prompt
            _ = system_prompt
            return (
                '{"answer":"Compare mode uses standard and advanced branches.",'
                '"confidence":0.81,"status":"answered"}'
            )

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

    assert (
        response.standard.answer == "Compare mode uses standard and advanced branches."
    )
    assert (
        response.advanced.answer == "Compare mode uses standard and advanced branches."
    )


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
    assert response.comparison.winner == "standard"
    assert response.comparison.standard_score is not None
    assert response.comparison.advanced_score is not None
    assert response.comparison.standard_score > response.comparison.advanced_score
    assert response.comparison.grounded_score_delta is not None
    assert response.comparison.grounded_score_delta < 0
    assert (
        response.comparison.note
        == "Standard is more reliable due to stronger citations and groundedness."
    )


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
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
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
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
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
    assert response.comparison.winner == "standard"
    assert (
        response.comparison.note
        == "Chuẩn đáng tin cậy hơn vì có trích dẫn và độ bám tài liệu cao hơn"
    )

    class _BothWeakStandard:
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
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
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
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
    assert response.comparison.winner == "both_weak"
    assert (
        response.comparison.note
        == "Cả hai cần kiểm tra lại vì thiếu bằng chứng đủ mạnh"
    )

    class _AdvancedStrong:
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
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
    assert response.comparison.winner == "advanced"
    assert (
        response.comparison.note
        == "Nâng cao đáng tin cậy hơn vì có trích dẫn và độ bám tài liệu cao hơn"
    )


def test_compare_zero_citation_high_confidence_loses_to_cited_answer() -> None:
    class _StandardCited:
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
            _ = query
            _ = chat_history
            _ = model
            return validate_query_response(
                {
                    "mode": "standard",
                    "answer": "Cited standard",
                    "citations": [
                        {"chunk_id": "s1", "doc_id": "d1", "source": "seeded://s1"}
                    ],
                    "confidence": 0.45,
                    "status": "answered",
                    "response_language": response_language or "en",
                    "grounded_score": 0.7,
                    "citation_count": 1,
                    "hallucination_detected": False,
                    "trace": [],
                }
            )

    class _AdvancedNoCitationButConfident:
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
            _ = query
            _ = chat_history
            _ = model
            return validate_query_response(
                {
                    "mode": "advanced",
                    "answer": "Uncited advanced",
                    "citations": [],
                    "confidence": 0.99,
                    "status": "answered",
                    "response_language": response_language or "en",
                    "grounded_score": 0.15,
                    "citation_count": 0,
                    "hallucination_detected": False,
                    "trace": [],
                }
            )

    compare = CompareWorkflow(
        standard_workflow=_StandardCited(),  # type: ignore[arg-type]
        advanced_workflow=_AdvancedNoCitationButConfident(),  # type: ignore[arg-type]
    )
    response = compare.run(query="compare", response_language="en")

    assert response.comparison.winner == "standard"
    assert response.comparison.preferred_mode == "standard"
    assert response.comparison.standard_score is not None
    assert response.comparison.advanced_score is not None
    assert response.comparison.standard_score > response.comparison.advanced_score


def test_compare_hallucination_loses_to_non_hallucinated_branch() -> None:
    class _StandardHallucinated:
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
            _ = query
            _ = chat_history
            _ = model
            return validate_query_response(
                {
                    "mode": "standard",
                    "answer": "Hallucinated standard",
                    "citations": [
                        {"chunk_id": "s1", "doc_id": "d1", "source": "seeded://s1"}
                    ],
                    "confidence": 0.9,
                    "status": "answered",
                    "response_language": response_language or "en",
                    "grounded_score": 0.12,
                    "citation_count": 1,
                    "hallucination_detected": True,
                    "trace": [],
                }
            )

    class _AdvancedSafe:
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
            _ = query
            _ = chat_history
            _ = model
            return validate_query_response(
                {
                    "mode": "advanced",
                    "answer": "Grounded advanced",
                    "citations": [
                        {"chunk_id": "a1", "doc_id": "d2", "source": "seeded://a1"}
                    ],
                    "confidence": 0.55,
                    "status": "answered",
                    "response_language": response_language or "en",
                    "grounded_score": 0.42,
                    "citation_count": 1,
                    "hallucination_detected": False,
                    "trace": [],
                }
            )

    compare = CompareWorkflow(
        standard_workflow=_StandardHallucinated(),  # type: ignore[arg-type]
        advanced_workflow=_AdvancedSafe(),  # type: ignore[arg-type]
    )
    response = compare.run(query="compare", response_language="en")

    assert response.comparison.winner == "advanced"
    assert response.comparison.preferred_mode == "advanced"


def test_compare_insufficient_evidence_loses_to_grounded_cited_answer() -> None:
    class _StandardInsufficient:
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
            _ = query
            _ = chat_history
            _ = model
            return validate_query_response(
                {
                    "mode": "standard",
                    "answer": "Insufficient",
                    "citations": [],
                    "confidence": 0.0,
                    "status": "insufficient_evidence",
                    "response_language": response_language or "en",
                    "grounded_score": 0.0,
                    "citation_count": 0,
                    "hallucination_detected": False,
                    "trace": [],
                }
            )

    class _AdvancedGrounded:
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
            _ = query
            _ = chat_history
            _ = model
            return validate_query_response(
                {
                    "mode": "advanced",
                    "answer": "Grounded advanced",
                    "citations": [
                        {"chunk_id": "a1", "doc_id": "d2", "source": "seeded://a1"}
                    ],
                    "confidence": 0.52,
                    "status": "answered",
                    "response_language": response_language or "en",
                    "grounded_score": 0.63,
                    "citation_count": 1,
                    "hallucination_detected": False,
                    "trace": [],
                }
            )

    compare = CompareWorkflow(
        standard_workflow=_StandardInsufficient(),  # type: ignore[arg-type]
        advanced_workflow=_AdvancedGrounded(),  # type: ignore[arg-type]
    )
    response = compare.run(query="compare", response_language="en")

    assert response.comparison.winner == "advanced"
    assert response.comparison.preferred_mode == "advanced"
    assert response.comparison.reasons


def test_compare_tie_when_both_similar_quality() -> None:
    class _StandardSimilar:
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
            _ = query
            _ = chat_history
            _ = model
            return validate_query_response(
                {
                    "mode": "standard",
                    "answer": "Standard similar",
                    "citations": [
                        {"chunk_id": "s1", "doc_id": "d1", "source": "seeded://s1"}
                    ],
                    "confidence": 0.61,
                    "status": "answered",
                    "response_language": response_language or "en",
                    "grounded_score": 0.46,
                    "citation_count": 1,
                    "hallucination_detected": False,
                    "trace": [],
                }
            )

    class _AdvancedSimilar:
        def run(
            self, query: str, chat_history=None, model=None, response_language=None
        ):
            _ = query
            _ = chat_history
            _ = model
            return validate_query_response(
                {
                    "mode": "advanced",
                    "answer": "Advanced similar",
                    "citations": [
                        {"chunk_id": "a1", "doc_id": "d2", "source": "seeded://a1"}
                    ],
                    "confidence": 0.63,
                    "status": "answered",
                    "response_language": response_language or "en",
                    "grounded_score": 0.45,
                    "citation_count": 1,
                    "hallucination_detected": False,
                    "trace": [],
                }
            )

    compare = CompareWorkflow(
        standard_workflow=_StandardSimilar(),  # type: ignore[arg-type]
        advanced_workflow=_AdvancedSimilar(),  # type: ignore[arg-type]
    )
    response = compare.run(query="compare", response_language="en")

    assert response.comparison.winner == "tie"
    assert response.comparison.preferred_mode == "review"
    assert response.comparison.reasons
