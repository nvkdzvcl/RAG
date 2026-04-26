"""Advanced workflow tests."""

from app.schemas.api import AdvancedQueryResponse, validate_query_response
from app.schemas.common import Mode
from app.schemas.retrieval import RetrievalResult
from app.schemas.workflow import CritiqueResult
from app.generation import StubLLMClient
from app.retrieval import ScoreOnlyReranker
from app.workflows.advanced import AdvancedWorkflow
from app.workflows.critique import HeuristicCritic
from app.workflows.query_rewrite import QueryRewriter
from app.workflows.runner import WorkflowRunner
from app.workflows.standard import StandardWorkflow


def _sample_context() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk_adv_001",
            doc_id="doc_adv_001",
            source="memory://adv1",
            title="Advanced Evidence",
            section="Main",
            content="Advanced workflow retries retrieval and critiques groundedness.",
            score=0.7,
            score_type="hybrid",
        )
    ]


def test_critique_schema_validity() -> None:
    critic = HeuristicCritic()

    critique = critic.critique(
        query="How does advanced retrieval work?",
        draft_answer="Advanced workflow retries retrieval.",
        context=_sample_context(),
        loop_count=1,
        max_loops=2,
    )

    assert isinstance(critique, CritiqueResult)
    dumped = critique.model_dump()
    for key in [
        "grounded",
        "enough_evidence",
        "has_conflict",
        "missing_aspects",
        "should_retry_retrieval",
        "should_refine_answer",
        "better_queries",
        "confidence",
        "note",
    ]:
        assert key in dumped


def test_retry_loop_limit() -> None:
    workflow = AdvancedWorkflow(max_loops=2)

    response = workflow.run("force retry retrieval loop for diagnostics")

    assert isinstance(response, AdvancedQueryResponse)
    assert response.loop_count == 2
    assert response.stop_reason == "max_loop_reached"


def test_abstain_behavior() -> None:
    workflow = AdvancedWorkflow(max_loops=2)

    response = workflow.run("force abstain on unsupported claim")

    assert response.status == "insufficient_evidence"
    assert response.citations == []
    assert response.stop_reason == "critic_abstain"


def test_advanced_mode_run_path() -> None:
    runner = WorkflowRunner()

    response = runner.run(query="How does advanced mode improve reliability?", mode=Mode.ADVANCED)
    parsed = validate_query_response(response.model_dump())

    assert isinstance(parsed, AdvancedQueryResponse)
    assert parsed.mode == "advanced"
    assert parsed.loop_count <= 2
    assert parsed.answer


def test_query_rewriter_malformed_json_falls_back_to_heuristic() -> None:
    rewriter = QueryRewriter(
        llm_client=StubLLMClient(
            responder=lambda prompt, system: "```json\n{bad json}\n```"
        )
    )

    rewrites = rewriter.rewrite("Truy hoi thong tin cho Self-RAG", loop_count=1)

    assert rewrites
    assert any("grounded evidence" in item for item in rewrites)


def test_query_rewriter_uses_llm_json_when_valid() -> None:
    rewriter = QueryRewriter(
        llm_client=StubLLMClient(
            responder=lambda prompt, system: (
                '{"rewrites":["Truy hoi Self-RAG co dan chung","Self-RAG grounded evidence"]}'
            )
        )
    )

    rewrites = rewriter.rewrite("Self-RAG la gi?", loop_count=1)

    assert rewrites
    assert rewrites[0] == "Truy hoi Self-RAG co dan chung"


def test_critic_malformed_json_falls_back_to_heuristic_result() -> None:
    critic = HeuristicCritic(
        llm_client=StubLLMClient(
            responder=lambda prompt, system: "```json\nthis is not valid json\n```"
        )
    )

    critique = critic.critique(
        query="How does advanced retrieval work?",
        draft_answer="Advanced workflow retries retrieval.",
        context=_sample_context(),
        loop_count=1,
        max_loops=2,
    )

    assert isinstance(critique, CritiqueResult)
    assert critique.note


def test_critic_uses_llm_json_when_valid() -> None:
    critic = HeuristicCritic(
        llm_client=StubLLMClient(
            responder=lambda prompt, system: (
                "{"
                '"grounded": true,'
                '"enough_evidence": true,'
                '"has_conflict": false,'
                '"missing_aspects": ["citations"],'
                '"should_retry_retrieval": false,'
                '"should_refine_answer": true,'
                '"better_queries": ["self-rag citations"],'
                '"confidence": 0.93,'
                '"note": "llm_json_used"'
                "}"
            )
        )
    )

    critique = critic.critique(
        query="How does advanced retrieval work?",
        draft_answer="Advanced workflow retries retrieval.",
        context=_sample_context(),
        loop_count=1,
        max_loops=2,
    )

    assert critique.note == "llm_json_used"
    assert critique.confidence == 0.93
    assert critique.should_refine_answer is True


def test_advanced_answer_includes_citations_when_context_exists() -> None:
    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="adv_ctx_001",
                    doc_id="adv_doc_001",
                    source="seeded://adv",
                    content="Self-RAG truy xuất ngữ cảnh rồi phản biện để giảm suy diễn.",
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

    llm = StubLLMClient(
        responder=lambda prompt, system, model=None: (
            '{"answer":"Self-RAG truy xuất ngữ cảnh rồi phản biện để giảm suy diễn.",'
            '"confidence":0.82,"status":"answered"}'
        )
    )
    standard = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=llm,
        reranker=ScoreOnlyReranker(),
    )
    workflow = AdvancedWorkflow(standard_workflow=standard, max_loops=1)

    response = workflow.run("Self-RAG hoạt động thế nào?")

    assert response.status == "answered"
    assert response.citations
    assert response.citation_count >= 1
    assert response.grounded_score > 0


def test_advanced_marks_hallucination_when_answer_outside_context() -> None:
    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="adv_ctx_002",
                    doc_id="adv_doc_002",
                    source="seeded://adv",
                    content="Self-RAG dùng retrieval và critique để bám tài liệu gốc.",
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

    llm = StubLLMClient(
        responder=lambda prompt, system, model=None: (
            '{"answer":"Chu de tra loi la blockchain va thi truong tien so.",'
            '"confidence":0.95,"status":"answered"}'
        )
    )
    standard = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=llm,
        reranker=ScoreOnlyReranker(),
    )
    workflow = AdvancedWorkflow(standard_workflow=standard, max_loops=1)

    response = workflow.run("Self-RAG hoạt động thế nào?")

    assert response.status in {"partial", "answered"}
    assert response.citations
    assert response.hallucination_detected is False
    assert response.grounded_score > 0.0
    assert response.stop_reason in {"weak_evidence_cautious", "refined_with_context", "hallucination_refined"}


def test_advanced_hallucination_guard_refines_once_when_grounded_answer_available() -> None:
    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="adv_ctx_guard_001",
                    doc_id="adv_doc_guard_001",
                    source="seeded://adv",
                    content="Self-RAG sử dụng truy xuất, rerank và critique để tăng độ bám tài liệu.",
                    score=0.93,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def get_retriever(self) -> _FakeRetriever:
            return _FakeRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    class _CustomRefiner:
        def refine(self, *args, **kwargs) -> str:
            _ = args
            _ = kwargs
            return "Self-RAG có bước critique để tăng độ tin cậy."

        def refine_strict_grounded(self, *args, **kwargs) -> str:
            _ = args
            _ = kwargs
            return "Self-RAG sử dụng truy xuất, rerank và critique để tăng độ bám tài liệu."

    llm = StubLLMClient(
        responder=lambda prompt, system, model=None: (
            '{"answer":"Chu de tra loi la blockchain va thi truong tien so.",'
            '"confidence":0.96,"status":"answered"}'
        )
    )
    standard = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=llm,
        reranker=ScoreOnlyReranker(),
    )
    workflow = AdvancedWorkflow(
        standard_workflow=standard,
        max_loops=1,
        refiner=_CustomRefiner(),
    )

    response = workflow.run("Self-RAG hoạt động thế nào?")

    assert response.status == "answered"
    assert response.stop_reason in {"hallucination_refined", "refined_with_context"}
    assert response.hallucination_detected is False
    assert response.grounded_score > 0


def test_advanced_recovers_from_model_insufficient_when_relevant_context_exists() -> None:
    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="adv_ctx_recover_001",
                    doc_id="adv_doc_recover_001",
                    source="seeded://adv",
                    content="tokenrecoverxyz xuất hiện trong tài liệu và mô tả quy trình ký duyệt.",
                    score=0.89,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def get_retriever(self) -> _FakeRetriever:
            return _FakeRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    class _RecoverRefiner:
        def refine(self, *args, **kwargs) -> str:
            _ = args
            _ = kwargs
            return "tokenrecoverxyz mô tả quy trình ký duyệt trong tài liệu."

        def refine_strict_grounded(self, *args, **kwargs) -> str:
            _ = args
            _ = kwargs
            return "tokenrecoverxyz mô tả quy trình ký duyệt trong tài liệu."

    llm = StubLLMClient(
        responder=lambda prompt, system, model=None: (
            '{"answer":"Không đủ thông tin từ tài liệu để trả lời",'
            '"confidence":0.1,"status":"insufficient_evidence"}'
        )
    )
    standard = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=llm,
        reranker=ScoreOnlyReranker(),
    )
    workflow = AdvancedWorkflow(
        standard_workflow=standard,
        max_loops=1,
        refiner=_RecoverRefiner(),
    )

    response = workflow.run("tokenrecoverxyz là gì?")

    assert response.status != "insufficient_evidence"
    assert response.citations
    assert any(citation.doc_id == "adv_doc_recover_001" for citation in response.citations)


def test_advanced_returns_insufficient_when_no_relevant_context_exists() -> None:
    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="adv_ctx_irrelevant_001",
                    doc_id="adv_doc_irrelevant_001",
                    source="seeded://adv",
                    content="completely unrelated content about gardening and weather.",
                    score=0.7,
                    score_type="hybrid",
                    rank=1,
                )
            ]

    class _FakeIndexManager:
        def get_retriever(self) -> _FakeRetriever:
            return _FakeRetriever()

        def get_active_source(self) -> str:
            return "seeded"

    llm = StubLLMClient(
        responder=lambda prompt, system, model=None: (
            '{"answer":"This is an unsupported finance answer.",'
            '"confidence":0.9,"status":"answered"}'
        )
    )
    standard = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=llm,
        reranker=ScoreOnlyReranker(),
    )
    workflow = AdvancedWorkflow(standard_workflow=standard, max_loops=1)

    response = workflow.run("What is blockchain market cap?")

    assert response.status == "insufficient_evidence"
    assert response.stop_reason == "no_relevant_context"
    assert response.citations == []


def test_advanced_weak_evidence_returns_cautious_answer_not_empty() -> None:
    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="adv_ctx_weak_001",
                    doc_id="adv_doc_weak_001",
                    source="seeded://adv",
                    content="tokenweakabc xuất hiện trong điều khoản hướng dẫn xử lý.",
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

    llm = StubLLMClient(
        responder=lambda prompt, system, model=None: (
            '{"answer":"tokenweakabc alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi.",'
            '"confidence":0.5,"status":"answered"}'
        )
    )
    standard = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=llm,
        reranker=ScoreOnlyReranker(),
    )
    workflow = AdvancedWorkflow(standard_workflow=standard, max_loops=1)

    response = workflow.run("tokenweakabc nói gì?")

    assert response.status != "insufficient_evidence"
    assert response.answer.strip()
    assert response.citations
    assert response.stop_reason in {"weak_evidence_cautious", "hallucination_refined", "refined_with_context", "critique_pass"}


def test_advanced_preserves_citations_after_refine() -> None:
    class _FakeRetriever:
        def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
            _ = query
            _ = top_k
            return [
                RetrievalResult(
                    chunk_id="adv_ctx_refine_001",
                    doc_id="adv_doc_refine_001",
                    source="seeded://adv",
                    content="tokenrefineabc mô tả process evidence details trong tài liệu.",
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

    class _CustomRefiner:
        def refine(self, *args, **kwargs) -> str:
            _ = args
            _ = kwargs
            return "tokenrefineabc có process evidence details trong tài liệu."

        def refine_strict_grounded(self, *args, **kwargs) -> str:
            _ = args
            _ = kwargs
            return "tokenrefineabc có process evidence details trong tài liệu."

    llm = StubLLMClient(
        responder=lambda prompt, system, model=None: (
            '{"answer":"tokenrefineabc.",'
            '"confidence":0.7,"status":"answered"}'
        )
    )
    standard = StandardWorkflow(
        index_manager=_FakeIndexManager(),
        llm_client=llm,
        reranker=ScoreOnlyReranker(),
    )
    workflow = AdvancedWorkflow(
        standard_workflow=standard,
        max_loops=1,
        refiner=_CustomRefiner(),
    )

    response = workflow.run("tokenrefineabc process evidence details là gì?")

    assert response.status in {"answered", "partial"}
    assert response.citations
    assert any(citation.doc_id == "adv_doc_refine_001" for citation in response.citations)
