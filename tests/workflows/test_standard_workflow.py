"""Tests for standard workflow end-to-end path."""

from app.schemas.retrieval import RetrievalResult
from app.schemas.api import StandardQueryResponse, validate_query_response
from app.schemas.common import Mode
from app.services import QueryService
from app.workflows.runner import WorkflowRunner
from app.workflows.standard import StandardWorkflow


def test_standard_workflow_run_path() -> None:
    workflow = StandardWorkflow()

    response = workflow.run(query="How does standard mode perform retrieval?", chat_history=None)

    assert isinstance(response, StandardQueryResponse)
    assert response.mode == "standard"
    assert response.answer
    assert isinstance(response.citations, list)
    assert response.status in {"answered", "partial", "insufficient_evidence"}


def test_standard_workflow_uses_ingested_files_instead_of_memory_corpus() -> None:
    workflow = StandardWorkflow()

    response = workflow.run(query="What modes are supported?", chat_history=None)

    assert response.trace[0]["step"] == "retrieve"
    assert response.trace[0]["count"] > 0
    assert all(not citation.source.startswith("memory://") for citation in response.citations)


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
    workflow = StandardWorkflow(index_manager=_FakeIndexManager(), llm_client=llm_client)

    response = workflow.run(query="Use qwen2.5:7b", model="qwen2.5:7b")

    assert response.answer == "Model override answer."
    assert llm_client.models
    assert set(llm_client.models) == {"qwen2.5:7b"}
