"""Import smoke tests for project scaffolding."""


def test_import_app_modules() -> None:
    from app.main import create_app  # noqa: F401
    from app.api.router import api_router  # noqa: F401
    from app.core.config import get_settings  # noqa: F401
    from app.generation import BaselineGenerator, CitationBuilder, StructuredOutputParser  # noqa: F401
    from app.indexing import (  # noqa: F401
        BM25Index,
        BaseEmbeddingProvider,
        HashEmbeddingProvider,
        IndexBuilder,
        SentenceTransformerEmbeddingProvider,
    )
    from app.ingestion import Chunker, MarkdownLoader, TextCleaner, TextLoader  # noqa: F401
    from app.retrieval import (  # noqa: F401
        CrossEncoderReranker,
        DenseRetriever,
        HybridRetriever,
        ScoreOnlyReranker,
        SparseRetriever,
    )
    from app.services import QueryService  # noqa: F401
    from app.schemas import GeneratedAnswer, LoadedDocument, QueryRequest, RetrievalResult, WorkflowState  # noqa: F401
    from app.workflows.advanced import AdvancedWorkflow  # noqa: F401
    from app.workflows.compare import CompareWorkflow  # noqa: F401
    from app.workflows.critique import HeuristicCritic  # noqa: F401
    from app.workflows.query_rewrite import QueryRewriter  # noqa: F401
    from app.workflows.refine import AnswerRefiner  # noqa: F401
    from app.workflows.retrieval_gate import HeuristicRetrievalGate  # noqa: F401
    from app.workflows.runner import WorkflowRunner  # noqa: F401
    from app.workflows.standard import StandardWorkflow  # noqa: F401


def test_create_app() -> None:
    from app.main import create_app

    app = create_app()
    assert app.title
