"""Import smoke tests for project scaffolding."""


def test_import_app_modules() -> None:
    from app.main import create_app as create_app
    from app.api.router import api_router as api_router
    from app.core.config import get_settings as get_settings
    from app.generation import (
        BaselineGenerator as BaselineGenerator,
        CitationBuilder as CitationBuilder,
        StructuredOutputParser as StructuredOutputParser,
    )
    from app.indexing import (
        BM25Index as BM25Index,
        BaseEmbeddingProvider as BaseEmbeddingProvider,
        HashEmbeddingProvider as HashEmbeddingProvider,
        IndexBuilder as IndexBuilder,
        SentenceTransformerEmbeddingProvider as SentenceTransformerEmbeddingProvider,
    )
    from app.ingestion import (
        Chunker as Chunker,
        MarkdownLoader as MarkdownLoader,
        TextCleaner as TextCleaner,
        TextLoader as TextLoader,
    )
    from app.retrieval import (
        CrossEncoderReranker as CrossEncoderReranker,
        DenseRetriever as DenseRetriever,
        HybridRetriever as HybridRetriever,
        ScoreOnlyReranker as ScoreOnlyReranker,
        SparseRetriever as SparseRetriever,
    )
    from app.services import QueryService as QueryService
    from app.schemas import (
        GeneratedAnswer as GeneratedAnswer,
        LoadedDocument as LoadedDocument,
        QueryRequest as QueryRequest,
        RetrievalResult as RetrievalResult,
        WorkflowState as WorkflowState,
    )
    from app.workflows.advanced import AdvancedWorkflow as AdvancedWorkflow
    from app.workflows.compare import CompareWorkflow as CompareWorkflow
    from app.workflows.critique import HeuristicCritic as HeuristicCritic
    from app.workflows.query_rewrite import QueryRewriter as QueryRewriter
    from app.workflows.refine import AnswerRefiner as AnswerRefiner
    from app.workflows.retrieval_gate import (
        HeuristicRetrievalGate as HeuristicRetrievalGate,
    )
    from app.workflows.runner import WorkflowRunner as WorkflowRunner
    from app.workflows.standard import StandardWorkflow as StandardWorkflow


def test_create_app() -> None:
    from app.main import create_app

    app = create_app()
    assert app.title
