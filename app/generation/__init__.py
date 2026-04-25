"""Generation package for grounded baseline answering."""

from app.generation.baseline import BaselineGenerator
from app.generation.citations import CitationBuilder
from app.generation.interfaces import Generator
from app.generation.llm_client import (
    FallbackLLMClient,
    LLMClient,
    OpenAICompatibleLLMClient,
    StubLLMClient,
    complete_with_model,
    create_llm_client,
    create_llm_client_from_settings,
    did_use_fallback,
)
from app.generation.parser import StructuredOutputParser

__all__ = [
    "BaselineGenerator",
    "CitationBuilder",
    "FallbackLLMClient",
    "Generator",
    "LLMClient",
    "OpenAICompatibleLLMClient",
    "StructuredOutputParser",
    "StubLLMClient",
    "complete_with_model",
    "create_llm_client",
    "create_llm_client_from_settings",
    "did_use_fallback",
]
