"""Baseline grounded answer generator."""

from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import get_settings
from app.core.prompting import PromptRepository
from app.generation.citations import CitationBuilder
from app.generation.interfaces import Generator
from app.generation.llm_client import LLMClient, complete_with_model
from app.generation.parser import StructuredOutputParser
from app.schemas.common import Mode
from app.schemas.generation import GeneratedAnswer
from app.schemas.retrieval import RetrievalResult
from app.workflows.shared import (
    build_language_system_prompt,
    localized_insufficient_evidence,
    response_language_name,
)

logger = logging.getLogger(__name__)


_STANDARD_PROMPT_FALLBACK = (
    "You are a grounded RAG assistant.\n"
    "Answer in the required response language only.\n"
    "If response_language is Vietnamese, write fully in Vietnamese.\n"
    "Do not answer in Chinese unless the user asks in Chinese.\n"
    "Use only the provided context; do not invent unsupported facts.\n"
    "If evidence is insufficient, set status to insufficient_evidence.\n"
    "Return strict JSON with keys: answer, confidence, status.\n"
    "response_language: $response_language ($response_language_name)\n"
    "mode: $mode\n"
    "question: $question\n"
    "context:\n$context"
)

_ADVANCED_PROMPT_FALLBACK = (
    "You are the advanced Self-RAG answerer.\n"
    "Answer in the required response language only.\n"
    "If response_language is Vietnamese, write fully in Vietnamese.\n"
    "Do not answer in Chinese unless the user asks in Chinese.\n"
    "Ground every claim in the context. If evidence is weak, abstain with status=insufficient_evidence.\n"
    "Return strict JSON with keys: answer, confidence, status.\n"
    "response_language: $response_language ($response_language_name)\n"
    "mode: $mode\n"
    "question: $question\n"
    "context:\n$context"
)


class BaselineGenerator(Generator):
    """Generate grounded answers from selected retrieval context."""

    def __init__(
        self,
        llm_client: LLMClient,
        parser: StructuredOutputParser | None = None,
        citation_builder: CitationBuilder | None = None,
        prompt_repository: PromptRepository | None = None,
        prompt_dir: str | Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.parser = parser or StructuredOutputParser()
        self.citation_builder = citation_builder or CitationBuilder()
        settings = get_settings()
        resolved_prompt_dir = prompt_dir or settings.prompt_dir
        self.prompt_repository = prompt_repository or PromptRepository(resolved_prompt_dir)

    def _build_prompt(
        self,
        query: str,
        context: list[RetrievalResult],
        mode: Mode,
        response_language: str,
    ) -> str:
        joined_context = "\n\n".join(
            f"[chunk_id={doc.chunk_id}] {doc.content}" for doc in context
        )
        prompt_name = "advanced_answer.md" if mode == Mode.ADVANCED else "standard_answer.md"
        prompt_fallback = _ADVANCED_PROMPT_FALLBACK if mode == Mode.ADVANCED else _STANDARD_PROMPT_FALLBACK
        return self.prompt_repository.render(
            prompt_name,
            fallback=prompt_fallback,
            mode=mode.value,
            question=query,
            context=joined_context,
            response_language=response_language,
            response_language_name=response_language_name(response_language),
        )

    def _insufficient(
        self,
        reason: str,
        *,
        response_language: str,
        raw_output: str | None = None,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=localized_insufficient_evidence(response_language),
            citations=[],
            confidence=0.0,
            status="insufficient_evidence",
            stop_reason=reason,
            raw_output=raw_output,
        )

    def generate_answer(
        self,
        query: str,
        context: list[RetrievalResult],
        mode: Mode,
        model: str | None = None,
        response_language: str = "en",
    ) -> GeneratedAnswer:
        non_empty_context = [doc for doc in context if doc.content.strip()]
        if not non_empty_context:
            return self._insufficient(
                "no_context",
                response_language=response_language,
            )

        prompt = self._build_prompt(
            query,
            non_empty_context,
            mode,
            response_language,
        )
        try:
            raw_output = complete_with_model(
                self.llm_client,
                prompt,
                system_prompt=build_language_system_prompt(response_language),
                model=model,
            )
        except Exception as exc:
            logger.warning("LLM completion failed in BaselineGenerator.", exc_info=exc)
            return self._insufficient(
                "llm_error",
                response_language=response_language,
            )
        parsed = self.parser.parse(raw_output)

        if parsed.status == "insufficient_evidence" or not parsed.answer.strip():
            return self._insufficient(
                "model_insufficient_evidence",
                response_language=response_language,
                raw_output=raw_output,
            )

        citations = self.citation_builder.build(non_empty_context)
        return GeneratedAnswer(
            answer=parsed.answer.strip(),
            citations=citations,
            confidence=parsed.confidence,
            status=parsed.status,
            stop_reason="generated",
            raw_output=raw_output,
        )
