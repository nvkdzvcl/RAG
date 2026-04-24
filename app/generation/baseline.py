"""Baseline grounded answer generator."""

from __future__ import annotations

from app.generation.citations import CitationBuilder
from app.generation.interfaces import Generator
from app.generation.llm_client import LLMClient
from app.generation.parser import StructuredOutputParser
from app.schemas.common import Mode
from app.schemas.generation import GeneratedAnswer
from app.schemas.retrieval import RetrievalResult


class BaselineGenerator(Generator):
    """Generate grounded answers from selected retrieval context."""

    def __init__(
        self,
        llm_client: LLMClient,
        parser: StructuredOutputParser | None = None,
        citation_builder: CitationBuilder | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.parser = parser or StructuredOutputParser()
        self.citation_builder = citation_builder or CitationBuilder()

    def _build_prompt(self, query: str, context: list[RetrievalResult], mode: Mode) -> str:
        joined_context = "\n\n".join(
            f"[chunk_id={doc.chunk_id}] {doc.content}" for doc in context
        )
        return (
            f"Mode: {mode.value}\n"
            f"Question: {query}\n"
            "Use only the provided context. If evidence is weak, respond with status=insufficient_evidence.\n"
            "Return strict JSON with keys: answer, confidence, status.\n"
            f"Context:\n{joined_context}"
        )

    def _insufficient(self, reason: str, raw_output: str | None = None) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer="Insufficient evidence to provide a grounded answer.",
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
    ) -> GeneratedAnswer:
        non_empty_context = [doc for doc in context if doc.content.strip()]
        if not non_empty_context:
            return self._insufficient("no_context")

        prompt = self._build_prompt(query, non_empty_context, mode)
        raw_output = self.llm_client.complete(prompt)
        parsed = self.parser.parse(raw_output)

        if parsed.status == "insufficient_evidence" or not parsed.answer.strip():
            return self._insufficient("model_insufficient_evidence", raw_output=raw_output)

        citations = self.citation_builder.build(non_empty_context)
        return GeneratedAnswer(
            answer=parsed.answer.strip(),
            citations=citations,
            confidence=parsed.confidence,
            status=parsed.status,
            stop_reason="generated",
            raw_output=raw_output,
        )
