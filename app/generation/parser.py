"""Structured output parsing helpers for generation responses."""

from __future__ import annotations

from app.core.json_utils import parse_json_object
from app.schemas.generation import ParsedAnswer


class StructuredOutputParser:
    """Parse raw LLM output into a typed answer object."""

    @staticmethod
    def _extract_nested_answer(answer: str) -> str:
        nested = parse_json_object(answer)
        if nested and isinstance(nested.get("answer"), str):
            extracted = nested["answer"].strip()
            if extracted:
                return extracted
        return answer

    def parse(self, raw_text: str) -> ParsedAnswer:
        payload = parse_json_object(raw_text)
        if payload:
            try:
                parsed = ParsedAnswer.model_validate(payload)
                parsed.answer = self._extract_nested_answer(parsed.answer).strip()
                return parsed
            except (TypeError, ValueError):
                pass

        # Fallback: treat raw text as direct answer content.
        answer = raw_text.strip()
        if not answer:
            return ParsedAnswer(
                answer="Insufficient evidence to answer reliably.",
                confidence=0.0,
                status="insufficient_evidence",
            )

        return ParsedAnswer(answer=answer, confidence=None, status="answered")
