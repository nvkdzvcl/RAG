"""Structured output parsing helpers for generation responses."""

from __future__ import annotations

import json
import re

from app.schemas.generation import ParsedAnswer


class StructuredOutputParser:
    """Parse raw LLM output into a typed answer object."""

    _fenced_json = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)

    def _extract_json_candidate(self, raw_text: str) -> str | None:
        text = raw_text.strip()

        fenced_match = self._fenced_json.search(text)
        if fenced_match:
            return fenced_match.group(1).strip()

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1].strip()

        return None

    def parse(self, raw_text: str) -> ParsedAnswer:
        json_candidate = self._extract_json_candidate(raw_text)

        if json_candidate:
            try:
                payload = json.loads(json_candidate)
                return ParsedAnswer.model_validate(payload)
            except (json.JSONDecodeError, TypeError, ValueError):
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
