"""Generator interface for baseline grounded answer generation."""

from __future__ import annotations

from typing import Protocol

from app.schemas.common import Mode
from app.schemas.generation import GeneratedAnswer
from app.schemas.retrieval import RetrievalResult


class Generator(Protocol):
    """Generator contract shared across workflow modes."""

    def generate_answer(
        self,
        query: str,
        context: list[RetrievalResult],
        mode: Mode,
        model: str | None = None,
        response_language: str = "en",
    ) -> GeneratedAnswer:
        """Generate grounded answer from query and selected context."""
