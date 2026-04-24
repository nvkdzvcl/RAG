"""LLM client abstraction and local stub implementation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class LLMClient(Protocol):
    """Abstraction for text generation providers."""

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Return completion text for prompt/system_prompt."""


class StubLLMClient:
    """Simple deterministic LLM client for local tests and scaffolding."""

    def __init__(
        self,
        responder: Callable[[str, str | None], str] | None = None,
    ) -> None:
        self._responder = responder or self._default_responder

    @staticmethod
    def _default_responder(prompt: str, system_prompt: str | None = None) -> str:
        _ = system_prompt
        _ = prompt
        return '{"answer":"Stub grounded answer.","confidence":0.5,"status":"answered"}'

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        return self._responder(prompt, system_prompt)
