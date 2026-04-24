"""Core interfaces for retrieval, generation, critique, and workflow orchestration."""

from typing import Protocol

from app.schemas.common import Mode


class Retriever(Protocol):
    """Retriever contract."""

    def retrieve_dense(self, query: str, top_k: int) -> list[dict]: ...

    def retrieve_sparse(self, query: str, top_k: int) -> list[dict]: ...

    def retrieve_hybrid(self, query: str, top_k: int) -> list[dict]: ...


class Reranker(Protocol):
    """Reranker contract."""

    def rerank(self, query: str, docs: list[dict]) -> list[dict]: ...


class Generator(Protocol):
    """Generator contract."""

    def generate_answer(self, query: str, context: list[dict], mode: Mode) -> dict: ...


class Critic(Protocol):
    """Critic contract."""

    def critique(self, query: str, draft_answer: str, context: list[dict]) -> dict: ...


class Workflow(Protocol):
    """Workflow contract."""

    def run(self, query: str, chat_history: list[dict[str, str]] | None = None) -> dict: ...
