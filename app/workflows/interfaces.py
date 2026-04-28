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

    async def generate_answer_async(
        self,
        query: str,
        context: list[dict],
        mode: Mode,
        model: str | None = None,
    ) -> dict: ...

    def generate_answer(
        self,
        query: str,
        context: list[dict],
        mode: Mode,
        model: str | None = None,
    ) -> dict: ...


class Critic(Protocol):
    """Critic contract."""

    async def critique_async(
        self, query: str, draft_answer: str, context: list[dict]
    ) -> dict: ...

    def critique(self, query: str, draft_answer: str, context: list[dict]) -> dict: ...


class Workflow(Protocol):
    """Workflow contract."""

    async def run_async(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
    ) -> dict: ...

    def run(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str | None = None,
    ) -> dict: ...
