"""Answer refinement step for advanced workflow."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.async_utils import run_coro_sync
from app.core.config import get_settings
from app.core.json_utils import parse_json_object
from app.core.prompting import PromptRepository
from app.generation.llm_client import LLMClient, complete_with_model
from app.schemas.retrieval import RetrievalResult
from app.schemas.workflow import CritiqueResult
from app.workflows.shared import (
    build_chat_history_context,
    build_language_system_prompt,
    localized_insufficient_evidence,
    response_language_name,
)

_REFINE_PROMPT_FALLBACK = (
    "Refine the draft answer to improve groundedness and coverage.\\n"
    "Return in $response_language_name only.\\n"
    "response_language: $response_language\\n"
    "Return strict JSON only with key: refined_answer.\\n"
    "question: $question\\n"
    "chat_history: $chat_history\\n"
    "draft_answer: $draft_answer\\n"
    "critique: $critique\\n"
    "selected_context: $selected_context"
)

_STRICT_GROUNDED_REWRITE_PROMPT_FALLBACK = (
    "Rewrite the draft answer strictly based on selected context.\\n"
    "ONLY use facts present in selected_context.\\n"
    "Do NOT use external knowledge.\\n"
    "If the context does not support the answer, return exactly: "
    "\"Không đủ thông tin từ tài liệu để trả lời\"\\n"
    "Return in $response_language_name only.\\n"
    "Return strict JSON only with key: refined_answer.\\n"
    "question: $question\\n"
    "chat_history: $chat_history\\n"
    "draft_answer: $draft_answer\\n"
    "selected_context: $selected_context"
)


class AnswerRefiner:
    """Refine draft answers using critique hints and context snippets."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        prompt_repository: PromptRepository | None = None,
        prompt_dir: str | Path | None = None,
        use_llm: bool = True,
    ) -> None:
        settings = get_settings()
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.memory_window = max(0, int(getattr(settings, "memory_window", 3)))
        self.max_tokens = max(1, int(getattr(settings, "llm_max_tokens", 2048)))
        resolved_prompt_dir = prompt_dir or settings.prompt_dir
        self.prompt_repository = prompt_repository or PromptRepository(resolved_prompt_dir)

    def _heuristic_refine(
        self,
        query: str,
        draft_answer: str,
        critique: CritiqueResult,
        context: list[RetrievalResult],
        response_language: str,
    ) -> str:
        _ = query
        refined = draft_answer.strip()

        if critique.missing_aspects:
            refined += "\\n\\nAdditional coverage: " + ", ".join(critique.missing_aspects[:3]) + "."

        if context:
            lead_source = context[0]
            refined += (
                f"\\n\\nEvidence note: supported by {lead_source.title or lead_source.doc_id}"
                f" ({lead_source.chunk_id})."
            )

        if response_language == "vi":
            refined = refined.replace("Additional coverage:", "Bổ sung nội dung:")
            refined = refined.replace("Evidence note: supported by", "Ghi chú bằng chứng: được hỗ trợ bởi")

        return refined

    async def refine_async(
        self,
        query: str,
        draft_answer: str,
        critique: CritiqueResult,
        context: list[RetrievalResult],
        *,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
    ) -> str:
        heuristic = self._heuristic_refine(
            query,
            draft_answer,
            critique,
            context,
            response_language=response_language,
        )

        if not self.use_llm or self.llm_client is None:
            return heuristic

        prompt = self.prompt_repository.render(
            "refine.md",
            fallback=_REFINE_PROMPT_FALLBACK,
            question=query,
            chat_history=build_chat_history_context(
                chat_history,
                memory_window=self.memory_window,
            ),
            draft_answer=draft_answer,
            critique=json.dumps(critique.model_dump(mode="json"), ensure_ascii=False),
            selected_context=json.dumps(
                [
                    {
                        "chunk_id": item.chunk_id,
                        "doc_id": item.doc_id,
                        "content": item.content,
                    }
                    for item in context
                ],
                ensure_ascii=False,
            ),
            response_language=response_language,
            response_language_name=response_language_name(response_language),
        )

        try:
            raw = await complete_with_model(
                self.llm_client,
                prompt,
                system_prompt=build_language_system_prompt(response_language),
                model=model,
                max_tokens=self.max_tokens,
            )
        except Exception:
            return heuristic

        payload = parse_json_object(raw)
        if payload and isinstance(payload.get("refined_answer"), str):
            refined_answer = payload["refined_answer"].strip()
            if refined_answer:
                return refined_answer

        return heuristic

    async def refine_strict_grounded_async(
        self,
        *,
        query: str,
        draft_answer: str,
        context: list[RetrievalResult],
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
    ) -> str:
        """One-shot strict grounding rewrite used by hallucination guard."""
        if not context:
            return localized_insufficient_evidence(response_language)

        default_answer = localized_insufficient_evidence(response_language)
        if not self.use_llm or self.llm_client is None:
            return default_answer

        prompt = self.prompt_repository.render(
            "refine_grounded.md",
            fallback=_STRICT_GROUNDED_REWRITE_PROMPT_FALLBACK,
            question=query,
            chat_history=build_chat_history_context(
                chat_history,
                memory_window=self.memory_window,
            ),
            draft_answer=draft_answer,
            critique="{}",
            selected_context=json.dumps(
                [
                    {
                        "chunk_id": item.chunk_id,
                        "doc_id": item.doc_id,
                        "content": item.content,
                    }
                    for item in context
                ],
                ensure_ascii=False,
            ),
            response_language=response_language,
            response_language_name=response_language_name(response_language),
        )
        try:
            raw = await complete_with_model(
                self.llm_client,
                prompt,
                system_prompt=build_language_system_prompt(response_language),
                model=model,
                max_tokens=min(self.max_tokens, 512),
            )
        except Exception:
            return default_answer

        payload = parse_json_object(raw)
        if payload and isinstance(payload.get("refined_answer"), str):
            refined_answer = payload["refined_answer"].strip()
            if refined_answer:
                return refined_answer
        return default_answer

    def refine(
        self,
        query: str,
        draft_answer: str,
        critique: CritiqueResult,
        context: list[RetrievalResult],
        *,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
    ) -> str:
        """Sync wrapper for legacy callers."""
        return run_coro_sync(
            self.refine_async(
                query=query,
                draft_answer=draft_answer,
                critique=critique,
                context=context,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        )

    def refine_strict_grounded(
        self,
        *,
        query: str,
        draft_answer: str,
        context: list[RetrievalResult],
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
    ) -> str:
        """Sync wrapper for legacy callers."""
        return run_coro_sync(
            self.refine_strict_grounded_async(
                query=query,
                draft_answer=draft_answer,
                context=context,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        )
