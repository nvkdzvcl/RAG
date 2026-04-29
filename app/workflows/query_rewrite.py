"""Query rewrite logic for advanced retrieval retries."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.async_utils import run_coro_sync
from app.core.cache import QueryCache
from app.core.config import get_settings
from app.core.json_utils import parse_json_list, parse_json_object
from app.core.prompting import PromptRepository
from app.generation.llm_client import LLMClient, complete_with_model
from app.schemas.workflow import CritiqueResult
from app.workflows.shared import (
    build_chat_history_context,
    build_language_system_prompt,
    response_language_name,
)

_REWRITE_PROMPT_FALLBACK = (
    "Rewrite the query to improve retrieval quality.\n"
    "Return strict JSON only.\n"
    'Schema: {"rewrites": ["string", ...]}\n'
    "Use chat history only to resolve follow-up references.\n"
    "Provide up to 3 concise alternatives in $response_language_name.\n"
    "response_language: $response_language\n"
    "chat_history: $chat_history\n"
    "question: $question\n"
    "loop_count: $loop_count\n"
    "critique: $critique"
)


class QueryRewriter:
    """Generate rewritten queries for retry loops."""

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        prompt_repository: PromptRepository | None = None,
        prompt_dir: str | Path | None = None,
        max_candidates: int = 4,
        use_llm: bool = True,
        llm_cache: QueryCache | None = None,
    ) -> None:
        settings = get_settings()
        self.llm_client = llm_client
        self.llm_cache = llm_cache
        self.use_llm = use_llm
        self.max_candidates = max_candidates
        self.memory_window = max(0, int(getattr(settings, "memory_window", 3)))
        self.max_tokens = max(1, int(getattr(settings, "llm_rewrite_max_tokens", 256)))
        resolved_prompt_dir = prompt_dir or settings.prompt_dir
        self.prompt_repository = prompt_repository or PromptRepository(
            resolved_prompt_dir
        )

    @staticmethod
    def _dedupe(candidates: list[str], max_candidates: int) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            cleaned = item.strip()
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            deduped.append(cleaned)
            seen.add(key)
            if len(deduped) >= max_candidates:
                break
        return deduped

    def _heuristic_rewrite(
        self,
        query: str,
        *,
        critique: CritiqueResult | None = None,
        loop_count: int = 0,
    ) -> list[str]:
        base = query.strip()
        candidates: list[str] = []

        if critique and critique.better_queries:
            candidates.extend(critique.better_queries)

        candidates.append(f"{base} with grounded evidence")
        candidates.append(f"{base} with source citations")

        if critique and critique.missing_aspects:
            for aspect in critique.missing_aspects[:2]:
                candidates.append(f"{base} focus on {aspect}")

        if "force retry" in base.lower():
            candidates.append(f"{base} retry iteration {loop_count}")

        return self._dedupe(candidates, max_candidates=self.max_candidates)

    async def _llm_rewrite(
        self,
        query: str,
        *,
        critique: CritiqueResult | None = None,
        loop_count: int = 0,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
    ) -> list[str]:
        if not self.use_llm or self.llm_client is None:
            return []

        critique_payload = (
            critique.model_dump(mode="json") if critique is not None else {}
        )
        prompt = self.prompt_repository.render(
            "query_rewrite.md",
            fallback=_REWRITE_PROMPT_FALLBACK,
            question=query,
            loop_count=loop_count,
            critique=json.dumps(critique_payload, ensure_ascii=False),
            chat_history=build_chat_history_context(
                chat_history,
                memory_window=self.memory_window,
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
                llm_cache=self.llm_cache,
            )
        except Exception:
            return []

        rewrites: list[str] = []
        payload = parse_json_object(raw)
        if payload and isinstance(payload.get("rewrites"), list):
            rewrites = [
                str(item).strip() for item in payload["rewrites"] if str(item).strip()
            ]
        else:
            payload_list = parse_json_list(raw)
            if payload_list:
                rewrites = [
                    str(item).strip() for item in payload_list if str(item).strip()
                ]
        return self._dedupe(rewrites, max_candidates=self.max_candidates)

    async def rewrite_async(
        self,
        query: str,
        *,
        critique: CritiqueResult | None = None,
        loop_count: int = 0,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
    ) -> list[str]:
        llm_candidates = await self._llm_rewrite(
            query,
            critique=critique,
            loop_count=loop_count,
            chat_history=chat_history,
            model=model,
            response_language=response_language,
        )
        heuristic_candidates = self._heuristic_rewrite(
            query,
            critique=critique,
            loop_count=loop_count,
        )
        return self._dedupe(
            llm_candidates + heuristic_candidates,
            max_candidates=self.max_candidates,
        )

    def rewrite(
        self,
        query: str,
        *,
        critique: CritiqueResult | None = None,
        loop_count: int = 0,
        chat_history: list[dict[str, str]] | None = None,
        model: str | None = None,
        response_language: str = "en",
    ) -> list[str]:
        """Sync wrapper for legacy callers."""
        return run_coro_sync(
            self.rewrite_async(
                query,
                critique=critique,
                loop_count=loop_count,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        )
