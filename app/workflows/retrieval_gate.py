"""Retrieval gate for advanced workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.core.async_utils import run_coro_sync
from app.core.config import get_settings
from app.core.json_utils import parse_json_object
from app.core.prompting import PromptRepository
from app.generation.llm_client import LLMClient, complete_with_model
from app.workflows.shared import build_language_system_prompt, response_language_name

_GATE_PROMPT_FALLBACK = (
    "Decide if retrieval is required before answering.\n"
    "Return strict JSON only with keys: need_retrieval (boolean), reason (string).\n"
    "Use query meaning, not stylistic preference.\n"
    "Keep reason in $response_language_name.\n"
    "response_language: $response_language\n"
    "question: $question\n"
    "chat_history: $chat_history"
)


class HeuristicRetrievalGate:
    """Decide whether advanced mode should run retrieval."""

    _token_pattern = re.compile(r"\w+")
    _small_talk_tokens = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank",
        "bye",
        "goodbye",
    }

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
        self.max_tokens = max(1, int(getattr(settings, "llm_gate_max_tokens", 128)))
        resolved_prompt_dir = prompt_dir or settings.prompt_dir
        self.prompt_repository = prompt_repository or PromptRepository(resolved_prompt_dir)

    def _heuristic_decide(
        self, query: str, chat_history: list[dict[str, str]] | None = None
    ) -> tuple[bool, str]:
        _ = chat_history
        normalized = query.strip().lower()
        if not normalized:
            return False, "empty_query"

        if "force retrieval" in normalized or "force retry" in normalized:
            return True, "forced_retrieval"

        tokens = set(self._token_pattern.findall(normalized))
        if tokens and tokens.issubset(self._small_talk_tokens):
            return False, "small_talk"

        if len(tokens) <= 2 and any(token in self._small_talk_tokens for token in tokens):
            return False, "small_talk_short"

        return True, "default_retrieval"

    async def _llm_decide(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        *,
        model: str | None = None,
        response_language: str = "en",
    ) -> tuple[bool, str] | None:
        if not self.use_llm or self.llm_client is None:
            return None

        prompt = self.prompt_repository.render(
            "retrieval_gate.md",
            fallback=_GATE_PROMPT_FALLBACK,
            question=query,
            chat_history=json.dumps(chat_history or [], ensure_ascii=False),
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
            return None

        payload = parse_json_object(raw)
        if payload is None:
            return None

        need_retrieval = payload.get("need_retrieval")
        reason = payload.get("reason")
        if not isinstance(need_retrieval, bool):
            return None
        if not isinstance(reason, str) or not reason.strip():
            reason = "llm_gate"
        return need_retrieval, reason.strip()

    async def decide_async(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        *,
        model: str | None = None,
        response_language: str = "en",
    ) -> tuple[bool, str]:
        heuristic_need, heuristic_reason = self._heuristic_decide(query, chat_history=chat_history)
        if heuristic_reason in {"empty_query", "forced_retrieval", "small_talk", "small_talk_short"}:
            return heuristic_need, heuristic_reason

        llm_decision = await self._llm_decide(
            query,
            chat_history=chat_history,
            model=model,
            response_language=response_language,
        )
        if llm_decision is not None:
            return llm_decision
        return heuristic_need, heuristic_reason

    def decide(
        self,
        query: str,
        chat_history: list[dict[str, str]] | None = None,
        *,
        model: str | None = None,
        response_language: str = "en",
    ) -> tuple[bool, str]:
        """Sync wrapper for legacy callers."""
        return run_coro_sync(
            self.decide_async(
                query,
                chat_history=chat_history,
                model=model,
                response_language=response_language,
            )
        )
