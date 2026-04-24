"""Retrieval gate for advanced workflow."""

from __future__ import annotations

import re


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

    def decide(self, query: str, chat_history: list[dict[str, str]] | None = None) -> tuple[bool, str]:
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
