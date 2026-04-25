"""Query rewrite logic for advanced retrieval retries."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.json_utils import parse_json_list, parse_json_object
from app.core.prompting import PromptRepository
from app.generation.llm_client import LLMClient, complete_with_model
from app.schemas.workflow import CritiqueResult

_REWRITE_PROMPT_FALLBACK = (
    "Rewrite the query to improve retrieval quality.\n"
    "Return strict JSON only.\n"
    "Schema: {\"rewrites\": [\"string\", ...]}\n"
    "Provide up to 3 concise alternatives, keep original language (Vietnamese compatible).\n"
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
    ) -> None:
        settings = get_settings()
        self.llm_client = llm_client
        self.use_llm = use_llm
        self.max_candidates = max_candidates
        resolved_prompt_dir = prompt_dir or settings.prompt_dir
        self.prompt_repository = prompt_repository or PromptRepository(resolved_prompt_dir)

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

    def _llm_rewrite(
        self,
        query: str,
        *,
        critique: CritiqueResult | None = None,
        loop_count: int = 0,
        model: str | None = None,
    ) -> list[str]:
        if not self.use_llm or self.llm_client is None:
            return []

        critique_payload = critique.model_dump(mode="json") if critique is not None else {}
        prompt = self.prompt_repository.render(
            "query_rewrite.md",
            fallback=_REWRITE_PROMPT_FALLBACK,
            question=query,
            loop_count=loop_count,
            critique=json.dumps(critique_payload, ensure_ascii=False),
        )

        try:
            raw = complete_with_model(
                self.llm_client,
                prompt,
                model=model,
            )
        except Exception:
            return []

        rewrites: list[str] = []
        payload = parse_json_object(raw)
        if payload and isinstance(payload.get("rewrites"), list):
            rewrites = [str(item).strip() for item in payload["rewrites"] if str(item).strip()]
        else:
            payload_list = parse_json_list(raw)
            if payload_list:
                rewrites = [str(item).strip() for item in payload_list if str(item).strip()]
        return self._dedupe(rewrites, max_candidates=self.max_candidates)

    def rewrite(
        self,
        query: str,
        *,
        critique: CritiqueResult | None = None,
        loop_count: int = 0,
        model: str | None = None,
    ) -> list[str]:
        llm_candidates = self._llm_rewrite(
            query,
            critique=critique,
            loop_count=loop_count,
            model=model,
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
