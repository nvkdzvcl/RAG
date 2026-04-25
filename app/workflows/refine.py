"""Answer refinement step for advanced workflow."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.json_utils import parse_json_object
from app.core.prompting import PromptRepository
from app.generation.llm_client import LLMClient, complete_with_model
from app.schemas.retrieval import RetrievalResult
from app.schemas.workflow import CritiqueResult

_REFINE_PROMPT_FALLBACK = (
    "Refine the draft answer to improve groundedness and coverage.\\n"
    "Keep response language aligned with the question (Vietnamese/English).\\n"
    "Return strict JSON only with key: refined_answer.\\n"
    "question: $question\\n"
    "draft_answer: $draft_answer\\n"
    "critique: $critique\\n"
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
        resolved_prompt_dir = prompt_dir or settings.prompt_dir
        self.prompt_repository = prompt_repository or PromptRepository(resolved_prompt_dir)

    def _heuristic_refine(
        self,
        query: str,
        draft_answer: str,
        critique: CritiqueResult,
        context: list[RetrievalResult],
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

        return refined

    def refine(
        self,
        query: str,
        draft_answer: str,
        critique: CritiqueResult,
        context: list[RetrievalResult],
        *,
        model: str | None = None,
    ) -> str:
        heuristic = self._heuristic_refine(query, draft_answer, critique, context)

        if not self.use_llm or self.llm_client is None:
            return heuristic

        prompt = self.prompt_repository.render(
            "refine.md",
            fallback=_REFINE_PROMPT_FALLBACK,
            question=query,
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
        )

        try:
            raw = complete_with_model(
                self.llm_client,
                prompt,
                model=model,
            )
        except Exception:
            return heuristic

        payload = parse_json_object(raw)
        if payload and isinstance(payload.get("refined_answer"), str):
            refined_answer = payload["refined_answer"].strip()
            if refined_answer:
                return refined_answer

        return heuristic
