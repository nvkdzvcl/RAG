"""Baseline grounded answer generator."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.core.config import get_settings
from app.core.prompting import PromptRepository
from app.generation.citations import CitationBuilder
from app.generation.interfaces import Generator
from app.generation.llm_client import LLMClient, complete_with_model, did_use_fallback
from app.generation.parser import StructuredOutputParser
from app.schemas.common import Mode
from app.schemas.generation import GeneratedAnswer
from app.schemas.retrieval import RetrievalResult
from app.workflows.shared import (
    build_language_system_prompt,
    localized_insufficient_evidence,
    response_language_name,
)

logger = logging.getLogger(__name__)


_STANDARD_PROMPT_FALLBACK = (
    "You are a grounded RAG assistant.\n"
    "Use chat history only to resolve follow-up references (for example: \"còn điều 3 thì sao\").\n"
    "Do not let chat history override provided evidence context.\n"
    "Answer in the required response language only.\n"
    "If response_language is Vietnamese, write fully in Vietnamese.\n"
    "Do not answer in Chinese unless the user asks in Chinese.\n"
    "Only for explicit title queries (e.g., \"tên của Điều 2 là gì\", \"điều 2 tên là gì\", \"tên mục/phần ... là gì\"), "
    "if context contains the exact heading/title, "
    "return it exactly as: \"Tên của Điều X là: <exact title>.\"\n"
    "Do not use the title format for compare/explain questions (for example: \"phân biệt\", \"so sánh\", \"giải thích\", \"trình bày\").\n"
    "For compare questions (for example \"Phân biệt A và B\"), explain differences based only on provided context.\n"
    "For factual definition/name queries (for example containing \"là gì\", \"định nghĩa\", \"tên\"), answer directly and concisely.\n"
    "Do not paraphrase official title text.\n"
    "Use only the provided context; do not invent unsupported facts.\n"
    "If evidence is insufficient, set status to insufficient_evidence.\n"
    "Return strict JSON with keys: answer, confidence, status.\n"
    "response_language: $response_language ($response_language_name)\n"
    "mode: $mode\n"
    "chat_history:\n$chat_history\n"
    "question: $question\n"
    "context:\n$context"
)

_ADVANCED_PROMPT_FALLBACK = (
    "You are the advanced Self-RAG answerer.\n"
    "ONLY use the provided context chunks.\n"
    "Every answer must be supported by context.\n"
    "Do NOT use external knowledge.\n"
    "Use chat history only to resolve follow-up references (for example: \"còn điều 3 thì sao\").\n"
    "Do not let chat history override provided evidence context.\n"
    "If the answer is not found in context, respond exactly: "
    "\"Không đủ thông tin từ tài liệu để trả lời\"\n"
    "Answer in the required response language only.\n"
    "If response_language is Vietnamese, write fully in Vietnamese.\n"
    "Do not answer in Chinese unless the user asks in Chinese.\n"
    "Only for explicit title queries (e.g., \"tên của Điều 2 là gì\", \"điều 2 tên là gì\", \"tên mục/phần ... là gì\"), "
    "if context contains the exact heading/title, "
    "return it exactly as: \"Tên của Điều X là: <exact title>.\"\n"
    "Do not use the title format for compare/explain questions (for example: \"phân biệt\", \"so sánh\", \"giải thích\", \"trình bày\").\n"
    "For compare questions (for example \"Phân biệt A và B\"), explain differences based only on provided context.\n"
    "For factual definition/name queries (for example containing \"là gì\", \"định nghĩa\", \"tên\"), answer directly and concisely.\n"
    "Do not paraphrase official title text.\n"
    "Ground every claim in the context. If evidence is weak, abstain with status=insufficient_evidence.\n"
    "Return strict JSON with keys: answer, confidence, status.\n"
    "response_language: $response_language ($response_language_name)\n"
    "mode: $mode\n"
    "chat_history:\n$chat_history\n"
    "question: $question\n"
    "context:\n$context"
)

_NON_TITLE_INTENT_KEYWORDS = (
    "phân biệt",
    "phan biet",
    "so sánh",
    "so sanh",
    "giải thích",
    "giai thich",
    "trình bày",
    "trinh bay",
    "compare",
    "comparison",
    "differentiate",
    "difference",
    "explain",
)

_EXPLICIT_TITLE_PATTERNS = (
    re.compile(r"\btên\s+(?:của\s+)?điều\s+(?P<article>\d+[a-z]?)\s+là\s+gì\b", flags=re.IGNORECASE),
    re.compile(r"\bten\s+(?:cua\s+)?dieu\s+(?P<article>\d+[a-z]?)\s+la\s+gi\b", flags=re.IGNORECASE),
    re.compile(r"\bđiều\s+(?P<article>\d+[a-z]?)\s+tên\s+là\s+gì\b", flags=re.IGNORECASE),
    re.compile(r"\bdieu\s+(?P<article>\d+[a-z]?)\s+ten\s+la\s+gi\b", flags=re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+the\s+(?:title|name)\s+of\s+article\s+(?P<article>\d+[a-z]?)\b", flags=re.IGNORECASE),
)


class BaselineGenerator(Generator):
    """Generate grounded answers from selected retrieval context."""

    def __init__(
        self,
        llm_client: LLMClient,
        parser: StructuredOutputParser | None = None,
        citation_builder: CitationBuilder | None = None,
        prompt_repository: PromptRepository | None = None,
        prompt_dir: str | Path | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.parser = parser or StructuredOutputParser()
        self.citation_builder = citation_builder or CitationBuilder()
        settings = get_settings()
        resolved_prompt_dir = prompt_dir or settings.prompt_dir
        self.prompt_repository = prompt_repository or PromptRepository(resolved_prompt_dir)

    def _build_prompt(
        self,
        query: str,
        context: list[RetrievalResult],
        mode: Mode,
        response_language: str,
        chat_history_context: str,
    ) -> str:
        joined_context = "\n\n".join(
            f"[chunk_id={doc.chunk_id}] {doc.content}" for doc in context
        )
        prompt_name = "advanced_answer.md" if mode == Mode.ADVANCED else "standard_answer.md"
        prompt_fallback = _ADVANCED_PROMPT_FALLBACK if mode == Mode.ADVANCED else _STANDARD_PROMPT_FALLBACK
        return self.prompt_repository.render(
            prompt_name,
            fallback=prompt_fallback,
            mode=mode.value,
            question=query,
            context=joined_context,
            chat_history=chat_history_context,
            response_language=response_language,
            response_language_name=response_language_name(response_language),
        )

    @staticmethod
    def _title_question_article_number(query: str) -> str | None:
        lowered = query.casefold()
        if any(keyword in lowered for keyword in _NON_TITLE_INTENT_KEYWORDS):
            return None

        for pattern in _EXPLICIT_TITLE_PATTERNS:
            match = pattern.search(query)
            if not match:
                continue
            article = match.group("article").strip()
            if article:
                return article.upper()
        return None

    @staticmethod
    def _extract_article_title(
        article_number: str,
        context: list[RetrievalResult],
    ) -> tuple[str, RetrievalResult] | None:
        escaped_number = re.escape(article_number)
        line_pattern = re.compile(
            rf"(?im)^\s*(?:điều|dieu|article)\s*{escaped_number}\s*[\.\:\-]\s*(.+?)\s*$"
        )
        fallback_pattern = re.compile(
            rf"(?im)(?:điều|dieu|article)\s*{escaped_number}\s*[\.\:\-]\s*([^\n]+)"
        )

        for item in context:
            for line in item.content.splitlines():
                matched = line_pattern.match(line.strip())
                if not matched:
                    continue
                title = matched.group(1).strip().rstrip(".")
                if title:
                    return title, item

            fallback_match = fallback_pattern.search(item.content)
            if fallback_match:
                title = fallback_match.group(1).strip().rstrip(".")
                if title:
                    return title, item

        return None

    def _generate_exact_title_answer(
        self,
        *,
        query: str,
        context: list[RetrievalResult],
        response_language: str,
    ) -> GeneratedAnswer | None:
        article_number = self._title_question_article_number(query)
        if not article_number:
            return None

        extracted = self._extract_article_title(article_number, context)
        if not extracted:
            return None

        title, supporting_doc = extracted
        if response_language == "vi":
            answer = f"Tên của Điều {article_number} là: {title}."
        else:
            answer = f"The title of Article {article_number} is: {title}."

        return GeneratedAnswer(
            answer=answer,
            citations=self.citation_builder.build([supporting_doc]),
            confidence=0.95,
            status="answered",
            stop_reason="heuristic_exact_title",
            raw_output=None,
            llm_fallback_used=False,
        )

    def _insufficient(
        self,
        reason: str,
        *,
        response_language: str,
        raw_output: str | None = None,
        llm_fallback_used: bool = False,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=localized_insufficient_evidence(response_language),
            citations=[],
            confidence=0.0,
            status="insufficient_evidence",
            stop_reason=reason,
            raw_output=raw_output,
            llm_fallback_used=llm_fallback_used,
        )

    def generate_answer(
        self,
        query: str,
        context: list[RetrievalResult],
        mode: Mode,
        model: str | None = None,
        response_language: str = "en",
        chat_history_context: str = "(empty)",
    ) -> GeneratedAnswer:
        non_empty_context = [doc for doc in context if doc.content.strip()]
        if not non_empty_context:
            return self._insufficient(
                "no_context",
                response_language=response_language,
            )

        exact_title_answer = self._generate_exact_title_answer(
            query=query,
            context=non_empty_context,
            response_language=response_language,
        )
        if exact_title_answer is not None:
            return exact_title_answer

        llm_fallback_used = False
        prompt = self._build_prompt(
            query,
            non_empty_context,
            mode,
            response_language,
            chat_history_context,
        )
        try:
            raw_output = complete_with_model(
                self.llm_client,
                prompt,
                system_prompt=build_language_system_prompt(response_language),
                model=model,
            )
            llm_fallback_used = did_use_fallback(self.llm_client)
        except Exception as exc:
            logger.warning("LLM completion failed in BaselineGenerator.", exc_info=exc)
            return self._insufficient(
                "llm_error",
                response_language=response_language,
                llm_fallback_used=did_use_fallback(self.llm_client),
            )
        parsed = self.parser.parse(raw_output)

        if parsed.status == "insufficient_evidence" or not parsed.answer.strip():
            return self._insufficient(
                "model_insufficient_evidence",
                response_language=response_language,
                raw_output=raw_output,
                llm_fallback_used=llm_fallback_used,
            )

        citations = self.citation_builder.build(non_empty_context)
        return GeneratedAnswer(
            answer=parsed.answer.strip(),
            citations=citations,
            confidence=parsed.confidence,
            status=parsed.status,
            stop_reason="generated",
            raw_output=raw_output,
            llm_fallback_used=llm_fallback_used,
        )
