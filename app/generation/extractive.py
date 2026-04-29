"""Deterministic extractive answerer for high-confidence simple questions."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.generation.citations import CitationBuilder
from app.schemas.generation import GeneratedAnswer
from app.schemas.retrieval import RetrievalResult

_WORD_PATTERN = re.compile(r"[A-Za-zÀ-ỹĐđ0-9]+")
_TITLE_QUERY_PATTERN = re.compile(
    r"\btên\s+(?:của\s+)?điều\s+(?P<article>\d+[a-z]?)\s+là\s+gì\b",
    flags=re.IGNORECASE,
)
_TITLE_QUERY_PATTERN_ASCII = re.compile(
    r"\bten\s+(?:cua\s+)?dieu\s+(?P<article>\d+[a-z]?)\s+la\s+gi\b",
    flags=re.IGNORECASE,
)
_ARTICLE_SCOPE_PATTERN = re.compile(
    r"(?im)^\s*(?:điều|dieu|article)\s+(?P<article>\d+[a-z]?)\s*[\.\:\-]\s*(?P<title>.+?)\s*$"
)
_ARTICLE_QUESTION_PATTERN = re.compile(
    r"\bđiều\s+(?P<article>\d+[a-z]?)\s+quy\s+định\s+gì\b",
    flags=re.IGNORECASE,
)
_ARTICLE_QUESTION_PATTERN_ASCII = re.compile(
    r"\bdieu\s+(?P<article>\d+[a-z]?)\s+quy\s+dinh\s+gi\b",
    flags=re.IGNORECASE,
)
_CLAUSE_ARTICLE_PATTERN = re.compile(
    r"\bkhoản\s+(?P<clause>\d+[a-z]?)\s+điều\s+(?P<article>\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
_CLAUSE_ARTICLE_PATTERN_ASCII = re.compile(
    r"\bkhoan\s+(?P<clause>\d+[a-z]?)\s+dieu\s+(?P<article>\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)*(?:\s*(?:đồng|dong|vnđ|vnd|triệu|trieu|tỷ|ty|tháng|thang|ngày|ngay|năm|nam|%))?",
    flags=re.IGNORECASE,
)
_DEFINITION_QUERY_PATTERN = re.compile(
    r"\bđịnh\s+nghĩa\s+(?P<term>.+?)(?:\s+là\s+gì|\?|$)",
    flags=re.IGNORECASE,
)
_DEFINITION_QUERY_PATTERN_ASCII = re.compile(
    r"\bdinh\s+nghia\s+(?P<term>.+?)(?:\s+la\s+gi|\?|$)",
    flags=re.IGNORECASE,
)
_WHAT_IS_QUERY_PATTERN = re.compile(
    r"\b(?P<term>[A-Za-zÀ-ỹĐđ0-9 _'\-]{2,80})\s+là\s+gì\b",
    flags=re.IGNORECASE,
)
_WHAT_IS_QUERY_PATTERN_ASCII = re.compile(
    r"\b(?P<term>[A-Za-zÀ-ỹĐđ0-9 _'\-]{2,80})\s+la\s+gi\b",
    flags=re.IGNORECASE,
)

_STOPWORDS = {
    "la",
    "là",
    "gì",
    "gi",
    "bao",
    "nhiêu",
    "bao_nhieu",
    "điều",
    "dieu",
    "khoản",
    "khoan",
    "mức",
    "muc",
    "phạt",
    "phat",
    "số",
    "so",
    "tiền",
    "tien",
    "thời",
    "thoi",
    "hạn",
    "han",
    "ngày",
    "ngay",
    "trích",
    "trich",
    "nguyên",
    "nguyen",
    "văn",
    "van",
    "quy",
    "định",
    "dinh",
}


@dataclass(frozen=True)
class ExtractiveDecision:
    attempted: bool
    used: bool
    reason: str
    answer: GeneratedAnswer | None = None


@dataclass(frozen=True)
class _Candidate:
    answer: str
    support_docs: list[RetrievalResult]
    reason: str
    answer_type: str


class ExtractiveAnswerer:
    """Rule-based extractor for simple high-confidence questions."""

    _MIN_OVERLAP_RATIO = 0.35
    _LOW_SCORE_THRESHOLD = 0.005
    _MAX_VERBATIM_CHARS = 900
    _MAX_SENTENCE_CHARS = 420

    def __init__(self, *, citation_builder: CitationBuilder | None = None) -> None:
        self.citation_builder = citation_builder or CitationBuilder()

    @staticmethod
    def _strip_diacritics(value: str) -> str:
        decomposed = unicodedata.normalize("NFD", value)
        stripped = "".join(
            char for char in decomposed if unicodedata.category(char) != "Mn"
        )
        return unicodedata.normalize("NFC", stripped)

    @classmethod
    def _normalized_text(cls, value: str) -> str:
        lowered = value.strip().lower()
        return cls._strip_diacritics(lowered)

    @classmethod
    def _tokens(cls, value: str) -> set[str]:
        normalized = cls._normalized_text(value)
        tokens: set[str] = set()
        for match in _WORD_PATTERN.findall(normalized):
            token = match.strip().lower()
            if len(token) <= 1:
                continue
            if token in _STOPWORDS:
                continue
            tokens.add(token)
        return tokens

    @classmethod
    def _lexical_overlap_ratio(cls, query: str, content: str) -> float:
        query_tokens = cls._tokens(query)
        if not query_tokens:
            return 0.0
        content_tokens = cls._tokens(content)
        if not content_tokens:
            return 0.0
        matched = len(query_tokens.intersection(content_tokens))
        return matched / len(query_tokens)

    @staticmethod
    def _best_score(result: RetrievalResult) -> float | None:
        candidates = (
            result.rerank_score,
            result.score,
            result.dense_score,
            result.sparse_score,
        )
        best: float | None = None
        for candidate in candidates:
            if not isinstance(candidate, (int, float)):
                continue
            numeric = float(candidate)
            if numeric != numeric:
                continue
            if best is None or numeric > best:
                best = numeric
        return best

    @classmethod
    def _score_confident_enough(cls, result: RetrievalResult) -> bool:
        score = cls._best_score(result)
        if score is None:
            return True
        if 0.0 <= score <= 1.0:
            return score >= cls._LOW_SCORE_THRESHOLD
        return score > 0.0

    @staticmethod
    def _article_from_query(query: str) -> str | None:
        for pattern in (_TITLE_QUERY_PATTERN, _TITLE_QUERY_PATTERN_ASCII):
            matched = pattern.search(query)
            if matched:
                return matched.group("article").upper()
        return None

    @staticmethod
    def _extract_article_heading(
        article: str,
        content: str,
    ) -> tuple[str, str] | None:
        article_norm = article.strip().upper()
        for match in _ARTICLE_SCOPE_PATTERN.finditer(content):
            matched_article = match.group("article").strip().upper()
            if matched_article != article_norm:
                continue
            title = match.group("title").strip().rstrip(".")
            if not title:
                continue
            heading_line = match.group(0).strip()
            return title, heading_line
        return None

    @staticmethod
    def _extract_article_excerpt(
        *,
        article: str,
        content: str,
        max_chars: int,
    ) -> str | None:
        article_norm = article.strip().upper()
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        start_index: int | None = None
        for idx, line in enumerate(lines):
            heading_match = _ARTICLE_SCOPE_PATTERN.match(line)
            if not heading_match:
                continue
            matched_article = heading_match.group("article").strip().upper()
            if matched_article == article_norm:
                start_index = idx
                break
        if start_index is None:
            return None

        collected: list[str] = []
        for idx in range(start_index + 1, len(lines)):
            line = lines[idx]
            if _ARTICLE_SCOPE_PATTERN.match(line):
                break
            collected.append(line)
            joined = " ".join(collected).strip()
            if len(joined) >= max_chars:
                break
        excerpt = " ".join(collected).strip()
        if not excerpt:
            return None
        return excerpt[:max_chars].strip()

    @staticmethod
    def _split_sentences(content: str) -> list[str]:
        raw = re.split(r"(?<=[\.\!\?])\s+|\n+", content)
        return [item.strip() for item in raw if item.strip()]

    def _match_title_query(
        self,
        query: str,
        selected_context: list[RetrievalResult],
        *,
        response_language: str,
    ) -> _Candidate | None:
        article = self._article_from_query(query)
        if article is None:
            return None
        top = selected_context[0]
        extracted = self._extract_article_heading(article, top.content)
        if extracted is None:
            return None
        title, _heading = extracted
        if response_language == "vi":
            answer = f"Tên của Điều {article} là: {title}."
        else:
            answer = f"The title of Article {article} is: {title}."
        return _Candidate(
            answer=answer,
            support_docs=[top],
            reason="title_article_match",
            answer_type="extractive_title",
        )

    def _match_article_rule_query(
        self,
        query: str,
        selected_context: list[RetrievalResult],
        *,
        response_language: str,
    ) -> _Candidate | None:
        matched = _ARTICLE_QUESTION_PATTERN.search(
            query
        ) or _ARTICLE_QUESTION_PATTERN_ASCII.search(query)
        if not matched:
            return None
        article = matched.group("article").upper()
        top = selected_context[0]
        excerpt = self._extract_article_excerpt(
            article=article,
            content=top.content,
            max_chars=self._MAX_VERBATIM_CHARS,
        )
        if not excerpt:
            return None
        if response_language == "vi":
            answer = f"Điều {article} quy định:\n{excerpt}"
        else:
            answer = f"Article {article} provides:\n{excerpt}"
        return _Candidate(
            answer=answer,
            support_docs=[top],
            reason="article_content_match",
            answer_type="extractive_article",
        )

    @staticmethod
    def _definition_term(query: str) -> str | None:
        for pattern in (
            _DEFINITION_QUERY_PATTERN,
            _DEFINITION_QUERY_PATTERN_ASCII,
            _WHAT_IS_QUERY_PATTERN,
            _WHAT_IS_QUERY_PATTERN_ASCII,
        ):
            matched = pattern.search(query)
            if not matched:
                continue
            term = matched.group("term").strip(" .,:;!?")
            if len(term) < 2:
                continue
            return term
        return None

    def _match_definition_query(
        self,
        query: str,
        selected_context: list[RetrievalResult],
    ) -> _Candidate | None:
        lowered = self._normalized_text(query)
        if "la gi" not in lowered:
            return None
        term = self._definition_term(query)
        if term is None:
            return None
        term_tokens = self._tokens(term)
        if not term_tokens:
            return None

        top = selected_context[0]
        for sentence in self._split_sentences(top.content):
            sentence_tokens = self._tokens(sentence)
            if not sentence_tokens:
                continue
            overlap = len(term_tokens.intersection(sentence_tokens)) / max(
                1, len(term_tokens)
            )
            if overlap < 0.75:
                continue
            sentence_norm = self._normalized_text(sentence)
            if " la " not in f" {sentence_norm} " and ":" not in sentence:
                continue
            resolved = sentence[: self._MAX_SENTENCE_CHARS].strip()
            if not resolved:
                continue
            return _Candidate(
                answer=resolved,
                support_docs=[top],
                reason="definition_sentence_match",
                answer_type="extractive_definition",
            )
        return None

    def _match_numeric_query(
        self,
        query: str,
        selected_context: list[RetrievalResult],
    ) -> _Candidate | None:
        query_norm = self._normalized_text(query)
        if "bao nhieu" not in query_norm:
            return None
        keyword_hits = any(
            keyword in query_norm
            for keyword in (
                "muc phat",
                "so tien",
                "thoi han",
                "ngay",
                "thang",
                "nam",
            )
        )
        if not keyword_hits:
            return None

        top = selected_context[0]
        for sentence in self._split_sentences(top.content):
            if not _NUMBER_PATTERN.search(sentence):
                continue
            if len(sentence) > self._MAX_SENTENCE_CHARS:
                sentence = sentence[: self._MAX_SENTENCE_CHARS].strip()
            if not sentence:
                continue
            return _Candidate(
                answer=sentence,
                support_docs=[top],
                reason="numeric_span_match",
                answer_type="extractive_numeric",
            )
        return None

    def _match_clause_article_query(
        self,
        query: str,
        selected_context: list[RetrievalResult],
    ) -> _Candidate | None:
        clause_match = _CLAUSE_ARTICLE_PATTERN.search(
            query
        ) or _CLAUSE_ARTICLE_PATTERN_ASCII.search(query)
        if not clause_match:
            return None
        clause = clause_match.group("clause").upper()
        article = clause_match.group("article").upper()
        pattern = re.compile(
            rf"(?im)^.*(?:khoản|khoan)\s+{re.escape(clause)}\s+(?:điều|dieu)\s+{re.escape(article)}.*$"
        )
        top = selected_context[0]
        for line in [line.strip() for line in top.content.splitlines() if line.strip()]:
            if not pattern.match(line):
                continue
            return _Candidate(
                answer=line[: self._MAX_SENTENCE_CHARS],
                support_docs=[top],
                reason="clause_article_match",
                answer_type="extractive_clause_article",
            )
        return None

    def _match_verbatim_query(
        self,
        query: str,
        selected_context: list[RetrievalResult],
    ) -> _Candidate | None:
        query_norm = self._normalized_text(query)
        if "trich nguyen van" not in query_norm:
            return None

        clause_candidate = self._match_clause_article_query(query, selected_context)
        if clause_candidate is not None:
            return _Candidate(
                answer=clause_candidate.answer,
                support_docs=clause_candidate.support_docs,
                reason="verbatim_clause_article_match",
                answer_type="extractive_verbatim",
            )

        top = selected_context[0]
        excerpt = top.content.strip()
        if not excerpt:
            return None
        excerpt = excerpt[: self._MAX_VERBATIM_CHARS].strip()
        return _Candidate(
            answer=excerpt,
            support_docs=[top],
            reason="verbatim_excerpt_match",
            answer_type="extractive_verbatim",
        )

    def _is_high_confidence(
        self,
        *,
        query: str,
        top_doc: RetrievalResult,
        candidate: _Candidate,
    ) -> bool:
        if not self._score_confident_enough(top_doc):
            return False
        overlap = self._lexical_overlap_ratio(query, top_doc.content)
        if overlap >= self._MIN_OVERLAP_RATIO:
            return True
        if candidate.reason in {
            "title_article_match",
            "article_content_match",
            "clause_article_match",
            "verbatim_clause_article_match",
        }:
            # Numbered structural matches are trusted even when overlap ratio is low.
            return True
        return False

    @staticmethod
    def _fast_path_stop_reason(answer_type: str) -> str:
        return f"heuristic_{answer_type}"

    @staticmethod
    def _looks_simple_query(query: str) -> bool:
        lowered = query.strip().lower()
        if not lowered:
            return False
        if (
            _TITLE_QUERY_PATTERN.search(query)
            or _TITLE_QUERY_PATTERN_ASCII.search(query)
            or _ARTICLE_QUESTION_PATTERN.search(query)
            or _ARTICLE_QUESTION_PATTERN_ASCII.search(query)
            or _CLAUSE_ARTICLE_PATTERN.search(query)
            or _CLAUSE_ARTICLE_PATTERN_ASCII.search(query)
            or _DEFINITION_QUERY_PATTERN.search(query)
            or _DEFINITION_QUERY_PATTERN_ASCII.search(query)
            or _WHAT_IS_QUERY_PATTERN.search(query)
            or _WHAT_IS_QUERY_PATTERN_ASCII.search(query)
        ):
            return True
        normalized = ExtractiveAnswerer._normalized_text(query)
        if "trich nguyen van" in normalized:
            return True
        if "bao nhieu" in normalized:
            return True
        return False

    def answer(
        self,
        *,
        query: str,
        selected_context: list[RetrievalResult],
        response_language: str = "en",
    ) -> ExtractiveDecision:
        if not selected_context:
            return ExtractiveDecision(
                attempted=False,
                used=False,
                reason="no_context",
                answer=None,
            )
        top = selected_context[0]
        detectors = (
            lambda: self._match_title_query(
                query, selected_context, response_language=response_language
            ),
            lambda: self._match_article_rule_query(
                query, selected_context, response_language=response_language
            ),
            lambda: self._match_clause_article_query(query, selected_context),
            lambda: self._match_definition_query(query, selected_context),
            lambda: self._match_numeric_query(query, selected_context),
            lambda: self._match_verbatim_query(query, selected_context),
        )
        attempted = False
        latest_reason = "not_simple_query"
        for detector in detectors:
            candidate = detector()
            if candidate is None:
                continue
            attempted = True
            latest_reason = candidate.reason
            if not self._is_high_confidence(
                query=query,
                top_doc=top,
                candidate=candidate,
            ):
                latest_reason = f"{candidate.reason}:low_confidence"
                continue

            answer = GeneratedAnswer(
                answer=candidate.answer,
                citations=self.citation_builder.build(candidate.support_docs),
                confidence=0.95,
                status="answered",
                stop_reason=self._fast_path_stop_reason(candidate.answer_type),
                llm_fallback_used=False,
                llm_cache_hit=False,
                fast_path=True,
                fast_path_type="extractive",
            )
            return ExtractiveDecision(
                attempted=True,
                used=True,
                reason=candidate.reason,
                answer=answer,
            )

        return ExtractiveDecision(
            attempted=attempted or self._looks_simple_query(query),
            used=False,
            reason=latest_reason,
            answer=None,
        )
