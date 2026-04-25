"""Generation baseline tests."""

from app.generation import BaselineGenerator, CitationBuilder, StubLLMClient, StructuredOutputParser
from app.schemas.common import Mode
from app.schemas.retrieval import RetrievalResult


def _sample_context() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="chunk_001",
            doc_id="doc_001",
            source="docs/a.md",
            title="Doc A",
            section="Intro",
            content="Self-RAG uses retrieval and critique to improve grounding.",
            score=0.8,
            score_type="hybrid",
        ),
        RetrievalResult(
            chunk_id="chunk_002",
            doc_id="doc_002",
            source="docs/b.md",
            title="Doc B",
            section="Methods",
            page=2,
            content="Citations should reference source chunks used by generation.",
            score=0.7,
            score_type="hybrid",
        ),
    ]


def test_citation_formatting() -> None:
    builder = CitationBuilder()
    citations = builder.build(_sample_context())
    lines = builder.format_citations(citations)

    assert len(citations) == 2
    assert lines[0].startswith("[1]")
    assert "chunk_id=chunk_001" in lines[0]
    assert "source=docs/a.md" in lines[0]


def test_structured_output_parsing() -> None:
    parser = StructuredOutputParser()
    raw = (
        "```json\n"
        '{"answer":"Grounded response.","confidence":0.82,"status":"answered"}'
        "\n```"
    )

    parsed = parser.parse(raw)

    assert parsed.answer == "Grounded response."
    assert parsed.confidence == 0.82
    assert parsed.status == "answered"


def test_structured_output_parsing_extracts_nested_answer_json() -> None:
    parser = StructuredOutputParser()
    raw = (
        '{"answer":"{\\"answer\\":\\"Noi dung tra loi\\",\\"confidence\\":0.6,\\"status\\":\\"answered\\"}",'
        '"confidence":0.9,"status":"answered"}'
    )

    parsed = parser.parse(raw)

    assert parsed.answer == "Noi dung tra loi"
    assert parsed.confidence == 0.9
    assert parsed.status == "answered"


def test_insufficient_evidence_path() -> None:
    llm = StubLLMClient(
        responder=lambda prompt, system: '{"answer":"","confidence":0.1,"status":"insufficient_evidence"}'
    )
    generator = BaselineGenerator(llm_client=llm)

    result = generator.generate_answer(
        query="What is the unsupported claim?",
        context=_sample_context(),
        mode=Mode.STANDARD,
    )

    assert result.status == "insufficient_evidence"
    assert result.citations == []
    assert result.stop_reason == "model_insufficient_evidence"


def test_title_question_returns_exact_article_title_from_context() -> None:
    llm = StubLLMClient(
        responder=lambda prompt, system, model=None: (
            '{"answer":"Dieu 2 giai thich tu ngu trong luat.","confidence":0.72,"status":"answered"}'
        )
    )
    generator = BaselineGenerator(llm_client=llm)
    context = [
        RetrievalResult(
            chunk_id="chunk_title_002",
            doc_id="doc_title_002",
            source="docs/law.md",
            title="Luật mẫu",
            content="Điều 2. Giải thích từ ngữ\nCác thuật ngữ được hiểu như sau...",
            score=0.95,
            score_type="hybrid",
            rank=1,
        )
    ]

    result = generator.generate_answer(
        query="tên của điều 2 là gì",
        context=context,
        mode=Mode.STANDARD,
        response_language="vi",
    )

    assert result.answer == "Tên của Điều 2 là: Giải thích từ ngữ."
    assert result.status == "answered"
    assert result.stop_reason == "heuristic_exact_title"
    assert result.citations
    assert result.citations[0].chunk_id == "chunk_title_002"
