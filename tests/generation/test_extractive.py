"""Tests for deterministic extractive fast-path answers."""

from app.generation.extractive import ExtractiveAnswerer
from app.schemas.retrieval import RetrievalResult


def _doc(content: str, *, score: float = 0.9) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_001",
        doc_id="doc_001",
        source="docs/law.md",
        title="Luật mẫu",
        content=content,
        score=score,
        score_type="hybrid",
        rank=1,
    )


def test_extractive_title_question() -> None:
    answerer = ExtractiveAnswerer()
    decision = answerer.answer(
        query="Tên Điều 2 là gì?",
        selected_context=[
            _doc(
                "Điều 2. Giải thích từ ngữ\nCác thuật ngữ trong luật được hiểu như sau."
            )
        ],
        response_language="vi",
    )

    assert decision.attempted is True
    assert decision.used is True
    assert decision.answer is not None
    assert decision.answer.fast_path is True
    assert decision.answer.fast_path_type == "extractive"
    assert "Tên của Điều 2 là: Giải thích từ ngữ." in decision.answer.answer


def test_extractive_article_rule_question() -> None:
    answerer = ExtractiveAnswerer()
    decision = answerer.answer(
        query="Điều 5 quy định gì?",
        selected_context=[
            _doc(
                "Điều 5. Nguyên tắc xử lý\n"
                "Tổ chức, cá nhân phải tuân thủ quy định pháp luật hiện hành.\n"
                "Việc xử lý phải bảo đảm khách quan, công khai và đúng thẩm quyền.\n"
                "Điều 6. Trách nhiệm thi hành"
            )
        ],
        response_language="vi",
    )

    assert decision.attempted is True
    assert decision.used is True
    assert decision.answer is not None
    assert "Điều 5 quy định:" in decision.answer.answer
    assert "khách quan" in decision.answer.answer


def test_extractive_definition_question() -> None:
    answerer = ExtractiveAnswerer()
    decision = answerer.answer(
        query="Định nghĩa dữ liệu cá nhân là gì?",
        selected_context=[
            _doc(
                "Dữ liệu cá nhân là thông tin dưới dạng ký hiệu, chữ viết, số, hình ảnh hoặc âm thanh."
            )
        ],
        response_language="vi",
    )

    assert decision.attempted is True
    assert decision.used is True
    assert decision.answer is not None
    assert "Dữ liệu cá nhân là thông tin" in decision.answer.answer


def test_extractive_numeric_question() -> None:
    answerer = ExtractiveAnswerer()
    decision = answerer.answer(
        query="Mức phạt tối đa là bao nhiêu?",
        selected_context=[
            _doc("Mức phạt tiền tối đa đối với hành vi này là 50.000.000 đồng.")
        ],
        response_language="vi",
    )

    assert decision.attempted is True
    assert decision.used is True
    assert decision.answer is not None
    assert "50.000.000 đồng" in decision.answer.answer


def test_extractive_verbatim_clause_article_question() -> None:
    answerer = ExtractiveAnswerer()
    decision = answerer.answer(
        query="Trích nguyên văn Khoản 2 Điều 10",
        selected_context=[
            _doc(
                "Khoản 2 Điều 10: Người thực hiện nghĩa vụ phải nộp hồ sơ trong thời hạn 15 ngày làm việc."
            )
        ],
        response_language="vi",
    )

    assert decision.attempted is True
    assert decision.used is True
    assert decision.answer is not None
    assert "Khoản 2 Điều 10" in decision.answer.answer
    assert "15 ngày" in decision.answer.answer


def test_extractive_returns_none_when_confidence_is_low() -> None:
    answerer = ExtractiveAnswerer()
    decision = answerer.answer(
        query="Điều 99 quy định gì?",
        selected_context=[_doc("Điều 5. Nguyên tắc xử lý và tổ chức thi hành.")],
        response_language="vi",
    )

    assert decision.attempted is True
    assert decision.used is False
    assert decision.answer is None
