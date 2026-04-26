"""Grounding and hallucination heuristic tests."""

from app.workflows.shared import assess_grounding


def test_vietnamese_paraphrase_not_falsely_flagged() -> None:
    assessment = assess_grounding(
        "Người ký thông tư là Bộ trưởng Nguyễn Văn A.",
        ["Thông tư này do Bộ trưởng Nguyễn Văn A ký ban hành."],
        citation_count=1,
        has_selected_context=True,
        status="answered",
    )

    assert assessment.hallucination_detected is False
    assert assessment.grounded_score >= 0.12


def test_unrelated_answer_is_flagged_even_with_context() -> None:
    assessment = assess_grounding(
        "Bitcoin đạt vốn hóa hơn 2 nghìn tỷ USD.",
        ["Điều 2 quy định trách nhiệm của đơn vị tiếp nhận hồ sơ."],
        citation_count=1,
        has_selected_context=True,
        status="answered",
    )

    assert assessment.hallucination_detected is True
    assert assessment.grounding_reason == "almost_no_overlap_even_with_citations"


def test_cited_answer_gets_positive_grounding_signal() -> None:
    assessment = assess_grounding(
        "Self-RAG dùng retrieval và critique để tăng độ bám tài liệu.",
        ["Self-RAG sử dụng retrieval, rerank và critique để tăng độ bám tài liệu."],
        citation_count=2,
        has_selected_context=True,
        status="answered",
    )

    assert assessment.hallucination_detected is False
    assert "citations" in assessment.grounding_reason


def test_zero_citation_generic_answer_gets_warning() -> None:
    assessment = assess_grounding(
        "Theo ngữ cảnh hiện có, cần thêm thông tin để trả lời chính xác.",
        ["Điều 3 mô tả thời hạn xử lý hồ sơ là 15 ngày làm việc."],
        citation_count=0,
        has_selected_context=True,
        status="answered",
    )

    assert assessment.hallucination_detected is True
    assert assessment.grounding_reason == "generic_answer_without_citations"
