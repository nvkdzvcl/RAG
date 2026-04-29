"""Grounding and hallucination heuristic tests."""

import app.workflows.shared.grounding as grounding_mod
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


def test_paraphrased_answer_gets_semantic_grounding_boost(monkeypatch) -> None:
    monkeypatch.setattr(
        grounding_mod, "grounded_overlap_score", lambda answer, context_chunks: 0.0
    )
    monkeypatch.setattr(
        grounding_mod,
        "_semantic_context_similarity",
        lambda answer, context_chunks: 0.86,
    )
    monkeypatch.setattr(grounding_mod, "_GROUNDING_SEMANTIC_WEIGHT", 0.35)
    monkeypatch.setattr(grounding_mod, "_GROUNDING_SEMANTIC_MIN_SIMILARITY", 0.58)

    assessment = assess_grounding(
        "Nội dung được phê chuẩn bởi cơ quan quản lý cấp bộ.",
        ["Thông tư này do Bộ trưởng Nguyễn Văn A ký ban hành."],
        citation_count=0,
        has_selected_context=True,
        status="answered",
    )

    assert assessment.hallucination_detected is False
    assert assessment.grounded_score >= 0.12
    assert assessment.grounding_reason == "strong_grounding_no_citations"


def test_grounding_falls_back_to_overlap_when_semantic_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        grounding_mod, "grounded_overlap_score", lambda answer, context_chunks: 0.13
    )
    monkeypatch.setattr(
        grounding_mod,
        "_semantic_context_similarity",
        lambda answer, context_chunks: None,
    )

    assessment = assess_grounding(
        "Self-RAG keeps answers grounded in retrieved context.",
        ["Self-RAG uses retrieved evidence and critique to stay grounded."],
        citation_count=0,
        has_selected_context=True,
        status="answered",
    )

    assert assessment.hallucination_detected is False
    assert assessment.grounded_score == 0.13
    assert assessment.grounding_reason == "strong_grounding_no_citations"


def test_standard_simple_policy_skips_semantic_encoder(monkeypatch) -> None:
    grounding_mod._clear_grounding_cache()
    monkeypatch.setattr(grounding_mod, "_GROUNDING_SEMANTIC_ENABLED", True)
    monkeypatch.setattr(grounding_mod, "_GROUNDING_SEMANTIC_STANDARD_ENABLED", True)

    load_calls = {"count": 0}

    def _record_loader():
        load_calls["count"] += 1
        return None

    monkeypatch.setattr(grounding_mod, "_load_semantic_encoder", _record_loader)

    evaluation = grounding_mod.assess_grounding_with_policy(
        "Tên của Điều 2 là: Quy định chung.",
        ["Điều 2. Quy định chung."],
        citation_count=1,
        has_selected_context=True,
        status="answered",
        policy=grounding_mod.GroundingPolicy(
            mode="standard",
            query_complexity="simple",
            generated_status="answered",
            answer_length=36,
            citation_count=1,
            retrieval_confidence=0.8,
            fast_path_used=False,
        ),
    )

    assert evaluation.grounding_semantic_used is False
    assert load_calls["count"] == 0


def test_standard_normal_risky_can_use_semantic_when_adaptive(monkeypatch) -> None:
    grounding_mod._clear_grounding_cache()
    monkeypatch.setattr(grounding_mod, "_GROUNDING_SEMANTIC_ENABLED", True)
    monkeypatch.setattr(grounding_mod, "_GROUNDING_SEMANTIC_STANDARD_ENABLED", True)
    monkeypatch.setattr(
        grounding_mod, "grounded_overlap_score", lambda answer, context_chunks: 0.01
    )

    semantic_calls = {"count": 0}

    def _semantic_score(answer: str, context_chunks: list[str]) -> float:
        _ = answer
        _ = context_chunks
        semantic_calls["count"] += 1
        return 0.83

    monkeypatch.setattr(grounding_mod, "_semantic_context_similarity", _semantic_score)
    evaluation = grounding_mod.assess_grounding_with_policy(
        "Câu trả lời dài nhưng thiếu trích dẫn cụ thể.",
        ["Ngữ cảnh có bằng chứng nhưng độ khớp lexical thấp."],
        citation_count=0,
        has_selected_context=True,
        status="answered",
        policy=grounding_mod.GroundingPolicy(
            mode="standard",
            query_complexity="normal",
            generated_status="answered",
            answer_length=420,
            citation_count=0,
            retrieval_confidence=0.19,
            fast_path_used=False,
        ),
    )

    assert semantic_calls["count"] == 1
    assert evaluation.grounding_semantic_used is True


def test_advanced_policy_can_use_semantic_grounding(monkeypatch) -> None:
    grounding_mod._clear_grounding_cache()
    monkeypatch.setattr(grounding_mod, "_GROUNDING_SEMANTIC_ENABLED", True)
    monkeypatch.setattr(grounding_mod, "_GROUNDING_SEMANTIC_ADVANCED_ENABLED", True)
    monkeypatch.setattr(
        grounding_mod, "grounded_overlap_score", lambda answer, context_chunks: 0.0
    )

    semantic_calls = {"count": 0}

    def _semantic_score(answer: str, context_chunks: list[str]) -> float:
        _ = answer
        _ = context_chunks
        semantic_calls["count"] += 1
        return 0.88

    monkeypatch.setattr(grounding_mod, "_semantic_context_similarity", _semantic_score)

    evaluation = grounding_mod.assess_grounding_with_policy(
        "Nội dung này được giải thích theo cách diễn giải.",
        ["Điều khoản gốc mô tả cùng ý nghĩa nhưng khác từ ngữ."],
        citation_count=0,
        has_selected_context=True,
        status="answered",
        policy=grounding_mod.GroundingPolicy(
            mode="advanced",
            query_complexity="normal",
            generated_status="answered",
            answer_length=240,
            citation_count=0,
            retrieval_confidence=0.21,
            fast_path_used=False,
        ),
    )

    assert semantic_calls["count"] == 1
    assert evaluation.grounding_semantic_used is True
    assert evaluation.assessment.grounded_score >= 0.12


def test_grounding_cache_hit_avoids_semantic_recompute(monkeypatch) -> None:
    grounding_mod._clear_grounding_cache()
    monkeypatch.setattr(grounding_mod, "_GROUNDING_SEMANTIC_ENABLED", True)
    monkeypatch.setattr(grounding_mod, "_GROUNDING_SEMANTIC_ADVANCED_ENABLED", True)
    monkeypatch.setattr(
        grounding_mod, "grounded_overlap_score", lambda answer, context_chunks: 0.03
    )

    semantic_calls = {"count": 0}

    def _semantic_score(answer: str, context_chunks: list[str]) -> float:
        _ = answer
        _ = context_chunks
        semantic_calls["count"] += 1
        return 0.84

    monkeypatch.setattr(grounding_mod, "_semantic_context_similarity", _semantic_score)

    policy = grounding_mod.GroundingPolicy(
        mode="advanced",
        query_complexity="complex",
        generated_status="answered",
        answer_length=420,
        citation_count=0,
        retrieval_confidence=0.18,
        fast_path_used=False,
    )
    kwargs = {
        "citation_count": 0,
        "has_selected_context": True,
        "status": "answered",
        "policy": policy,
    }

    first = grounding_mod.assess_grounding_with_policy(
        "Phần trả lời đã được diễn giải dài hơn từ cùng bằng chứng.",
        ["Bằng chứng nguồn mô tả cùng nội dung theo cách khác."],
        **kwargs,
    )
    second = grounding_mod.assess_grounding_with_policy(
        "Phần trả lời đã được diễn giải dài hơn từ cùng bằng chứng.",
        ["Bằng chứng nguồn mô tả cùng nội dung theo cách khác."],
        **kwargs,
    )

    assert first.grounding_cache_hit is False
    assert second.grounding_cache_hit is True
    assert semantic_calls["count"] == 1
