"""Tests for deterministic query complexity and dynamic budget policy."""

from app.workflows.query_budget import classify_query_complexity, choose_query_budget


def test_query_complexity_simple_extractive_patterns() -> None:
    assert classify_query_complexity("Tên của Điều 3 là gì?") == "simple_extractive"
    assert classify_query_complexity("What is the definition of retrieval?") == (
        "simple_extractive"
    )
    assert classify_query_complexity("Hạn chót nộp hồ sơ là ngày nào?") == (
        "simple_extractive"
    )
    assert classify_query_complexity("Trích nguyên văn Clause 2.") == (
        "simple_extractive"
    )


def test_query_complexity_complex_patterns() -> None:
    assert classify_query_complexity("Compare standard mode and advanced mode.") == (
        "complex"
    )
    assert classify_query_complexity("Phân tích ưu nhược điểm của hai phương án.") == (
        "complex"
    )


def test_query_complexity_defaults_to_normal() -> None:
    assert classify_query_complexity("How does retrieval work in this system?") == (
        "normal"
    )


def test_choose_query_budget_dynamic_simple() -> None:
    budget = choose_query_budget(
        "What is the title of Article 2?",
        dynamic_enabled=True,
        base_hybrid_top_k=8,
        base_rerank_top_k=6,
        base_context_top_k=4,
        base_context_max_chars=4000,
        base_llm_max_tokens=2048,
        simple_max_tokens=384,
        normal_max_tokens=768,
        complex_max_tokens=1536,
        simple_context_chars=1600,
        normal_context_chars=3000,
    )

    assert budget.complexity == "simple_extractive"
    assert budget.hybrid_top_k == 3
    assert budget.rerank_top_k == 3
    assert budget.context_top_k == 2
    assert budget.context_max_chars == 1600
    assert budget.max_tokens == 384


def test_choose_query_budget_dynamic_normal_and_complex() -> None:
    normal_budget = choose_query_budget(
        "How does retrieval work in this system?",
        dynamic_enabled=True,
        base_hybrid_top_k=8,
        base_rerank_top_k=6,
        base_context_top_k=4,
        base_context_max_chars=4000,
        base_llm_max_tokens=2048,
        simple_max_tokens=384,
        normal_max_tokens=768,
        complex_max_tokens=1536,
        simple_context_chars=1600,
        normal_context_chars=3000,
    )
    complex_budget = choose_query_budget(
        "Compare and synthesize pros and cons across both approaches.",
        dynamic_enabled=True,
        base_hybrid_top_k=8,
        base_rerank_top_k=6,
        base_context_top_k=4,
        base_context_max_chars=4000,
        base_llm_max_tokens=2048,
        simple_max_tokens=384,
        normal_max_tokens=768,
        complex_max_tokens=1536,
        simple_context_chars=1600,
        normal_context_chars=3000,
    )

    assert normal_budget.complexity == "normal"
    assert normal_budget.hybrid_top_k == 5
    assert normal_budget.rerank_top_k == 5
    assert normal_budget.context_top_k == 3
    assert normal_budget.context_max_chars == 3000
    assert normal_budget.max_tokens == 768

    assert complex_budget.complexity == "complex"
    assert complex_budget.hybrid_top_k == 8
    assert complex_budget.rerank_top_k == 6
    assert complex_budget.context_top_k == 4
    assert complex_budget.context_max_chars == 4000
    assert complex_budget.max_tokens == 1536


def test_choose_query_budget_keeps_retrieval_when_locked() -> None:
    budget = choose_query_budget(
        "Tên của Điều 2 là gì?",
        dynamic_enabled=True,
        base_hybrid_top_k=4,
        base_rerank_top_k=4,
        base_context_top_k=4,
        base_context_max_chars=4000,
        base_llm_max_tokens=2048,
        simple_max_tokens=384,
        normal_max_tokens=768,
        complex_max_tokens=1536,
        simple_context_chars=1600,
        normal_context_chars=3000,
        retrieval_top_k_locked=True,
    )

    assert budget.complexity == "simple_extractive"
    assert budget.hybrid_top_k == 4
    assert budget.rerank_top_k == 4
    assert budget.context_top_k == 2


def test_choose_query_budget_disabled_keeps_legacy_budget() -> None:
    budget = choose_query_budget(
        "Tên của Điều 2 là gì?",
        dynamic_enabled=False,
        base_hybrid_top_k=8,
        base_rerank_top_k=6,
        base_context_top_k=4,
        base_context_max_chars=4000,
        base_llm_max_tokens=2048,
        simple_max_tokens=384,
        normal_max_tokens=768,
        complex_max_tokens=1536,
        simple_context_chars=1600,
        normal_context_chars=3000,
    )

    assert budget.dynamic_enabled is False
    assert budget.hybrid_top_k == 8
    assert budget.rerank_top_k == 6
    assert budget.context_top_k == 4
    assert budget.context_max_chars == 4000
    assert budget.max_tokens == 2048
