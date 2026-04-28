import math

from app.evaluation.metrics import compute_retrieval_metrics, extract_trace_fields
from app.evaluation.schemas import RetrievedSourceTrace


def test_compute_retrieval_metrics_no_results() -> None:
    hit, mrr, ndcg = compute_retrieval_metrics([], ["source1"])
    assert hit is False
    assert mrr == 0.0
    assert ndcg == 0.0


def test_compute_retrieval_metrics_no_expecteds() -> None:
    hit, mrr, ndcg = compute_retrieval_metrics(["doc1_chunk_001"], [])
    assert hit is False
    assert mrr == 0.0
    assert ndcg == 0.0


def test_compute_retrieval_metrics_no_relevant_chunks() -> None:
    hit, mrr, ndcg = compute_retrieval_metrics(
        ["wrong1_chunk_001", "wrong2_chunk_001"], ["correct_source"]
    )
    assert hit is False
    assert mrr == 0.0
    assert ndcg == 0.0


def test_compute_retrieval_metrics_first_result_relevant() -> None:
    hit, mrr, ndcg = compute_retrieval_metrics(
        ["correctdoc_chunk_0001", "wrong_chunk_0001"], ["correctdoc"]
    )
    assert hit is True
    assert math.isclose(mrr, 1.0)
    assert math.isclose(ndcg, 1.0)


def test_compute_retrieval_metrics_later_result_relevant() -> None:
    hit, mrr, ndcg = compute_retrieval_metrics(
        ["wrong_chunk_001", "wrong_chunk_002", "targetdoc_chunk_001"], ["targetdoc"]
    )
    assert hit is True
    assert math.isclose(mrr, 1.0 / 3)
    # DCG = 1/log2(4) = 0.5. IDCG = 1.0
    assert math.isclose(ndcg, 0.5)


def test_compute_retrieval_metrics_ndcg_reward_order() -> None:
    hit1, mrr1, ndcg1 = compute_retrieval_metrics(
        ["a_chunk_01", "b_chunk_01"], ["a", "b"]
    )  # Best possible order

    hit2, mrr2, ndcg2 = compute_retrieval_metrics(
        ["wrong_chunk_01", "b_chunk_01"], ["a", "b"]
    )  # Worse order (missing one, rank 2)

    assert hit1 is True and hit2 is True
    assert math.isclose(mrr1, 1.0)
    assert math.isclose(mrr2, 0.5)
    assert ndcg1 > ndcg2
    assert math.isclose(ndcg1, 1.0)


def test_compute_retrieval_metrics_ndcg_no_duplicate_reward() -> None:
    hit, mrr, ndcg = compute_retrieval_metrics(
        ["doc_chunk_001", "doc_chunk_002", "doc_chunk_003"], ["doc"]
    )
    # nDCG should cap at 1.0 instead of accumulating 1/log2(i+2) infinitely
    assert hit is True
    assert math.isclose(mrr, 1.0)
    assert math.isclose(ndcg, 1.0)


def test_compute_retrieval_metrics_matches_chunk_id_when_structured() -> None:
    hit, mrr, ndcg = compute_retrieval_metrics(
        ["docalpha_chunk_0001_hash"],
        ["chunk_id=docalpha_chunk_0001_hash"],
        [
            RetrievedSourceTrace(
                chunk_id="docalpha_chunk_0001_hash",
                doc_id="docalpha",
                source="docs/alpha.md",
            )
        ],
    )
    assert hit is True
    assert math.isclose(mrr, 1.0)
    assert math.isclose(ndcg, 1.0)


def test_compute_retrieval_metrics_matches_doc_id_when_structured() -> None:
    hit, mrr, ndcg = compute_retrieval_metrics(
        ["docbeta_chunk_0007_hash"],
        ["doc_id=docbeta"],
        [
            RetrievedSourceTrace(
                chunk_id="docbeta_chunk_0007_hash",
                doc_id="docbeta",
                source="docs/beta.md",
            )
        ],
    )
    assert hit is True
    assert math.isclose(mrr, 1.0)
    assert math.isclose(ndcg, 1.0)


def test_compute_retrieval_metrics_matches_source_or_path_when_structured() -> None:
    hit, mrr, ndcg = compute_retrieval_metrics(
        ["docmodes_chunk_0003_hash"],
        ["path=docs/MODES.md"],
        [
            RetrievedSourceTrace(
                chunk_id="docmodes_chunk_0003_hash",
                doc_id="docmodes",
                source="docs/MODES.md",
            )
        ],
    )
    assert hit is True
    assert math.isclose(mrr, 1.0)
    assert math.isclose(ndcg, 1.0)


def test_compute_retrieval_metrics_matches_section_when_structured() -> None:
    hit, mrr, ndcg = compute_retrieval_metrics(
        ["docmodes_chunk_0004_hash"],
        ["section=Advanced Mode"],
        [
            RetrievedSourceTrace(
                chunk_id="docmodes_chunk_0004_hash",
                doc_id="docmodes",
                source="docs/MODES.md",
                section="advanced mode",
            )
        ],
    )
    assert hit is True
    assert math.isclose(mrr, 1.0)
    assert math.isclose(ndcg, 1.0)


def test_compute_retrieval_metrics_keeps_legacy_fallback_string_match() -> None:
    hit, mrr, ndcg = compute_retrieval_metrics(
        ["docmodes_chunk_0005_abcd"],
        ["modes.md#docmodes_chunk_0005"],
        [
            RetrievedSourceTrace(
                chunk_id="docmodes_chunk_0005_abcd",
                doc_id="docmodes",
                source="docs/MODES.md",
            )
        ],
    )
    assert hit is True
    assert math.isclose(mrr, 1.0)
    assert math.isclose(ndcg, 1.0)


def test_extract_trace_fields_collects_retrieved_source_metadata() -> None:
    trace = [
        {
            "step": "retrieve",
            "count": 1,
            "chunk_ids": ["doc1_chunk_0001"],
            "docs": [
                {
                    "chunk_id": "doc1_chunk_0001",
                    "doc_id": "doc1",
                    "source": "docs/guide.md",
                    "title": "Guide",
                    "section": "Intro",
                }
            ],
        }
    ]
    extracted = extract_trace_fields(trace)
    assert extracted.retrieved_chunk_ids == ["doc1_chunk_0001"]
    assert len(extracted.retrieved_sources) == 1
    first = extracted.retrieved_sources[0]
    assert first.chunk_id == "doc1_chunk_0001"
    assert first.doc_id == "doc1"
    assert first.source == "docs/guide.md"
    assert first.title == "Guide"
    assert first.section == "Intro"
