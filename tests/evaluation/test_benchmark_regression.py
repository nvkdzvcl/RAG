"""Lightweight retrieval benchmark regression test."""

import json
from pathlib import Path
from typing import Any

from app.evaluation.schemas import RetrievedSourceTrace
from app.evaluation.metrics import compute_retrieval_metrics
from app.indexing.bm25_index import BM25Index
from app.indexing.vector_index import InMemoryVectorIndex
from app.indexing.providers.hash_embedding import HashEmbeddingProvider
from app.retrieval.dense import DenseRetriever
from app.retrieval.sparse import SparseRetriever
from app.retrieval.hybrid import HybridRetriever
from app.schemas.ingestion import DocumentChunk


def run_benchmark() -> dict[str, Any]:
    dataset_path = Path("tests/fixtures/eval/benchmark.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    corpus = [DocumentChunk(**item) for item in data["corpus"]]

    provider = HashEmbeddingProvider()

    try:
        vectors = provider.embed_documents([chunk.content for chunk in corpus])
    except AttributeError:
        vectors = [provider.embed_query(chunk.content) for chunk in corpus]

    vector_index = InMemoryVectorIndex()
    bm25_index = BM25Index()

    vector_index.build(corpus, vectors)
    bm25_index.build(corpus)

    dense_retriever = DenseRetriever(
        vector_index=vector_index, embedding_provider=provider
    )
    sparse_retriever = SparseRetriever(bm25_index=bm25_index)

    retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
    )

    results = []
    total_hit = 0
    total_mrr = 0.0
    total_ndcg = 0.0

    for item in queries:
        query = item["query"]
        gold = item["gold_sources"]

        scored_chunks = retriever.retrieve(query, top_k=3)
        retrieved_ids = []
        retrieved_sources = []
        for sc in scored_chunks:
            retrieved_ids.append(sc.chunk.chunk_id)
            retrieved_sources.append(
                RetrievedSourceTrace(
                    chunk_id=sc.chunk.chunk_id,
                    doc_id=sc.chunk.doc_id,
                    source=sc.chunk.source,
                    title=sc.chunk.title,
                    section=sc.chunk.section,
                )
            )

        hit, mrr, ndcg = compute_retrieval_metrics(
            retrieved_chunk_ids=retrieved_ids,
            gold_sources=gold,
            retrieved_sources=retrieved_sources,
        )

        if hit:
            total_hit += 1
        total_mrr += mrr
        total_ndcg += ndcg

        results.append(
            {
                "query": query,
                "gold_sources": gold,
                "retrieved_ids": retrieved_ids,
                "hit": hit,
                "mrr": mrr,
                "ndcg": ndcg,
            }
        )

    n = len(queries)
    metrics = {
        "hit_rate": total_hit / n,
        "avg_mrr": total_mrr / n,
        "avg_ndcg": total_ndcg / n,
    }

    return {"metrics": metrics, "runs": results}


def test_retrieval_benchmark_regression() -> None:
    report = run_benchmark()
    metrics = report["metrics"]

    # Conservative thresholds for deterministic BM25+Hash retrieval
    assert metrics["hit_rate"] >= 0.70
    assert metrics["avg_mrr"] >= 0.50
    assert metrics["avg_ndcg"] >= 0.50


if __name__ == "__main__":
    report_out = run_benchmark()
    out_file = Path("data/eval/results/benchmark_regression.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_out, f, indent=2)
    print(f"Benchmark completed successfully. Results saved to {out_file}")
    for key, value in report_out["metrics"].items():
        print(f"{key}: {value:.2f}")
