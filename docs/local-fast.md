# local-fast.md

## Mục tiêu

Profile này tối ưu cho chạy local với Ollama `qwen2.5:3b`, ưu tiên độ trễ thấp và trải nghiệm phản hồi nhanh.

## Quick Start (Fast Profile)

1. Copy cấu hình fast:
   `cp .env.fast.example .env`
2. Kéo model và chạy Ollama:
   - `ollama pull qwen2.5:3b`
   - `ollama serve`
3. Chạy backend:
   - dev loop: `uvicorn app.main:app --reload`
   - benchmark latency: `uvicorn app.main:app --host 127.0.0.1 --port 8000`

## Fast Mode vs Strict Mode

### Fast mode (khuyến nghị local)

- `LLM_MODEL=qwen2.5:3b`
- `LLM_MAX_TOKENS=512` hoặc `768`
- `RAG_DYNAMIC_BUDGET_ENABLED=true`
- `GROUNDING_POLICY=adaptive`
- `GROUNDING_SEMANTIC_STANDARD_ENABLED=false`
- `RERANK_CASCADE_ENABLED=true`
- `RERANK_SIMPLE_SKIP_CROSS_ENCODER=true`
- `ADVANCED_ADAPTIVE_ENABLED=true`
- `COMPARE_PARALLEL_ENABLED=false`

Kỳ vọng: độ trễ thấp hơn, chi phí CPU/GPU thấp hơn, kiểm chứng ngữ nghĩa ở standard bớt nghiêm ngặt hơn strict mode.

### Strict mode (độ tin cậy ưu tiên hơn tốc độ)

Ví dụ cấu hình:

- `GROUNDING_POLICY=strict`
- `GROUNDING_SEMANTIC_STANDARD_ENABLED=true`
- `RERANK_CASCADE_ENABLED=false` (hoặc giữ `true` nhưng giảm skip)
- `RERANK_SIMPLE_SKIP_CROSS_ENCODER=false`
- `ADVANCED_FORCE_LLM_GATE=true`
- `ADVANCED_FORCE_LLM_CRITIC=true`

Kỳ vọng: kiểm tra chặt hơn, nhưng độ trễ và số LLM call tăng.

## Standard vs Advanced trong Local Fast

- `standard`: nhanh nhất, phù hợp truy vấn factual/simple, baseline cho benchmark.
- `advanced`: chậm hơn do gate/critique/refine, nhưng an toàn hơn khi câu hỏi mơ hồ/xung đột bằng chứng.

## Khi nào semantic grounding bị skip

Theo policy grounding hiện tại:

- Global tắt semantic (`GROUNDING_SEMANTIC_ENABLED=false`), hoặc policy lexical-only.
- Fast-path extractive được dùng.
- `standard` query complexity là `simple_extractive`.
- `standard` đang bật `GROUNDING_SEMANTIC_STANDARD_ENABLED=false`.
- `standard normal` có tín hiệu lexical đủ tốt (không rơi vào nhóm risky: thiếu citation, answer dài, retrieval confidence thấp, lexical score thấp).

Khi semantic bị skip, hệ thống dùng lexical overlap để giảm latency.

## Khi nào cross-encoder reranker bị skip

Theo `rerank_policy` hiện tại, cross-encoder có thể bị bỏ qua khi:

- query thuộc `simple_extractive` và bật `RERANK_SIMPLE_SKIP_CROSS_ENCODER=true`.
- số candidate ít (`few_candidates_skip`).
- top score cao và chênh lệch rõ (`high_confidence_clear_gap_skip`).
- mode standard/compare không cần cross-encoder theo policy.

`advanced` vẫn có đường chất lượng để dùng cross-encoder khi tín hiệu mơ hồ/rủi ro.

## Benchmark before/after optimization

Chạy backend không `--reload` rồi đo:

- non-stream:
  `python scripts/benchmark_latency.py --api-base-url http://127.0.0.1:8000/api/v1 --mode compare --runs 5 --warmup 1 --concurrency 1`
- stream:
  `python scripts/benchmark_latency.py --api-base-url http://127.0.0.1:8000/api/v1 --mode compare --stream --runs 5 --warmup 1 --concurrency 1`

Lưu kết quả:

- before:
  `python scripts/benchmark_latency.py --mode compare --runs 5 --output-json data/eval/results/latency_before.json`
- after:
  `python scripts/benchmark_latency.py --mode compare --runs 5 --output-json data/eval/results/latency_after.json`

Các số chính cần so:

- `client_latency_ms` (p50/p95)
- `backend_latency_ms` (nếu response có)
- stream: `time_to_first_event_ms`, `time_to_first_token_ms`, `total_stream_ms`
