# ARCHITECTURE.md

## Tổng Quan

Dự án này là một hệ thống **RAG mã nguồn mở với 3 chế độ** theo kiến trúc module:

- **Standard Mode**: RAG nền tảng
- **Advanced Mode**: Self-RAG Level 2 theo hướng thực dụng
- **Compare Mode**: chạy song song standard và advanced để so sánh

Kiến trúc tách biệt hạ tầng dùng chung khỏi lớp điều phối workflow.

## Các Tầng Kiến Trúc Cấp Cao

1. ingestion
2. indexing
3. retrieval
4. generation
5. critique (advanced)
6. workflows (standard/advanced/compare)
7. API
8. frontend
9. evaluation

## Thành Phần Dùng Chung vs Theo Chế Độ

Dùng chung:

- schemas
- loaders/chunking
- indexes/retrieval/reranking
- contract generation và citations
- config/logging
- công cụ evaluation

Theo từng mode:

- luồng điều khiển
- hành vi retry/abstain
- logic tổng hợp so sánh

## Hợp Đồng Dữ Liệu Compare Mode

Response của compare mode trả về:

- nhánh `standard`
- nhánh `advanced`
- tóm tắt `comparison`

Cách này tránh triển khai trùng lặp bằng cách tái sử dụng workflow standard và advanced hiện có.

## Trạng Thái Triển Khai Hiện Tại

Các lớp MVP end-to-end đã có:

- scaffold dự án, config, logging có cấu trúc
- schema typed cho ingestion/retrieval/generation/workflow/API
- ingestion layer (loader text/markdown, cleaner, chunker, giữ metadata)
- indexing layer (interface embeddings, vector index abstraction, BM25 index, local persistence)
- retrieval layer (dense, sparse, hybrid fusion, hook reranker, context selector)
- generation baseline (LLM client abstraction, output có cấu trúc, citations, xử lý thiếu bằng chứng)
- workflows:
  - standard (`retrieve -> rerank -> select -> generate -> cite`)
  - advanced (retrieval gate, rewrite, critique, retry/refine/abstain với vòng lặp có giới hạn)
  - compare (chạy standard + advanced và tổng hợp summary)
- backend API (`/api/v1/health`, `/api/v1/query`)
- frontend tích hợp chọn mode và panel hiển thị theo mode
- dataset evaluation + runner + regression checks cho payload standard/advanced/compare

Ghi chú backend embedding:

- mặc định dense embeddings dùng sentence-transformers (`intfloat/multilingual-e5-base`) với tiền tố kiểu E5 (`passage:` cho chunk, `query:` cho truy vấn)
- chọn provider runtime theo config
- nếu import/load model sentence-transformers lỗi, hệ thống tự fallback về hash embedding có tính xác định và không làm crash API

Ghi chú backend reranker:

- mặc định rerank bằng cross-encoder (`BAAI/bge-reranker-v2-m3`) trên top candidate sau retrieval
- điểm reranker được gắn vào output (`rerank_score`), đồng thời giữ lại điểm retrieval ban đầu trong metadata
- nếu khởi tạo cross-encoder lỗi, workflow fallback về score-only reranking và vẫn giữ API/query hoạt động

Khoảng trống MVP đã biết:

- retrieval/generation hiện là baseline có tính xác định, ưu tiên cho local dev và testing
- prompt files đã có nhưng một số module business logic vẫn dùng heuristic nhẹ
- hardening production (auth, rate limiting, streaming response, triển khai vận hành) chủ động để ngoài phạm vi MVP
