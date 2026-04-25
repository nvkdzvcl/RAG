# RUNBOOK.md

## Thiết Lập Backend Local

1. Tạo và kích hoạt môi trường ảo:
   `python3 -m venv .venv && source .venv/bin/activate`
2. Cài dependencies:
   `pip install -r requirements-dev.txt`
3. Sao chép file mẫu biến môi trường:
   `cp .env.example .env`
4. Khởi chạy backend:
   `uvicorn app.main:app --reload`

## Thiết Lập Frontend Local

1. Vào thư mục frontend:
   `cd frontend`
2. Cài dependencies:
   `npm install`
3. Sao chép file env mẫu cho frontend:
   `cp .env.example .env`
4. Chạy dev server:
   `npm run dev`

## Quy Trình Dev Thường Dùng

1. Upload một hoặc nhiều tài liệu bằng `POST /api/v1/documents/upload`
2. Kiểm tra trạng thái xử lý qua:
   - `GET /api/v1/documents`
   - `GET /api/v1/documents/{document_id}/status`
3. Chạy truy vấn trên cả 3 mode (`standard`, `advanced`, `compare`)
4. Chạy evaluation:
   `python scripts/run_eval.py --dataset data/eval/golden_dataset.jsonl --modes standard advanced compare`
5. Chạy test:
   - vòng lặp local nhanh: `pytest -m "not slow and not e2e"` (hoặc `make test-fast`)
   - chỉ integration: `make test-integration`
   - kiểm tra đầy đủ: `pytest` (hoặc `make test-full`)

## Cấu Hình Embedding

Cấu hình embedding đa ngôn ngữ/Vietnamese khuyến nghị:

- `EMBEDDING_PROVIDER=sentence_transformers`
- `EMBEDDING_MODEL=intfloat/multilingual-e5-base`
- `EMBEDDING_DEVICE=cpu`
- `EMBEDDING_BATCH_SIZE=16`
- `EMBEDDING_NORMALIZE=true`

Hành vi fallback:

- nếu chưa cài `sentence-transformers`, hoặc khởi tạo model thất bại, hệ thống ghi cảnh báo và tự động fallback sang `HashEmbeddingProvider`
- API vẫn khởi động bình thường (không crash), nên môi trường dev local vẫn dùng được

Kỳ vọng hiệu năng trên CPU:

- lần chạy đầu có thể mất thời gian để tải/khởi tạo model
- tốc độ indexing trên CPU chậm hơn hash embedding
- chất lượng retrieval cho tài liệu tiếng Việt hoặc pha trộn Việt-Anh thường tốt hơn rõ rệt so với hash fallback

## Cấu Hình Reranker

Cấu hình reranker theo hướng production khuyến nghị:

- `RERANKER_PROVIDER=cross_encoder`
- `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`
- `RERANKER_DEVICE=cpu`
- `RERANKER_BATCH_SIZE=8`
- `RERANKER_TOP_N=6`

Vai trò của reranker:

- sắp xếp lại top candidate sau retrieval bằng điểm cặp query-document
- nâng chất lượng context trước bước generation

Đánh đổi về độ trễ:

- cross-encoder reranking chậm hơn so với sắp xếp chỉ dựa trên điểm ban đầu
- nên giữ `RERANKER_TOP_N` nhỏ (ví dụ `4-8`) trên máy local CPU 16GB RAM

Hành vi fallback:

- nếu khởi tạo model cross-encoder lỗi, backend ghi cảnh báo và dùng score-only reranker
- các mode `standard`, `advanced`, `compare` vẫn chạy bình thường

## Trước Khi Mở Pull Request

- chạy test và đảm bảo pass
- chạy evaluation ít nhất một lần (`stub` hoặc `workflow` predictor)
- cập nhật docs khi kiến trúc hoặc hành vi thay đổi
- giữ phạm vi thay đổi gọn, tránh refactor không liên quan
- đảm bảo không commit secrets

## Thứ Tự Ưu Tiên Khi Debug

Nếu chất lượng output/hành vi chưa đúng kỳ vọng, kiểm tra theo thứ tự:

1. lệch schema request/response
2. metadata ingestion và đầu ra chunking
3. trạng thái persistence/load của index
4. đầu ra retrieval dense/sparse/hybrid
5. thứ tự reranker và context selection
6. parse output của generator và xử lý thiếu bằng chứng
7. chuyển trạng thái advanced workflow (gate/rewrite/critique/retry/refine/abstain)
8. giả định mapping/render dữ liệu phía frontend

## Trạng Thái Xử Lý Tài Liệu

Backend trả về các trạng thái upload sau:

- `uploaded`
- `splitting`
- `embedding`
- `indexing`
- `ready`
- `failed`

Khi có ít nhất một tài liệu `ready`, workflow truy vấn sẽ dùng index của tài liệu đã upload.
Nếu không có tài liệu nào `ready`, workflow sẽ fallback về index seeded corpus.

## Ghi Chú Về Ingestion Nội Dung Hỗn Hợp

- Định dạng file hỗ trợ: `.pdf`, `.docx`, `.txt`, `.md`, `.markdown`
- Tầng parser trích xuất block có cấu trúc: `text`, `table`, `image`
- Bảng được giữ ở dạng table block và chunk mà không tách nhỏ
- OCR hiện chưa bật mặc định; ảnh được theo dõi dưới dạng metadata/image placeholder block
