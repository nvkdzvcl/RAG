# Self-RAG (Mã Nguồn Mở, 3 Chế Độ)

Ứng dụng Self-RAG theo kiến trúc module với 3 chế độ:

- `standard`: pipeline RAG cơ bản
- `advanced`: vòng lặp Self-RAG thực dụng (gate/rewrite/critique/retry/refine/abstain)
- `compare`: chạy đồng thời `standard` + `advanced` và trả kết quả so sánh song song

## Công Nghệ Sử Dụng

- Backend: Python, FastAPI, Pydantic
- Frontend: React, Vite, Tailwind CSS, cấu trúc tương thích shadcn/ui
- Evaluation: golden dataset + CLI runner + regression tests

## Cấu Trúc Repository

```text
app/
  api/
  core/
  evaluation/
  generation/
  indexing/
  ingestion/
  retrieval/
  schemas/
  services/
  workflows/
data/
  eval/
docs/
frontend/
prompts/
tests/
```

## Backend: Cài Đặt và Chạy

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
```

URL backend mặc định: `http://127.0.0.1:8000`

Thư mục tài liệu nguồn mặc định: `docs/` (có thể cấu hình qua `CORPUS_DIR`).
Tài liệu upload được lưu ở `data/raw/` (hoặc `<DATA_DIR>/raw`) và được xử lý để tạo index truy hồi.
Hành vi truy hồi ở runtime:
- nếu có ít nhất một tài liệu upload ở trạng thái `ready`, workflow query sẽ dùng index của tài liệu upload
- nếu không, hệ thống fallback về corpus mặc định (`CORPUS_DIR`)

Vector index và BM25 index được lưu bền vững tại `INDEX_DIR` (mặc định `data/indexes/`).

## Embedding Backend (Tiếng Việt + Đa Ngôn Ngữ)

Embedding backend mặc định dùng sentence-transformers:

- provider: `sentence_transformers`
- model: `intfloat/multilingual-e5-base`
- device: `cpu`
- batch size: `16`
- normalize vector: `true`

Định dạng E5 được áp dụng nội bộ:

- document chunk -> `passage: {text}`
- user query -> `query: {text}`

Nếu sentence-transformers không khả dụng hoặc load model lỗi, backend sẽ log cảnh báo và fallback về `hash-embedding` có tính xác định, để API vẫn khởi động được.

Lưu ý khi chạy trên CPU (mang tính tham khảo):

- lần chạy đầu có thể chậm hơn do tải/khởi tạo model artifact
- chất lượng truy hồi cho tiếng Việt và truy vấn Việt-Anh thường tốt hơn hash embedding
- thời gian indexing cao hơn chế độ hash, đặc biệt với upload lớn

## Reranker Backend (Sau Truy Hồi)

Reranker mặc định là cross-encoder từ sentence-transformers:

- provider: `cross_encoder`
- model: `BAAI/bge-reranker-v2-m3`
- device: `cpu`
- batch size: `8`
- số candidate rerank tối đa: `6`

Vai trò của reranker:

- nhận `query + các chunk đã retrieve`
- chấm điểm từng cặp query-chunk
- sắp xếp lại theo `rerank_score` để cải thiện chất lượng context đầu vào cho generation

Đánh đổi:

- chất lượng context và độ grounded thường tốt hơn
- độ trễ mỗi query cao hơn fallback score-only, đặc biệt trên CPU

Fallback:

- nếu load model cross-encoder lỗi, backend log cảnh báo và fallback sang score-only reranking
- API và các workflow query vẫn chạy, không crash

## Run Với Qwen (OpenAI-Compatible)

Backend hỗ trợ client OpenAI-compatible, dùng được với:

- Ollama
- vLLM
- SGLang

Biến môi trường chính:

- `LLM_PROVIDER=openai_compatible`
- `LLM_MODEL=qwen2.5:3b` (hoặc model Qwen khác)
- `LLM_API_BASE=http://localhost:11434/v1` (Ollama)
- `LLM_API_KEY=ollama`
- `LLM_TEMPERATURE=0.2`
- `LLM_MAX_TOKENS=2048`
- `LLM_TIMEOUT_SECONDS=120`

### Ollama (mặc định dễ chạy local)

```bash
ollama pull qwen2.5:3b
ollama serve
```

Thiết lập trong `.env`:

```bash
LLM_PROVIDER=openai_compatible
LLM_MODEL=qwen2.5:3b
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=ollama
```

### vLLM (GPU)

Ví dụ endpoint OpenAI-compatible:

```bash
LLM_PROVIDER=openai_compatible
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_API_BASE=http://localhost:8000/v1
LLM_API_KEY=EMPTY
```

### SGLang (GPU)

Ví dụ endpoint OpenAI-compatible:

```bash
LLM_PROVIDER=openai_compatible
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_API_BASE=http://localhost:30000/v1
LLM_API_KEY=EMPTY
```

Lưu ý fallback:

- nếu endpoint/model Qwen lỗi hoặc trả dữ liệu không hợp lệ, workflow vẫn chạy nhờ fallback heuristic + stub client
- `standard`, `advanced`, `compare` không crash chỉ vì LLM lỗi runtime

## Chiến Lược Parsing Ingestion

Pipeline ingestion hiện dùng parser abstraction:

- `BaseDocumentParser`
- `PDFParser` (`pdfplumber`)
- `DocxParser` (`python-docx`)
- `TextParser`
- `MarkdownParser`

Parser phát ra các block có cấu trúc (`text`, `table`, `image`) kèm metadata (`page`, `section`, `bbox`).
Chunking có nhận biết cấu trúc và giữ metadata block (`block_type`, `language`, `section`, `page`).
Block bảng được giữ nguyên (không tách đôi qua nhiều chunk).

### OCR Tùy Chọn Cho PDF Scan

- OCR mặc định **tắt** (`OCR_ENABLED=false`).
- Khi bật OCR, parser PDF vẫn ưu tiên text/table từ `pdfplumber`.
- Nếu trang PDF có quá ít text (`OCR_MIN_TEXT_CHARS`) thì hệ thống thử OCR bằng Tesseract + PyMuPDF.
- OCR thất bại sẽ chỉ ghi cảnh báo và bỏ qua block OCR, không làm crash upload/query.
- OCR yêu cầu cài hệ thống: `tesseract-ocr` và gói tiếng Việt `tesseract-ocr-vie` (hoặc tương đương theo OS).
- OCR hiện chỉ áp dụng cho PDF; OCR ảnh trong DOCX được giữ lại cho giai đoạn sau.
- Sau khi bật OCR trong `.env`, cần **upload lại tài liệu PDF** để index mới chứa nội dung OCR.

Xem hướng dẫn chi tiết tại [docs/OCR.md](docs/OCR.md).

## API Endpoints

- `GET /api/v1/health`
- `POST /api/v1/query`
- `POST /api/v1/documents/upload`
- `GET /api/v1/documents`
- `GET /api/v1/documents/{document_id}/status`

Ví dụ gọi query:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does advanced mode differ from standard mode?",
    "mode": "advanced",
    "chat_history": []
  }'
```

Giá trị `mode` hỗ trợ: `standard`, `advanced`, `compare`.

Ví dụ upload tài liệu:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload \
  -F "file=@./sample.md"
```

Định dạng file hỗ trợ: `pdf`, `docx`, `txt`, `md`, `markdown`.

## Frontend: Cài Đặt và Chạy

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

URL frontend mặc định: `http://127.0.0.1:5173`

Mặc định frontend gọi `VITE_API_BASE_URL=/api/v1` và Vite proxy `/api/*` về `http://localhost:8000`.

## Evaluation

Dataset evaluation:

- dataset chính: `data/eval/golden_dataset.jsonl`
- dataset tương thích: `data/eval/golden.jsonl`

Chạy evaluation (`standard` + `advanced`):

```bash
python scripts/run_eval.py --dataset data/eval/golden_dataset.jsonl --modes standard advanced
```

Chạy evaluation (bao gồm `compare`):

```bash
python scripts/run_eval.py --dataset data/eval/golden_dataset.jsonl --modes standard advanced compare
```

Smoke test offline nhanh (stub predictor, deterministic):

```bash
python -m app.evaluation.runner --dataset data/eval/golden.jsonl --predictor stub
```

Chạy bằng workflow thật với dataset tương thích:

```bash
python -m app.evaluation.runner --dataset data/eval/golden.jsonl --predictor workflow
```

Artifact sinh ra:

- `data/eval/results/results.json`
- `data/eval/results/report.md`
- `data/eval/results/summary.csv`

Cách thêm sample evaluation:

1. Thêm một dòng JSON mới vào `data/eval/golden_dataset.jsonl`.
2. Bao gồm các trường bắt buộc: `id`, `question`, `expected_behavior`, `category`.
3. Tùy chọn: `reference_answer`, `gold_sources`, `notes`.
4. Giữ cân bằng các nhóm `category`, bao gồm sample tiếng Việt.

Cách đọc kết quả:

- So sánh `avg_latency_delta_ms` và `avg_confidence_delta` giữa `standard` và `advanced`.
- Kiểm tra `advanced_retry_rate`, `abstain_rate`, `citation_rate` trong `report.md`.
- Xem `groundedness_proxy` như chỉ báo heuristic, không phải thước đo factuality tuyệt đối.

## Tests

Vòng lặp dev thông thường (nhanh, bỏ test nặng/e2e):

```bash
pytest -m "not slow and not e2e"
# hoặc
make test-fast
```

Chỉ kiểm tra backend logic cốt lõi (schema/retrieval/generation):

```bash
pytest tests/schemas tests/retrieval tests/generation -m "not slow"
```

Kiểm tra integration backend (không gồm slow/e2e):

```bash
make test-integration
```

Trước khi push/release (full suite):

```bash
pytest
# hoặc
make test-full
```

## Ghi Chú

- `requirements.txt` và `requirements-dev.txt` là hai file phụ thuộc chính thức.
- `requirement.txt` cố ý không dùng để tránh nhầm lẫn cài đặt.
- OCR cho image hiện chưa bật; image block đang được lưu dưới dạng metadata placeholder.
