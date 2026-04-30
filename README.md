# Self-RAG Level 2 (Standard / Advanced / Compare)

Ứng dụng RAG mã nguồn mở với 3 chế độ truy vấn: `standard`, `advanced`, `compare`.
Mục tiêu của repo là một hệ thống rõ ràng, module hóa, dễ test và dễ mở rộng cho bài toán hỏi đáp có trích dẫn nguồn.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=111111)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)

<p align="center">
  <img src="./assets/RAG-pipeline.gif" alt="RAG pipeline demo" width="100%" />
</p>

## 1) Ba chế độ hoạt động

- `standard`:
  `query -> retrieve -> rerank -> select context -> generate -> return`
- `advanced`:
  `query -> retrieval gate -> rewrite -> retrieve -> rerank -> draft -> critique -> retry/refine/abstain -> return`
- `compare`:
  chạy cả `standard` và `advanced`, trả về kết quả song song kèm phần so sánh.

## 2) Thành phần chính

- Ingestion: loader cho `pdf`, `docx`, `txt`, `md`, làm sạch và chunking.
- Indexing: dense embeddings + BM25, lưu local.
- Retrieval: dense + sparse + RRF fusion.
- Reranking: cross-encoder (có fallback an toàn).
- Generation: câu trả lời có citations + xử lý thiếu bằng chứng.
- Workflow runner: điều phối theo mode.
- API: FastAPI (`/api/v1/...`) + SSE stream.
- Frontend: React + Vite + Tailwind + shadcn/ui + lucide-react.

## 3) Kiến trúc lưu trữ hiện tại

- Vector index: `InMemoryVectorIndex` và persist JSON tại `data/indexes`.
- Sparse index: BM25 local.
- Chưa dùng ChromaDB/FAISS trong codebase hiện tại.

## 4) Cài đặt nhanh

### Yêu cầu

- Python 3.12 (khuyến nghị)
- Node.js 18+ / npm
- Ollama hoặc OpenAI-compatible endpoint (nếu dùng LLM thật)

### Backend

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Backend chạy tại `http://127.0.0.1:8000`.

Kiểm tra nhanh:

```bash
curl http://127.0.0.1:8000/api/v1/health
```

### Khởi động Backend API trên Windows (Ollama + WSL)

Nếu bạn chạy backend trong WSL nhưng muốn dùng Ollama từ Windows (GPU), dùng lại luồng này:

1. Windows PowerShell (Admin):

```powershell
Get-Process ollama | Stop-Process -Force
$env:OLLAMA_HOST="0.0.0.0:11434"
ollama serve
```

2. Trong WSL, đảm bảo không chạy Ollama local:

```bash
sudo snap stop ollama
curl http://127.0.0.1:11434/api/tags
```

3. Kiểm tra WSL gọi được Ollama Windows (thay IP cho đúng máy bạn):

```bash
curl http://172.25.80.1:11434/api/tags
```

4. Cập nhật `.env`:

```env
OLLAMA_BASE_URL=http://172.25.80.1:11434
LLM_PROVIDER=openai_compatible
LLM_MODEL=qwen2.5:3b
LLM_API_BASE=http://172.25.80.1:11434/v1
LLM_API_KEY=ollama
```

5. Chạy backend từ WSL:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Frontend mặc định chạy tại `http://127.0.0.1:5173`.

## 5) API mẫu

### Upload tài liệu

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/documents/upload" \
  -F "file=@./tests/test-files/testocr.pdf"
```

### Query (standard / advanced / compare)

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tóm tắt nội dung chính của tài liệu",
    "mode": "advanced"
  }'
```

### Stream qua SSE

```bash
curl -N -X POST "http://127.0.0.1:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Cho tôi câu trả lời có trích dẫn",
    "mode": "compare"
  }'
```

## 6) Chạy test

```bash
make test-fast
make test-integration
make test-full
```

Frontend check:

```bash
cd frontend
npm run build
```

## 7) Cấu trúc thư mục

```text
app/
  api/          # routes FastAPI
  ingestion/    # loaders, cleaner, chunker
  indexing/     # embeddings, vector/bm25 index, persistence
  retrieval/    # dense, sparse, hybrid, reranker, context selector
  generation/   # llm client, baseline generation, citations
  workflows/    # standard, advanced, compare orchestration
  services/     # query/document/runtime services
prompts/        # prompt templates
tests/          # unit/integration tests
frontend/       # React app
docs/           # architecture, modes, runbook, evaluation...
```

## 8) Tài liệu thêm

- [Architecture](./docs/ARCHITECTURE.md)
- [Modes](./docs/MODES.md)
- [Runbook](./docs/RUNBOOK.md)
- [Evaluation](./docs/EVALUATION.md)
- [OCR](./docs/OCR.md)

## 9) License

MIT - xem [LICENSE](./LICENSE).
