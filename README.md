# Enterprise-Grade Self-RAG Pipeline

> Hệ thống RAG (Retrieval-Augmented Generation) đáp ứng tiêu chuẩn Production, được tối ưu hóa cho khả năng truy xuất độ chính xác cao, truy hồi cực kỳ ổn định (deterministic) và chống sinh ảo giác (hallucination).

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Pytest](https://img.shields.io/badge/Tests-190%2B_Passing-10B981?logo=pytest&logoColor=white)
![Stability](https://img.shields.io/badge/CI-Security_%26_Dependency_Audits-2088FF)

Repository này **không phải là bản demo hay tutorial RAG cơ bản**. Đây là một hệ thống thiết kế theo module hướng tới cấp độ doanh nghiệp (enterprise-oriented), giải quyết các bài toán hóc búa nhất của RAG: trôi ngữ cảnh (lost-in-the-middle), trả lời ảo giác, bóc tách cấu trúc dữ liệu, và quản trị rủi ro giảm sút chất lượng âm thầm (silent degradation).

---

## 🌟 Công Nghệ Sử Dụng

**Backend**
![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-Schemas-E92063?logo=pydantic&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-Tests-0A9EDC?logo=pytest&logoColor=white)

**Frontend**
![React](https://img.shields.io/badge/React-UI-61DAFB?logo=react&logoColor=111111)
![Vite](https://img.shields.io/badge/Vite-Build-646CFF?logo=vite&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-Frontend-3178C6?logo=typescript&logoColor=white)
![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-Styling-06B6D4?logo=tailwindcss&logoColor=white)

**RAG / LLM**
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-Embeddings-FF6F00)
![BM25](https://img.shields.io/badge/BM25-Sparse_Retrieval-4B5563)
![Qwen](https://img.shields.io/badge/Qwen-LLM-7C3AED)
![Ollama](https://img.shields.io/badge/Ollama-Local_Runtime-000000?logo=ollama&logoColor=white)

**CI/CD & Code Quality**
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI-2088FF?logo=github-actions&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-Linter-FCC21B?logo=python&logoColor=black)
![Mypy](https://img.shields.io/badge/Mypy-Type_Check-2B5B84?logo=python&logoColor=white)

---

## 💡 Các Năng Lực Cốt Lõi

Khác biệt với các hệ thống Single-Vector RAG tiêu chuẩn, kiến trúc của hệ thống này sở hữu 3 tầng đánh giá và xử lý:

- **Bóc tách Tự Động Nhận Diện Cấu Trúc (Structure-Aware Ingestion):** Duy trì ý nghĩa văn cảnh bằng cách sử dụng các sliding chunk thông minh - không bao giờ cắt đôi các bảng dữ liệu (data tables) và tự động nối các Heading liên quan (`[Title: X | Section: Y]`) vào đầu mỗi chunk để định hướng chính xác mô hình Vector Dense.
- **Truy hồi Lai kết hợp Reranking (Hybrid RRF & Cross-Encoder):** Dung hợp Dense Vectors (tìm kiếm ngữ nghĩa) và Sparse Keywords (BM25) qua thuật toán thẻ lai cực nhạy (Reciprocal Rank Fusion - RRF). Sau đó chấm điểm chặt chẽ lần 2 bởi Cross-Encoder Reranker để đẩy các đoạn văn chính xác nhất lên đầu.
- **Agent Tự Phản Tư (Self-Reflective RAG):** Tự động đo mức độ bằng chứng (Grounding) của contexts đã truy hồi. Pipeline sẽ cực kỳ dứt khoát **chối từ trả lời (Abstain)** hoặc nhờ người dùng làm rõ thay vì "bịa" ra kết quả (hallucinate). Nếu ngữ cảnh hợp lệ, tự động đánh giá (critique) và tinh chỉnh prompt đầu ra (rewrite/refine).
- **Caching Đa Tầng (LRU Cache):** Giảm thiểu tối đa áp lực tài nguyên và độ trễ do mạng thông qua cache In-memory tích hợp song song cả 3 chặng: Embedding, Retrieval và LLM API.
- **Evaluation Framework Chuyên Sâu:** Trang bị sẵn engine tính toán và báo cáo tự động cho các chỉ số Truy Hồi Thông Tin (IR Metrics: **Hit Rate, MRR, nDCG**) rạch ròi, độc lập hoàn toàn với lớp AI phán đoán. Thêm vào đó là pipeline đánh giá độ tin cậy dựa trên phương pháp LLM-as-a-judge.

---

## 🏗️ Tổng Quan Kiến Trúc

 Hệ thống phân chia thành các Provider cực kỳ Module hóa (Hot-swappable):

```text
┌─────────────────┐       ┌──────────────────┐       ┌─────────────────┐
│  Ingestion      │       │  Retrieval Tiers │       │ Generation      │
│                 │       │                  │       │                 │
│ 1. Parse Layout ├──────►│ 1. Dense (E5/VN) ├──────►│ 1. Caching      │
│ 2. Smart Chunk  │       │ 2. Sparse (BM25) │       │ 2. Self-Critique│
│ 3. Inject Context       │ 3. Fusion (RRF)  │       │ 3. Grounding    │
└─────────────────┘       │ 4. Cross-Encoder │       │ 4. Fallback     │
                          └──────────────────┘       └─────────────────┘
```

## 📊 Evaluation & Benchmarks

Chúng tôi sử dụng mô hình Deterministic Benchmarking (đo lường tính kiên định) nhằm chống lại vấn đề suy giảm truy xuất thầm lặng qua thời gian. Một cơ chế CI Guard tự động chặn đứng và kiểm chứng toán học các ngưỡng an toàn này.

*Metrics Baseline ví dụ trên dataset nội bộ:*

| Metric | Chỉ Sử Dụng Dense | Hybrid + Reranker |
| :--- | :--- | :--- |
| **Hit Rate@K** | `0.78` | **`>0.98`** |
| **Mean Reciprocal Rank (MRR)** | `0.65` | **`>0.88`** |
| **nDCG@K** | `0.60` | **`>0.85`** |
| **Tỷ Lệ Sinh Ảo Giác** | `12%` | **`<1%`** |

---

## 🚀 Khởi Động Nhanh

### 1. Cài Đặt Môi Trường
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

### 2. Khởi Động Backend API
Khởi chạy server FastAPI (mặc định tại cổng `8000`):
```bash
uvicorn app.main:app --reload
```

### 3. Cung Cấp Nguồn Cấp LLM
Dự án được kết nối tự do với mọi LLM sử dụng format API của OpenAI (vLLM, Groq, Ollama v.v...). Thiết lập trên local dễ nhất:
```bash
ollama serve
ollama pull qwen2.5:3b
```
*(Cần cập nhật `.env`: cấu hình biến `LLM_API_BASE=http://localhost:11434/v1` và `LLM_PROVIDER=openai_compatible`)*

### 4. Bơm Dữ Liệu & Hỏi Đáp (Ingest & Query)
Upload một tài liệu vào kho:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents/upload -F "file=@./data/sample.pdf"
```

Chạy một query yêu cầu phân tích cao trong chế độ Self-RAG:
```bash
curl -X POST http://127.0.0.1:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Tóm tắt các phát hiện quan trọng trong báo cáo tài chính.", "mode": "advanced"}'
```

---

## 🛡️ Kiểm Định Kiểm Phủ & Mã Nguồn (Testing & Security Guardrails)

Tính chính xác của dự án RAG được bảo kê chéo bằng tổ hợp hơn **190+ bài Unit Test / Integration Test mạch lạc** song song trên Github Actions.

**Lệnh lập trình hằng ngày (Fast Tests vòng lặp Local):**
```bash
python -m pytest -m "not slow and not e2e" \
  --cov=app --cov-report=term-missing --cov-report=xml --cov-fail-under=75
```

**Bot CI/CD Kiểm Soát Định Kỳ:**
- **Deterministic Regression Guard:** Xác nhận bảo vệ sàn `Hit Rate`, `MRR`, và `nDCG` với Mock Data JSON (Fixture) nội bộ mà không cần tải bất cứ Model Internet hay API ngoài luồng nào.
- **Strict Typing:** `Mypy` chạy toàn phần strict mode đảm bảo tính ổn định bộ nhớ.
- **Code Format:** Tự format theo `Ruff`.
- **Security & Dependency Auditing:** CI Pipeline tự động trigger ứng dụng `bandit` (Quét tấn công AST code) và `pip-audit` (Quét phơi bày lỗ hổng thư viện mở).

---

## 🗺️ Tầm Nhìn & Kế Hoạch (Roadmap)

Bộ Pipeline RAG này đã gần như đủ bản lĩnh để plug-and-play vào production của Doanh Nghiệp. Các cột mốc nâng cấp quy mô ngang dự tính trong chặng đường tới:

- **Lưu trữ Cụm (Distributed Vector & Graph Stores):** Di dời `InMemoryVectorIndex` lên Qdrant, Milvus. Cân nhắc tích hợp mô hình `Neo4j` giúp Query GraphRAG kết nối vòng lặp rộng.
- **Bơm dữ liệu hướng Sự Kiện (Event-Driven Ingestion):** Add RabbitMQ/Kafka tách rời Job bóc tách File PDF nặng nhọc ra khỏi Backend Thread HTTP.
- **Phân luồng Semantic Routing Agent:** Viết Gateway Agent phân loại sớm mục đích truy vấn (VD: Yêu cầu Thống kê báo cáo -> Ném sang đường Text-to-SQL thay vì ném sang Vector Database).

---
*Kiến trúc RAG chất lượng cao - Dành cho doanh nghiệp làm sản phẩm chân chính.*
