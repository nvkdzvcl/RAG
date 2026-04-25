# QWEN.md

## Mục Đích

Tài liệu này hướng dẫn cách chạy mô hình Qwen với backend thông qua API tương thích OpenAI.
Các server được hỗ trợ:

- Ollama
- vLLM
- SGLang

## Mô Hình Khuyến Nghị

Cho môi trường local và vòng lặp phát triển nhanh:

- `qwen2.5:3b` (Ollama, tài nguyên thấp)
- `qwen2.5:7b-instruct` (chất lượng tốt hơn, cần RAM/VRAM cao hơn)

Cho chất lượng cao hơn trên server GPU:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct` (cần VRAM lớn hơn)

## Biến Môi Trường

```bash
LLM_PROVIDER=openai_compatible
LLM_MODEL=qwen2.5:3b
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2048
LLM_TIMEOUT_SECONDS=120
```

URL `API base` thay thế:

- vLLM: `http://localhost:8000/v1`
- SGLang: `http://localhost:30000/v1`

## Cấu Hình Ollama (CPU/GPU)

```bash
ollama pull qwen2.5:3b
ollama serve
```

Sau đó chạy backend như bình thường:

```bash
uvicorn app.main:app --reload
```

## Gợi Ý CPU / RAM

- `qwen2.5:3b` là lựa chọn mặc định thực tế nhất cho laptop/máy chỉ có CPU.
- Hiệu năng trên CPU sẽ thấp hơn và phản hồi ở advanced mode thường chậm hơn.
- Truy vấn đầu tiên có thể chậm do thời gian warm-up model.

## Gợi Ý GPU / vLLM

- Dùng vLLM/SGLang khi cần độ trễ tốt hơn hoặc chạy Qwen kích thước lớn.
- Đảm bảo endpoint tương thích OpenAI được bật (`/v1/chat/completions`).
- Đảm bảo tên model trong `LLM_MODEL` khớp chính xác với model server đang expose.

## Khắc Phục Sự Cố

### 1) Lỗi connection refused / timeout

- Kiểm tra server đã chạy chưa (`ollama serve`, tiến trình vLLM hoặc SGLang).
- Kiểm tra lại `LLM_API_BASE` và cổng.

### 2) Lỗi 404 không tìm thấy model

- Xác nhận `LLM_MODEL` có tồn tại trên server.
- Với Ollama, chạy `ollama list`.

### 3) Model trả JSON lỗi định dạng

- Hệ thống sẽ tự fallback sang logic parse/heuristic.
- Workflow `standard`/`advanced`/`compare` vẫn tiếp tục chạy, không crash.

### 4) Có chạy được khi không bật Qwen không?

Có. Ứng dụng có cơ chế fallback an toàn:

- Lỗi runtime của LLM sẽ fallback về stub output.
- Các bước critique/rewrite/gate/refine ở advanced sẽ fallback về heuristic logic.
