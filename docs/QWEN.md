# QWEN.md

## Purpose

This guide explains how to run Qwen models with the backend through an OpenAI-compatible API.
Supported servers:

- Ollama
- vLLM
- SGLang

## Recommended Models

For local development and quick iteration:

- `qwen2.5:3b` (Ollama, low resource)
- `qwen2.5:7b-instruct` (better quality, higher RAM/VRAM)

For stronger quality on GPU servers:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct` (requires larger VRAM)

## Environment Variables

```bash
LLM_PROVIDER=openai_compatible
LLM_MODEL=qwen2.5:3b
LLM_API_BASE=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=2048
LLM_TIMEOUT_SECONDS=120
```

Alternative API base URLs:

- vLLM: `http://localhost:8000/v1`
- SGLang: `http://localhost:30000/v1`

## Ollama Setup (CPU/GPU)

```bash
ollama pull qwen2.5:3b
ollama serve
```

Then run backend normally:

```bash
uvicorn app.main:app --reload
```

## CPU / RAM Guidance

- `qwen2.5:3b` is the most practical default for laptops/CPU-only hosts.
- Expect lower throughput on CPU and slower responses in advanced mode.
- First query may be slower due to model warm-up.

## GPU / vLLM Guidance

- Use vLLM/SGLang when you need better latency or larger Qwen models.
- Keep OpenAI-compatible endpoint enabled (`/v1/chat/completions`).
- Ensure model name in `LLM_MODEL` exactly matches what your server exposes.

## Troubleshooting

### 1) Connection refused / timeout

- Verify server is running (`ollama serve`, vLLM, or SGLang process).
- Check `LLM_API_BASE` and port.

### 2) 404 model not found

- Verify `LLM_MODEL` exists on the server.
- For Ollama, run `ollama list`.

### 3) Malformed JSON from model

- System will auto-fallback to heuristic parsing/logic.
- Standard/advanced/compare workflows should continue without crashing.

### 4) Works without Qwen?

Yes. The app includes safe fallback behavior:

- LLM runtime errors fall back to stub output.
- Advanced critique/rewrite/gate/refine fall back to heuristic logic.
