"""Tests for OpenAI-compatible LLM client and fallback behavior."""

from __future__ import annotations

import asyncio
import json

import httpx

from app.generation.llm_client import (
    FallbackLLMClient,
    OpenAICompatibleLLMClient,
    StubLLMClient,
    complete_with_model,
    create_llm_client,
)


def test_openai_compatible_chat_completion_success() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path.endswith("/chat/completions")

        payload = json.loads(request.content.decode("utf-8"))
        assert payload["model"] == "qwen2.5:3b"
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"

        body = {
            "choices": [
                {
                    "message": {
                        "content": '{"answer":"Xin chao","confidence":0.8,"status":"answered"}'
                    }
                }
            ]
        }
        return httpx.Response(status_code=200, json=body)

    client = httpx.AsyncClient(
        base_url="http://localhost:11434/v1/",
        transport=httpx.MockTransport(_handler),
    )
    llm = OpenAICompatibleLLMClient(
        model="qwen2.5:3b",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0.2,
        max_tokens=256,
        timeout_seconds=10,
        client=client,
    )

    result = asyncio.run(llm.complete("xin chao", system_prompt="ban la tro ly"))
    asyncio.run(client.aclose())

    assert "Xin chao" in result


def test_openai_compatible_chat_completion_supports_model_override() -> None:
    captured_models: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        captured_models.append(payload["model"])
        return httpx.Response(
            status_code=200,
            json={"choices": [{"message": {"content": '{"answer":"ok","confidence":0.7,"status":"answered"}'}}]},
        )

    client = httpx.AsyncClient(
        base_url="http://localhost:11434/v1/",
        transport=httpx.MockTransport(_handler),
    )
    llm = OpenAICompatibleLLMClient(
        model="qwen2.5:3b",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0.2,
        max_tokens=256,
        timeout_seconds=10,
        client=client,
    )

    asyncio.run(llm.complete("default model"))
    asyncio.run(llm.complete("override model", model="qwen3.5:9b"))
    asyncio.run(client.aclose())

    assert captured_models == ["qwen2.5:3b", "qwen3.5:9b"]


def test_openai_compatible_supports_list_content_blocks() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        _ = request
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "part-1 "},
                                {"type": "text", "text": "part-2"},
                            ]
                        }
                    }
                ]
            },
        )

    client = httpx.AsyncClient(
        base_url="http://localhost:11434/v1/",
        transport=httpx.MockTransport(_handler),
    )
    llm = OpenAICompatibleLLMClient(
        model="qwen2.5:3b",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0.2,
        max_tokens=128,
        timeout_seconds=10,
        client=client,
    )

    assert asyncio.run(llm.complete("hello")) == "part-1 part-2"
    asyncio.run(client.aclose())


def test_timeout_or_http_error_uses_fallback_client() -> None:
    def _timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    client = httpx.AsyncClient(
        base_url="http://localhost:11434/v1/",
        transport=httpx.MockTransport(_timeout_handler),
    )
    primary = OpenAICompatibleLLMClient(
        model="qwen2.5:3b",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0.2,
        max_tokens=128,
        timeout_seconds=2,
        client=client,
    )
    fallback = StubLLMClient(
        responder=lambda prompt, system: '{"answer":"fallback","confidence":0.1,"status":"answered"}'
    )
    wrapped = FallbackLLMClient(primary=primary, fallback=fallback)

    output = asyncio.run(wrapped.complete("question", system_prompt="system"))
    asyncio.run(client.aclose())

    assert "fallback" in output


def test_create_llm_client_unknown_provider_falls_back_to_stub() -> None:
    fallback = StubLLMClient(
        responder=lambda prompt, system: '{"answer":"stub","confidence":0.2,"status":"answered"}'
    )
    client = create_llm_client(
        provider_name="unknown-provider",
        model="qwen2.5:3b",
        api_base="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0.2,
        max_tokens=128,
        timeout_seconds=10,
        fallback_client=fallback,
    )

    assert asyncio.run(client.complete("test")) == '{"answer":"stub","confidence":0.2,"status":"answered"}'


def test_complete_with_model_filters_unsupported_kwargs_for_legacy_client() -> None:
    class _LegacyClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str | None]] = []

        def complete(self, prompt: str, system_prompt: str | None = None) -> str:
            self.calls.append((prompt, system_prompt))
            return "legacy-ok"

    llm = _LegacyClient()
    result = asyncio.run(
        complete_with_model(
            llm,
            "legacy prompt",
            system_prompt="legacy system",
            model="qwen2.5:3b",
            max_tokens=128,
        )
    )

    assert result == "legacy-ok"
    assert llm.calls == [("legacy prompt", "legacy system")]


def test_stub_responder_supports_legacy_system_param_name() -> None:
    client = StubLLMClient(responder=lambda prompt, system: f"{prompt}|{system}")

    result = asyncio.run(client.complete("hello", system_prompt="world", model="ignored", max_tokens=99))

    assert result == "hello|world"
