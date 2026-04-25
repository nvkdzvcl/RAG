"""Tests for OpenAI-compatible LLM client and fallback behavior."""

from __future__ import annotations

import json

import httpx

from app.generation.llm_client import (
    FallbackLLMClient,
    OpenAICompatibleLLMClient,
    StubLLMClient,
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

    client = httpx.Client(
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

    result = llm.complete("xin chao", system_prompt="ban la tro ly")
    client.close()

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

    client = httpx.Client(
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

    llm.complete("default model")
    llm.complete("override model", model="qwen3.5:9b")
    client.close()

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

    client = httpx.Client(
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

    assert llm.complete("hello") == "part-1 part-2"
    client.close()


def test_timeout_or_http_error_uses_fallback_client() -> None:
    def _timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    client = httpx.Client(
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

    output = wrapped.complete("question", system_prompt="system")
    client.close()

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

    assert client.complete("test") == '{"answer":"stub","confidence":0.2,"status":"answered"}'
