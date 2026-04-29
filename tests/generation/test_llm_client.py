"""Tests for OpenAI-compatible LLM client and fallback behavior."""

from __future__ import annotations

import asyncio
import json

import httpx

from app.core.cache import QueryCache
from app.generation.llm_client import (
    FallbackLLMClient,
    OpenAICompatibleLLMClient,
    StubLLMClient,
    _build_llm_cache_key,
    _llm_provider_signature,
    complete_with_model,
    create_llm_client,
    did_use_cache,
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
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"answer":"ok","confidence":0.7,"status":"answered"}'
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


def test_openai_compatible_streams_partial_deltas_when_callback_provided() -> None:
    def _handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload.get("stream") is True
        stream_body = (
            'data: {"choices":[{"delta":{"content":"part-1 "}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"part-2"}}]}\n\n'
            "data: [DONE]\n\n"
        ).encode("utf-8")
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=stream_body,
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
    deltas: list[str] = []

    result = asyncio.run(llm.complete("stream", on_delta=deltas.append))
    asyncio.run(client.aclose())

    assert result == "part-1 part-2"
    assert deltas == ["part-1 ", "part-2"]


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
        responder=lambda prompt, system: (
            '{"answer":"fallback","confidence":0.1,"status":"answered"}'
        )
    )
    wrapped = FallbackLLMClient(primary=primary, fallback=fallback)

    output = asyncio.run(wrapped.complete("question", system_prompt="system"))
    asyncio.run(client.aclose())

    assert "fallback" in output


def test_create_llm_client_unknown_provider_falls_back_to_stub() -> None:
    fallback = StubLLMClient(
        responder=lambda prompt, system: (
            '{"answer":"stub","confidence":0.2,"status":"answered"}'
        )
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

    assert (
        asyncio.run(complete_with_model(client, "test"))
        == '{"answer":"stub","confidence":0.2,"status":"answered"}'
    )


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

    result = asyncio.run(
        client.complete("hello", system_prompt="world", model="ignored", max_tokens=99)
    )

    assert result == "hello|world"


def test_complete_with_model_uses_llm_cache_on_repeated_prompt() -> None:
    class _CountingClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            model: str | None = None,
            max_tokens: int | None = None,
            temperature: float | None = None,
            response_format: dict[str, str] | None = None,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = model
            _ = max_tokens
            _ = temperature
            _ = response_format
            self.calls += 1
            return "cached-output"

    client = _CountingClient()
    cache = QueryCache(maxsize=4, enabled=True)

    first = asyncio.run(
        complete_with_model(
            client,
            "hello",
            system_prompt="system",
            model="qwen2.5:3b",
            max_tokens=128,
            temperature=0.2,
            response_format={"type": "json_object"},
            llm_cache=cache,
        )
    )
    second = asyncio.run(
        complete_with_model(
            client,
            "hello",
            system_prompt="system",
            model="qwen2.5:3b",
            max_tokens=128,
            temperature=0.2,
            response_format={"type": "json_object"},
            llm_cache=cache,
        )
    )

    assert first == "cached-output"
    assert second == "cached-output"
    assert client.calls == 1
    assert did_use_cache(client) is True


def test_llm_cache_key_changes_when_api_base_differs() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(  # pragma: no cover - network not exercised.
            status_code=200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )
    )
    client_a = httpx.AsyncClient(base_url="http://backend-a/v1/", transport=transport)
    client_b = httpx.AsyncClient(base_url="http://backend-b/v1/", transport=transport)
    llm_a = OpenAICompatibleLLMClient(
        model="qwen2.5:3b",
        api_base="http://backend-a/v1",
        api_key="secret-a",
        temperature=0.2,
        max_tokens=128,
        timeout_seconds=10,
        client=client_a,
    )
    llm_b = OpenAICompatibleLLMClient(
        model="qwen2.5:3b",
        api_base="http://backend-b/v1",
        api_key="secret-b",
        temperature=0.2,
        max_tokens=128,
        timeout_seconds=10,
        client=client_b,
    )
    payload = {
        "prompt": "same prompt",
        "system_prompt": "same system",
        "model": "qwen2.5:3b",
        "max_tokens": 128,
    }

    key_a = _build_llm_cache_key(llm_a, payload)
    key_b = _build_llm_cache_key(llm_b, payload)

    asyncio.run(client_a.aclose())
    asyncio.run(client_b.aclose())

    assert key_a is not None
    assert key_b is not None
    assert key_a != key_b


def test_fallback_provider_signature_changes_when_nested_clients_change() -> None:
    class _StaticClient:
        def __init__(self, *, model: str, api_base: str) -> None:
            self.model = model
            self.api_base = api_base

        def complete(
            self,
            prompt: str,
            system_prompt: str | None = None,
            **kwargs: object,
        ) -> str:
            _ = prompt
            _ = system_prompt
            _ = kwargs
            return "ok"

    wrapped_a = FallbackLLMClient(
        primary=_StaticClient(model="model-a", api_base="http://api-a/v1"),
        fallback=_StaticClient(
            model="model-fallback-a", api_base="http://fallback-a/v1"
        ),
    )
    wrapped_b = FallbackLLMClient(
        primary=_StaticClient(model="model-b", api_base="http://api-b/v1"),
        fallback=_StaticClient(
            model="model-fallback-a", api_base="http://fallback-a/v1"
        ),
    )
    wrapped_c = FallbackLLMClient(
        primary=_StaticClient(model="model-a", api_base="http://api-a/v1"),
        fallback=_StaticClient(
            model="model-fallback-c", api_base="http://fallback-c/v1"
        ),
    )

    sig_a = _llm_provider_signature(wrapped_a)
    sig_b = _llm_provider_signature(wrapped_b)
    sig_c = _llm_provider_signature(wrapped_c)

    assert sig_a != sig_b
    assert sig_a != sig_c
    assert sig_b != sig_c


def test_llm_provider_signature_excludes_secrets() -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(  # pragma: no cover - network not exercised.
            status_code=200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )
    )
    async_client = httpx.AsyncClient(
        base_url="https://public.example.com/v1/",
        transport=transport,
    )
    secret_api_key = "super-secret-api-key"
    llm = OpenAICompatibleLLMClient(
        model="qwen2.5:3b",
        api_base="https://user:very-secret-pass@public.example.com/v1?token=abc123",
        api_key=secret_api_key,
        temperature=0.2,
        max_tokens=128,
        timeout_seconds=10,
        client=async_client,
    )
    signature = _llm_provider_signature(llm)
    cache_key = _build_llm_cache_key(
        llm,
        {
            "prompt": "secret safety",
            "system_prompt": "system",
            "model": "qwen2.5:3b",
            "max_tokens": 128,
        },
    )

    asyncio.run(async_client.aclose())

    assert secret_api_key not in signature
    assert "very-secret-pass" not in signature
    assert "token=abc123" not in signature
    assert "user@" not in signature
    assert cache_key is not None
    assert secret_api_key not in cache_key
