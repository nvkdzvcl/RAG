"""LLM client abstraction, OpenAI-compatible client, and local stub fallback."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

import httpx

from app.core.async_utils import await_if_needed

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Abstraction for text generation providers."""

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Return completion text for prompt/system_prompt."""


class StubLLMClient:
    """Simple deterministic LLM client for local tests and scaffolding."""

    def __init__(
        self,
        responder: Callable[..., str | Awaitable[str]] | None = None,
    ) -> None:
        self._responder = responder or self._default_responder
        self.last_call_used_fallback = False

    @staticmethod
    def _default_responder(
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        _ = system_prompt
        _ = prompt
        _ = model
        _ = max_tokens
        return '{"answer":"Stub grounded answer.","confidence":0.5,"status":"answered"}'

    @staticmethod
    def _supported_kwargs(function: Callable[..., Any], candidates: dict[str, Any]) -> dict[str, Any]:
        try:
            signature = inspect.signature(function)
        except (TypeError, ValueError):
            return dict(candidates)

        supports_var_keywords = False
        selected: dict[str, Any] = {}
        for parameter in signature.parameters.values():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                supports_var_keywords = True
                continue
            if parameter.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            if parameter.name in candidates:
                selected[parameter.name] = candidates[parameter.name]

        if supports_var_keywords:
            for key, value in candidates.items():
                selected.setdefault(key, value)
        return selected

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        candidates: dict[str, Any] = {
            "prompt": prompt,
            "system_prompt": system_prompt,
            # Keep backward compatibility for legacy responder lambdas: (prompt, system, ...)
            "system": system_prompt,
            "model": model,
            "max_tokens": max_tokens,
        }
        candidates.update(kwargs)
        selected = self._supported_kwargs(self._responder, candidates)
        return await await_if_needed(self._responder(**selected))


def _normalize_complete_args(
    *,
    prompt: str,
    system_prompt: str | None,
    model: str | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if system_prompt is not None and not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string or None")

    payload: dict[str, Any] = {"prompt": prompt, "system_prompt": system_prompt}
    normalized_model = model.strip() if isinstance(model, str) else ""
    if normalized_model:
        payload["model"] = normalized_model

    if isinstance(max_tokens, bool):
        return payload
    if isinstance(max_tokens, int) and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    return payload


def _supported_complete_kwargs(llm_client: LLMClient, payload: dict[str, Any]) -> dict[str, Any]:
    complete_fn = llm_client.complete
    try:
        signature = inspect.signature(complete_fn)
    except (TypeError, ValueError):
        return dict(payload)

    supports_var_keywords = False
    selected: dict[str, Any] = {}
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            supports_var_keywords = True
            continue
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if parameter.name in payload:
            selected[parameter.name] = payload[parameter.name]

    if supports_var_keywords:
        for key, value in payload.items():
            selected.setdefault(key, value)
    return selected


async def complete_with_model(
    llm_client: LLMClient,
    prompt: str,
    *,
    system_prompt: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
) -> str:
    """Invoke completion while safely supporting optional per-call model override."""
    payload = _normalize_complete_args(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
    )
    selected = _supported_complete_kwargs(llm_client, payload)
    output = await await_if_needed(llm_client.complete(**selected))
    if not isinstance(output, str):
        raise TypeError("LLM completion output must be a string.")
    return output


class OpenAICompatibleLLMClient:
    """OpenAI-compatible chat client (Ollama/vLLM/SGLang)."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str | None,
        temperature: float,
        max_tokens: int,
        timeout_seconds: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not api_base.strip():
            raise ValueError("api_base must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        self.model = model.strip()
        self.api_base = api_base.rstrip("/")
        self.api_key = (api_key or "").strip()
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = int(timeout_seconds)
        self.last_call_used_fallback = False
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=f"{self.api_base}/",
            timeout=httpx.Timeout(self.timeout_seconds),
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @staticmethod
    def _extract_content(body: dict[str, Any]) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI-compatible response is missing choices.")

        first = choices[0]
        if not isinstance(first, dict):
            raise RuntimeError("OpenAI-compatible choice is not an object.")

        message = first.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("OpenAI-compatible choice is missing message.")

        content = message.get("content")
        if isinstance(content, str):
            return content

        # Some OpenAI-compatible servers return list[{'type': 'text', 'text': '...'}].
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text_part = item.get("text")
                    if isinstance(text_part, str):
                        parts.append(text_part)
            if parts:
                return "".join(parts)

        raise RuntimeError("Invalid response content type from OpenAI-compatible API.")

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        messages: list[dict[str, str]] = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.append({"role": "user", "content": prompt})
        selected_model = model.strip() if isinstance(model, str) and model.strip() else self.model

        payload: dict[str, object] = {
            "model": selected_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": int(max_tokens) if isinstance(max_tokens, int) and max_tokens > 0 else self.max_tokens,
        }

        try:
            response = await self._client.post(
                "chat/completions",
                headers=self._build_headers(),
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict):
                raise RuntimeError("OpenAI-compatible response is not a JSON object.")
            return self._extract_content(body)
        except Exception as exc:
            raise RuntimeError("OpenAI-compatible completion request failed.") from exc

    async def aclose(self) -> None:
        """Close owned async HTTP resources."""
        if self._owns_client:
            await self._client.aclose()


class FallbackLLMClient:
    """Use primary client and fall back to secondary client on runtime errors."""

    def __init__(self, primary: LLMClient, fallback: LLMClient) -> None:
        self.primary = primary
        self.fallback = fallback
        self.last_call_used_fallback = False

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        _ = kwargs
        self.last_call_used_fallback = False
        try:
            return await complete_with_model(
                self.primary,
                prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            self.last_call_used_fallback = True
            logger.warning(
                "Primary LLM client failed; falling back to stub client.",
                exc_info=exc,
            )
            return await complete_with_model(
                self.fallback,
                prompt,
                system_prompt=system_prompt,
                model=model,
                max_tokens=max_tokens,
            )

    async def aclose(self) -> None:
        """Close underlying clients when they expose async close hooks."""
        await close_llm_client(self.primary)
        await close_llm_client(self.fallback)


def did_use_fallback(llm_client: LLMClient) -> bool:
    """Best-effort flag indicating whether the latest call used fallback."""
    value = getattr(llm_client, "last_call_used_fallback", False)
    return bool(value)


async def close_llm_client(llm_client: LLMClient) -> None:
    """Close optional client resources when supported."""
    async_closer = getattr(llm_client, "aclose", None)
    if callable(async_closer):
        await await_if_needed(async_closer())
        return

    sync_closer = getattr(llm_client, "close", None)
    if callable(sync_closer):
        sync_closer()


OPENAI_COMPATIBLE_PROVIDER_NAMES = {
    "openai_compatible",
    "openai-compatible",
    "openai",
}
STUB_PROVIDER_NAMES = {"stub", "mock", "none", "local_stub"}


def create_llm_client(
    *,
    provider_name: str,
    model: str,
    api_base: str,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    fallback_client: LLMClient | None = None,
) -> LLMClient:
    """Create configured LLM client with safe local fallback."""
    normalized = provider_name.strip().lower() if provider_name else ""
    fallback = fallback_client or StubLLMClient()

    if normalized in OPENAI_COMPATIBLE_PROVIDER_NAMES:
        try:
            primary = OpenAICompatibleLLMClient(
                model=model,
                api_base=api_base,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            return FallbackLLMClient(primary=primary, fallback=fallback)
        except Exception as exc:
            logger.warning(
                "Failed to initialize OpenAI-compatible LLM client. Falling back to stub client.",
                exc_info=exc,
            )
            return fallback

    if normalized in STUB_PROVIDER_NAMES:
        return fallback

    logger.warning("Unknown LLM provider '%s'. Falling back to stub client.", provider_name)
    return fallback


def create_llm_client_from_settings(settings: Any, fallback_client: LLMClient | None = None) -> LLMClient:
    """Create a runtime LLM client from application settings."""
    return create_llm_client(
        provider_name=getattr(settings, "llm_provider"),
        model=getattr(settings, "llm_model"),
        api_base=getattr(settings, "llm_api_base"),
        api_key=getattr(settings, "llm_api_key", None),
        temperature=float(getattr(settings, "llm_temperature")),
        max_tokens=int(getattr(settings, "llm_max_tokens")),
        timeout_seconds=int(getattr(settings, "llm_timeout_seconds")),
        fallback_client=fallback_client,
    )
