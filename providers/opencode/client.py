"""OpenCode Zen provider implementation (OpenAI-compatible Chat Completions)."""

from __future__ import annotations

import json
import types
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger

from providers.base import ProviderConfig
from providers.defaults import OPENCODE_DEFAULT_BASE
from providers.transports.openai_chat import OpenAIChatTransport

from .request import build_request_body


def _tool_call_obj(index=0, tool_id=None, name=None, arguments=""):
    return types.SimpleNamespace(
        index=index,
        id=tool_id,
        function=types.SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeDelta:
    __slots__ = ("content", "reasoning_content", "tool_calls")

    def __init__(self):
        self.content: str | None = None
        self.reasoning_content: str | None = None
        self.tool_calls: list[Any] | None = None


class _FakeChoice:
    __slots__ = ("index", "delta", "finish_reason")

    def __init__(self):
        self.index = 0
        self.delta = _FakeDelta()
        self.finish_reason: str | None = None


class _FakeChunk:
    __slots__ = ("usage", "choices")

    def __init__(self):
        self.usage = None
        self.choices = [_FakeChoice()]


async def _iter_anthropic_json_as_chunks(
    response: httpx.Response,
) -> AsyncIterator[_FakeChunk]:
    body_bytes = await response.aread()
    try:
        data = json.loads(body_bytes)
    except json.JSONDecodeError:
        raise httpx.HTTPStatusError(
            f"Anthropic fallback: invalid JSON response",
            request=response.request,
            response=response,
        )

    if "error" in data:
        error_data = data["error"]
        raise httpx.HTTPStatusError(
            f"Anthropic fallback error: {error_data.get('message', 'unknown')}",
            request=response.request,
            response=response,
        )

    usage = data.get("usage", {})
    if usage:
        chunk = _FakeChunk()
        chunk.usage = types.SimpleNamespace(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=0,
        )
        yield chunk

    content_blocks = data.get("content", [])
    for cb in content_blocks:
        chunk = _FakeChunk()
        cb_type = cb.get("type", "")

        if cb_type == "text":
            chunk.choices[0].delta.content = cb.get("text", "")
        elif cb_type == "thinking":
            chunk.choices[0].delta.reasoning_content = cb.get("thinking", "")
        elif cb_type == "tool_use":
            chunk.choices[0].delta.tool_calls = [
                _tool_call_obj(
                    tool_id=cb.get("id", ""),
                    name=cb.get("name", ""),
                    arguments=json.dumps(cb.get("input", {})),
                )
            ]

        yield chunk

    # Final chunk: usage + stop reason
    if usage:
        chunk = _FakeChunk()
        chunk.usage = types.SimpleNamespace(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
        )
        chunk.choices[0].finish_reason = data.get("stop_reason", "end_turn")
        yield chunk


class OpenCodeProvider(OpenAIChatTransport):
    """OpenCode Zen provider using ``https://opencode.ai/zen/v1/chat/completions``."""

    def __init__(
        self,
        config: ProviderConfig,
        provider_name: str = "OPENCODE",
        *,
        enable_anthropic_fallback: bool = False,
    ):
        if enable_anthropic_fallback:
            base_url = (config.base_url or OPENCODE_DEFAULT_BASE).rstrip("/")
        else:
            base_url = config.base_url or OPENCODE_DEFAULT_BASE
        super().__init__(
            config,
            provider_name=provider_name,
            base_url=base_url,
            api_key=config.api_key,
        )
        self._enable_anthropic_fallback = enable_anthropic_fallback
        self._pending_fallback_request: Any = None
        self._fallback_httpx: httpx.AsyncClient | None = None

    async def cleanup(self) -> None:
        if self._fallback_httpx is not None:
            await self._fallback_httpx.aclose()
            self._fallback_httpx = None
        await super().cleanup()

    def _get_fallback_httpx(self) -> httpx.AsyncClient:
        if self._fallback_httpx is None:
            self._fallback_httpx = httpx.AsyncClient(
                base_url=self._base_url,
                proxy=self._config.proxy or None,
                timeout=httpx.Timeout(
                    self._config.http_read_timeout,
                    connect=self._config.http_connect_timeout,
                    read=self._config.http_read_timeout,
                    write=self._config.http_write_timeout,
                ),
            )
        return self._fallback_httpx

    async def _create_anthropic_stream(self) -> tuple[Any, dict]:
        from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
        from core.anthropic.native_messages_request import (
            build_base_native_anthropic_request_body,
        )

        request = self._pending_fallback_request
        self._pending_fallback_request = None

        anthropic_body = build_base_native_anthropic_request_body(
            request,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            thinking_enabled=self._is_thinking_enabled(request),
        )
        # Remove streaming keys so the response is a single JSON body, not SSE.
        anthropic_body.pop("stream", None)
        anthropic_body.pop("stream_options", None)
        anthropic_body.pop("extra_body", None)

        client = self._get_fallback_httpx()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

        response = await client.send(
            client.build_request(
                "POST", "/messages", json=anthropic_body, headers=headers
            ),
            stream=True,
        )

        if response.status_code != 200:
            try:
                error_body = await response.aread()
                error_text = error_body.decode("utf-8", errors="replace")
            except Exception:
                error_text = "unable to read error body"
            raise httpx.HTTPStatusError(
                f"Anthropic fallback failed: HTTP {response.status_code}: {error_text[:500]}",
                request=response.request,
                response=response,
            )

        stream = _iter_anthropic_json_as_chunks(response)
        return stream, {}

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        if self._enable_anthropic_fallback:
            self._pending_fallback_request = request
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    async def _create_stream(self, body: dict) -> tuple[Any, dict]:
        try:
            return await super()._create_stream(body)
        except Exception as openai_error:
            if not self._enable_anthropic_fallback:
                raise
            logger.warning(
                "{}_FALLBACK: OpenAI Chat failed ({}), retrying with Anthropic Messages",
                self._provider_name,
                type(openai_error).__name__,
            )
            try:
                return await self._create_anthropic_stream()
            except Exception:
                raise openai_error
