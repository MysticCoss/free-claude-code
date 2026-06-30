"""OpenCode Zen provider implementation.

Dispatches to the correct SDK endpoint based on model:
- ``@ai-sdk/openai-compatible`` models → ``/chat/completions`` (OpenAI Chat)
- ``@ai-sdk/anthropic`` models → ``/messages`` (Anthropic Messages)
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx

from core.anthropic import iter_provider_stream_error_sse_events
from core.anthropic.native_sse_block_policy import (
    NativeSseBlockPolicyState,
    transform_native_sse_block_event,
)
from providers.base import ProviderConfig
from providers.defaults import OPENCODE_DEFAULT_BASE
from providers.error_mapping import (
    extract_provider_error_detail,
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from providers.transports.anthropic_messages.http import raise_for_status_with_body
from providers.transports.anthropic_messages.stream import (
    AnthropicMessagesStreamAdapter,
)
from providers.transports.http import maybe_await_aclose
from providers.transports.openai_chat import OpenAIChatTransport

from .request import (
    build_anthropic_request_body,
    build_request_body,
    is_anthropic_native_model,
)


class OpenCodeProvider(OpenAIChatTransport):
    """OpenCode Zen provider dispatching to the correct endpoint per model."""

    stream_chunk_mode = "line"

    def __init__(
        self,
        config: ProviderConfig,
        provider_name: str = "OPENCODE",
    ):
        super().__init__(
            config,
            provider_name=provider_name,
            base_url=config.base_url or OPENCODE_DEFAULT_BASE,
            api_key=config.api_key,
        )
        self._anthropic_httpx: httpx.AsyncClient | None = None

    async def cleanup(self) -> None:
        if self._anthropic_httpx is not None:
            await self._anthropic_httpx.aclose()
            self._anthropic_httpx = None
        await super().cleanup()

    # ------------------------------------------------------------------
    # Anthropic Messages transport interface
    # ------------------------------------------------------------------

    def _get_anthropic_httpx(self) -> httpx.AsyncClient:
        if self._anthropic_httpx is None:
            self._anthropic_httpx = httpx.AsyncClient(
                base_url=self._base_url,
                proxy=self._config.proxy or None,
                timeout=httpx.Timeout(
                    self._config.http_read_timeout,
                    connect=self._config.http_connect_timeout,
                    read=self._config.http_read_timeout,
                    write=self._config.http_write_timeout,
                ),
            )
        return self._anthropic_httpx

    def _request_headers(self) -> dict[str, str]:
        return {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            "x-api-key": self._api_key,
        }

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        request = self._get_anthropic_httpx().build_request(
            "POST",
            "/messages",
            json=body,
            headers=self._request_headers(),
        )
        return await self._get_anthropic_httpx().send(request, stream=True)

    async def _validated_stream_send(
        self, body: dict, *, req_tag: str
    ) -> httpx.Response:
        send_response = await self._send_stream_request(body)
        if send_response.status_code != 200:
            try:
                await raise_for_status_with_body(
                    send_response,
                    provider_name=self._provider_name,
                    req_tag=req_tag,
                    log_api_error_tracebacks=self._config.log_api_error_tracebacks,
                )
            finally:
                if not send_response.is_closed:
                    await maybe_await_aclose(send_response)
        return send_response

    def _new_stream_state(self, _request: Any, *, thinking_enabled: bool) -> Any:
        return NativeSseBlockPolicyState()

    def _transform_stream_event(
        self,
        event: str,
        state: Any,
        *,
        thinking_enabled: bool,
    ) -> str | None:
        return transform_native_sse_block_event(
            event, state, thinking_enabled=thinking_enabled
        )

    def _get_error_message(self, error: Exception, request_id: str | None) -> str:
        mapped_error = map_error(error, rate_limiter=self._global_rate_limiter)
        return user_visible_message_for_mapped_provider_error(
            mapped_error,
            provider_name=self._provider_name,
            read_timeout_s=self._config.http_read_timeout,
            detail=extract_provider_error_detail(error),
            request_id=request_id,
        )

    def _emit_error_events(
        self,
        *,
        request: Any,
        input_tokens: int,
        error_message: str,
        sent_any_event: bool,
    ) -> Iterator[str]:
        yield from iter_provider_stream_error_sse_events(
            request=request,
            input_tokens=input_tokens,
            error_message=error_message,
            sent_any_event=sent_any_event,
            log_raw_sse_events=self._config.log_raw_sse_events,
        )

    # ------------------------------------------------------------------
    # Request body dispatch
    # ------------------------------------------------------------------

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        thinking = self._is_thinking_enabled(request, thinking_enabled)
        model = getattr(request, "model", "")
        if is_anthropic_native_model(model):
            return build_anthropic_request_body(
                request,
                thinking_enabled=thinking,
            )
        return build_request_body(request, thinking_enabled=thinking)

    # ------------------------------------------------------------------
    # Stream dispatch
    # ------------------------------------------------------------------

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream response, dispatching to the correct adapter per model."""
        model = getattr(request, "model", "")
        if is_anthropic_native_model(model):
            adapter = AnthropicMessagesStreamAdapter(
                self,
                request=request,
                input_tokens=input_tokens,
                request_id=request_id,
                thinking_enabled=thinking_enabled,
            )
            async for event in adapter.run():
                yield event
        else:
            async for event in super().stream_response(
                request,
                input_tokens=input_tokens,
                request_id=request_id,
                thinking_enabled=thinking_enabled,
            ):
                yield event
