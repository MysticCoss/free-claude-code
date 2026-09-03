"""Tests for the OpenCode OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.providers.base import ProviderConfig
from tests.providers.support import (
    capture_openai_chat_wire_body,
    immediate_admission,
    profiled_provider,
    reasoning_for,
)


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_build_request_body_replays_tool_reasoning_natively(
    provider_id: str,
) -> None:
    provider = profiled_provider(
        provider_id,
        ProviderConfig(
            api_key="test_opencode_key",
            base_url="https://example.invalid/v1",
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "I should inspect the file.",
                            "signature": "sig",
                        },
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "Read",
                            "input": {"path": "README.md"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "file contents",
                        }
                    ],
                },
            ],
            "thinking": {"type": "enabled"},
        }
    )

    body = provider._build_request_body(request, reasoning=reasoning_for(request))

    assistant = body["messages"][0]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == "I should inspect the file."
    assert "<think>" not in assistant["content"]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert body["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "file contents",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
async def test_tool_only_history_sends_empty_reasoning_content_on_wire(
    provider_id: str,
) -> None:
    provider = profiled_provider(
        provider_id,
        ProviderConfig(
            api_key="test_opencode_key",
            base_url="https://example.invalid/v1",
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_missing",
                            "name": "Read",
                            "input": {"path": "README.md"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_missing",
                            "content": "file contents",
                        }
                    ],
                },
            ],
        }
    )

    body = provider._build_request_body(request, reasoning=reasoning_for(request))
    wire = await capture_openai_chat_wire_body(body)

    assistant = wire["messages"][0]
    assert assistant["content"] == ""
    assert assistant["reasoning_content"] == ""
    assert assistant["tool_calls"][0]["id"] == "call_missing"
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"
    assert wire["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_missing",
        "content": "file contents",
    }


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_reasoning_delta_preserves_empty_field_as_onset_signal(
    provider_id: str,
) -> None:
    provider = profiled_provider(
        provider_id,
        ProviderConfig(
            api_key="test_opencode_key",
            base_url="https://example.invalid/v1",
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    profile = provider._profile

    assert profile.reasoning_delta(SimpleNamespace(reasoning_content="think")) == (
        "think"
    )
    # Present-but-empty stays an explicit onset signal (""), not None; the
    # stream loop decides when that signal may open a thinking block.
    assert profile.reasoning_delta(SimpleNamespace(reasoning_content="")) == ""
    assert profile.reasoning_delta(SimpleNamespace(reasoning_content=None)) is None
    assert profile.reasoning_delta(SimpleNamespace()) is None


def _chunk(delta: SimpleNamespace, *, finish_reason: str | None = None) -> object:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
        usage=SimpleNamespace(completion_tokens=5, prompt_tokens=8),
    )


async def _stream(*chunks: object):
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
async def test_stream_keeps_one_text_block_when_content_chunks_carry_empty_reasoning(
    provider_id: str,
) -> None:
    provider = profiled_provider(
        provider_id,
        ProviderConfig(
            api_key="test_opencode_key",
            base_url="https://example.invalid/v1",
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )
    chunks = (
        _chunk(
            SimpleNamespace(content=None, reasoning_content="plan", tool_calls=None)
        ),
        _chunk(SimpleNamespace(content="Hello", reasoning_content="", tool_calls=None)),
        _chunk(
            SimpleNamespace(content=" world", reasoning_content="", tool_calls=None)
        ),
        _chunk(SimpleNamespace(content=None, reasoning_content="", tool_calls=None)),
        _chunk(
            SimpleNamespace(content=None, reasoning_content=None, tool_calls=None),
            finish_reason="stop",
        ),
    )
    request = MessagesRequest(
        model="qwen3.8-flash",
        max_tokens=100,
        messages=[Message(role="user", content="hi")],
    )

    with patch.object(
        provider._client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=_stream(*chunks),
    ):
        events = [event async for event in provider.stream_response(request)]

    parsed = parse_sse_text("".join(events))
    text_starts = [
        event
        for event in parsed
        if event.event == "content_block_start"
        and event.data.get("content_block", {}).get("type") == "text"
    ]
    thinking_starts = [
        event
        for event in parsed
        if event.event == "content_block_start"
        and event.data.get("content_block", {}).get("type") == "thinking"
    ]
    thinking_deltas = [
        event.data.get("delta", {}).get("thinking", "")
        for event in parsed
        if event.event == "content_block_delta"
        and event.data.get("delta", {}).get("type") == "thinking_delta"
    ]
    text_deltas = [
        event.data.get("delta", {}).get("text", "")
        for event in parsed
        if event.event == "content_block_delta"
        and event.data.get("delta", {}).get("type") == "text_delta"
    ]

    assert len(text_starts) == 1
    assert len(thinking_starts) == 1
    assert thinking_deltas == ["plan"]
    assert text_deltas == ["Hello", " world"]
