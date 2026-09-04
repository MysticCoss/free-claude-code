"""Tests for the OpenCode OpenAI-compatible provider."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.session_id import claude_to_opencode_session_id
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


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("deepseek-v4-flash", {"role": "latest_reminder", "content": "Reminder"}),
        ("DeepSeek/deepseek-chat", {"role": "latest_reminder", "content": "Reminder"}),
        ("qwen3.8-flash", None),
    ],
)
def test_gateway_mid_conversation_system_selection(
    model: str,
    expected: dict | None,
) -> None:
    """Gateway model name alone selects the mid-conversation system behavior.

    DeepSeek-family models get the native ``latest_reminder`` role even when
    served through a non-DeepSeek gateway profile; other models drop the
    message (the top-level system prompt stays at index zero).
    """
    provider = profiled_provider(
        "opencode_go",
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
            "model": model,
            "system": "S",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Reminder"},
            ],
        }
    )

    body = provider._build_request_body(request, reasoning=reasoning_for(request))

    assert body["messages"][0] == {"role": "system", "content": "S"}
    assert body["messages"][1] == {"role": "user", "content": "Hello"}
    if expected is None:
        assert len(body["messages"]) == 2
    else:
        assert body["messages"][2] == expected


def _profiled(provider_id: str):
    return profiled_provider(
        provider_id,
        ProviderConfig(
            api_key="test_opencode_key",
            base_url="https://example.invalid/v1",
            rate_limit=1,
            rate_window=1,
        ),
        admission=immediate_admission(),
    )


def _system_policy_request(model: str) -> MessagesRequest:
    return MessagesRequest.model_validate(
        {
            "model": model,
            "system": "Conversation-wide instructions",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "system",
                    "content": "<system-reminder>tick</system-reminder>",
                },
            ],
        }
    )


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_mid_conversation_system_dropped_for_non_deepseek_model(
    provider_id: str,
) -> None:
    provider = _profiled(provider_id)
    request = _system_policy_request("qwen3.8-flash")

    body = provider._build_request_body(request, reasoning=reasoning_for(request))

    assert body["messages"] == [
        {"role": "system", "content": "Conversation-wide instructions"},
        {"role": "user", "content": "hi"},
    ]


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_mid_conversation_system_becomes_latest_reminder_for_deepseek_model(
    provider_id: str,
) -> None:
    provider = _profiled(provider_id)
    request = _system_policy_request("deepseek-v4-flash")

    body = provider._build_request_body(request, reasoning=reasoning_for(request))

    assert body["messages"] == [
        {"role": "system", "content": "Conversation-wide instructions"},
        {"role": "user", "content": "hi"},
        {
            "role": "latest_reminder",
            "content": "<system-reminder>tick</system-reminder>",
        },
    ]


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_opencode_session_headers_forward_mapped_session_id(
    provider_id: str,
) -> None:
    provider = _profiled(provider_id)
    request = MessagesRequest.model_validate(
        {
            "model": "some-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "fcc_session_id": "session-abc",
        }
    )

    body = provider._build_request_body(request, reasoning=reasoning_for(request))

    assert body["extra_headers"] == {
        "x-opencode-client": "fcc",
        "x-opencode-session": claude_to_opencode_session_id("session-abc"),
    }


def test_non_opencode_profile_does_not_inject_session_headers() -> None:
    provider = _profiled("xai")
    request = MessagesRequest.model_validate(
        {
            "model": "grok",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "fcc_session_id": "session-abc",
        }
    )

    body = provider._build_request_body(request, reasoning=reasoning_for(request))

    assert "extra_headers" not in body
