"""Fork regression tests for OpenCode-plan behaviors.

Kept in a dedicated file so upstream rewrites of ``test_opencode.py``
cannot silently drop these guarantees. Covers:

* ``x-opencode-*`` session-header injection (feature 3)
* mid-conversation system-role policy: latest_reminder for DeepSeek-family
  models (even proxied through gateway plans), drop everywhere else
* empty ``reasoning_content`` churn gate in the shared stream assembler
  (feature 10)
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import parse_sse_text
from free_claude_code.core.session_id import claude_to_opencode_session_id
from free_claude_code.providers.openai_chat import (
    OpenAIChatProfile,
    OpenAIChatProvider,
    OpenAIChatRequestPolicy,
)
from free_claude_code.providers.openai_chat.reasoning import NO_REASONING
from free_claude_code.providers.opencode import create_opencode_provider
from tests.providers.support import (
    immediate_admission,
    make_provider_config,
    reasoning_for,
)


def _config():
    return make_provider_config(
        api_key="test_opencode_key",
        base_url="https://example.invalid/v1",
        rate_limit=100,
        rate_window=1,
    )


def _opencode_provider(provider_id: str) -> OpenAIChatProvider:
    """Build a real OpenCode provider without touching the network."""
    with (
        patch("free_claude_code.providers.openai_chat.provider.AsyncOpenAI"),
        patch("httpx.AsyncClient"),
    ):
        return create_opencode_provider(
            provider_id,
            _config(),
            immediate_admission(provider_name=provider_id),
        )


def _chat_body(provider: OpenAIChatProvider, payload: dict) -> dict:
    request = MessagesRequest.model_validate(payload)
    return provider._build_request_body(request, reasoning=reasoning_for(request))


_SYSTEM_POLICY_MESSAGES = [
    {"role": "user", "content": "hi"},
    {"role": "system", "content": "<system-reminder>tick</system-reminder>"},
]


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_mid_conversation_system_dropped_for_non_deepseek_model(
    provider_id: str,
) -> None:
    body = _chat_body(
        _opencode_provider(provider_id),
        {
            "model": "qwen3.8-flash",
            "system": "Conversation-wide instructions",
            "max_tokens": 100,
            "messages": _SYSTEM_POLICY_MESSAGES,
        },
    )

    assert body["messages"] == [
        {"role": "system", "content": "Conversation-wide instructions"},
        {"role": "user", "content": "hi"},
    ]


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_mid_conversation_system_becomes_latest_reminder_for_deepseek_model(
    provider_id: str,
) -> None:
    body = _chat_body(
        _opencode_provider(provider_id),
        {
            "model": "deepseek-v4-flash",
            "system": "Conversation-wide instructions",
            "max_tokens": 100,
            "messages": _SYSTEM_POLICY_MESSAGES,
        },
    )

    assert body["messages"] == [
        {"role": "system", "content": "Conversation-wide instructions"},
        {"role": "user", "content": "hi"},
        {
            "role": "latest_reminder",
            "content": "<system-reminder>tick</system-reminder>",
        },
    ]


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("deepseek-v4-flash", {"role": "latest_reminder", "content": "Reminder"}),
        ("DeepSeek/deepseek-chat", {"role": "latest_reminder", "content": "Reminder"}),
        ("qwen3.8-flash", None),
    ],
)
def test_gateway_model_name_selects_mid_conversation_behavior(
    model: str,
    expected: dict | None,
) -> None:
    """Gateway model names alone select the behavior on an OpenCode plan.

    DeepSeek-family model names keep the native ``latest_reminder`` role even
    through a non-DeepSeek gateway profile; other names drop the message (the
    top-level system prompt stays at index zero).
    """
    body = _chat_body(
        _opencode_provider("opencode_go"),
        {
            "model": model,
            "system": "S",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "Reminder"},
            ],
        },
    )

    assert body["messages"][0] == {"role": "system", "content": "S"}
    assert body["messages"][1] == {"role": "user", "content": "Hello"}
    if expected is None:
        assert len(body["messages"]) == 2
    else:
        assert body["messages"][2] == expected


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_opencode_session_headers_forward_mapped_session_id(
    provider_id: str,
) -> None:
    body = _chat_body(
        _opencode_provider(provider_id),
        {
            "model": "some-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "fcc_session_id": "session-abc",
        },
    )

    assert body["extra_headers"] == {
        "x-opencode-client": "fcc",
        "x-opencode-session": claude_to_opencode_session_id("session-abc"),
    }


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_opencode_session_headers_include_request_id(provider_id: str) -> None:
    body = _chat_body(
        _opencode_provider(provider_id),
        {
            "model": "some-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "fcc_session_id": "session-abc",
            "fcc_request_id": "req-123",
        },
    )

    assert body["extra_headers"]["x-opencode-request"] == "req-123"


def test_non_opencode_profile_does_not_inject_session_headers() -> None:
    provider = OpenAIChatProvider(
        _config(),
        profile=OpenAIChatProfile(
            OpenAIChatRequestPolicy(
                provider_name="NOT_OPENCODE",
                reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
            ),
            NO_REASONING,
        ),
        admission=immediate_admission(provider_name="NOT_OPENCODE"),
    )

    body = _chat_body(
        provider,
        {
            "model": "grok",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
            "fcc_session_id": "session-abc",
        },
    )

    assert "extra_headers" not in body


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
def test_reasoning_delta_preserves_empty_field_as_onset_signal(
    provider_id: str,
) -> None:
    profile = _opencode_provider(provider_id)._profile

    assert profile.reasoning_delta(SimpleNamespace(reasoning_content="think")) == (
        "think"
    )
    # Present-but-empty stays an explicit onset signal (""), not None; the
    # assembler decides when that signal may open a thinking block.
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
async def test_stream_keeps_one_text_block_when_content_chunks_carry_empty_reasoning() -> (
    None
):
    """Regression: relays that echo ``reasoning_content: ""`` on every chunk
    must not churn content blocks (the OpenCode qwen gateway symptom).

    Drives the shared ``_OpenAIChatStreamAssembler`` through a base
    OpenAI-chat provider because the gate lives in the assembler that every
    OpenAI-chat provider (OpenCode included) uses.
    """
    provider = OpenAIChatProvider(
        _config(),
        profile=OpenAIChatProfile(
            OpenAIChatRequestPolicy(
                provider_name="OPENCODE_GO",
                reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
            ),
            NO_REASONING,
        ),
        admission=immediate_admission(provider_name="OPENCODE_GO"),
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
        events = [
            event
            async for event in provider.stream_messages(
                request, reasoning=reasoning_for(request)
            )
        ]

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
