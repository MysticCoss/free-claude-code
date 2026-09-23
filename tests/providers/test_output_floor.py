"""Tests for upstream output-token minimum recovery.

Covers the pure parse/raise helpers and the Responses transport behavior that
raises ``max_output_tokens`` to the upstream-required minimum, retries once,
and succeeds (e.g. muse-spark via OpenCode Go requires ``>= 16``).
"""

import json
from collections.abc import Mapping

import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.anthropic.stream_contracts import (
    parse_sse_text,
    text_content,
)
from free_claude_code.core.openai_responses import ResponsesToolPolicy
from free_claude_code.providers.openai_responses import OpenAIResponsesTransport
from free_claude_code.providers.output_floor import (
    parse_output_token_floor,
    raise_output_tokens,
)
from tests.providers.support import REASONING_ON, immediate_admission


class _BadRequest(Exception):
    """Stand-in for openai.BadRequestError (status_code + optional JSON body)."""

    def __init__(self, message: str, body: object | None = None):
        super().__init__(message)
        self.status_code = 400
        self.body = body


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_parse_floor_from_observed_message():
    error = _BadRequest(
        "Upstream request failed: [invalid_request_error] `max_output_tokens` "
        "The number must be `>= 16`.",
        body={
            "param": "max_output_tokens",
            "type": "invalid_request_error",
            "message": "Upstream request failed: [invalid_request_error] "
            "`max_output_tokens` The number must be `>= 16`.",
        },
    )
    assert parse_output_token_floor(error) == 16


@pytest.mark.parametrize(
    "message,expected",
    [
        ("max_output_tokens must be >= 16", 16),
        ("`max_output_tokens` should be >= `32`.", 32),
        ("max_output_tokens must be greater than or equal to 64", 64),
        ("max_output_tokens must be at least 128", 128),
        ("minimum value is 16 for max_output_tokens", 16),
        ("minimum allowed value of 24 for max_tokens", 24),
        ("minimum of 48 for max_completion_tokens", 48),
    ],
)
def test_parse_floor_various_phrasings(message, expected):
    assert parse_output_token_floor(_BadRequest(message)) == expected


def test_parse_floor_reads_structured_param_scope():
    error = _BadRequest(
        "invalid request",
        body={"param": "max_output_tokens", "message": ">= 16"},
    )
    assert parse_output_token_floor(error) == 16


def test_parse_floor_selects_matching_parameter_from_structured_error_list():
    error = _BadRequest(
        "invalid request",
        body={
            "errors": [
                {
                    "param": "temperature",
                    "message": "max_output_tokens must be >= 2",
                },
                {"param": "max_output_tokens", "message": ">= 16"},
            ]
        },
    )
    assert parse_output_token_floor(error) == 16


def test_parse_floor_respects_structured_non_output_parameter():
    error = _BadRequest(
        "invalid request",
        body={
            "param": "temperature",
            "message": "max_output_tokens must be >= 2",
        },
    )
    assert parse_output_token_floor(error) is None


def test_parse_floor_ignores_text_outside_structured_error_schema():
    error = _BadRequest(
        "invalid request",
        body={"request": {"message": "max_output_tokens must be >= 16"}},
    )
    assert parse_output_token_floor(error) is None


def test_parse_floor_ignores_non_400():
    error = _BadRequest("max_output_tokens must be >= 16")
    error.status_code = 429
    assert parse_output_token_floor(error) is None


def test_parse_floor_ignores_unrelated_400():
    assert parse_output_token_floor(_BadRequest("temperature must be <= 2")) is None


def test_parse_floor_returns_none_without_number():
    assert (
        parse_output_token_floor(
            _BadRequest("max_output_tokens is smaller than allowed")
        )
        is None
    )


def test_parse_floor_does_not_bind_cap_comparators():
    assert (
        parse_output_token_floor(_BadRequest("max_output_tokens must be <= 4096"))
        is None
    )


def test_raise_increases_max_output_tokens():
    assert raise_output_tokens({"max_output_tokens": 8}, 16) == {
        "max_output_tokens": 16
    }


def test_raise_increases_chat_fields():
    assert raise_output_tokens({"max_tokens": 4}, 16) == {"max_tokens": 16}
    assert raise_output_tokens({"max_completion_tokens": 1}, 16) == {
        "max_completion_tokens": 16
    }


def test_raise_noop_when_at_floor_returns_none():
    assert raise_output_tokens({"max_output_tokens": 16}, 16) is None
    assert raise_output_tokens({"max_output_tokens": 64}, 16) is None


def test_raise_does_not_mutate_input():
    body = {"max_output_tokens": 8, "model": "m"}
    raised = raise_output_tokens(body, 16)
    assert body["max_output_tokens"] == 8
    assert raised is not None
    assert raised["max_output_tokens"] == 16


def test_raise_ignores_bool_values():
    assert raise_output_tokens({"max_output_tokens": True}, 16) is None


# --------------------------------------------------------------------------- #
# Responses transport integration
# --------------------------------------------------------------------------- #

_FLOOR_ERROR = {
    "param": "max_output_tokens",
    "type": "invalid_request_error",
    "message": "Upstream request failed: [invalid_request_error] "
    "`max_output_tokens` The number must be `>= 16`.",
}


def _sse(*events: Mapping[str, object]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events)


def _completed_sse() -> str:
    return _sse(
        {
            "type": "response.output_text.delta",
            "sequence_number": 0,
            "item_id": "item_text",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
            "logprobs": [],
        },
        {
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "resp_floor",
                "model": "upstream-model",
                "object": "response",
                "status": "completed",
                "output": [],
                "usage": {
                    "input_tokens": 8,
                    "output_tokens": 16,
                    "total_tokens": 24,
                },
            },
        },
    )


@pytest.mark.asyncio
async def test_responses_transport_raises_to_floor_and_retries():
    bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("max_output_tokens", 0) < 16:
            return httpx2.Response(400, json=_FLOOR_ERROR)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_completed_sse(),
        )

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    transport = OpenAIResponsesTransport(
        client=client,
        admission=immediate_admission(provider_name="TEST_FLOOR"),
        provider_name="TEST_FLOOR",
        read_timeout_s=120.0,
        log_raw_sse_events=False,
        tool_policy=ResponsesToolPolicy(),
    )
    try:
        output = [
            event
            async for event in transport.stream_messages(
                MessagesRequest.model_validate(
                    {
                        "model": "upstream-model",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 8,
                    }
                ),
                input_tokens=11,
                request_id="req_floor",
                response_model="public-model",
                reasoning=REASONING_ON,
            )
        ]
        assert text_content(parse_sse_text("".join(output))) == "ok"
        assert len(bodies) == 2
        assert bodies[0]["max_output_tokens"] == 8
        assert bodies[1]["max_output_tokens"] == 16
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_responses_transport_does_not_retry_compliant_body_twice():
    bodies: list[dict[str, object]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_completed_sse(),
        )

    client = AsyncOpenAI(
        api_key="test-key",
        base_url="https://provider.invalid/v1",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    transport = OpenAIResponsesTransport(
        client=client,
        admission=immediate_admission(provider_name="TEST_FLOOR"),
        provider_name="TEST_FLOOR",
        read_timeout_s=120.0,
        log_raw_sse_events=False,
        tool_policy=ResponsesToolPolicy(),
    )
    try:
        request = MessagesRequest.model_validate(
            {
                "model": "upstream-model",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 64,
            }
        )
        output = [
            event
            async for event in transport.stream_messages(
                request,
                input_tokens=11,
                request_id="req_floor_ok",
                response_model="public-model",
                reasoning=REASONING_ON,
            )
        ]
        assert text_content(parse_sse_text("".join(output))) == "ok"
        assert len(bodies) == 1
        assert bodies[0]["max_output_tokens"] == 64
    finally:
        await client.close()
