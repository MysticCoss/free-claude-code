"""Session identity across real ingress, routing, provider, and SDK boundaries."""

import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import patch

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from free_claude_code.config.settings import Settings
from free_claude_code.core.session_id import claude_to_opencode_session_id
from free_claude_code.providers.opencode import create_opencode_provider
from free_claude_code.providers.opencode.user_agent import opencode_user_agent
from tests.api.support import create_test_app, provider_manager_for_app
from tests.providers.support import immediate_admission, make_provider_config
from tests.providers.test_opencode import (
    _catalog_payload,
    _chat_event_stream,
    _responses_event_stream,
)

pytestmark = pytest.mark.asyncio

VALID_SESSION = "ses_98cd96b77333egPHPVV7mm2Css"
NATIVE_SESSION = claude_to_opencode_session_id("native-session")
CONVERSATION_A = claude_to_opencode_session_id("conversation-a")
CONVERSATION_B = claude_to_opencode_session_id("conversation-b")
CONVERSATION_C = claude_to_opencode_session_id("conversation-c")


def _ua() -> str:
    return opencode_user_agent()


def successful_response(request):
    body = (
        _responses_event_stream("hello")
        if request.url.path.endswith("/responses")
        else _chat_event_stream("hello")
    )
    return httpx2.Response(
        200, headers={"content-type": "text/event-stream"}, text=body
    )


@asynccontextmanager
async def wire_client(provider_id="opencode_zen", handler=None):
    requests = []
    catalog_requests = []

    async def generation(request):
        requests.append(request)
        if handler is not None:
            return await handler(request)
        if not request.headers.get("x-opencode-session"):
            return httpx2.Response(
                400,
                json={
                    "type": "MissingSessionID",
                    "message": "Missing x-opencode-session",
                },
            )
        return successful_response(request)

    def catalog(request):
        catalog_requests.append(request)
        return httpx.Response(
            200,
            json=_catalog_payload(
                provider_key="opencode-go"
                if provider_id == "opencode_go"
                else "opencode"
            ),
        )

    def sdk(**kwargs):
        kwargs["http_client"] = httpx2.AsyncClient(
            transport=httpx2.MockTransport(generation)
        )
        return AsyncOpenAI(**kwargs)

    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI", side_effect=sdk
    ):
        provider = create_opencode_provider(
            provider_id,
            make_provider_config(
                api_key="test_opencode_key",
                base_url="https://opencode.ai/zen/v1",
            ),
            immediate_admission(provider_name=provider_id),
            catalog_client=httpx.AsyncClient(transport=httpx.MockTransport(catalog)),
        )
    app = create_test_app(
        Settings(
            model=f"{provider_id}/responses-selector",
            opencode_api_key="test_opencode_key",
            proxy_auth_enabled=True,
            proxy_auth_token="fcc-private-token",
        ),
        providers={provider_id: provider},
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://fcc.test",
            headers={"Authorization": "Bearer fcc-private-token"},
        ) as client:
            yield client, provider, requests, catalog_requests
    finally:
        await provider_manager_for_app(app).close()


def payload(ingress, provider_id, selector):
    request = {"model": f"{provider_id}/{selector}", "stream": True}
    if ingress == "responses":
        request["input"] = "hello"
    else:
        request.update(messages=[{"role": "user", "content": "hello"}], max_tokens=128)
    return request


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize("ingress", ["responses", "messages"])
@pytest.mark.parametrize(
    "header_name,user_agent",
    [
        ("User-Agent", _ua()),
        ("uSeR-aGeNt", "claude-cli/2.1.0"),
        ("User-Agent", "codex-cli/1.0"),
    ],
)
async def test_upstream_always_receives_first_party_versioned_user_agent(
    provider_id, selector, ingress, header_name, user_agent
):
    assert opencode_user_agent() == "opencode/1.18.32"

    async def upstream(request):
        if request.headers.get_list("user-agent") != [_ua()]:
            return httpx2.Response(
                403,
                json={
                    "type": "FreeTierError",
                    "message": "OpenCode's free tier can only be used from within OpenCode",
                },
            )
        return successful_response(request)

    async with wire_client(provider_id, upstream) as (
        client,
        provider,
        requests,
        catalogs,
    ):
        response = await client.post(
            f"/v1/{ingress}",
            json=payload(ingress, provider_id, selector),
            headers={header_name: user_agent, "X-Session-Id": "native-session"},
        )
        assert response.status_code == 200, response.text
        assert requests[-1].headers.get_list("user-agent") == [_ua()]
        assert requests[-1].headers["x-opencode-session"] == NATIVE_SESSION
        assert requests[-1].headers["authorization"] == "Bearer test_opencode_key"
        assert provider._client.default_headers["User-Agent"] == _ua()
        assert all(
            request.headers.get_list("user-agent") == [_ua()] for request in catalogs
        )


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize("user_agent", [None, b"", b" \t ", b"client/caf\xe9"])
async def test_unusable_client_user_agent_still_sends_first_party_value(
    provider_id, selector, user_agent
):
    async with wire_client(provider_id) as (client, provider, requests, _catalogs):
        request = payload("responses", provider_id, selector)
        previous = await client.post(
            "/v1/responses",
            json=request,
            headers={"User-Agent": "claude-cli/2.1.0", "session-id": "previous"},
        )
        assert previous.status_code == 200, previous.text
        headers = [
            (b"Authorization", b"Bearer fcc-private-token"),
            (b"Session-Id", b"current"),
        ]
        if user_agent is not None:
            headers.append((b"User-Agent", user_agent))
        # A raw Request avoids the API client's automatic HTTPX User-Agent.
        response = await client.send(
            httpx.Request(
                "POST", "http://fcc.test/v1/responses", headers=headers, json=request
            )
        )
        assert response.status_code == 200, response.text
        assert requests[-1].headers.get_list("user-agent") == [_ua()]
        assert requests[-1].headers["x-opencode-session"] == (
            claude_to_opencode_session_id("current")
        )
        assert provider._client.default_headers["User-Agent"] == _ua()


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize("ingress", ["responses", "messages"])
async def test_existing_session_id_reaches_opencode_without_forwarding_other_headers(
    provider_id, selector, ingress
):
    async with wire_client(provider_id) as (client, provider, requests, catalogs):
        defaults = {
            name: value
            for name, value in provider._client.default_headers.items()
            if isinstance(value, str)
        }
        for name in (
            "Session-Id",
            "X-OpenCode-Session",
            "X-Session-Id",
            "X-Claude-Code-Session-Id",
            "session_id",
            "X-Grok-Session-Id",
            "X-Meta-Ai-Gateway-Session-Id",
            "X-Tbh-Session-Id",
        ):
            for conversation in ("conversation-a", "conversation-a", "conversation-b"):
                response = await client.post(
                    f"/v1/{ingress}",
                    json=payload(ingress, provider_id, selector),
                    headers=[
                        (name.encode(), conversation.encode()),
                        (b"X-Fcc-Launch-Id", b"launcher-fallback"),
                        (b"Cookie", b"private=cookie"),
                        (b"X-Private", b"caf\xe9"),
                    ],
                )
                assert response.status_code == 200, response.text
                upstream = requests[-1]
                assert upstream.headers["x-opencode-session"] == (
                    claude_to_opencode_session_id(conversation)
                )
                assert upstream.headers["authorization"] == "Bearer test_opencode_key"
                assert "cookie" not in upstream.headers
                assert "x-private" not in upstream.headers
                assert "x-fcc-launch-id" not in upstream.headers
        assert {
            name: value
            for name, value in provider._client.default_headers.items()
            if isinstance(value, str)
        } == defaults
        assert all("x-opencode-session" not in request.headers for request in catalogs)


@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
async def test_session_header_precedence_and_absence_fall_back_to_conversation_seed(
    selector,
):
    async with wire_client() as (client, _provider, requests, _catalogs):
        request = payload("responses", "opencode_zen", selector)
        response = await client.post(
            "/v1/responses",
            json=request,
            headers={
                "x-opencode-session": VALID_SESSION,
                "session-id": "native-session",
                "X-Claude-Code-Session-Id": "claude-session",
                "session_id": "dsh-session",
                "x-grok-session-id": "grok-session",
                "x-meta-ai-gateway-session-id": "muse-session",
                "x-fcc-launch-id": "launcher-fallback",
                "User-Agent": "claude-cli/2.1.0",
            },
        )
        assert response.status_code == 200, response.text
        assert requests[-1].headers["x-opencode-session"] == VALID_SESSION
        assert requests[-1].headers.get_list("user-agent") == [_ua()]
        response = await client.post(
            "/v1/responses",
            json={**request, "prompt_cache_key": "not-a-conversation"},
            headers={"User-Agent": _ua()},
        )
        assert response.status_code == 200, response.text
        assert "x-opencode-session" in requests[-1].headers
        assert requests[-1].headers["x-opencode-session"].startswith("ses_")
        assert requests[-1].headers.get_list("user-agent") == [_ua()]
        first_seed = requests[-1].headers["x-opencode-session"]
        response = await client.post(
            "/v1/responses",
            json={**request, "prompt_cache_key": "not-a-conversation"},
            headers={"User-Agent": _ua()},
        )
        assert response.status_code == 200, response.text
        assert requests[-1].headers["x-opencode-session"] == first_seed


@pytest.mark.parametrize("provider_id", ["opencode_zen", "opencode_go"])
@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize("ingress", ["responses", "messages"])
async def test_client_without_any_session_header_still_sends_conversation_seed(
    provider_id, selector, ingress
):
    async with wire_client(provider_id) as (client, _provider, requests, _catalogs):
        for _ in range(2):
            response = await client.post(
                f"/v1/{ingress}",
                json=payload(ingress, provider_id, selector),
                headers={"User-Agent": _ua()},
            )
            assert response.status_code == 200, response.text
        assert "x-opencode-session" in requests[-1].headers
        assert requests[-1].headers["x-opencode-session"].startswith("ses_")
        assert (
            requests[-1].headers["x-opencode-session"]
            == requests[-2].headers["x-opencode-session"]
        )


@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize("ingress", ["responses", "messages"])
async def test_minimal_tools_are_injected_when_client_sends_none(selector, ingress):
    async with wire_client() as (client, _provider, requests, _catalogs):
        response = await client.post(
            f"/v1/{ingress}",
            json=payload(ingress, "opencode_zen", selector),
            headers={"User-Agent": _ua()},
        )
        assert response.status_code == 200, response.text
        body = json.loads(requests[-1].content)
        tools = body.get("tools") or []
        names = set()
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if isinstance(tool.get("name"), str):
                names.add(tool["name"])
            function = tool.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.add(function["name"])
        assert {"bash", "edit", "glob", "grep", "read"} <= names


@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize("ingress", ["responses", "messages"])
async def test_capitalized_client_tools_gain_lowercase_free_tier_names(
    selector, ingress
):
    async with wire_client() as (client, _provider, requests, _catalogs):
        request = payload(ingress, "opencode_zen", selector)
        if ingress == "responses":
            request["tools"] = [
                {
                    "type": "function",
                    "name": "Bash",
                    "description": "Run a command",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "type": "function",
                    "name": "Read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            ]
        else:
            request["tools"] = [
                {
                    "name": "Bash",
                    "description": "Run a command",
                    "input_schema": {"type": "object", "properties": {}},
                },
                {
                    "name": "Read",
                    "description": "Read a file",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ]
        response = await client.post(
            f"/v1/{ingress}",
            json=request,
            headers={"User-Agent": _ua()},
        )
        assert response.status_code == 200, response.text
        body = json.loads(requests[-1].content)
        names = set()
        for tool in body.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            if isinstance(tool.get("name"), str):
                names.add(tool["name"])
            function = tool.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.add(function["name"])
        assert "Bash" in names
        assert "Read" in names
        assert {"bash", "read"} <= names


@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize("ingress", ["responses", "messages"])
async def test_non_conforming_session_values_map_deterministically(selector, ingress):
    async with wire_client() as (client, _provider, requests, _catalogs):
        for conversation in ("explicit-session", "explicit-session", "other-value"):
            response = await client.post(
                f"/v1/{ingress}",
                json=payload(ingress, "opencode_zen", selector),
                headers={"x-opencode-session": conversation},
            )
            assert response.status_code == 200, response.text
            assert requests[-1].headers["x-opencode-session"] == (
                claude_to_opencode_session_id(conversation)
            )


@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize("ingress", ["responses", "messages"])
async def test_launch_fallback_is_stable_until_a_native_conversation_id_is_available(
    selector, ingress
):
    async with wire_client() as (client, _provider, requests, _catalogs):
        for launch_id, native_id in (
            ("launch-a", ""),
            ("launch-a", ""),
            ("launch-a", "native-conversation"),
            ("launch-b", ""),
        ):
            response = await client.post(
                f"/v1/{ingress}",
                json=payload(ingress, "opencode_zen", selector),
                headers={"x-fcc-launch-id": launch_id, "session-id": native_id},
            )
            assert response.status_code == 200, response.text
            expected_source = native_id or launch_id
            assert requests[-1].headers["x-opencode-session"] == (
                claude_to_opencode_session_id(expected_source)
            )
            assert "x-fcc-launch-id" not in requests[-1].headers


@pytest.mark.parametrize("selector", ["responses-selector", "chat-selector"])
@pytest.mark.parametrize(
    "session_header",
    [
        "session-id",
        "X-Claude-Code-Session-Id",
        "session_id",
        "X-Grok-Session-Id",
        "X-Meta-Ai-Gateway-Session-Id",
        "X-Tbh-Session-Id",
        "x-fcc-launch-id",
    ],
)
async def test_concurrent_conversations_keep_their_session_ids_during_retry(
    selector, session_header
):
    first_arrived = asyncio.Event()
    second_arrived = asyncio.Event()
    attempts = []
    user_agents = []

    async def upstream(request):
        session = request.headers.get("x-opencode-session")
        attempts.append(session)
        user_agents.append(request.headers.get_list("user-agent"))
        if session == CONVERSATION_A and attempts.count(session) == 1:
            first_arrived.set()
            await second_arrived.wait()
            return httpx2.Response(503, json={"error": {"message": "Try again"}})
        if session == CONVERSATION_B:
            second_arrived.set()
        return successful_response(request)

    async with wire_client(handler=upstream) as (
        client,
        provider,
        requests,
        _catalogs,
    ):
        request = payload("responses", "opencode_zen", selector)
        async with asyncio.timeout(5):
            first = asyncio.create_task(
                client.post(
                    "/v1/responses",
                    json=request,
                    headers={
                        session_header: "conversation-a",
                        "User-Agent": _ua(),
                    },
                )
            )
            try:
                await first_arrived.wait()
                second = await client.post(
                    "/v1/responses",
                    json=request,
                    headers={
                        session_header: "conversation-b",
                        "User-Agent": "codex-cli/1.0",
                    },
                )
                assert second.status_code == 200, second.text
                result = await first
                assert result.status_code == 200, result.text
            finally:
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
        assert attempts == [CONVERSATION_A, CONVERSATION_B, CONVERSATION_A]
        assert len(requests) == 3
        assert user_agents == [
            [_ua()],
            [_ua()],
            [_ua()],
        ]
        response = await client.post(
            "/v1/responses",
            json=request,
            headers={session_header: "conversation-c", "User-Agent": ""},
        )
        assert response.status_code == 200, response.text
        assert requests[-1].headers.get_list("user-agent") == [_ua()]
        assert requests[-1].headers["x-opencode-session"] == CONVERSATION_C
        assert provider._client.default_headers["User-Agent"] == _ua()
