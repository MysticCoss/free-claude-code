"""Fork regression coverage for Claude Desktop 3P model-id compatibility.

Kept in its own file so upstream rewrites of ``test_model_listing.py`` or
``test_routing.py`` cannot drop it. Covers the desktop-safe id codec
interop with every catalog provider, the dedicated-port request detection,
the /v1/models desktop view, the supervisor listener plan, and inbound
routing. The hex id scheme itself is upstream's; this file pins the fork's
integration of it (second listener, port default view, [1m] variants).
"""

import http.client
import re
import socket
import threading
import time

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.types import Receive, Scope, Send

from free_claude_code.api.dependencies import is_claude_desktop_request
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.routing import ModelRouter
from free_claude_code.cli import commands as cli_commands
from free_claude_code.cli.commands import ServerSupervisor, desktop_listener_port
from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.provider_catalog import SUPPORTED_PROVIDER_IDS
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import (
    decode_gateway_model_id,
    desktop_model_id,
)
from tests.api.support import create_test_app, provider_manager_for_app

_PROVIDER_IDS = frozenset(SUPPORTED_PROVIDER_IDS)
_DESKTOP_BASE_URL = "http://testserver:8083"
_MAIN_BASE_URL = "http://testserver"


def _settings(
    *,
    desktop: bool,
    model: str = "deepseek/deepseek-chat",
    model_sonnet: str | None = None,
    model_opus: str | None = None,
    model_haiku: str | None = None,
    model_fable: str | None = None,
    model_compact: str | None = None,
    fcc_1m_models: str | None = None,
    port: int = 8082,
    claude_desktop_port: int = 8083,
) -> Settings:
    return Settings.model_construct(
        model=model,
        model_fable=model_fable,
        model_opus=model_opus,
        model_sonnet=model_sonnet,
        model_haiku=model_haiku,
        model_compact=model_compact,
        model_fallbacks=None,
        fcc_1m_models=fcc_1m_models,
        enable_claude_desktop_3p=desktop,
        claude_desktop_port=claude_desktop_port,
        port=port,
        proxy_auth_enabled=False,
        proxy_auth_token="freecc",
        deepseek_api_key="deepseek-key",
        groq_api_key="groq-key",
        open_router_api_key="open-router-key",
    )


@pytest.mark.parametrize("provider_id", sorted(_PROVIDER_IDS))
def test_every_catalog_provider_round_trips(provider_id: str) -> None:
    # The hex desktop ids must decode cleanly for every catalog provider and
    # pass Desktop's discovery name filter — the fork's desktop view serves
    # exactly these ids on the dedicated port.
    ref = f"{provider_id}/some-model"
    model_id = desktop_model_id(ref)
    assert _desktop_name_filter_passes(model_id)
    decoded = decode_gateway_model_id(model_id)
    assert decoded is not None
    assert decoded.provider_id == provider_id
    assert decoded.provider_model == "some-model"
    assert not decoded.force_reasoning_off
    no_thinking_id = desktop_model_id(ref, no_thinking=True)
    assert _desktop_name_filter_passes(no_thinking_id)
    no_thinking = decode_gateway_model_id(no_thinking_id)
    assert no_thinking is not None
    assert no_thinking.provider_id == provider_id
    assert no_thinking.provider_model == "some-model"
    assert no_thinking.force_reasoning_off


@pytest.mark.parametrize(
    "model_name",
    [
        "claude-sonnet-4-20250514",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
        "claude-fable-5",
        "claude-haiku-4-5-20251001",
        "claude-not-a-provider-model",
        "claude-deepseek-",
        "deepseek-v4-flash",
    ],
)
def test_non_desktop_ids_are_not_decoded(model_name: str) -> None:
    # Plain claude-* aliases and bare model names carry no provider segment
    # in the desktop/gateway id schemes, so the decoder passes them through
    # as None and they route through the alias/shortcut machinery instead.
    assert decode_gateway_model_id(model_name) is None


def _request_from_port(port: int | None) -> Request:
    scope: dict = {"type": "http", "method": "GET", "path": "/v1/models", "headers": []}
    if port is not None:
        scope["server"] = ("127.0.0.1", port)
    return Request(scope)


@pytest.mark.parametrize(
    ("port", "expected"),
    [
        (8083, True),
        (8082, False),
        (443, False),
    ],
)
def test_is_claude_desktop_request_matches_listener_port(
    port: int, expected: bool
) -> None:
    settings = _settings(desktop=True)

    assert is_claude_desktop_request(_request_from_port(port), settings) is expected


def test_is_claude_desktop_request_requires_feature_enabled() -> None:
    assert (
        is_claude_desktop_request(_request_from_port(8083), _settings(desktop=False))
        is False
    )


def test_is_claude_desktop_request_handles_missing_server_scope() -> None:
    settings = _settings(desktop=True)

    assert is_claude_desktop_request(_request_from_port(None), settings) is False


def test_is_claude_desktop_request_honors_custom_desktop_port() -> None:
    settings = _settings(desktop=True, claude_desktop_port=9099)

    assert is_claude_desktop_request(_request_from_port(9099), settings) is True


@pytest.mark.parametrize(
    ("desktop", "expected"),
    [
        (False, None),
        (True, 8083),
    ],
)
def test_desktop_listener_plan(desktop: bool, expected: int | None) -> None:
    assert desktop_listener_port(_settings(desktop=desktop)) is expected


def test_desktop_listener_refused_when_ports_collide() -> None:
    # model_construct bypasses validators; the plan must still refuse a clash.
    settings = _settings(desktop=True, port=8083, claude_desktop_port=8083)

    assert desktop_listener_port(settings) is None


def test_settings_reject_same_port_when_desktop_enabled() -> None:
    with pytest.raises(ValidationError, match="CLAUDE_DESKTOP_PORT"):
        Settings(
            port=8083,
            claude_desktop_port=8083,
            enable_claude_desktop_3p=True,
        )

    settings = Settings(
        port=8083,
        claude_desktop_port=8083,
        enable_claude_desktop_3p=False,
    )
    assert settings.claude_desktop_port == 8083


def _get_model_entries(
    settings: Settings,
    cache: dict[str, list[str]],
    base_url: str = _MAIN_BASE_URL,
) -> list[dict[str, object]]:
    app = create_test_app(settings)
    for provider_id, model_ids in cache.items():
        provider_manager_for_app(app).cache_model_infos(
            provider_id,
            {ProviderModelInfo(model_id) for model_id in model_ids},
        )
    response = TestClient(app, base_url=base_url).get("/v1/models")
    assert response.status_code == 200
    return list(response.json()["data"])


def _get_model_ids(
    settings: Settings,
    cache: dict[str, list[str]],
    base_url: str = _MAIN_BASE_URL,
) -> list[str]:
    return [str(entry["id"]) for entry in _get_model_entries(settings, cache, base_url)]


# Claude Desktop 3P discovery keeps a /v1/models entry only when its id
# passes a "recognizably Claude" name check: the lowercase id must contain no
# third-party vendor token (blacklist wins even when the id starts with
# "claude-"), and it must contain a family/vendor name. The gateway id codec
# makes every advertised desktop id pass on its name alone.
# The constants below replicate the filter shipped in Claude Desktop 1.46388.
_DESKTOP_FAMILY_TIERS = ("sonnet", "opus", "haiku", "fable", "mythos")
_DESKTOP_VENDOR_BLACKLIST = re.compile(
    r"ark-code|astron|command-r|deepseek|doubao|gemini|gemma|glm|gpt|grok|hermes|hy3|kimi|lfm"
    r"|\bling\b|llama|longcat|mimo|minimax|mistral|mixtral|moonshot|nemotron|openai|phi-|qianfan"
    r"|qwen|tc-code|\bunic\b|yi-|stepfun|step-3|seed-|bytedance|hunyuan|granite|amazon\.nova"
    r"|nova-|devstral|ministral|ernie|codex|arcee|trinity|abab|phi\d|\bk2\.|\bm2\.|jamba|arctic"
    r"|solar|mercury|zamba|kat-coder|\bds-|dpsk"
)


def _desktop_name_filter_passes(model_id: str) -> bool:
    lowered = model_id.lower()
    if _DESKTOP_VENDOR_BLACKLIST.search(lowered):
        return False
    return any(
        token in lowered for token in ("claude", *_DESKTOP_FAMILY_TIERS, "anthropic")
    )


def test_desktop_catalog_entries_all_survive_desktop_discovery_filter() -> None:
    entries = _get_model_entries(
        _settings(desktop=True),
        {"open_router": ["deepseek/deepseek-v4-pro", "qwen/qwen3.8-flash"]},
        base_url=_DESKTOP_BASE_URL,
    )

    assert entries
    rejected = [
        str(entry["id"])
        for entry in entries
        if not _desktop_name_filter_passes(str(entry["id"]))
    ]
    assert rejected == []


def test_desktop_ids_encode_blacklisted_vendor_names() -> None:
    # Desktop's blacklist wins over the claude-substring pass, so raw
    # vendor-carrying ids would be filtered out — the desktop view must
    # serve the encoded hex ids instead, which pass on the name alone.
    assert not _desktop_name_filter_passes("claude-deepseek-deepseek-chat")
    assert not _desktop_name_filter_passes(
        "claude-3-freecc-no-thinking/open_router/qwen/qwen3.8-flash"
    )
    assert _desktop_name_filter_passes(desktop_model_id("deepseek/deepseek-chat"))
    assert _desktop_name_filter_passes(
        desktop_model_id("open_router/qwen/qwen3.8-flash", no_thinking=True)
    )

    entries = _get_model_entries(
        _settings(desktop=True),
        {"open_router": ["qwen/qwen3.8-flash"]},
        base_url=_DESKTOP_BASE_URL,
    )
    by_id = {str(entry["id"]): entry for entry in entries}
    assert all("anthropic_family_tier" not in entry for entry in entries)
    for model_id in (
        desktop_model_id("deepseek/deepseek-chat"),
        desktop_model_id("open_router/qwen/qwen3.8-flash"),
        desktop_model_id("deepseek/deepseek-chat", no_thinking=True),
        desktop_model_id("open_router/qwen/qwen3.8-flash", no_thinking=True),
    ):
        assert model_id in by_id, model_id
    for raw_id in (
        "anthropic/deepseek/deepseek-chat",
        "anthropic/open_router/qwen/qwen3.8-flash",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/open_router/qwen/qwen3.8-flash",
    ):
        assert raw_id not in by_id, raw_id


def test_main_port_catalog_keeps_raw_ids_without_tier_field() -> None:
    entries = _get_model_entries(
        _settings(desktop=True),
        {"open_router": ["qwen/qwen3.8-flash"]},
    )

    assert entries
    assert all("anthropic_family_tier" not in entry for entry in entries)
    ids = {str(entry["id"]) for entry in entries}
    assert "anthropic/deepseek/deepseek-chat" in ids
    assert "anthropic/open_router/qwen/qwen3.8-flash" in ids
    # The encoded desktop ids stay exclusive to the desktop view.
    assert desktop_model_id("deepseek/deepseek-chat") not in ids
    assert desktop_model_id("open_router/qwen/qwen3.8-flash") not in ids


def test_disabled_desktop_port_catalog_keeps_main_port_form() -> None:
    entries = _get_model_entries(
        _settings(desktop=False),
        {"open_router": ["qwen/qwen3.8-flash"]},
        base_url=_DESKTOP_BASE_URL,
    )

    assert entries
    assert all("anthropic_family_tier" not in entry for entry in entries)
    ids = {str(entry["id"]) for entry in entries}
    assert "anthropic/open_router/qwen/qwen3.8-flash" in ids
    assert desktop_model_id("open_router/qwen/qwen3.8-flash") not in ids


def test_desktop_listener_advertises_claude_prefixed_ids() -> None:
    ids = _get_model_ids(
        _settings(desktop=True),
        {"open_router": ["meta/llama-3.3"]},
        base_url=_DESKTOP_BASE_URL,
    )

    assert desktop_model_id("deepseek/deepseek-chat") in ids
    assert desktop_model_id("open_router/meta/llama-3.3") in ids
    assert desktop_model_id("open_router/meta/llama-3.3[1m]") not in ids
    assert not any(model_id.startswith("anthropic/") for model_id in ids)
    # The no-thinking variant shares the desktop prefix scheme.
    assert desktop_model_id("deepseek/deepseek-chat", no_thinking=True) in ids
    # Genuine Claude aliases are untouched.
    assert "claude-sonnet-4-20250514" in ids


def test_desktop_listener_prefixes_1m_variants() -> None:
    ids = _get_model_ids(
        _settings(desktop=True, fcc_1m_models="deepseek/deepseek-chat"),
        {},
        base_url=_DESKTOP_BASE_URL,
    )

    assert desktop_model_id("deepseek/deepseek-chat[1m]") in ids


def test_main_port_stays_normal_while_desktop_enabled() -> None:
    ids = _get_model_ids(
        _settings(desktop=True),
        {"open_router": ["meta/llama-3.3"]},
    )

    assert "anthropic/deepseek/deepseek-chat" in ids
    assert "anthropic/open_router/meta/llama-3.3" in ids
    assert "claude-deepseek-deepseek-chat" not in ids
    assert desktop_model_id("deepseek/deepseek-chat") not in ids


def test_desktop_port_normal_when_feature_disabled() -> None:
    ids = _get_model_ids(_settings(desktop=False), {}, base_url=_DESKTOP_BASE_URL)

    assert "anthropic/deepseek/deepseek-chat" in ids
    assert "claude-deepseek-deepseek-chat" not in ids


def test_router_routes_desktop_hex_id() -> None:
    router = ModelRouter(_settings(desktop=False, model="groq/llama-3.3-70b"))

    model_id = desktop_model_id("deepseek/deepseek-v4-flash")
    resolved = router.resolve(model_id)

    assert resolved.original_model == model_id
    assert resolved.primary.provider_id == "deepseek"
    assert resolved.primary.provider_model == "deepseek-v4-flash"
    assert resolved.primary.provider_model_ref == "deepseek/deepseek-v4-flash"


def test_router_strips_1m_suffix_from_desktop_id() -> None:
    router = ModelRouter(_settings(desktop=True))

    resolved = router.resolve(desktop_model_id("deepseek/deepseek-v4-flash[1m]"))

    assert resolved.primary.provider_model == "deepseek-v4-flash"


def test_router_strips_1m_suffix_from_plain_ref() -> None:
    router = ModelRouter(_settings(desktop=True))

    resolved = router.resolve("deepseek/deepseek-v4-flash[1m]")

    assert resolved.primary.provider_id == "deepseek"
    assert resolved.primary.provider_model == "deepseek-v4-flash"


def test_router_routes_no_thinking_desktop_id() -> None:
    router = ModelRouter(_settings(desktop=True))

    resolved = router.resolve(
        desktop_model_id("open_router/qwen/qwen3.8-flash", no_thinking=True)
    )

    assert resolved.primary.provider_id == "open_router"
    assert resolved.primary.provider_model == "qwen/qwen3.8-flash"
    assert resolved.reasoning_preference is ReasoningPreference.OFF


def test_router_keeps_existing_id_forms_working_alongside_desktop_ids() -> None:
    router = ModelRouter(_settings(desktop=True, model_sonnet="deepseek/deepseek-chat"))

    gateway = router.resolve("anthropic/deepseek/deepseek-v4-flash")
    no_thinking = router.resolve(
        "claude-3-freecc-no-thinking/deepseek/deepseek-v4-flash"
    )
    desktop = router.resolve(desktop_model_id("deepseek/deepseek-v4-flash"))
    sonnet = router.resolve("claude-sonnet-4-20250514")

    assert gateway.primary.provider_model == "deepseek-v4-flash"
    assert no_thinking.primary.provider_model == "deepseek-v4-flash"
    assert desktop.primary.provider_id == "deepseek"
    assert desktop.primary.provider_model == "deepseek-v4-flash"
    assert sonnet.primary.provider_model == "deepseek-chat"


def test_desktop_admin_fields_require_restart() -> None:
    # The desktop listener exists only from a supervisor generation start,
    # so Admin Apply must trigger the automatic restart for both fields;
    # without restart_required the toggle would silently do nothing until a
    # manual process restart.
    assert FIELD_BY_KEY["ENABLE_CLAUDE_DESKTOP_3P"].restart_required is True
    assert FIELD_BY_KEY["CLAUDE_DESKTOP_PORT"].restart_required is True


async def _ok_asgi_app(scope: Scope, receive: Receive, send: Send) -> None:
    """Minimal ASGI app with a lifespan handshake for listener-lifecycle tests."""

    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
        return
    assert scope["type"] == "http"
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": b"desktop-ok"})


def _ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _listener_settings(desktop_port: int) -> Settings:
    return _settings(
        desktop=True, port=_ephemeral_port(), claude_desktop_port=desktop_port
    )


def _await_http_ok(port: int, timeout_s: float = 10.0) -> None:
    """Poll until the desktop listener serves HTTP, or fail loudly."""

    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            try:
                connection.request("GET", "/")
                response = connection.getresponse()
                body = response.read()
            finally:
                connection.close()
            if response.status == 200 and body == b"desktop-ok":
                return
        except Exception as exc:
            last_error = exc
            time.sleep(0.05)
    raise AssertionError(f"desktop listener never served on {port}: {last_error!r}")


def _hold_port(port: int) -> socket.socket:
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("0.0.0.0", port))
    holder.listen(16)
    return holder


def test_desktop_listener_serves_when_port_free() -> None:
    supervisor = ServerSupervisor(console_logging=False)
    settings = _listener_settings(_ephemeral_port())
    server, thread = supervisor._start_desktop_listener(_ok_asgi_app, settings)
    try:
        assert server is not None and thread is not None
        _await_http_ok(settings.claude_desktop_port)
    finally:
        supervisor._stop_desktop_listener(server, thread)
    assert not thread.is_alive()


def test_desktop_listener_retries_contended_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repro: the previous generation's listener still holds the desktop port.

    The old code spawned the serving thread without reserving the port, so
    the bind died on EADDRINUSE while the logs already claimed "starting" —
    first run with the feature enabled came up with a dead 8083 until the
    next toggle/restart freed the port.
    """

    monkeypatch.setattr(cli_commands, "DESKTOP_LISTENER_BIND_ATTEMPTS", 40)
    monkeypatch.setattr(cli_commands, "DESKTOP_LISTENER_BIND_RETRY_SECONDS", 0.05)
    port = _ephemeral_port()
    holder = _hold_port(port)
    releaser = threading.Timer(0.3, holder.close)
    releaser.daemon = True
    releaser.start()
    supervisor = ServerSupervisor(console_logging=False)
    server, thread = supervisor._start_desktop_listener(
        _ok_asgi_app, _listener_settings(port)
    )
    try:
        assert server is not None and thread is not None
        _await_http_ok(port)
    finally:
        supervisor._stop_desktop_listener(server, thread)
        holder.close()


def test_desktop_listener_gives_up_gracefully_when_port_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permanently-held port must not raise and must not kill the main server."""

    monkeypatch.setattr(cli_commands, "DESKTOP_LISTENER_BIND_ATTEMPTS", 2)
    monkeypatch.setattr(cli_commands, "DESKTOP_LISTENER_BIND_RETRY_SECONDS", 0.01)
    holder = _hold_port(_ephemeral_port())
    try:
        supervisor = ServerSupervisor(console_logging=False)
        server, thread = supervisor._start_desktop_listener(
            _ok_asgi_app, _listener_settings(holder.getsockname()[1])
        )
        assert (server, thread) == (None, None)
    finally:
        holder.close()


def test_desktop_fields_live_in_runtime_section() -> None:
    # The 3P toggle + port belong with the server-process settings (next to
    # PORT), not with model routing: they control a second listener, not
    # which model a route resolves to.
    assert FIELD_BY_KEY["ENABLE_CLAUDE_DESKTOP_3P"].section_id == "runtime"
    assert FIELD_BY_KEY["CLAUDE_DESKTOP_PORT"].section_id == "runtime"
