"""Fork regression coverage for Claude Desktop 3P model-id compatibility.

Kept in its own file so upstream rewrites of ``test_model_listing.py`` or
``test_routing.py`` cannot drop it. Covers the ``claude-<provider>-<model>``
id codec, the dedicated-port request detection, the /v1/models desktop view,
the supervisor listener plan, and inbound routing.
"""

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from free_claude_code.api.dependencies import is_claude_desktop_request
from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.routing import ModelRouter
from free_claude_code.cli.commands import desktop_listener_port
from free_claude_code.config.admin.manifest import FIELD_BY_KEY
from free_claude_code.config.provider_catalog import SUPPORTED_PROVIDER_IDS
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import (
    claude_desktop_model_id,
    decode_claude_desktop_model_id,
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
    fcc_1m_models: str | None = None,
    port: int = 8082,
    claude_desktop_port: int = 8083,
) -> Settings:
    return Settings.model_construct(
        model=model,
        model_fable=None,
        model_opus=None,
        model_sonnet=model_sonnet,
        model_haiku=None,
        model_compact=None,
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


def test_claude_desktop_model_id_requires_provider_ref() -> None:
    with pytest.raises(ValueError):
        claude_desktop_model_id("deepseek-chat")


@pytest.mark.parametrize(
    ("provider_id", "provider_model"),
    [
        ("deepseek", "deepseek-v4-flash"),
        ("kimi", "k2"),
        ("kimi_code", "k2-turbo"),
        ("open_router", "meta-llama/llama-3.3-70b-instruct"),
        ("ollama", "qwen2:7b"),
        ("ollama_cloud", "llama3.1"),
        ("opencode_go", "qwen3.8-flash"),
        ("deepseek", "deepseek-v4-flash[1m]"),
    ],
)
def test_claude_desktop_id_round_trips(provider_id: str, provider_model: str) -> None:
    model_id = claude_desktop_model_id(f"{provider_id}/{provider_model}")
    assert model_id.startswith("claude-")
    decoded = decode_claude_desktop_model_id(model_id, _PROVIDER_IDS)
    assert decoded is not None
    assert decoded.provider_id == provider_id
    assert decoded.provider_model == provider_model
    assert not decoded.force_reasoning_off


@pytest.mark.parametrize("provider_id", sorted(_PROVIDER_IDS))
def test_every_catalog_provider_round_trips(provider_id: str) -> None:
    model_id = claude_desktop_model_id(f"{provider_id}/some-model")
    decoded = decode_claude_desktop_model_id(model_id, _PROVIDER_IDS)
    assert decoded is not None
    assert decoded.provider_id == provider_id
    assert decoded.provider_model == "some-model"


@pytest.mark.parametrize(
    "model_name",
    [
        "claude-sonnet-4-20250514",
        "claude-3-haiku-20240307",
        "claude-3-5-sonnet-20241022",
        "claude-fable-5",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "anthropic/deepseek/deepseek-chat",
        "claude-not-a-provider-model",
        "claude-deepseek-",
        "deepseek-v4-flash",
    ],
)
def test_non_desktop_ids_are_not_decoded(model_name: str) -> None:
    assert decode_claude_desktop_model_id(model_name, _PROVIDER_IDS) is None


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


def _get_model_ids(
    settings: Settings,
    cache: dict[str, list[str]],
    base_url: str = _MAIN_BASE_URL,
) -> list[str]:
    app = create_test_app(settings)
    for provider_id, model_ids in cache.items():
        provider_manager_for_app(app).cache_model_infos(
            provider_id,
            {ProviderModelInfo(model_id) for model_id in model_ids},
        )
    response = TestClient(app, base_url=base_url).get("/v1/models")
    assert response.status_code == 200
    return [item["id"] for item in response.json()["data"]]


def test_desktop_listener_advertises_claude_prefixed_ids() -> None:
    ids = _get_model_ids(
        _settings(desktop=True),
        {"open_router": ["meta/llama-3.3"]},
        base_url=_DESKTOP_BASE_URL,
    )

    assert "claude-deepseek-deepseek-chat" in ids
    assert "claude-open_router-meta/llama-3.3" in ids
    assert "claude-open_router-meta/llama-3.3[1m]" not in ids
    assert not any(model_id.startswith("anthropic/") for model_id in ids)
    # The no-thinking variant already starts with "claude-" and is unchanged.
    assert "claude-3-freecc-no-thinking/deepseek/deepseek-chat" in ids
    # Genuine Claude aliases are untouched.
    assert "claude-sonnet-4-20250514" in ids


def test_desktop_listener_prefixes_1m_variants() -> None:
    ids = _get_model_ids(
        _settings(desktop=True, fcc_1m_models="deepseek/deepseek-chat"),
        {},
        base_url=_DESKTOP_BASE_URL,
    )

    assert "claude-deepseek-deepseek-chat[1m]" in ids


def test_main_port_stays_normal_while_desktop_enabled() -> None:
    ids = _get_model_ids(
        _settings(desktop=True),
        {"open_router": ["meta/llama-3.3"]},
    )

    assert "anthropic/deepseek/deepseek-chat" in ids
    assert "anthropic/open_router/meta/llama-3.3" in ids
    assert "claude-deepseek-deepseek-chat" not in ids


def test_desktop_port_normal_when_feature_disabled() -> None:
    ids = _get_model_ids(_settings(desktop=False), {}, base_url=_DESKTOP_BASE_URL)

    assert "anthropic/deepseek/deepseek-chat" in ids
    assert "claude-deepseek-deepseek-chat" not in ids


def test_router_routes_desktop_id_with_desktop_mode() -> None:
    router = ModelRouter(
        _settings(desktop=False, model="groq/llama-3.3-70b"),
        desktop_mode=True,
    )

    resolved = router.resolve("claude-deepseek-deepseek-v4-flash")

    assert resolved.original_model == "claude-deepseek-deepseek-v4-flash"
    assert resolved.primary.provider_id == "deepseek"
    assert resolved.primary.provider_model == "deepseek-v4-flash"
    assert resolved.primary.provider_model_ref == "deepseek/deepseek-v4-flash"


def test_router_strips_1m_suffix_from_desktop_id() -> None:
    router = ModelRouter(_settings(desktop=True), desktop_mode=True)

    resolved = router.resolve("claude-deepseek-deepseek-v4-flash[1m]")

    assert resolved.primary.provider_model == "deepseek-v4-flash"


def test_router_ignores_desktop_ids_without_desktop_mode() -> None:
    router = ModelRouter(
        _settings(desktop=True, model="groq/llama-3.3-70b"),
        desktop_mode=False,
    )

    resolved = router.resolve("claude-deepseek-deepseek-v4-flash")

    # Falls through to the default configured model, not to DeepSeek.
    assert resolved.primary.provider_id == "groq"
    assert resolved.primary.provider_model == "llama-3.3-70b"


def test_router_keeps_existing_id_forms_working_in_desktop_mode() -> None:
    router = ModelRouter(
        _settings(desktop=True, model_sonnet="deepseek/deepseek-chat"),
        desktop_mode=True,
    )

    gateway = router.resolve("anthropic/deepseek/deepseek-v4-flash")
    no_thinking = router.resolve(
        "claude-3-freecc-no-thinking/deepseek/deepseek-v4-flash"
    )
    sonnet = router.resolve("claude-sonnet-4-20250514")

    assert gateway.primary.provider_model == "deepseek-v4-flash"
    assert no_thinking.primary.provider_model == "deepseek-v4-flash"
    assert sonnet.primary.provider_model == "deepseek-chat"


def test_desktop_admin_fields_require_restart() -> None:
    # The desktop listener exists only from a supervisor generation start,
    # so Admin Apply must trigger the automatic restart for both fields;
    # without restart_required the toggle would silently do nothing until a
    # manual process restart.
    assert FIELD_BY_KEY["ENABLE_CLAUDE_DESKTOP_3P"].restart_required is True
    assert FIELD_BY_KEY["CLAUDE_DESKTOP_PORT"].restart_required is True
