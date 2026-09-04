"""Fork regression coverage for Claude Desktop 3P model-id compatibility.

Kept in its own file so upstream rewrites of ``test_model_listing.py`` or
``test_routing.py`` cannot drop it. Covers the ``claude-<provider>-<model>``
id codec, the dedicated-port request detection, the /v1/models desktop view,
the supervisor listener plan, and inbound routing.
"""

import re

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
# "claude-"), and it must match one of the family names. A valid
# ``anthropic_family_tier`` on the entry bypasses the name check entirely.
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


def _desktop_keeps_entry(entry: dict[str, object]) -> bool:
    tier = entry.get("anthropic_family_tier")
    if isinstance(tier, str) and tier.lower() in _DESKTOP_FAMILY_TIERS:
        return True
    return _desktop_name_filter_passes(str(entry["id"]))


def test_desktop_catalog_entries_all_survive_desktop_discovery_filter() -> None:
    entries = _get_model_entries(
        _settings(desktop=True),
        {"open_router": ["deepseek/deepseek-v4-pro", "qwen/qwen3.8-flash"]},
        base_url=_DESKTOP_BASE_URL,
    )

    assert entries
    rejected = [
        str(entry["id"]) for entry in entries if not _desktop_keeps_entry(entry)
    ]
    assert rejected == []


def test_desktop_family_tier_rescues_blacklisted_vendor_names() -> None:
    # The codec's "claude-" prefix alone is not enough: Desktop's blacklist
    # wins over the claude-substring pass, so these ids need the tier field.
    assert not _desktop_name_filter_passes("claude-deepseek-deepseek-chat")
    assert not _desktop_name_filter_passes(
        "claude-3-freecc-no-thinking/open_router/qwen/qwen3.8-flash"
    )

    entries = _get_model_entries(
        _settings(desktop=True),
        {"open_router": ["qwen/qwen3.8-flash"]},
        base_url=_DESKTOP_BASE_URL,
    )
    by_id = {str(entry["id"]): entry for entry in entries}
    for model_id in (
        "claude-deepseek-deepseek-chat",
        "claude-open_router-qwen/qwen3.8-flash",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/open_router/qwen/qwen3.8-flash",
    ):
        assert model_id in by_id, model_id
        assert by_id[model_id].get("anthropic_family_tier") == "sonnet", model_id


def test_desktop_family_tier_follows_configured_aliases() -> None:
    entries = _get_model_entries(
        _settings(
            desktop=True,
            model="deepseek/deepseek-chat",
            model_opus="groq/llama-3.3-70b",
            model_haiku="groq/llama-3.3-70b",
            model_fable="open_router/qwen/qwen3.8-flash",
            model_compact="deepseek/deepseek-chat",
            fcc_1m_models="groq/llama-3.3-70b",
        ),
        {},
        base_url=_DESKTOP_BASE_URL,
    )
    tiers = {str(entry["id"]): entry.get("anthropic_family_tier") for entry in entries}

    # model_compact claims the default chat ref first in table order.
    assert tiers["claude-deepseek-deepseek-chat"] == "haiku"
    assert tiers["claude-3-freecc-no-thinking/deepseek/deepseek-chat"] == "haiku"
    # Aliases that share a ref resolve in declared order: opus before haiku.
    assert tiers["claude-groq-llama-3.3-70b"] == "opus"
    # [1m] catalog variants keep the alias tier of their base ref.
    assert tiers["claude-groq-llama-3.3-70b[1m]"] == "opus"
    assert tiers["claude-open_router-qwen/qwen3.8-flash"] == "fable"
    # Unmapped models fall back to the neutral everyday-work tier.
    assert tiers["claude-3-freecc-no-thinking/groq/llama-3.3-70b"] == "opus"


def test_main_port_catalog_omits_family_tier_field() -> None:
    entries = _get_model_entries(
        _settings(desktop=True),
        {"open_router": ["qwen/qwen3.8-flash"]},
    )

    assert entries
    assert all("anthropic_family_tier" not in entry for entry in entries)


def test_disabled_desktop_port_catalog_omits_family_tier_field() -> None:
    entries = _get_model_entries(
        _settings(desktop=False),
        {"open_router": ["qwen/qwen3.8-flash"]},
        base_url=_DESKTOP_BASE_URL,
    )

    assert all("anthropic_family_tier" not in entry for entry in entries)


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
