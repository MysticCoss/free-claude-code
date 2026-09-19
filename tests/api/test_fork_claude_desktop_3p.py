"""Fork regression coverage for Claude Desktop 3P model-id compatibility.

Kept in its own file so upstream rewrites of ``test_model_listing.py`` or
``test_routing.py`` cannot drop it. Covers the obfuscated
``claude-<provider>-<model>`` id codec, the dedicated-port request detection,
the /v1/models desktop view, the supervisor listener plan, and inbound
routing.
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
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import (
    claude_desktop_model_id,
    claude_desktop_no_thinking_model_id,
    decode_claude_desktop_model_id,
    decode_claude_desktop_no_thinking_model_id,
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
    with pytest.raises(ValueError):
        claude_desktop_no_thinking_model_id("deepseek-chat")


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
        # Vendor tokens in the provider id itself.
        ("mistral", "mistral-large-latest"),
        ("kimi", "kimi-k2.6"),
        ("minimax", "minimax-m2.5"),
        ("openai", "gpt-5-mini"),
        # Vowelless tokens (hyphen is inserted after the first character).
        ("open_router", "zai/glm-4.6"),
        ("open_router", "openai/gpt-5"),
        ("opencode_go", "hy3-preview"),
        ("opencode_go", "dpsk-v2"),
        ("open_router", "ds-1.5b"),
        ("open_router", "arcee-ai/M2.5"),
        # Uppercase tokens must still be obfuscated (Desktop filters the
        # lowercased id).
        ("open_router", "Qwen3-235B"),
        # Cross-segment token: "nova-" spans provider tail, separator, and
        # model head, so encode/decode must treat the whole id as one string.
        ("sambanova", "ova-pro"),
        # Slash-carrying model refs with tokens in both path segments.
        ("open_router", "qwen/qwen3.8-flash"),
        ("open_router", "deepseek/deepseek-v4-pro"),
    ],
)
def test_claude_desktop_id_round_trips(provider_id: str, provider_model: str) -> None:
    model_id = claude_desktop_model_id(f"{provider_id}/{provider_model}")
    assert model_id.startswith("claude-")
    assert _desktop_name_filter_passes(model_id)
    decoded = decode_claude_desktop_model_id(model_id, _PROVIDER_IDS)
    assert decoded is not None
    assert decoded.provider_id == provider_id
    assert decoded.provider_model == provider_model
    assert not decoded.force_reasoning_off


@pytest.mark.parametrize(
    ("provider_id", "provider_model"),
    [
        ("deepseek", "deepseek-chat"),
        ("open_router", "qwen/qwen3.8-flash"),
        ("open_router", "zai/glm-4.6"),
        ("sambanova", "ova-pro"),
    ],
)
def test_claude_desktop_no_thinking_id_round_trips(
    provider_id: str, provider_model: str
) -> None:
    model_id = claude_desktop_no_thinking_model_id(f"{provider_id}/{provider_model}")
    assert model_id.startswith("claude-3-freecc-no-thinking/")
    assert _desktop_name_filter_passes(model_id)
    decoded = decode_claude_desktop_no_thinking_model_id(model_id, _PROVIDER_IDS)
    assert decoded is not None
    assert decoded.provider_id == provider_id
    assert decoded.provider_model == provider_model
    assert decoded.force_reasoning_off


def test_obfuscation_replaces_first_vowel_or_inserts_after_first_char() -> None:
    # Tokens with a vowel: the first vowel becomes a hyphen.
    assert claude_desktop_model_id("deepseek/deepseek-chat") == (
        "claude-d-epseek-d-epseek-chat"
    )
    assert claude_desktop_model_id("opencode_go/qwen3.8-flash") == (
        "claude-opencode_go-qw-n3.8-flash"
    )
    assert claude_desktop_model_id("open_router/Qwen3-235B") == (
        "claude-open_router-Qw-n3-235B"
    )
    # Vowelless tokens: a hyphen is inserted after the first character.
    assert claude_desktop_model_id("open_router/glm-4.6") == (
        "claude-open_router-g-lm-4.6"
    )
    assert claude_desktop_model_id("open_router/gpt-5") == "claude-open_router-g-pt-5"
    assert claude_desktop_model_id("opencode_go/hy3") == "claude-opencode_go-h-y3"
    assert claude_desktop_model_id("open_router/dpsk-v2") == (
        "claude-open_router-d-psk-v2"
    )


def test_clean_vendor_names_are_not_obfuscated() -> None:
    # "hy4" is not the blacklisted "hy3"; hyphens in clean names stay put.
    assert claude_desktop_model_id("opencode_go/hy4-preview") == (
        "claude-opencode_go-hy4-preview"
    )
    assert claude_desktop_no_thinking_model_id("opencode_go/muse-spark-1-1") == (
        "claude-3-freecc-no-thinking/opencode_go/muse-spark-1-1"
    )
    # Tokens that themselves contain hyphens are broken at the vowel, so the
    # name's own hyphens survive untouched next to the marker.
    # "llama"'s first vowel is the "a" at index 2: l l - m a.
    assert claude_desktop_model_id("groq/llama-3.3-70b") == (
        "claude-groq-ll-ma-3.3-70b"
    )


@pytest.mark.parametrize("provider_id", sorted(_PROVIDER_IDS))
def test_every_catalog_provider_round_trips(provider_id: str) -> None:
    ref = f"{provider_id}/some-model"
    model_id = claude_desktop_model_id(ref)
    assert _desktop_name_filter_passes(model_id)
    decoded = decode_claude_desktop_model_id(model_id, _PROVIDER_IDS)
    assert decoded is not None
    assert decoded.provider_id == provider_id
    assert decoded.provider_model == "some-model"
    no_thinking_id = claude_desktop_no_thinking_model_id(ref)
    assert _desktop_name_filter_passes(no_thinking_id)
    no_thinking = decode_claude_desktop_no_thinking_model_id(
        no_thinking_id, _PROVIDER_IDS
    )
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
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "anthropic/deepseek/deepseek-chat",
        "claude-not-a-provider-model",
        "claude-deepseek-",
        "deepseek-v4-flash",
    ],
)
def test_non_desktop_ids_are_not_decoded(model_name: str) -> None:
    assert decode_claude_desktop_model_id(model_name, _PROVIDER_IDS) is None


@pytest.mark.parametrize(
    "model_name",
    [
        "deepseek/deepseek-chat",
        "anthropic/deepseek/deepseek-chat",
        "claude-deepseek-deepseek-chat",
        "claude-3-freecc-no-thinking/not-a-provider/deepseek-chat",
        "claude-3-freecc-no-thinking/foobar/deepseek-chat",
        "claude-3-freecc-no-thinking/deepseek/",
    ],
)
def test_non_desktop_no_thinking_ids_are_not_decoded(model_name: str) -> None:
    assert decode_claude_desktop_no_thinking_model_id(model_name, _PROVIDER_IDS) is None


@pytest.mark.parametrize(
    ("model_name", "provider_id", "provider_model"),
    [
        # Ids from before obfuscation shipped (and ids users typed by hand):
        # still decode through the original provider-prefix match.
        ("claude-deepseek-deepseek-chat", "deepseek", "deepseek-chat"),
        ("claude-mistral-mistral-large", "mistral", "mistral-large"),
        ("claude-open_router-qwen/qwen3.8-flash", "open_router", "qwen/qwen3.8-flash"),
    ],
)
def test_legacy_raw_desktop_ids_still_decode(
    model_name: str, provider_id: str, provider_model: str
) -> None:
    decoded = decode_claude_desktop_model_id(model_name, _PROVIDER_IDS)
    assert decoded is not None
    assert decoded.provider_id == provider_id
    assert decoded.provider_model == provider_model


def test_legacy_raw_no_thinking_ids_still_decode() -> None:
    decoded = decode_claude_desktop_no_thinking_model_id(
        "claude-3-freecc-no-thinking/open_router/qwen/qwen3.8-flash", _PROVIDER_IDS
    )
    assert decoded is not None
    assert decoded.provider_id == "open_router"
    assert decoded.provider_model == "qwen/qwen3.8-flash"
    assert decoded.force_reasoning_off


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


def test_desktop_ids_obfuscate_blacklisted_vendor_names() -> None:
    # The codec's "claude-" prefix alone is not enough: Desktop's blacklist
    # wins over the claude-substring pass, so the ids carry the obfuscated
    # vendor tokens and pass on the name check alone, with no tier field.
    assert not _desktop_name_filter_passes("claude-deepseek-deepseek-chat")
    assert not _desktop_name_filter_passes(
        "claude-3-freecc-no-thinking/open_router/qwen/qwen3.8-flash"
    )
    assert _desktop_name_filter_passes("claude-d-epseek-d-epseek-chat")
    assert _desktop_name_filter_passes(
        "claude-3-freecc-no-thinking/open_router/qw-n/qw-n3.8-flash"
    )

    entries = _get_model_entries(
        _settings(desktop=True),
        {"open_router": ["qwen/qwen3.8-flash"]},
        base_url=_DESKTOP_BASE_URL,
    )
    by_id = {str(entry["id"]): entry for entry in entries}
    assert all("anthropic_family_tier" not in entry for entry in entries)
    for model_id in (
        "claude-d-epseek-d-epseek-chat",
        "claude-open_router-qw-n/qw-n3.8-flash",
        "claude-3-freecc-no-thinking/d-epseek/d-epseek-chat",
        "claude-3-freecc-no-thinking/open_router/qw-n/qw-n3.8-flash",
    ):
        assert model_id in by_id, model_id
    for raw_id in (
        "claude-deepseek-deepseek-chat",
        "claude-open_router-qwen/qwen3.8-flash",
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
    assert not any("d-epseek" in model_id for model_id in ids)
    assert not any("qw-n" in model_id for model_id in ids)


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
    assert not any(str(model_id).startswith("claude-d-") for model_id in ids)


def test_desktop_listener_advertises_claude_prefixed_ids() -> None:
    ids = _get_model_ids(
        _settings(desktop=True),
        {"open_router": ["meta/llama-3.3"]},
        base_url=_DESKTOP_BASE_URL,
    )

    assert "claude-d-epseek-d-epseek-chat" in ids
    assert "claude-open_router-meta/ll-ma-3.3" in ids
    assert "claude-open_router-meta/llama-3.3[1m]" not in ids
    assert not any(model_id.startswith("anthropic/") for model_id in ids)
    # The no-thinking variant keeps the shared prefix but carries the same
    # obfuscated provider/model segments.
    assert "claude-3-freecc-no-thinking/d-epseek/d-epseek-chat" in ids
    # Genuine Claude aliases are untouched.
    assert "claude-sonnet-4-20250514" in ids


def test_desktop_listener_prefixes_1m_variants() -> None:
    ids = _get_model_ids(
        _settings(desktop=True, fcc_1m_models="deepseek/deepseek-chat"),
        {},
        base_url=_DESKTOP_BASE_URL,
    )

    assert "claude-d-epseek-d-epseek-chat[1m]" in ids


def test_main_port_stays_normal_while_desktop_enabled() -> None:
    ids = _get_model_ids(
        _settings(desktop=True),
        {"open_router": ["meta/llama-3.3"]},
    )

    assert "anthropic/deepseek/deepseek-chat" in ids
    assert "anthropic/open_router/meta/llama-3.3" in ids
    assert "claude-deepseek-deepseek-chat" not in ids
    assert "claude-d-epseek-d-epseek-chat" not in ids


def test_desktop_port_normal_when_feature_disabled() -> None:
    ids = _get_model_ids(_settings(desktop=False), {}, base_url=_DESKTOP_BASE_URL)

    assert "anthropic/deepseek/deepseek-chat" in ids
    assert "claude-deepseek-deepseek-chat" not in ids


def test_router_routes_desktop_id_with_desktop_mode() -> None:
    router = ModelRouter(
        _settings(desktop=False, model="groq/llama-3.3-70b"),
        desktop_mode=True,
    )

    resolved = router.resolve("claude-d-epseek-d-epseek-v4-flash")

    assert resolved.original_model == "claude-d-epseek-d-epseek-v4-flash"
    assert resolved.primary.provider_id == "deepseek"
    assert resolved.primary.provider_model == "deepseek-v4-flash"
    assert resolved.primary.provider_model_ref == "deepseek/deepseek-v4-flash"


def test_router_strips_1m_suffix_from_desktop_id() -> None:
    router = ModelRouter(_settings(desktop=True), desktop_mode=True)

    resolved = router.resolve("claude-d-epseek-d-epseek-v4-flash[1m]")

    assert resolved.primary.provider_model == "deepseek-v4-flash"


def test_router_routes_obfuscated_no_thinking_desktop_id() -> None:
    # A clean provider with an obfuscated model would be half-decoded by the
    # generic gateway parser, so the desktop decoder must run first.
    router = ModelRouter(_settings(desktop=True), desktop_mode=True)

    resolved = router.resolve(
        "claude-3-freecc-no-thinking/open_router/qw-n/qw-n3.8-flash"
    )

    assert resolved.primary.provider_id == "open_router"
    assert resolved.primary.provider_model == "qwen/qwen3.8-flash"
    assert resolved.reasoning_preference is ReasoningPreference.OFF


def test_router_ignores_desktop_ids_without_desktop_mode() -> None:
    router = ModelRouter(
        _settings(desktop=True, model="groq/llama-3.3-70b"),
        desktop_mode=False,
    )

    resolved = router.resolve("claude-d-epseek-d-epseek-v4-flash")

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
    legacy_desktop = router.resolve("claude-deepseek-deepseek-v4-flash")
    sonnet = router.resolve("claude-sonnet-4-20250514")

    assert gateway.primary.provider_model == "deepseek-v4-flash"
    assert no_thinking.primary.provider_model == "deepseek-v4-flash"
    assert legacy_desktop.primary.provider_id == "deepseek"
    assert legacy_desktop.primary.provider_model == "deepseek-v4-flash"
    assert sonnet.primary.provider_model == "deepseek-chat"


def test_desktop_admin_fields_require_restart() -> None:
    # The desktop listener exists only from a supervisor generation start,
    # so Admin Apply must trigger the automatic restart for both fields;
    # without restart_required the toggle would silently do nothing until a
    # manual process restart.
    assert FIELD_BY_KEY["ENABLE_CLAUDE_DESKTOP_3P"].restart_required is True
    assert FIELD_BY_KEY["CLAUDE_DESKTOP_PORT"].restart_required is True
