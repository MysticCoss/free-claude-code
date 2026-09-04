"""Fork regression coverage for Claude Desktop 3P model-id compatibility.

Kept in its own file so upstream rewrites of ``test_model_listing.py`` or
``test_routing.py`` cannot drop it. Covers the ``claude-<provider>-<model>``
id codec, the /v1/models desktop view, and inbound routing.
"""

import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.provider_catalog import SUPPORTED_PROVIDER_IDS
from free_claude_code.config.settings import Settings
from free_claude_code.core.gateway_model_ids import (
    claude_desktop_model_id,
    decode_claude_desktop_model_id,
)
from tests.api.support import create_test_app, provider_manager_for_app

_PROVIDER_IDS = frozenset(SUPPORTED_PROVIDER_IDS)


def _settings(
    *,
    desktop: bool,
    model: str = "deepseek/deepseek-chat",
    model_sonnet: str | None = None,
    fcc_1m_models: str | None = None,
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


def _get_model_ids(settings: Settings, cache: dict[str, list[str]]) -> list[str]:
    app = create_test_app(settings)
    for provider_id, model_ids in cache.items():
        provider_manager_for_app(app).cache_model_infos(
            provider_id,
            {ProviderModelInfo(model_id) for model_id in model_ids},
        )
    response = TestClient(app).get("/v1/models")
    assert response.status_code == 200
    return [item["id"] for item in response.json()["data"]]


def test_desktop_mode_advertises_claude_prefixed_ids() -> None:
    ids = _get_model_ids(
        _settings(desktop=True),
        {"open_router": ["meta/llama-3.3"]},
    )

    assert "claude-deepseek-deepseek-chat" in ids
    assert "claude-open_router-meta/llama-3.3" in ids
    assert "claude-open_router-meta/llama-3.3[1m]" not in ids
    assert not any(model_id.startswith("anthropic/") for model_id in ids)
    # The no-thinking variant already starts with "claude-" and is unchanged.
    assert "claude-3-freecc-no-thinking/deepseek/deepseek-chat" in ids
    # Genuine Claude aliases are untouched.
    assert "claude-sonnet-4-20250514" in ids


def test_desktop_mode_prefixes_1m_variants() -> None:
    ids = _get_model_ids(
        _settings(desktop=True, fcc_1m_models="deepseek/deepseek-chat"),
        {},
    )

    assert "claude-deepseek-deepseek-chat[1m]" in ids


def test_disabled_desktop_mode_keeps_gateway_ids() -> None:
    ids = _get_model_ids(_settings(desktop=False), {})

    assert "anthropic/deepseek/deepseek-chat" in ids
    assert "claude-deepseek-deepseek-chat" not in ids


def test_router_routes_desktop_id_to_exact_provider_model() -> None:
    router = ModelRouter(_settings(desktop=True, model="groq/llama-3.3-70b"))

    resolved = router.resolve("claude-deepseek-deepseek-v4-flash")

    assert resolved.original_model == "claude-deepseek-deepseek-v4-flash"
    assert resolved.primary.provider_id == "deepseek"
    assert resolved.primary.provider_model == "deepseek-v4-flash"
    assert resolved.primary.provider_model_ref == "deepseek/deepseek-v4-flash"


def test_router_strips_1m_suffix_from_desktop_id() -> None:
    router = ModelRouter(_settings(desktop=True))

    resolved = router.resolve("claude-deepseek-deepseek-v4-flash[1m]")

    assert resolved.primary.provider_model == "deepseek-v4-flash"


def test_router_ignores_desktop_ids_when_disabled() -> None:
    router = ModelRouter(_settings(desktop=False, model="groq/llama-3.3-70b"))

    resolved = router.resolve("claude-deepseek-deepseek-v4-flash")

    # Falls through to the default configured model, not to DeepSeek.
    assert resolved.primary.provider_id == "groq"
    assert resolved.primary.provider_model == "llama-3.3-70b"


def test_router_keeps_existing_id_forms_working_when_desktop_enabled() -> None:
    router = ModelRouter(_settings(desktop=True, model_sonnet="deepseek/deepseek-chat"))

    gateway = router.resolve("anthropic/deepseek/deepseek-v4-flash")
    no_thinking = router.resolve(
        "claude-3-freecc-no-thinking/deepseek/deepseek-v4-flash"
    )
    sonnet = router.resolve("claude-sonnet-4-20250514")

    assert gateway.primary.provider_model == "deepseek-v4-flash"
    assert no_thinking.primary.provider_model == "deepseek-v4-flash"
    assert sonnet.primary.provider_model == "deepseek-chat"
