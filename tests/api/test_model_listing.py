from fastapi.testclient import TestClient

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.config.settings import Settings
from tests.api.support import create_test_app, provider_manager_for_app


def _settings(
    *,
    model: str = "deepseek/deepseek-chat",
    model_fable: str | None = None,
    model_opus: str | None = "open_router/anthropic/claude-opus",
    model_haiku: str | None = "deepseek/deepseek-chat",
    fcc_1m_models: str = "",
    opencode_api_key: str = "",
) -> Settings:
    return Settings.model_construct(
        model=model,
        model_fable=model_fable,
        model_opus=model_opus,
        model_sonnet=None,
        model_haiku=model_haiku,
        fcc_1m_models=fcc_1m_models,
        anthropic_auth_token="",
        deepseek_api_key="deepseek-key",
        open_router_api_key="open-router-key",
        wafer_api_key="wafer-key",
        opencode_api_key=opencode_api_key,
    )


def _cache_models(app, provider_id: str, *model_ids: str) -> None:
    provider_manager_for_app(app).cache_model_infos(
        provider_id,
        {ProviderModelInfo(model_id) for model_id in model_ids},
    )


def test_models_list_includes_configured_refs_cached_provider_models_and_aliases():
    app = create_test_app(_settings())
    _cache_models(app, "deepseek", "deepseek-chat")
    _cache_models(
        app,
        "open_router",
        "meta/llama-3.3",
        "anthropic/claude-opus",
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    data = response.json()
    ids = [item["id"] for item in data["data"]]
    assert ids[:6] == [
        "anthropic/deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "anthropic/open_router/anthropic/claude-opus",
        "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus",
        "anthropic/open_router/meta/llama-3.3",
        "claude-3-freecc-no-thinking/open_router/meta/llama-3.3",
    ]
    assert ids.count("anthropic/deepseek/deepseek-chat") == 1
    assert ids.count("anthropic/open_router/anthropic/claude-opus") == 1
    display_names = {item["id"]: item["display_name"] for item in data["data"]}
    assert (
        display_names["anthropic/open_router/meta/llama-3.3"]
        == "open_router/meta/llama-3.3"
    )
    assert (
        display_names["claude-3-freecc-no-thinking/open_router/meta/llama-3.3"]
        == "open_router/meta/llama-3.3 (no thinking)"
    )
    assert "claude-sonnet-4-20250514" in ids
    assert "claude-fable-5" in ids
    assert data["first_id"] == ids[0]
    assert data["last_id"] == ids[-1]
    assert data["has_more"] is False


def test_models_list_uses_thinking_metadata_for_cached_models():
    app = create_test_app(_settings(model_opus=None))
    manager = provider_manager_for_app(app)
    _cache_models(app, "deepseek", "deepseek-chat")
    manager.cache_model_infos(
        "open_router",
        {
            ProviderModelInfo("reasoning-model", supports_thinking=True),
            ProviderModelInfo("plain-model", supports_thinking=False),
        },
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/open_router/reasoning-model" in ids
    assert "claude-3-freecc-no-thinking/open_router/reasoning-model" in ids
    assert "anthropic/open_router/plain-model" not in ids
    assert "claude-3-freecc-no-thinking/open_router/plain-model" in ids


def test_models_list_uses_cached_metadata_for_configured_refs():
    app = create_test_app(
        _settings(
            model="open_router/plain-model",
            model_opus=None,
            model_haiku=None,
        )
    )
    provider_manager_for_app(app).cache_model_infos(
        "open_router",
        {ProviderModelInfo("plain-model", supports_thinking=False)},
    )

    response = TestClient(app).get("/v1/models")

    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/open_router/plain-model" not in ids
    assert ids[0] == "claude-3-freecc-no-thinking/open_router/plain-model"


def test_models_list_includes_cached_wafer_models():
    app = create_test_app(
        _settings(
            model="wafer/DeepSeek-V4-Pro",
            model_opus=None,
            model_haiku=None,
        )
    )
    _cache_models(app, "wafer", "DeepSeek-V4-Pro", "MiniMax-M2.7")

    response = TestClient(app).get("/v1/models")

    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/wafer/DeepSeek-V4-Pro" in ids
    assert "claude-3-freecc-no-thinking/wafer/DeepSeek-V4-Pro" in ids
    assert "anthropic/wafer/MiniMax-M2.7" in ids
    assert "claude-3-freecc-no-thinking/wafer/MiniMax-M2.7" in ids


def test_models_list_works_with_empty_discovery_catalog():
    app = create_test_app(_settings())

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert ids[:4] == [
        "anthropic/deepseek/deepseek-chat",
        "claude-3-freecc-no-thinking/deepseek/deepseek-chat",
        "anthropic/open_router/anthropic/claude-opus",
        "claude-3-freecc-no-thinking/open_router/anthropic/claude-opus",
    ]
    assert "claude-sonnet-4-20250514" in ids


# ----------------- [1m] variants -----------------


def test_models_list_includes_1m_suffixed_variants_for_configured_refs():
    """Configured ref in FCC_1M_MODELS gets a [1m]-suffixed variant in /v1/models."""
    app = create_test_app(
        _settings(
            model="opencode_go/deepseek-v4-pro",
            model_opus=None,
            model_haiku=None,
            fcc_1m_models="opencode_go/deepseek-v4-pro",
        )
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/opencode_go/deepseek-v4-pro[1m]" in ids
    assert "claude-3-freecc-no-thinking/opencode_go/deepseek-v4-pro[1m]" in ids


def test_models_list_includes_1m_suffixed_variants_for_cached_models():
    """Cached provider model in FCC_1M_MODELS also gets a [1m] variant."""
    from free_claude_code.application.model_metadata import ProviderModelInfo

    app = create_test_app(
        _settings(
            fcc_1m_models="opencode_go/deepseek-v4-pro",
            opencode_api_key="opencode-key",
        )
    )
    provider_manager_for_app(app).cache_model_infos(
        "opencode_go", {ProviderModelInfo("deepseek-v4-pro")}
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert "anthropic/opencode_go/deepseek-v4-pro[1m]" in ids
    assert "claude-3-freecc-no-thinking/opencode_go/deepseek-v4-pro[1m]" in ids


def test_models_list_no_1m_variants_when_not_configured():
    """Empty FCC_1M_MODELS produces no [1m] variants in /v1/models."""
    app = create_test_app(_settings(fcc_1m_models=""))

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    assert not any("[1m]" in id_ for id_ in ids)


def test_models_list_1m_variants_deduped_against_configured_chat_models():
    """A configured chat model ref also in FCC_1M_MODELS does not produce duplicates."""
    app = create_test_app(
        _settings(
            model="opencode_go/deepseek-v4-pro",
            model_opus=None,
            model_haiku=None,
            fcc_1m_models="opencode_go/deepseek-v4-pro",
        )
    )

    response = TestClient(app).get("/v1/models")

    assert response.status_code == 200
    ids = [item["id"] for item in response.json()["data"]]
    # Both the standard variant and the [1m] variant are present, each once.
    assert ids.count("anthropic/opencode_go/deepseek-v4-pro") == 1
    assert ids.count("anthropic/opencode_go/deepseek-v4-pro[1m]") == 1
