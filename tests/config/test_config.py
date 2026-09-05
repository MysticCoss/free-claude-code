"""Contracts for pure Settings and canonical source composition."""

from enum import Enum
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from free_claude_code.application.routing import ModelRouter
from free_claude_code.config import loader
from free_claude_code.config.constants import (
    ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    HTTP_CONNECT_TIMEOUT_DEFAULT,
)
from free_claude_code.config.env_files import dotenv_values_from_file
from free_claude_code.config.loader import (
    ConfigSource,
    clear_settings_cache,
    compose_settings_snapshot,
    get_settings,
    repair_invalid_managed_provider_proxies,
)
from free_claude_code.config.model_refs import (
    configured_chat_model_refs,
    parse_model_name,
    parse_provider_type,
)
from free_claude_code.config.nim import NimSettings
from free_claude_code.config.paths import (
    managed_env_path,
    messaging_state_dir_path,
    server_log_path,
)
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings


@pytest.mark.parametrize("source", [ConfigSource.MANAGED, ConfigSource.PROCESS])
@pytest.mark.parametrize("tier", ["FABLE", "OPUS", "SONNET", "HAIKU"])
def test_retirement_normalizes_sources_without_resurrecting_overrides(source, tier):
    retired = {
        "MODEL": "github_models/openai/old",
        f"MODEL_{tier}": "github_models/vendor/opus",
        "MODEL_FALLBACKS": "github_models/old",
        f"REASONING_{tier}": "off",
    }
    managed = {
        "MODEL": "groq/managed",
        f"MODEL_{tier}": "groq/tier",
        "MODEL_FALLBACKS": "groq/backup",
    }
    snapshot = compose_settings_snapshot(
        retired if source is ConfigSource.MANAGED else managed,
        retired if source is ConfigSource.PROCESS else {},
    )
    assert snapshot.settings.model == DEFAULT_MODEL
    assert getattr(snapshot.settings, f"model_{tier.lower()}") is None
    assert (
        getattr(snapshot.settings, f"reasoning_{tier.lower()}")
        is ReasoningPreference.OFF
    )
    assert snapshot.settings.model_fallbacks is None
    expected = (
        ConfigSource.PROCESS if source is ConfigSource.PROCESS else ConfigSource.DEFAULT
    )
    assert snapshot.sources["model"] is expected
    assert snapshot.sources[f"model_{tier.lower()}"] is expected
    assert snapshot.sources["model_fallbacks"] is expected
    assert retired["MODEL"] == "github_models/openai/old"


def test_retirement_preserves_effective_default_order_and_process_environment(
    monkeypatch,
):
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.setenv("MODEL_OPUS", "github_models/opus")
    snapshot = compose_settings_snapshot(
        {
            "MODEL": "groq/default",
            "MODEL_OPUS": "groq/managed-tier",
            "MODEL_FALLBACKS": "groq/first, github_models/old, deepseek/last",
        }
    )
    assert snapshot.settings.model == "groq/default"
    assert snapshot.settings.model_opus is None
    assert snapshot.sources["model_opus"] is ConfigSource.PROCESS
    assert snapshot.settings.model_fallbacks == ("groq/first", "deepseek/last")
    assert loader.os.environ["MODEL_OPUS"] == "github_models/opus"


@pytest.mark.parametrize(
    "values",
    [
        {"MODEL": "github_models/"},
        {"MODEL": "other/unknown"},
        {"MODEL_FALLBACKS": "github_models/old,groq/a,groq/a"},
        {"MODEL_FALLBACKS": "github_models/old,,groq/a"},
        {"MODEL_FALLBACKS": "github_models/old,"},
        {"MODEL_FALLBACKS": ",github_models/old"},
        {"MODEL_FALLBACKS": "github_models/old, "},
        {"MODEL_FALLBACKS": "github_models/old,github_models/"},
    ],
)
def test_retirement_does_not_hide_invalid_configuration(values):
    with pytest.raises(ValidationError):
        compose_settings_snapshot(values, {})


def test_direct_settings_still_reject_retired_provider():
    with pytest.raises(ValidationError):
        Settings(MODEL="github_models/openai/old")


def test_settings_defaults_are_valid_and_nonempty() -> None:
    settings = Settings()

    assert settings.provider_rate_limit == 1
    assert settings.provider_rate_window == 2
    assert settings.provider_max_concurrency == 2
    assert settings.provider_progress_timeout == 600.0
    assert settings.http_read_timeout == 120.0
    assert settings.http_write_timeout == 10.0
    assert settings.http_connect_timeout == HTTP_CONNECT_TIMEOUT_DEFAULT
    assert settings.voice_note_enabled is True
    assert settings.whisper_device == "cpu"
    assert settings.whisper_model == "base"
    assert settings.enable_web_server_tools is True
    assert settings.proxy_auth_enabled is False
    assert settings.proxy_auth_token == "freecc"
    assert [
        name for name, value in settings if isinstance(value, str) and not value
    ] == []
    assert [
        name
        for name, field in Settings.model_fields.items()
        if field.get_default(call_default_factory=True) == ""
    ] == []


def test_every_external_setting_has_one_explicit_alias() -> None:
    aliases = [
        field.validation_alias
        for name, field in Settings.model_fields.items()
        if name != "nim"
    ]
    assert all(isinstance(alias, str) for alias in aliases)
    assert len(aliases) == len(set(aliases))
    assert Settings.model_fields["nim"].validation_alias is None


def test_direct_settings_construction_performs_no_environment_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL", "deepseek/process-model")
    monkeypatch.setenv("PROVIDER_RATE_LIMIT", "99")

    settings = Settings()

    assert settings.model.startswith("nvidia_nim/")
    assert settings.provider_rate_limit == 1


@pytest.mark.parametrize(
    ("key", "attribute", "value", "expected"),
    [
        ("MODEL", "model", "deepseek/deepseek-chat", "deepseek/deepseek-chat"),
        ("PROVIDER_RATE_LIMIT", "provider_rate_limit", "20", 20),
        (
            "PROVIDER_PROGRESS_TIMEOUT",
            "provider_progress_timeout",
            "900",
            900.0,
        ),
        ("HTTP_READ_TIMEOUT", "http_read_timeout", "600", 600.0),
        ("FCC_OPEN_BROWSER", "open_admin_browser", "false", False),
        ("REASONING_POLICY", "reasoning_policy", "off", ReasoningPreference.OFF),
        ("GROQ_API_KEY", "groq_api_key", " secret ", "secret"),
        ("OPENROUTER_PROXY", "open_router_proxy", " http://proxy ", "http://proxy"),
    ],
)
def test_process_values_are_parsed_at_the_loader_boundary(
    key: str,
    attribute: str,
    value: str,
    expected: object,
) -> None:
    snapshot = compose_settings_snapshot({}, {key: value})

    assert getattr(snapshot.settings, attribute) == expected
    assert snapshot.sources[attribute] is ConfigSource.PROCESS


@pytest.mark.parametrize(
    "value",
    [
        0.0,
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        float(1 << 64),
    ],
)
def test_provider_progress_timeout_must_be_representable(value: float) -> None:
    with pytest.raises(ValidationError):
        Settings(provider_progress_timeout=value)


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "inf", "-inf", "nan", str(1 << 64)],
)
def test_loader_rejects_invalid_provider_progress_timeout(value: str) -> None:
    with pytest.raises(ValidationError):
        compose_settings_snapshot({}, {"PROVIDER_PROGRESS_TIMEOUT": value})


@pytest.mark.parametrize(
    "key",
    [
        "GROQ_API_KEY",
        "OPENROUTER_PROXY",
        "MODEL_OPUS",
        "TELEGRAM_BOT_TOKEN",
        "ALLOWED_DIR",
    ],
)
def test_optional_blank_process_values_normalize_to_none(key: str) -> None:
    snapshot = compose_settings_snapshot({}, {key: "  "})
    attribute = next(
        name
        for name, field in Settings.model_fields.items()
        if field.validation_alias == key
    )

    assert getattr(snapshot.settings, attribute) is None


def test_blank_required_process_value_is_rejected() -> None:
    with pytest.raises(ValidationError, match="MODEL"):
        compose_settings_snapshot({}, {"MODEL": " "})


def test_blank_process_auth_token_uses_retained_default() -> None:
    snapshot = compose_settings_snapshot({}, {"ANTHROPIC_AUTH_TOKEN": ""})

    assert snapshot.settings.proxy_auth_token == "freecc"
    assert snapshot.sources["proxy_auth_token"] is ConfigSource.DEFAULT


def test_process_precedence_and_managed_token_exception() -> None:
    snapshot = compose_settings_snapshot(
        {
            "MODEL": "deepseek/managed",
            "ANTHROPIC_AUTH_TOKEN": "managed-token",
        },
        {
            "MODEL": "groq/process",
            "ANTHROPIC_AUTH_TOKEN": "stale-process-token",
        },
    )

    assert snapshot.settings.model == "groq/process"
    assert snapshot.sources["model"] is ConfigSource.PROCESS
    assert snapshot.settings.proxy_auth_token == "managed-token"
    assert snapshot.sources["proxy_auth_token"] is ConfigSource.MANAGED


def test_get_settings_is_cached_and_creates_managed_schema() -> None:
    clear_settings_cache()

    first = get_settings()
    second = get_settings()

    assert first is second


def _write_managed_config(text: str) -> Path:
    path = managed_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_repair_invalid_managed_provider_proxies_removes_all_eligible_values() -> None:
    invalid_openai = "invalid://user:leaked-secret@proxy.example:8080"
    managed = _write_managed_config(
        "\n".join(
            (
                "FCC_CONFIG_SCHEMA=1",
                "MODEL=nvidia_nim/test-model",
                "OPENROUTER_PROXY=http://proxy.example:notaport",
                "GROQ_PROXY=https://proxy.example:8443",
                f"OPENAI_PROXY={invalid_openai}",
                "PRESERVE_UNKNOWN=present",
                "",
            )
        )
    )

    removed = repair_invalid_managed_provider_proxies({})

    values = dotenv_values_from_file(managed)
    assert removed == ("OPENROUTER_PROXY", "OPENAI_PROXY")
    assert "OPENROUTER_PROXY" not in values
    assert "OPENAI_PROXY" not in values
    assert values["GROQ_PROXY"] == "https://proxy.example:8443"
    assert values["MODEL"] == "nvidia_nim/test-model"
    assert values["PRESERVE_UNKNOWN"] == "present"
    assert list(managed.parent.glob(f".{managed.name}.*.tmp")) == []


def test_repair_valid_managed_provider_proxy_leaves_file_unchanged() -> None:
    managed = _write_managed_config(
        "# Keep this exact text on a no-op.\n"
        "FCC_CONFIG_SCHEMA=1\n"
        "OPENAI_PROXY=https://proxy.example:8443\n"
    )
    baseline = managed.read_bytes()

    assert repair_invalid_managed_provider_proxies({}) == ()
    assert managed.read_bytes() == baseline


def test_repair_without_managed_file_is_a_noop() -> None:
    managed = managed_env_path()

    assert repair_invalid_managed_provider_proxies({}) == ()
    assert not managed.exists()


@pytest.mark.parametrize("process_value", ("", "invalid://process-proxy"))
def test_repair_preserves_process_owned_managed_proxy(
    process_value: str,
) -> None:
    invalid_openai = "invalid://managed-proxy"
    managed = _write_managed_config(
        "FCC_CONFIG_SCHEMA=1\n"
        f"OPENAI_PROXY={invalid_openai}\n"
        "OPENROUTER_PROXY=invalid://unshadowed\n"
    )
    process = {"OPENAI_PROXY": process_value, "KEEP_PROCESS": "unchanged"}
    baseline_process = dict(process)

    assert repair_invalid_managed_provider_proxies(process) == ("OPENROUTER_PROXY",)

    values = dotenv_values_from_file(managed)
    assert values["OPENAI_PROXY"] == invalid_openai
    assert "OPENROUTER_PROXY" not in values
    assert process == baseline_process


def test_repair_propagates_atomic_write_failure_without_changing_source() -> None:
    managed = _write_managed_config(
        "FCC_CONFIG_SCHEMA=1\nOPENAI_PROXY=invalid://managed-proxy\n"
    )
    baseline = managed.read_bytes()

    with (
        patch.object(
            loader,
            "atomic_write_managed_config",
            side_effect=OSError("disk full"),
        ),
        pytest.raises(OSError, match="disk full"),
    ):
        repair_invalid_managed_provider_proxies({})

    assert managed.read_bytes() == baseline


def test_repair_is_idempotent_and_writes_only_once() -> None:
    managed = _write_managed_config(
        "FCC_CONFIG_SCHEMA=1\nOPENAI_PROXY=invalid://managed-proxy\n"
    )

    with patch.object(
        loader,
        "atomic_write_managed_config",
        wraps=loader.atomic_write_managed_config,
    ) as writer:
        assert repair_invalid_managed_provider_proxies({}) == ("OPENAI_PROXY",)
        repaired = managed.read_bytes()
        assert repair_invalid_managed_provider_proxies({}) == ()

    assert writer.call_count == 1
    assert managed.read_bytes() == repaired


def test_repair_propagates_malformed_managed_config() -> None:
    managed = _write_managed_config('FCC_CONFIG_SCHEMA=1\nOPENAI_PROXY="unterminated\n')
    baseline = managed.read_bytes()

    with pytest.raises(ValueError, match="Could not parse configuration file"):
        repair_invalid_managed_provider_proxies({})

    assert managed.read_bytes() == baseline


def test_repair_propagates_config_lock_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = _write_managed_config(
        "FCC_CONFIG_SCHEMA=1\nOPENAI_PROXY=invalid://managed-proxy\n"
    )
    baseline = managed.read_bytes()

    class UnavailableLock:
        def __init__(self, _path: Path) -> None:
            pass

        def acquire(self, *, wait: bool, timeout: float) -> bool:
            assert wait is True
            assert timeout == 10.0
            return False

    monkeypatch.setattr(loader, "InterprocessFileLock", UnavailableLock)

    with pytest.raises(TimeoutError, match="Could not acquire managed-config lock"):
        repair_invalid_managed_provider_proxies({})

    assert managed.read_bytes() == baseline


def test_optional_strings_share_one_normalization_rule() -> None:
    settings = Settings.model_validate(
        {
            "GROQ_API_KEY": "  key  ",
            "OPENROUTER_PROXY": " ",
            "MODEL_OPUS": "",
            "ALLOWED_DIR": None,
        }
    )

    assert settings.groq_api_key == "key"
    assert settings.open_router_proxy is None
    assert settings.model_opus is None
    assert settings.allowed_dir is None


@pytest.mark.parametrize("value", [None, "", "   ", (), []])
def test_model_fallbacks_empty_values_disable_fallback(value: object) -> None:
    settings = Settings.model_validate({"MODEL_FALLBACKS": value})

    assert settings.model_fallbacks is None


@pytest.mark.parametrize(
    "value",
    [
        "open_router/vendor/model-a, groq/vendor/model-b ",
        ("open_router/vendor/model-a", " groq/vendor/model-b "),
        ["open_router/vendor/model-a", "groq/vendor/model-b"],
    ],
)
def test_model_fallbacks_preserve_order_and_trim_members(value: object) -> None:
    settings = Settings.model_validate({"MODEL_FALLBACKS": value})

    assert settings.model_fallbacks == (
        "open_router/vendor/model-a",
        "groq/vendor/model-b",
    )


@pytest.mark.parametrize(
    "value",
    [
        "open_router/vendor/model-a,,groq/vendor/model-b",
        "open_router/vendor/model-a,open_router/vendor/model-a",
    ],
)
def test_model_fallbacks_reject_blank_and_duplicate_members(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({"MODEL_FALLBACKS": value})


@pytest.mark.parametrize(
    "field",
    ["MODEL", "HOST", "WHISPER_MODEL", "LOG_LEVEL", "ANTHROPIC_AUTH_TOKEN"],
)
def test_required_strings_reject_blank(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate({field: "   "})


def test_model_validation_and_routing() -> None:
    settings = Settings(
        model="deepseek/fallback",
        model_opus="open_router/anthropic/claude-opus",
    )

    router = ModelRouter(settings)
    assert router.resolve("claude-opus-4").primary.provider_model_ref == (
        "open_router/anthropic/claude-opus"
    )
    assert router.resolve("unknown").primary.provider_model_ref == "deepseek/fallback"
    with pytest.raises(ValidationError, match="Invalid provider"):
        Settings(model="unknown/model")


@pytest.mark.parametrize(
    "field",
    ["MODEL", "MODEL_FABLE", "MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU"],
)
def test_model_settings_reject_empty_model_suffix(field: str) -> None:
    with pytest.raises(ValidationError, match="model suffix"):
        Settings.model_validate({field: "open_router/"})


def test_configured_chat_model_refs_are_unique() -> None:
    settings = Settings(
        model="deepseek/fallback",
        model_fable="open_router/anthropic/claude-fable",
        model_sonnet="deepseek/fallback",
        model_fallbacks=(
            "groq/vendor/model-a",
            "open_router/anthropic/claude-fable",
            "lmstudio/vendor/model-b",
        ),
    )

    refs = configured_chat_model_refs(settings)

    assert [ref.model_ref for ref in refs] == [
        "deepseek/fallback",
        "open_router/anthropic/claude-fable",
        "groq/vendor/model-a",
        "lmstudio/vendor/model-b",
    ]


@pytest.mark.parametrize(
    ("model_ref", "provider", "model"),
    [
        ("nvidia_nim/meta/llama", "nvidia_nim", "meta/llama"),
        ("open_router/deepseek/r1", "open_router", "deepseek/r1"),
        ("ollama_cloud/qwen3-coder:480b", "ollama_cloud", "qwen3-coder:480b"),
    ],
)
def test_model_ref_parsing(model_ref: str, provider: str, model: str) -> None:
    assert parse_provider_type(model_ref) == provider
    assert parse_model_name(model_ref) == model


def test_paths_are_owned_by_fcc_home(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    assert messaging_state_dir_path() == tmp_path / ".fcc" / "agent_workspace"
    assert server_log_path() == tmp_path / ".fcc" / "logs" / "server.log"


def test_nim_settings_keep_request_local_validation() -> None:
    settings = NimSettings.model_validate(
        {
            "max_tokens": "1024",
            "temperature": "0.5",
            "seed": "7",
            "stop": "",
        }
    )

    assert settings.max_tokens == 1024
    assert settings.temperature == 0.5
    assert settings.seed == 7
    assert settings.stop is None
    assert NimSettings().max_tokens == ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
    assert NimSettings().top_p == 0.95
    for unsupported_top_p in (0.0, 0.9, 1.0):
        with pytest.raises(ValidationError):
            NimSettings(top_p=unsupported_top_p)


class TestSettingsEmptyStringNormalization:
    """Blank optional env values normalize to None at the loader boundary.

    Upstream moved Settings off BaseSettings: direct construction performs no
    environment I/O, so these fork regression checks go through the loader.
    """

    @pytest.mark.parametrize(
        ("key", "attribute", "value", "expected"),
        [
            ("TELEGRAM_BOT_TOKEN", "telegram_bot_token", "abc123", "abc123"),
            ("TELEGRAM_BOT_TOKEN", "telegram_bot_token", "", None),
            ("ALLOWED_TELEGRAM_USER_ID", "allowed_telegram_user_id", "", None),
            (
                "DISCORD_BOT_TOKEN",
                "discord_bot_token",
                "discord_token_123",
                "discord_token_123",
            ),
            ("DISCORD_BOT_TOKEN", "discord_bot_token", "", None),
            (
                "ALLOWED_DISCORD_CHANNELS",
                "allowed_discord_channels",
                "111,222,333",
                "111,222,333",
            ),
            ("MESSAGING_PLATFORM", "messaging_platform", "discord", "discord"),
            ("WHISPER_DEVICE", "whisper_device", "cpu", "cpu"),
            ("WHISPER_DEVICE", "whisper_device", "cuda", "cuda"),
        ],
    )
    def test_optional_env_values_at_loader_boundary(
        self,
        key: str,
        attribute: str,
        value: str,
        expected: object,
    ) -> None:
        snapshot = compose_settings_snapshot({}, {key: value})

        assert getattr(snapshot.settings, attribute) == expected

    def test_whisper_device_auto_rejected(self) -> None:
        with pytest.raises(ValidationError, match="whisper_device"):
            compose_settings_snapshot({}, {"WHISPER_DEVICE": "auto"})


class TestPerModelMapping:
    """Test per-model settings and model-ref helpers."""

    def test_model_fields_default_none(self):
        """Per-model fields default to None."""
        from free_claude_code.config.settings import Settings

        s = Settings()
        assert s.model_fable is None
        assert s.model_opus is None
        assert s.model_sonnet is None
        assert s.model_haiku is None

    def test_model_opus_loader(self):
        """MODEL_OPUS env var is loaded at the loader boundary."""
        snapshot = compose_settings_snapshot(
            {}, {"MODEL_OPUS": "open_router/deepseek/deepseek-r1"}
        )
        assert snapshot.settings.model_opus == "open_router/deepseek/deepseek-r1"

    def test_model_fable_loader(self):
        """MODEL_FABLE env var is loaded at the loader boundary."""
        snapshot = compose_settings_snapshot(
            {}, {"MODEL_FABLE": "open_router/anthropic/claude-fable-5"}
        )
        assert snapshot.settings.model_fable == "open_router/anthropic/claude-fable-5"

    def test_model_sonnet_loader(self):
        """MODEL_SONNET env var is loaded at the loader boundary."""
        snapshot = compose_settings_snapshot(
            {}, {"MODEL_SONNET": "nvidia_nim/meta/llama-3.3-70b-instruct"}
        )
        assert (
            snapshot.settings.model_sonnet == "nvidia_nim/meta/llama-3.3-70b-instruct"
        )

    def test_model_haiku_loader(self):
        """MODEL_HAIKU env var is loaded at the loader boundary."""
        snapshot = compose_settings_snapshot({}, {"MODEL_HAIKU": "lmstudio/qwen2.5-7b"})
        assert snapshot.settings.model_haiku == "lmstudio/qwen2.5-7b"

    @pytest.mark.parametrize(
        "env_var", ["MODEL_FABLE", "MODEL_OPUS", "MODEL_SONNET", "MODEL_HAIKU"]
    )
    def test_empty_model_override_env_is_unset(self, env_var: str):
        """Empty per-model override env vars are treated as unset."""
        from free_claude_code.application.routing import ModelRouter

        settings = compose_settings_snapshot({}, {env_var: ""}).settings
        assert getattr(settings, env_var.lower()) is None
        model_name = env_var.removeprefix("MODEL_").lower()
        assert (
            ModelRouter(settings)
            .resolve(f"claude-{model_name}-4")
            .primary.provider_model_ref
            == settings.model
        )

    @pytest.mark.parametrize(
        "env_vars,expected_model,expected_haiku",
        [
            (
                {"MODEL": "nvidia_nim/meta/llama3-70b-instruct"},
                "nvidia_nim/meta/llama3-70b-instruct",
                None,
            ),
            (
                {
                    "MODEL": "open_router/anthropic/claude-3-opus",
                    "MODEL_HAIKU": "open_router/anthropic/claude-3-haiku",
                },
                "open_router/anthropic/claude-3-opus",
                "open_router/anthropic/claude-3-haiku",
            ),
            ({"MODEL": "deepseek/deepseek-chat"}, "deepseek/deepseek-chat", None),
            ({"MODEL": "wafer/DeepSeek-V4-Pro"}, "wafer/DeepSeek-V4-Pro", None),
            (
                {"MODEL": "cloudflare/@cf/moonshotai/kimi-k2.6"},
                "cloudflare/@cf/moonshotai/kimi-k2.6",
                None,
            ),
            (
                # Retired provider refs are migrated to the default model at
                # the loader boundary (upstream: retire github_models #1668).
                {"MODEL": "github_models/openai/gpt-4.1"},
                DEFAULT_MODEL,
                None,
            ),
            (
                {"MODEL": "sambanova/Meta-Llama-3.3-70B-Instruct"},
                "sambanova/Meta-Llama-3.3-70B-Instruct",
                None,
            ),
            ({"MODEL": "lmstudio/qwen2.5-7b"}, "lmstudio/qwen2.5-7b", None),
            ({"MODEL": "llamacpp/local-model"}, "llamacpp/local-model", None),
            ({"MODEL": "ollama/llama3.1"}, "ollama/llama3.1", None),
            (
                {"MODEL": "ollama_cloud/qwen3-coder:480b"},
                "ollama_cloud/qwen3-coder:480b",
                None,
            ),
        ],
    )
    def test_settings_models_from_env(
        self,
        env_vars: dict[str, str],
        expected_model: str,
        expected_haiku: str | None,
    ):
        """Environment variables override model defaults."""
        settings = compose_settings_snapshot({}, env_vars).settings
        assert settings.model == expected_model
        assert settings.model_haiku == expected_haiku

    @pytest.mark.parametrize(
        ("env_var", "value", "message"),
        [
            ("MODEL_OPUS", "bad_provider/some-model", "Invalid provider"),
            ("MODEL_OPUS", "noprefix", "provider type"),
            ("MODEL_HAIKU", "invalid/model", "Invalid provider"),
            ("MODEL_FABLE", "invalid/model", "Invalid provider"),
            ("MODEL_COMPACT", "invalid/model", "Invalid provider"),
        ],
    )
    def test_invalid_model_refs_raise_at_loader_boundary(
        self, env_var: str, value: str, message: str
    ):
        """Malformed per-model refs are rejected during validation."""
        with pytest.raises(ValidationError, match=message):
            compose_settings_snapshot({}, {env_var: value})

    def test_model_compact_loader(self):
        """MODEL_COMPACT loads through the boundary and blanks to None."""
        loaded = compose_settings_snapshot(
            {}, {"MODEL_COMPACT": "opencode_go/anthropic/claude-fable-5"}
        ).settings
        assert loaded.model_compact == "opencode_go/anthropic/claude-fable-5"

        blank = compose_settings_snapshot({}, {"MODEL_COMPACT": ""}).settings
        assert blank.model_compact is None

    def test_model_compact_default_is_none(self):
        """MODEL_COMPACT defaults to None when unset."""
        from free_claude_code.config.settings import Settings

        assert Settings().model_compact is None

    def test_fcc_1m_models_default_is_none(self):
        """FCC_1M_MODELS defaults to None when unset (upstream ban on empty strings)."""
        from free_claude_code.config.settings import Settings

        assert Settings().fcc_1m_models is None

    def test_fcc_1m_models_loaded_from_env(self):
        """FCC_1M_MODELS env var is loaded into settings."""
        snapshot = compose_settings_snapshot(
            {},
            {
                "FCC_1M_MODELS": (
                    "opencode_go/deepseek-v4-pro,opencode_go/deepseek-v4-flash"
                )
            },
        )
        assert snapshot.settings.fcc_1m_models == (
            "opencode_go/deepseek-v4-pro,opencode_go/deepseek-v4-flash"
        )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (
                "opencode_go/deepseek-v4-pro,opencode_go/deepseek-v4-flash",
                frozenset(
                    {
                        "opencode_go/deepseek-v4-pro",
                        "opencode_go/deepseek-v4-flash",
                    }
                ),
            ),
            (
                "opencode_go/deepseek-v4-pro[1m]",
                frozenset({"opencode_go/deepseek-v4-pro"}),
            ),
            (" a/b , c/d ", frozenset({"a/b", "c/d"})),
            ("", frozenset()),
        ],
    )
    def test_one_m_model_refs_parsing(self, value: str, expected: frozenset[str]):
        """one_m_model_refs parses, trims, strips [1m], and tolerates blanks."""
        settings = compose_settings_snapshot({}, {"FCC_1M_MODELS": value}).settings
        assert settings.one_m_model_refs() == expected

    def test_resolve_model_fable_override(self):
        """ModelRouter returns model_fable for Fable model names."""
        from free_claude_code.application.routing import ModelRouter
        from free_claude_code.config.settings import Settings

        s = Settings(model_fable="open_router/anthropic/claude-fable-5")
        assert (
            ModelRouter(s).resolve("claude-fable-5").primary.provider_model_ref
            == "open_router/anthropic/claude-fable-5"
        )

    def test_resolve_model_opus_override(self):
        """ModelRouter returns model_opus for opus model names."""
        from free_claude_code.application.routing import ModelRouter
        from free_claude_code.config.settings import Settings

        s = Settings(model_opus="open_router/deepseek/deepseek-r1")
        router = ModelRouter(s)
        for name in (
            "claude-opus-4-20250514",
            "claude-3-opus",
            "claude-3-opus-20240229",
        ):
            assert router.resolve(name).primary.provider_model_ref == (
                "open_router/deepseek/deepseek-r1"
            )

    def test_resolve_model_sonnet_override(self):
        """ModelRouter returns model_sonnet for sonnet model names."""
        from free_claude_code.application.routing import ModelRouter
        from free_claude_code.config.settings import Settings

        s = Settings(model_sonnet="nvidia_nim/meta/llama-3.3-70b-instruct")
        router = ModelRouter(s)
        for name in ("claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022"):
            assert router.resolve(name).primary.provider_model_ref == (
                "nvidia_nim/meta/llama-3.3-70b-instruct"
            )

    def test_resolve_model_haiku_override(self):
        """ModelRouter returns model_haiku for haiku model names."""
        from free_claude_code.application.routing import ModelRouter
        from free_claude_code.config.settings import Settings

        s = Settings(model_haiku="lmstudio/qwen2.5-7b")
        router = ModelRouter(s)
        for name in (
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022",
            "claude-haiku-4-20250514",
        ):
            assert router.resolve(name).primary.provider_model_ref == (
                "lmstudio/qwen2.5-7b"
            )

    def test_resolve_model_fallback_when_override_not_set(self):
        """ModelRouter falls back to MODEL when model override is None."""
        from free_claude_code.application.routing import ModelRouter
        from free_claude_code.config.settings import Settings

        s = Settings(model="nvidia_nim/fallback-model")
        router = ModelRouter(s)
        for name in (
            "claude-fable-5",
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            "claude-3-haiku-20240307",
        ):
            assert router.resolve(name).primary.provider_model_ref == (
                "nvidia_nim/fallback-model"
            )

    def test_resolve_model_unknown_model_falls_back(self):
        """ModelRouter falls back to MODEL for unrecognized model names."""
        from free_claude_code.application.routing import ModelRouter
        from free_claude_code.config.settings import Settings

        s = Settings(
            model="nvidia_nim/fallback-model",
            model_opus="open_router/opus-model",
        )
        router = ModelRouter(s)
        assert router.resolve("claude-2.1").primary.provider_model_ref == (
            "nvidia_nim/fallback-model"
        )
        assert router.resolve("some-unknown-model").primary.provider_model_ref == (
            "nvidia_nim/fallback-model"
        )

    def test_resolve_model_case_insensitive(self):
        """Model classification is case-insensitive."""
        from free_claude_code.application.routing import ModelRouter
        from free_claude_code.config.settings import Settings

        s = Settings(model_opus="open_router/opus-model")
        assert ModelRouter(s).resolve("Claude-OPUS-4").primary.provider_model_ref == (
            "open_router/opus-model"
        )

    def test_parse_provider_type(self):
        """parse_provider_type extracts provider from model string."""

        assert parse_provider_type("nvidia_nim/meta/llama") == "nvidia_nim"
        assert parse_provider_type("open_router/deepseek/r1") == "open_router"
        assert parse_provider_type("mistral/devstral-small-latest") == "mistral"
        assert (
            parse_provider_type("mistral_codestral/codestral-latest")
            == "mistral_codestral"
        )
        assert parse_provider_type("deepseek/deepseek-chat") == "deepseek"
        assert parse_provider_type("lmstudio/qwen") == "lmstudio"
        assert parse_provider_type("llamacpp/model") == "llamacpp"
        assert parse_provider_type("ollama/llama3.1") == "ollama"
        assert parse_provider_type("ollama_cloud/qwen3-coder:480b") == "ollama_cloud"
        assert parse_provider_type("wafer/DeepSeek-V4-Pro") == "wafer"
        assert parse_provider_type("minimax/MiniMax-M3") == "minimax"
        assert (
            parse_provider_type("cloudflare/@cf/moonshotai/kimi-k2.6") == "cloudflare"
        )
        assert parse_provider_type("vercel/openai/gpt-5.5") == "vercel"
        assert (
            parse_provider_type("huggingface/openai/gpt-oss-120b:fastest")
            == "huggingface"
        )
        assert parse_provider_type("cohere/command-a-plus-05-2026") == "cohere"
        assert parse_provider_type("github_models/openai/gpt-4.1") == ("github_models")
        assert parse_provider_type("gemini/models/gemini-3.1-flash-lite") == "gemini"
        assert parse_provider_type("groq/llama-3.3-70b-versatile") == "groq"
        assert (
            parse_provider_type("sambanova/Meta-Llama-3.3-70B-Instruct") == "sambanova"
        )
        assert parse_provider_type("cerebras/llama3.1-8b") == "cerebras"

    def test_parse_model_name(self):
        """parse_model_name extracts model name from model string."""

        assert parse_model_name("nvidia_nim/meta/llama") == "meta/llama"
        assert parse_model_name("mistral/devstral-small-latest") == (
            "devstral-small-latest"
        )
        assert (
            parse_model_name("mistral_codestral/codestral-latest") == "codestral-latest"
        )
        assert parse_model_name("deepseek/deepseek-chat") == "deepseek-chat"
        assert parse_model_name("lmstudio/qwen") == "qwen"
        assert parse_model_name("llamacpp/model") == "model"
        assert parse_model_name("ollama/llama3.1") == "llama3.1"
        assert parse_model_name("ollama_cloud/qwen3-coder:480b") == "qwen3-coder:480b"
        assert parse_model_name("wafer/DeepSeek-V4-Pro") == "DeepSeek-V4-Pro"
        assert parse_model_name("minimax/MiniMax-M3") == "MiniMax-M3"
        assert (
            parse_model_name("cloudflare/@cf/moonshotai/kimi-k2.6")
            == "@cf/moonshotai/kimi-k2.6"
        )
        assert parse_model_name("vercel/openai/gpt-5.5") == "openai/gpt-5.5"
        assert (
            parse_model_name("huggingface/openai/gpt-oss-120b:fastest")
            == "openai/gpt-oss-120b:fastest"
        )
        assert parse_model_name("cohere/command-a-plus-05-2026") == (
            "command-a-plus-05-2026"
        )
        assert parse_model_name("github_models/openai/gpt-4.1") == "openai/gpt-4.1"
        assert (
            parse_model_name("gemini/models/gemini-3.1-flash-lite")
            == "models/gemini-3.1-flash-lite"
        )
        assert (
            parse_model_name("groq/llama-3.3-70b-versatile")
            == "llama-3.3-70b-versatile"
        )
        assert (
            parse_model_name("sambanova/Meta-Llama-3.3-70B-Instruct")
            == "Meta-Llama-3.3-70B-Instruct"
        )
        assert parse_model_name("cerebras/llama3.1-8b") == "llama3.1-8b"

    def test_configured_chat_model_refs_collects_unique_models(self):
        """Model discovery is limited to configured chat references."""
        from free_claude_code.config.settings import Settings

        s = Settings()
        s.model = "nvidia_nim/fallback"
        s.model_fable = "open_router/anthropic/claude-fable-5"
        s.model_opus = "open_router/anthropic/claude-opus"
        s.model_sonnet = "nvidia_nim/fallback"
        s.model_haiku = None

        refs = configured_chat_model_refs(s)

        assert [ref.model_ref for ref in refs] == [
            "nvidia_nim/fallback",
            "open_router/anthropic/claude-fable-5",
            "open_router/anthropic/claude-opus",
        ]
        assert refs[0].provider_id == "nvidia_nim"
        assert refs[0].model_id == "fallback"
        assert refs[1].provider_id == "open_router"
        assert refs[1].model_id == "anthropic/claude-fable-5"
        assert refs[2].provider_id == "open_router"
        assert refs[2].model_id == "anthropic/claude-opus"


def test_settings_defaults_do_not_contain_empty_enum_strings() -> None:
    for _name, value in Settings():
        if isinstance(value, Enum):
            assert value.value
