from unittest.mock import patch

import pytest

from free_claude_code.application.errors import UnknownProviderError
from free_claude_code.application.routing import ModelRouter
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.models import (
    Message,
    MessagesRequest,
    TokenCountRequest,
)
from free_claude_code.core.openai_responses import OpenAIResponsesRequest
from free_claude_code.core.reasoning import ReasoningControl, ReasoningEffort


@pytest.fixture
def settings():
    settings = Settings()
    settings.model = "nvidia_nim/fallback-model"
    settings.model_fable = None
    settings.model_opus = None
    settings.model_sonnet = None
    settings.model_haiku = None
    settings.model_compact = None
    settings.fcc_1m_models = None
    settings.reasoning_policy = ReasoningPreference.CLIENT
    settings.reasoning_fable = ReasoningPreference.INHERIT
    settings.reasoning_opus = ReasoningPreference.INHERIT
    settings.reasoning_sonnet = ReasoningPreference.INHERIT
    settings.reasoning_haiku = ReasoningPreference.INHERIT
    return settings


def test_model_router_resolves_default_model(settings):
    resolved = ModelRouter(settings).resolve("claude-3-opus")

    assert resolved.original_model == "claude-3-opus"
    assert resolved.primary.provider_id == "nvidia_nim"
    assert resolved.primary.provider_model == "fallback-model"
    assert resolved.primary.provider_model_ref == "nvidia_nim/fallback-model"
    assert resolved.fallbacks == ()
    assert resolved.reasoning_preference is ReasoningPreference.CLIENT


def test_model_router_applies_opus_override(settings):
    settings.model_opus = "open_router/deepseek/deepseek-r1"

    request = MessagesRequest(
        model="claude-opus-4-20250514",
        max_tokens=100,
        messages=[Message(role="user", content="hello")],
    )
    routed = ModelRouter(settings).resolve_messages_request(request)

    assert routed.request.model == "deepseek/deepseek-r1"
    assert (
        routed.resolved.primary.provider_model_ref == "open_router/deepseek/deepseek-r1"
    )
    assert routed.resolved.original_model == "claude-opus-4-20250514"
    assert routed.reasoning.control is ReasoningControl.DEFAULT
    assert request.model == "claude-opus-4-20250514"


def test_model_router_applies_fable_override(settings):
    settings.model_fable = "open_router/anthropic/claude-fable-5"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-fable-5",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "anthropic/claude-fable-5"
    assert (
        routed.resolved.primary.provider_model_ref
        == "open_router/anthropic/claude-fable-5"
    )
    assert routed.resolved.original_model == "claude-fable-5"


def test_model_router_resolves_route_reasoning_preferences(settings):
    settings.reasoning_policy = ReasoningPreference.OFF
    settings.reasoning_fable = ReasoningPreference.HIGH
    settings.reasoning_opus = ReasoningPreference.MAX
    settings.reasoning_haiku = ReasoningPreference.OFF

    router = ModelRouter(settings)

    assert (
        router.resolve("claude-fable-5").reasoning_preference
        is ReasoningPreference.HIGH
    )
    assert (
        router.resolve("claude-opus-4-20250514").reasoning_preference
        is ReasoningPreference.MAX
    )
    assert (
        router.resolve("claude-sonnet-4-20250514").reasoning_preference
        is ReasoningPreference.OFF
    )
    assert (
        router.resolve("claude-3-haiku-20240307").reasoning_preference
        is ReasoningPreference.OFF
    )
    assert router.resolve("claude-2.1").reasoning_preference is ReasoningPreference.OFF


def test_model_router_applies_haiku_override(settings):
    settings.model_haiku = "lmstudio/qwen2.5-7b"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "qwen2.5-7b"
    assert routed.resolved.primary.provider_model_ref == "lmstudio/qwen2.5-7b"


def test_model_router_applies_sonnet_override(settings):
    settings.model_sonnet = "nvidia_nim/meta/llama-3.3-70b-instruct"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "meta/llama-3.3-70b-instruct"
    assert (
        routed.resolved.primary.provider_model_ref
        == "nvidia_nim/meta/llama-3.3-70b-instruct"
    )


def test_model_router_routes_prefixed_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="deepseek/deepseek-chat",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "deepseek-chat"
    assert routed.resolved.original_model == "deepseek/deepseek-chat"
    assert routed.resolved.primary.provider_id == "deepseek"
    assert routed.resolved.primary.provider_model == "deepseek-chat"
    assert routed.resolved.primary.provider_model_ref == "deepseek/deepseek-chat"


def test_model_router_routes_explicit_opencode_zen_prefix(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="opencode_zen/kimi-k2.6",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "kimi-k2.6"
    assert routed.resolved.primary.provider_id == "opencode_zen"
    assert routed.resolved.primary.provider_model_ref == "opencode_zen/kimi-k2.6"


def test_model_router_routes_wafer_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="wafer/DeepSeek-V4-Pro",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "DeepSeek-V4-Pro"
    assert routed.resolved.primary.provider_id == "wafer"
    assert routed.resolved.primary.provider_model == "DeepSeek-V4-Pro"
    assert routed.resolved.primary.provider_model_ref == "wafer/DeepSeek-V4-Pro"


def test_model_router_routes_minimax_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="minimax/MiniMax-M3",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "MiniMax-M3"
    assert routed.resolved.primary.provider_id == "minimax"
    assert routed.resolved.primary.provider_model == "MiniMax-M3"
    assert routed.resolved.primary.provider_model_ref == "minimax/MiniMax-M3"


def test_model_router_routes_gateway_encoded_provider_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="anthropic/nvidia_nim/deepseek-ai/deepseek-v4-pro",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "deepseek-ai/deepseek-v4-pro"
    assert (
        routed.resolved.original_model
        == "anthropic/nvidia_nim/deepseek-ai/deepseek-v4-pro"
    )
    assert routed.resolved.primary.provider_id == "nvidia_nim"
    assert routed.resolved.primary.provider_model == "deepseek-ai/deepseek-v4-pro"
    assert (
        routed.resolved.primary.provider_model_ref
        == "nvidia_nim/deepseek-ai/deepseek-v4-pro"
    )


def test_model_router_routes_no_thinking_gateway_model_directly(settings):
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-3-freecc-no-thinking/nvidia_nim/deepseek-ai/deepseek-v4-pro",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.request.model == "deepseek-ai/deepseek-v4-pro"
    assert (
        routed.resolved.original_model
        == "claude-3-freecc-no-thinking/nvidia_nim/deepseek-ai/deepseek-v4-pro"
    )
    assert routed.resolved.primary.provider_id == "nvidia_nim"
    assert routed.resolved.primary.provider_model == "deepseek-ai/deepseek-v4-pro"
    assert routed.reasoning.control is ReasoningControl.OFF


def test_direct_provider_model_uses_root_policy_without_model_name_guessing(settings):
    settings.reasoning_policy = ReasoningPreference.LOW
    settings.reasoning_opus = ReasoningPreference.MAX

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="open_router/anthropic/claude-opus-4",
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.resolved.primary.provider_id == "open_router"
    assert routed.resolved.primary.provider_model == "anthropic/claude-opus-4"
    assert routed.reasoning.effort is ReasoningEffort.LOW


def test_model_router_routes_token_count_request(settings):
    settings.model_haiku = "lmstudio/qwen2.5-7b"

    request = TokenCountRequest(
        model="claude-3-haiku-20240307",
        messages=[Message(role="user", content="hello")],
    )
    routed = ModelRouter(settings).resolve_token_count_request(request)

    assert routed.request.model == "qwen2.5-7b"
    assert request.model == "claude-3-haiku-20240307"


def test_model_router_routes_responses_request_without_protocol_conversion(settings):
    settings.model_fallbacks = ("open_router/vendor/fallback",)
    request = OpenAIResponsesRequest(
        model="opencode_zen/responses-model",
        input=[{"role": "user", "content": "hello"}],
        reasoning={"effort": "high"},
        previous_response_id="resp_previous",
    )

    routed = ModelRouter(settings).resolve_responses_request(request)

    assert routed.request.model == "responses-model"
    assert routed.request.input == request.input
    assert routed.request.previous_response_id == "resp_previous"
    assert routed.resolved.original_model == "opencode_zen/responses-model"
    assert routed.resolved.primary.provider_id == "opencode_zen"
    assert routed.resolved.primary.provider_model == "responses-model"
    assert [target.provider_model_ref for target in routed.resolved.fallbacks] == [
        "open_router/vendor/fallback"
    ]
    assert routed.reasoning.control is ReasoningControl.DEFAULT
    assert routed.reasoning.effort is ReasoningEffort.HIGH
    assert request.model == "opencode_zen/responses-model"


def test_model_router_appends_global_fallbacks_and_skips_only_primary(settings):
    settings.model_fallbacks = (
        "nvidia_nim/fallback-model",
        "nvidia_nim/alternate-model",
        "open_router/vendor/model-b",
    )

    resolved = ModelRouter(settings).resolve("claude-2.1")

    assert [target.provider_model_ref for target in resolved.fallbacks] == [
        "nvidia_nim/alternate-model",
        "open_router/vendor/model-b",
    ]


def test_direct_gateway_route_preserves_original_and_deduplicates_canonical_target(
    settings,
):
    settings.model_fallbacks = (
        "nvidia_nim/deepseek-ai/deepseek-v4-pro",
        "groq/vendor/model-b",
    )
    model = "anthropic/nvidia_nim/deepseek-ai/deepseek-v4-pro"

    resolved = ModelRouter(settings).resolve(model)

    assert resolved.original_model == model
    assert (
        resolved.primary.provider_model_ref == "nvidia_nim/deepseek-ai/deepseek-v4-pro"
    )
    assert [target.provider_model_ref for target in resolved.fallbacks] == [
        "groq/vendor/model-b"
    ]


def test_model_router_logs_mapping(settings):
    with patch("free_claude_code.application.routing.logger.debug") as mock_log:
        ModelRouter(settings).resolve("claude-2.1")

    mock_log.assert_called()
    args = mock_log.call_args[0]
    assert "MODEL MAPPING" in args[0]
    assert args[1] == "claude-2.1"
    assert args[2] == "fallback-model"


def test_model_router_preserves_typed_error_for_unknown_mapped_provider(settings):
    settings.model = "unknown/model"

    with pytest.raises(UnknownProviderError) as exc_info:
        ModelRouter(settings).resolve("claude-2.1")

    supported = "', '".join(PROVIDER_CATALOG)
    assert str(exc_info.value) == (
        f"Unknown provider_type: 'unknown'. Supported: '{supported}'"
    )


# ----------------- [1m] suffix stripping -----------------


def test_resolve_strips_1m_suffix_from_direct_provider_model(settings):
    """Direct provider model with [1m] suffix is stripped before upstream send."""
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="anthropic/opencode_go/deepseek-v4-pro[1m]",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.resolved.primary.provider_id == "opencode_go"
    assert routed.resolved.primary.provider_model == "deepseek-v4-pro"
    assert routed.resolved.primary.provider_model_ref == "opencode_go/deepseek-v4-pro"
    assert routed.resolved.original_model == (
        "anthropic/opencode_go/deepseek-v4-pro[1m]"
    )
    assert routed.request.model == "deepseek-v4-pro"


def test_resolve_strips_1m_suffix_from_mapped_model(settings):
    """A configured MODEL_SONNET that ends in [1m] is stripped before upstream send."""
    settings.model_sonnet = "opencode_go/deepseek-v4-pro[1m]"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert routed.resolved.primary.provider_id == "opencode_go"
    assert routed.resolved.primary.provider_model == "deepseek-v4-pro"
    assert routed.resolved.primary.provider_model_ref == "opencode_go/deepseek-v4-pro"
    assert routed.request.model == "deepseek-v4-pro"


def test_resolve_keeps_1m_only_in_original_model(settings):
    """Canonical targets are suffix-stripped; the raw request id keeps [1m]."""
    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="anthropic/opencode_go/deepseek-v4-pro[1m]",
            max_tokens=100,
            messages=[Message(role="user", content="hello")],
        )
    )

    assert "[1m]" in routed.resolved.original_model
    assert "[1m]" not in routed.resolved.primary.provider_model_ref


# ----------------- MODEL_COMPACT override -----------------


def test_resolve_messages_request_uses_compact_model_on_compaction_request(settings):
    """Compaction-shaped request is rerouted to settings.model_compact."""
    settings.model_compact = "opencode_go/deepseek-v4-flash"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system="You are a helpful assistant summarizing conversations.",
            messages=[Message(role="user", content="Summarize this thread")],
        )
    )

    assert routed.resolved.original_model == "<compact>"
    assert routed.resolved.primary.provider_id == "opencode_go"
    assert routed.resolved.primary.provider_model == "deepseek-v4-flash"
    assert routed.resolved.primary.provider_model_ref == "opencode_go/deepseek-v4-flash"
    assert routed.request.model == "deepseek-v4-flash"


def test_resolve_messages_request_falls_back_when_compact_unset(settings):
    """Compaction-shaped request falls through to normal routing when MODEL_COMPACT is unset."""
    assert settings.model_compact is None

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system="You are a helpful assistant summarizing conversations.",
            messages=[Message(role="user", content="Summarize this thread")],
        )
    )

    assert routed.resolved.original_model == "claude-sonnet-4-20250514"
    # Default model in the test fixture is "nvidia_nim/fallback-model".
    assert routed.resolved.primary.provider_id == "nvidia_nim"
    assert routed.resolved.primary.provider_model == "fallback-model"


def test_resolve_messages_request_no_compaction_when_marker_absent(settings):
    """Non-compaction request with MODEL_COMPACT set still uses normal routing."""
    settings.model_compact = "opencode_go/deepseek-v4-flash"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[Message(role="user", content="Hello, world")],
        )
    )

    assert routed.resolved.original_model == "claude-sonnet-4-20250514"
    assert routed.resolved.primary.provider_model_ref == "nvidia_nim/fallback-model"


def test_resolve_messages_request_detects_compact_user_marker(settings):
    """Last user message starting with 'CRITICAL: Respond with TEXT ONLY' triggers override."""
    settings.model_compact = "opencode_go/deepseek-v4-flash"

    routed = ModelRouter(settings).resolve_messages_request(
        MessagesRequest(
            model="claude-sonnet-4-20250514",
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[
                Message(
                    role="user",
                    content="CRITICAL: Respond with TEXT ONLY. Do not use tools.",
                )
            ],
        )
    )

    assert routed.resolved.original_model == "<compact>"
    assert routed.resolved.primary.provider_model == "deepseek-v4-flash"


def test_resolve_messages_request_preserves_request_body_for_compaction(settings):
    """Compaction request body is passed through unchanged (no sanitization)."""
    from free_claude_code.core.anthropic.models import (
        ContentBlockToolUse,
        Tool,
    )

    settings.model_compact = "opencode_go/deepseek-v4-flash"

    request = MessagesRequest(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        system="You are a helpful assistant summarizing conversations.",
        tools=[Tool(name="search", description="search the web")],
        messages=[
            Message(
                role="user",
                content=[
                    ContentBlockToolUse(
                        type="tool_use",
                        id="toolu_1",
                        name="search",
                        input={"q": "weather"},
                    )
                ],
            )
        ],
    )

    original_tools = request.tools
    original_content = request.messages[0].content

    routed = ModelRouter(settings).resolve_messages_request(request)

    # Model is replaced, but tools and tool blocks are NOT stripped.
    assert routed.request.model == "deepseek-v4-flash"
    assert routed.request.tools == original_tools
    assert isinstance(routed.request.messages[0].content, list)
    assert isinstance(routed.request.messages[0].content[0], ContentBlockToolUse)
    # Original request is untouched.
    assert request.model == "claude-sonnet-4-20250514"
    assert request.tools is not None
    assert request.messages[0].content is original_content
