"""Request builder for OpenCode Zen provider."""

from typing import Any

from loguru import logger

from api.models.anthropic import ContentBlockText
from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.content import get_block_type
from core.anthropic.conversion import OpenAIConversionError
from core.anthropic.native_messages_request import (
    build_base_native_anthropic_request_body,
)
from providers.exceptions import InvalidRequestError

# Anthropic adaptive thinking effort → DeepSeek reasoning_effort mapping.
# DeepSeek only has two effective tiers (low/medium → high, high/xhigh/max → max).
_ANTHROPIC_TO_DEEPSEEK_EFFORT = {
    "low": "high",
    "medium": "high",
    "high": "max",
    "xhigh": "max",
    "max": "max",
}

# budget_tokens thresholds for deriving effort when output_config.effort is absent.
_BUDGET_EFFORT_THRESHOLDS = (
    (4000, "low"),
    (12000, "medium"),
    (24000, "high"),
    (float("inf"), "max"),
)

_DEEPSEEK_V4_MODEL_PREFIXES = ("deepseek-v4",)

_IMAGE_STRIP_HINT = (
    "[Image attachment removed: DeepSeek models cannot view images natively. "
    "Use the `understand_image` tool to analyze images — call it with the "
    "image path or URL provided by the user.]"
)
_TOOL_IMAGE_STRIP_HINT = _IMAGE_STRIP_HINT

# Models that use Anthropic Messages API (@ai-sdk/anthropic) natively,
# rather than OpenAI Chat Completions (@ai-sdk/openai-compatible).
# Source: https://opencode.ai/docs/go
_ANTHROPIC_NATIVE_MODELS = frozenset(
    {
        "minimax-m3",
        "minimax-m2.7",
        "minimax-m2.5",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
    }
)


def _normalize_model_name(model: str) -> str:
    """Strip ``provider/`` prefix if present, returning the bare model id."""
    return model.rpartition("/")[-1] if "/" in model else model


def is_anthropic_native_model(model: str) -> bool:
    """Return True when *model* should use the Anthropic Messages endpoint."""
    return _normalize_model_name(model) in _ANTHROPIC_NATIVE_MODELS


def build_anthropic_request_body(
    request_data: Any,
    *,
    thinking_enabled: bool,
) -> dict:
    """Build a native Anthropic Messages request body for Anthropic-native models."""
    logger.debug(
        "OPENCODE_ANTHROPIC_REQUEST: native build model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    body = build_base_native_anthropic_request_body(
        request_data,
        default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
        thinking_enabled=thinking_enabled,
    )

    # Forward session / request identifiers to OpenCode Go.
    extra_headers: dict[str, str] = {}
    extra_headers["x-opencode-client"] = "fcc"
    session_id = getattr(request_data, "fcc_session_id", None)
    if session_id:
        extra_headers["x-opencode-session"] = session_id
    else:
        request_id = getattr(request_data, "fcc_request_id", None)
        if request_id:
            extra_headers["x-opencode-session"] = request_id
    request_id = getattr(request_data, "fcc_request_id", None)
    if request_id:
        extra_headers["x-opencode-request"] = request_id
    body["extra_headers"] = extra_headers

    logger.debug(
        "OPENCODE_ANTHROPIC_REQUEST: build done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body


def _is_deepseek_v4_model(model: str) -> bool:
    """Return True for deepseek-v4-pro, deepseek-v4-flash, or prefixed variants."""
    name = model.rpartition("/")[-1] if "/" in model else model
    return any(name.startswith(prefix) for prefix in _DEEPSEEK_V4_MODEL_PREFIXES)


def _extract_anthropic_effort(request_data: Any) -> str | None:
    """Extract effort from ``output_config.effort`` or derive from ``thinking.budget_tokens``."""
    output_config = getattr(request_data, "output_config", None)
    if isinstance(output_config, dict) and "effort" in output_config:
        return output_config["effort"]

    thinking = getattr(request_data, "thinking", None)
    if thinking is None:
        return None
    budget = getattr(thinking, "budget_tokens", None)
    if not isinstance(budget, int) or budget <= 0:
        return None
    for threshold, level in _BUDGET_EFFORT_THRESHOLDS:
        if budget <= threshold:
            return level
    return None


def _apply_deepseek_reasoning_effort(request_data: Any, body: dict) -> None:
    """Inject ``reasoning_effort`` for DeepSeek V4 models when thinking is enabled."""
    effort = _extract_anthropic_effort(request_data)
    if effort is None:
        return
    deepseek_effort = _ANTHROPIC_TO_DEEPSEEK_EFFORT.get(effort)
    if deepseek_effort is None:
        logger.debug(
            "OPENCODE_REQUEST: unknown Anthropic effort={} for model={}, skipping",
            effort,
            body.get("model"),
        )
        return
    body["reasoning_effort"] = deepseek_effort
    logger.debug(
        "OPENCODE_REQUEST: set reasoning_effort={} (from Anthropic effort={})",
        deepseek_effort,
        effort,
    )


def _strip_image_blocks_and_hint(request_data: Any) -> bool:
    """Strip image blocks from a MessagesRequest and inject hints for DeepSeek V4.

    Modifies messages in-place so the OpenAI converter never sees image blocks.
    Handles both top-level image blocks in user messages and images nested inside
    ``tool_result.content`` lists (e.g. Firefox MCP screenshot_page output).
    Returns True if any images were stripped.
    """
    messages = getattr(request_data, "messages", [])
    if not messages:
        return False

    stripped_any = False
    for msg in messages:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue

        had_image = False
        new_content: list[Any] = []
        for block in content:
            btype = get_block_type(block)
            if btype == "image":
                had_image = True
                stripped_any = True
                continue
            if btype == "tool_result":
                tool_content = getattr(block, "content", None)
                if isinstance(tool_content, list):
                    filtered_tool_content: list[Any] = []
                    tool_had_image = False
                    for sub in tool_content:
                        if get_block_type(sub) == "image":
                            tool_had_image = True
                            stripped_any = True
                            continue
                        filtered_tool_content.append(sub)
                    if tool_had_image:
                        if not filtered_tool_content:
                            filtered_tool_content = [
                                ContentBlockText(
                                    type="text",
                                    text=_TOOL_IMAGE_STRIP_HINT,
                                ),
                            ]
                        block.content = filtered_tool_content
            new_content.append(block)

        if had_image:
            new_content.append(ContentBlockText(type="text", text=_IMAGE_STRIP_HINT))
            msg.content = new_content

    return stripped_any


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from Anthropic request for OpenCode Zen."""
    logger.debug(
        "OPENCODE_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )

    # DeepSeek V4 models lack vision support — strip image blocks and inject
    # a hint so the model knows it can use the understand_image tool instead.
    model = getattr(request_data, "model", "")
    if _is_deepseek_v4_model(model):
        _strip_image_blocks_and_hint(request_data)

    try:
        body = build_base_request_body(
            request_data,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT
            if thinking_enabled
            else ReasoningReplayMode.DISABLED,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    model = body.get("model", "")
    if thinking_enabled and _is_deepseek_v4_model(model):
        _apply_deepseek_reasoning_effort(request_data, body)

    # Forward session / request identifiers to OpenCode Go so the billing
    # dashboard correlates requests originating from the same Claude Code session.
    extra_headers: dict[str, str] = {}
    extra_headers["x-opencode-client"] = "fcc"
    session_id = getattr(request_data, "fcc_session_id", None)
    if session_id:
        extra_headers["x-opencode-session"] = session_id
    else:
        # Fall back to request ID when Claude Code doesn't provide a session header.
        request_id = getattr(request_data, "fcc_request_id", None)
        if request_id:
            extra_headers["x-opencode-session"] = request_id
    request_id = getattr(request_data, "fcc_request_id", None)
    if request_id:
        extra_headers["x-opencode-request"] = request_id
    body["extra_headers"] = extra_headers

    logger.debug(
        "OPENCODE_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
