"""Request builder for OpenCode Zen provider."""

from typing import Any

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
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


def _is_deepseek_v4_model(model: str) -> bool:
    """Return True for deepseek-v4-pro, deepseek-v4-flash, or prefixed variants."""
    name = model.rpartition("/")[-1] if "/" in model else model
    return any(name.startswith(prefix) for prefix in _DEEPSEEK_V4_MODEL_PREFIXES)


def _extract_anthropic_effort(request_data: Any) -> str | None:
    """Extract effort from ``output_config.effort`` or derive from ``thinking.budget_tokens``."""
    output_config = getattr(request_data, "output_config", None)
    if isinstance(output_config, dict) and output_config.get("effort"):
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


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict:
    """Build OpenAI-format request body from Anthropic request for OpenCode Zen."""
    logger.debug(
        "OPENCODE_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
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

    logger.debug(
        "OPENCODE_REQUEST: conversion done model={} msgs={} tools={}",
        body.get("model"),
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )
    return body
