"""Model routing for Claude-compatible requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from config.provider_ids import SUPPORTED_PROVIDER_IDS
from config.settings import Settings

from .gateway_model_ids import decode_gateway_model_id
from .models.anthropic import MessagesRequest, TokenCountRequest


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    original_model: str
    provider_id: str
    provider_model: str
    provider_model_ref: str
    thinking_enabled: bool


@dataclass(frozen=True, slots=True)
class RoutedMessagesRequest:
    request: MessagesRequest
    resolved: ResolvedModel


@dataclass(frozen=True, slots=True)
class RoutedTokenCountRequest:
    request: TokenCountRequest
    resolved: ResolvedModel


COMPACT_SYSTEM_MARKER = "summarizing conversations"
COMPACT_USER_MARKER = "CRITICAL: Respond with TEXT ONLY"


def _extract_system_text(system: str | list[Any] | None) -> str:
    """Flatten the system prompt to a single searchable string."""
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    parts: list[str] = []
    for block in system:
        if isinstance(block, dict):
            parts.append(str(block.get("text", "")))
        elif hasattr(block, "text"):
            parts.append(str(block.text))
    return " ".join(parts)


def _extract_last_user_text(messages: list[Any]) -> str:
    """Return text content of the last user message for compaction detection."""
    for msg in reversed(messages):
        role = getattr(msg, "role", None)
        if role != "user":
            continue
        content = getattr(msg, "content", None)
        if content is None:
            continue
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            elif hasattr(block, "text"):
                parts.append(str(block.text))
        return " ".join(parts)
    return ""


def _is_compaction_request(request: MessagesRequest) -> bool:
    """Detect a compaction / summarization request from Claude Code.

    Claude Code sends compaction with:
      1. System prompt: "You are a helpful AI assistant tasked with summarizing conversations."
      2. Last user message starts with: "CRITICAL: Respond with TEXT ONLY..."

    Either signal is sufficient to classify the request as compaction.
    """
    system_text = _extract_system_text(request.system)
    if COMPACT_SYSTEM_MARKER in system_text:
        return True
    user_text = _extract_last_user_text(request.messages)
    if COMPACT_USER_MARKER in user_text:
        return True
    return False


class ModelRouter:
    """Resolve incoming Claude model names to configured provider/model pairs."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def resolve(self, claude_model_name: str) -> ResolvedModel:
        (
            direct_provider_id,
            direct_provider_model,
            force_thinking_enabled,
        ) = self._direct_provider_model(claude_model_name)
        if direct_provider_id is not None and direct_provider_model is not None:
            thinking_enabled = (
                force_thinking_enabled
                if force_thinking_enabled is not None
                else self._settings.resolve_thinking(direct_provider_model)
            )
            logger.debug(
                "MODEL DIRECT: '{}' -> provider='{}' model='{}' thinking={}",
                claude_model_name,
                direct_provider_id,
                direct_provider_model,
                thinking_enabled,
            )
            return ResolvedModel(
                original_model=claude_model_name,
                provider_id=direct_provider_id,
                provider_model=direct_provider_model,
                provider_model_ref=claude_model_name,
                thinking_enabled=thinking_enabled,
            )

        provider_model_ref = self._settings.resolve_model(claude_model_name)
        thinking_enabled = self._settings.resolve_thinking(claude_model_name)
        provider_id = Settings.parse_provider_type(provider_model_ref)
        provider_model = Settings.parse_model_name(provider_model_ref)
        if provider_model != claude_model_name:
            logger.debug(
                "MODEL MAPPING: '{}' -> '{}'", claude_model_name, provider_model
            )
        return ResolvedModel(
            original_model=claude_model_name,
            provider_id=provider_id,
            provider_model=provider_model,
            provider_model_ref=provider_model_ref,
            thinking_enabled=thinking_enabled,
        )

    def _resolve_compact_override(self, claude_model_name: str) -> ResolvedModel | None:
        """Return a compaction-specific model resolution when configured."""
        compact_ref = self._settings.model_compact
        if compact_ref is None:
            return None
        provider_id = Settings.parse_provider_type(compact_ref)
        provider_model = Settings.parse_model_name(compact_ref)
        thinking_enabled = self._settings.resolve_thinking(compact_ref)
        logger.info(
            "COMPACT ROUTE: '{}' -> provider='{}' model='{}'",
            claude_model_name,
            provider_id,
            provider_model,
        )
        return ResolvedModel(
            original_model=claude_model_name,
            provider_id=provider_id,
            provider_model=provider_model,
            provider_model_ref=compact_ref,
            thinking_enabled=thinking_enabled,
        )

    def _direct_provider_model(
        self, model_name: str
    ) -> tuple[str | None, str | None, bool | None]:
        decoded = decode_gateway_model_id(model_name)
        if decoded is not None:
            if decoded.provider_id not in SUPPORTED_PROVIDER_IDS:
                return None, None, None
            return (
                decoded.provider_id,
                decoded.provider_model,
                decoded.force_thinking_enabled,
            )

        provider_id, separator, provider_model = model_name.partition("/")
        if not separator:
            return None, None, None
        if provider_id not in SUPPORTED_PROVIDER_IDS:
            return None, None, None
        if not provider_model:
            return None, None, None
        return provider_id, provider_model, None

    def resolve_messages_request(
        self, request: MessagesRequest
    ) -> RoutedMessagesRequest:
        """Return an internal routed request context.

        Compaction requests (detected by system prompt) are routed to
        MODEL_COMPACT when configured, otherwise fall through to normal
        model resolution.
        """
        if self._settings.model_compact is not None and _is_compaction_request(
            request
        ):
            resolved = self._resolve_compact_override(request.model)
            if resolved is not None:
                routed = request.model_copy(deep=True)
                routed.model = resolved.provider_model
                return RoutedMessagesRequest(request=routed, resolved=resolved)

        resolved = self.resolve(request.model)
        routed = request.model_copy(deep=True)
        routed.model = resolved.provider_model
        return RoutedMessagesRequest(request=routed, resolved=resolved)

    def resolve_token_count_request(
        self, request: TokenCountRequest
    ) -> RoutedTokenCountRequest:
        """Return an internal token-count request context."""
        resolved = self.resolve(request.model)
        routed = request.model_copy(
            update={"model": resolved.provider_model}, deep=True
        )
        return RoutedTokenCountRequest(request=routed, resolved=resolved)
