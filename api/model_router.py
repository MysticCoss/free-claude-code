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


def _sanitize_compact_messages(messages: list[Any]) -> list[Any]:
    """Strip orphan ``tool_result`` blocks whose ``tool_use`` is missing.

    When a session is terminated mid-flight, assistant messages with
    ``tool_use`` blocks may be absent from the history while their
    corresponding ``tool_result`` blocks remain.  Upstream providers
    (especially those using strict Jinja chat templates) reject these
    orphan tool messages because a tool result must follow an assistant
    or tool message.

    This walker collects every ``tool_use.id`` from assistant content
    blocks, then drops ``tool_result`` blocks whose ``tool_use_id``
    doesn't reference a known id.  User messages that become empty
    after stripping are removed entirely.
    """
    # Collect known tool_use ids from assistant messages.
    known_tool_use_ids: set[str] = set()
    for msg in messages:
        role = getattr(msg, "role", None)
        if role != "assistant":
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "tool_use":
                    tid = block.get("id")
                    if isinstance(tid, str):
                        known_tool_use_ids.add(tid)
            elif hasattr(block, "type") and getattr(block, "type", None) == "tool_use":
                tid = getattr(block, "id", None)
                if isinstance(tid, str):
                    known_tool_use_ids.add(tid)

    if not known_tool_use_ids:
        return list(messages)

    sanitized: list[Any] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role != "user":
            sanitized.append(msg)
            continue
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            sanitized.append(msg)
            continue

        kept_blocks: list[Any] = []
        for block in content:
            block_type = None
            tool_use_id = None
            if isinstance(block, dict):
                block_type = block.get("type")
                tool_use_id = block.get("tool_use_id")
            elif hasattr(block, "type"):
                block_type = getattr(block, "type", None)
                tool_use_id = getattr(block, "tool_use_id", None)

            if block_type == "tool_result" and tool_use_id not in known_tool_use_ids:
                logger.debug(
                    "COMPACT SANITIZE: dropping orphan tool_result tool_use_id={}",
                    tool_use_id,
                )
                continue
            kept_blocks.append(block)

        if kept_blocks:
            # Preserve the message structure but replace content with kept blocks.
            if hasattr(msg, "model_copy"):
                sanitized.append(msg.model_copy(update={"content": kept_blocks}))
            elif isinstance(msg, dict):
                sanitized.append({**msg, "content": kept_blocks})
            else:
                sanitized.append(msg)
        else:
            logger.debug("COMPACT SANITIZE: dropping empty user message after orphan cleanup")

    return sanitized


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
                sanitized = _sanitize_compact_messages(routed.messages)
                if len(sanitized) != len(routed.messages):
                    logger.info(
                        "COMPACT SANITIZE: removed {} orphan messages",
                        len(routed.messages) - len(sanitized),
                    )
                routed.messages = sanitized
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
