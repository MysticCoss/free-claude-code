"""Model routing for Claude-compatible requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loguru import logger

from config.provider_ids import SUPPORTED_PROVIDER_IDS
from config.settings import Settings

from .gateway_model_ids import decode_gateway_model_id
from .models.anthropic import ContentBlockText, MessagesRequest, TokenCountRequest


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
    is_compaction_override: bool = False


@dataclass(frozen=True, slots=True)
class RoutedTokenCountRequest:
    request: TokenCountRequest
    resolved: ResolvedModel


COMPACT_SYSTEM_MARKER = "summarizing conversations"
COMPACT_USER_MARKER = "CRITICAL: Respond with TEXT ONLY"


def _strip_1m_suffix(model: str) -> str:
    """Remove the ``[1m]`` context-window marker so upstream receives the real model id."""
    return model.removesuffix("[1m]")


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


def _tool_input_summary(inp: Any, max_len: int = 200) -> str:
    """Render tool input as a compact string for a text block."""
    try:
        from json import dumps as _json_dumps

        raw = _json_dumps(inp, ensure_ascii=False, default=str)
    except Exception:
        raw = str(inp)
    if len(raw) > max_len:
        raw = raw[:max_len] + "..."
    return raw


def _make_text_block(text: str) -> ContentBlockText:
    """Create a ``ContentBlockText`` block."""
    return ContentBlockText(type="text", text=text)


def _sanitize_compact_messages(messages: list[Any]) -> list[Any]:
    """Convert ``tool_use`` / ``tool_result`` blocks into plain-text blocks.

    Upstream providers — especially llama.cpp with strict Jinja chat
    templates — reject tool messages that don't follow the exact
    "assistant(tool_use) → user(tool_result)" sequence.  Compaction /
    summarization models don't need to call tools, so we rewrite every
    tool block as a human-readable text block and drop the outer
    ``tools`` definition later in the caller.
    """
    sanitized: list[Any] = []
    tool_count = 0

    for msg in messages:
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            sanitized.append(msg)
            continue

        new_blocks: list[Any] = []
        for block in content:
            block_type = None
            if isinstance(block, dict):
                block_type = block.get("type")
            elif hasattr(block, "type"):
                block_type = getattr(block, "type", None)

            if block_type == "tool_use":
                name = (
                    block.get("name", "unknown")
                    if isinstance(block, dict)
                    else getattr(block, "name", "unknown")
                )
                inp = (
                    block.get("input", {})
                    if isinstance(block, dict)
                    else getattr(block, "input", {})
                )
                inp_str = _tool_input_summary(inp)
                text = f"[Tool call: {name} — {inp_str}]"
                new_blocks.append(_make_text_block(text))
                tool_count += 1
            elif block_type == "tool_result":
                tid = (
                    block.get("tool_use_id", "?")
                    if isinstance(block, dict)
                    else getattr(block, "tool_use_id", "?")
                )
                result = (
                    block.get("content", "")
                    if isinstance(block, dict)
                    else getattr(block, "content", "")
                )
                if isinstance(result, list):
                    parts: list[str] = []
                    for c in result:
                        if isinstance(c, dict):
                            if c.get("type") == "text":
                                parts.append(str(c.get("text", "")))
                            elif c.get("type") == "image":
                                parts.append("[image]")
                            else:
                                parts.append(f"[{c.get('type', '?')}]")
                        else:
                            parts.append(str(c))
                    result = "\n".join(parts)
                result_str = str(result)
                is_error = (
                    block.get("is_error", False)
                    if isinstance(block, dict)
                    else getattr(block, "is_error", False)
                )
                label = "Tool error" if is_error else "Tool result"
                new_blocks.append(
                    _make_text_block(f"[{label} for {tid}]:\n{result_str}")
                )
                tool_count += 1
            else:
                new_blocks.append(block)

        if not new_blocks:
            logger.debug("COMPACT SANITIZE: dropping empty {} message", role)
            continue

        if hasattr(msg, "model_copy"):
            sanitized.append(msg.model_copy(update={"content": new_blocks}))
        elif isinstance(msg, dict):
            sanitized.append({**msg, "content": new_blocks})
        else:
            sanitized.append(msg)

    if tool_count:
        logger.info("COMPACT SANITIZE: converted {} tool blocks to text", tool_count)
    return sanitized


def _merge_system_messages(
    messages: list[Any], system: str | list[Any] | None
) -> tuple[list[Any], str | list[Any] | None]:
    """Extract system messages from ``messages`` and merge their text into the top-level ``system`` field.

    Handles both Pydantic model objects (``.role`` / ``.content``
    attributes) and plain dicts (``["role"]`` / ``["content"]`` keys)
    so callers can pass any message shape.

    Returns a ``(filtered_messages, merged_system)`` tuple where
    *filtered_messages* has no system-role messages and *merged_system*
    preserves the original system content plus any newly extracted text.
    """
    filtered: list[Any] = []
    system_contents: list[str] = []

    for msg in messages:
        role = getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")

        if role == "system":
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if content is not None:
                system_contents.append(str(content))
            continue
        filtered.append(msg)

    if system_contents:
        merged_system: str | list[Any] | None = system
        for content in system_contents:
            if merged_system is None:
                merged_system = content
            elif isinstance(merged_system, str):
                merged_system = f"{merged_system}\n\n---\n\n{content}"
            elif isinstance(merged_system, list):
                merged_system = [*merged_system, {"type": "text", "text": content}]
        logger.debug(
            "Merged {} system message(s) into system field",
            len(system_contents),
        )
        return filtered, merged_system

    return filtered, system


def _rewrite_system_messages_as_user(messages: list[Any]) -> list[Any]:
    """Convert system-role messages to user-role so OpenAI providers accept them.

    System reminders injected mid-conversation by Claude Code (skills list,
    task hints, ``<system-reminder>`` tags) are kept in-place as user
    messages so the message prefix stays byte-stable for upstream caching.
    """
    result: list[Any] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")

        if role == "system":
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if content is not None:
                if isinstance(msg, dict):
                    result.append({"role": "user", "content": str(content)})
                else:
                    result.append(msg.model_copy(update={"role": "user"}))
            continue
        result.append(msg)
    return result


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
            direct_provider_model = _strip_1m_suffix(direct_provider_model)
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
        provider_model = _strip_1m_suffix(Settings.parse_model_name(provider_model_ref))
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
        if self._settings.model_compact is not None and _is_compaction_request(request):
            resolved = self._resolve_compact_override(request.model)
            if resolved is not None:
                routed = request.model_copy(deep=True)
                routed.model = resolved.provider_model
                sanitized = _sanitize_compact_messages(routed.messages)
                routed.messages = sanitized
                # Compaction models don't need tool definitions — and strict
                # Jinja templates reject requests that advertise tools but
                # then receive tool blocks in text form.
                routed.tools = None
                routed.tool_choice = None
                # Anthropic-specific fields unknown to generic providers.
                routed.output_config = None
                routed.mcp_servers = None
                # Flatten structured system prompts to a plain string so
                # providers that only accept string systems can handle them.
                if isinstance(routed.system, list):
                    parts: list[str] = []
                    for block in routed.system:
                        if isinstance(block, dict):
                            parts.append(str(block.get("text", "")))
                        elif hasattr(block, "text"):
                            parts.append(str(block.text))
                    routed.system = "\n\n".join(parts)
                routed.messages, routed.system = _merge_system_messages(
                    routed.messages, routed.system
                )
                return RoutedMessagesRequest(
                    request=routed, resolved=resolved, is_compaction_override=True
                )

        resolved = self.resolve(request.model)
        routed = request.model_copy(deep=True)
        routed.model = resolved.provider_model
        routed.messages = _rewrite_system_messages_as_user(routed.messages)
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
