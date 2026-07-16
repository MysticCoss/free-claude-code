"""Model routing for Claude-compatible requests."""

from dataclasses import dataclass

from loguru import logger

from free_claude_code.application.errors import UnknownProviderError
from free_claude_code.config.model_refs import parse_model_name, parse_provider_type
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic import MessagesRequest, TokenCountRequest
from free_claude_code.core.gateway_model_ids import decode_gateway_model_id

# Markers Claude Code injects into compaction / summarization requests.
# When either marker is present, the request is redirected to ``model_compact``
# (if configured) so a cheaper model handles context summarization.
COMPACT_SYSTEM_MARKER = "summarizing conversations"
COMPACT_USER_MARKER = "CRITICAL: Respond with TEXT ONLY"

# Suffix Claude Code reads in /v1/models to grant the 1M-token context window.
# Stripped from ``provider_model`` before sending to upstream so the real
# provider only sees the bare model id.
ONE_M_CONTEXT_SUFFIX = "[1m]"


def _strip_1m_suffix(model: str) -> str:
    """Remove the ``[1m]`` context-window marker so upstream receives the real model id."""
    return model.removesuffix(ONE_M_CONTEXT_SUFFIX)


def _is_compaction_request(request: MessagesRequest) -> bool:
    """True when the request looks like a Claude Code compaction/summarization call."""
    system = request.system
    if isinstance(system, str) and COMPACT_SYSTEM_MARKER in system:
        return True
    if isinstance(system, list):
        for block in system:
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text and COMPACT_SYSTEM_MARKER in text:
                return True
    if request.messages:
        last = request.messages[-1]
        if last.role == "user" and isinstance(last.content, str):
            return last.content.startswith(COMPACT_USER_MARKER)
    return False


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
            stripped_provider_model = _strip_1m_suffix(direct_provider_model)
            thinking_enabled = (
                force_thinking_enabled
                if force_thinking_enabled is not None
                else self._resolve_thinking(direct_provider_model)
            )
            logger.debug(
                "MODEL DIRECT: '{}' -> provider='{}' model='{}' thinking={}",
                claude_model_name,
                direct_provider_id,
                stripped_provider_model,
                thinking_enabled,
            )
            return ResolvedModel(
                original_model=claude_model_name,
                provider_id=direct_provider_id,
                provider_model=stripped_provider_model,
                provider_model_ref=claude_model_name,
                thinking_enabled=thinking_enabled,
            )

        provider_model_ref = self._resolve_model_ref(claude_model_name)
        thinking_enabled = self._resolve_thinking(claude_model_name)
        provider_id = parse_provider_type(provider_model_ref)
        self._validate_provider_id(provider_id)
        provider_model = _strip_1m_suffix(parse_model_name(provider_model_ref))
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

    def _resolve_compact_override(self) -> ResolvedModel | None:
        """Resolve ``settings.model_compact`` into a ResolvedModel, validating the provider."""
        ref = self._settings.model_compact
        if ref is None:
            return None
        provider_id = parse_provider_type(ref)
        self._validate_provider_id(provider_id)
        return ResolvedModel(
            original_model="<compact>",
            provider_id=provider_id,
            provider_model=_strip_1m_suffix(parse_model_name(ref)),
            provider_model_ref=ref,
            thinking_enabled=self._settings.enable_model_thinking,
        )

    @staticmethod
    def _validate_provider_id(provider_id: str) -> None:
        if provider_id not in PROVIDER_CATALOG:
            raise UnknownProviderError.for_provider(provider_id, PROVIDER_CATALOG)

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

    def _resolve_model_ref(self, claude_model_name: str) -> str:
        """Resolve a Claude model name to the configured provider/model ref."""

        name_lower = claude_model_name.lower()
        if "fable" in name_lower and self._settings.model_fable is not None:
            return self._settings.model_fable
        if "opus" in name_lower and self._settings.model_opus is not None:
            return self._settings.model_opus
        if "haiku" in name_lower and self._settings.model_haiku is not None:
            return self._settings.model_haiku
        if "sonnet" in name_lower and self._settings.model_sonnet is not None:
            return self._settings.model_sonnet
        return self._settings.model

    def _resolve_thinking(self, claude_model_name: str) -> bool:
        """Resolve whether thinking is enabled for an incoming Claude model name."""

        name_lower = claude_model_name.lower()
        if "fable" in name_lower and self._settings.enable_fable_thinking is not None:
            return self._settings.enable_fable_thinking
        if "opus" in name_lower and self._settings.enable_opus_thinking is not None:
            return self._settings.enable_opus_thinking
        if "haiku" in name_lower and self._settings.enable_haiku_thinking is not None:
            return self._settings.enable_haiku_thinking
        if "sonnet" in name_lower and self._settings.enable_sonnet_thinking is not None:
            return self._settings.enable_sonnet_thinking
        return self._settings.enable_model_thinking

    def resolve_messages_request(
        self, request: MessagesRequest
    ) -> RoutedMessagesRequest:
        """Return an internal routed request context."""
        if self._settings.model_compact is not None and _is_compaction_request(request):
            resolved = self._resolve_compact_override()
            if resolved is not None:
                routed = request.model_copy(deep=True)
                routed.model = resolved.provider_model
                logger.debug(
                    "MODEL COMPACT: routing compaction request -> provider='{}' model='{}'",
                    resolved.provider_id,
                    resolved.provider_model,
                )
                return RoutedMessagesRequest(
                    request=routed,
                    resolved=resolved,
                    is_compaction_override=True,
                )
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
