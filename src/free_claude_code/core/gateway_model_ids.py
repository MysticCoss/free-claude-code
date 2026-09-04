"""Gateway-safe model ID encoding shared by API and CLI adapters."""

from collections.abc import Iterable
from dataclasses import dataclass

GATEWAY_MODEL_ID_PREFIX = "anthropic"

# Claude Code currently treats any model id containing ``claude-3-`` as not
# supporting thinking. This intentionally uses that client-side capability
# heuristic while keeping the real provider/model ref reversible for routing.
NO_THINKING_GATEWAY_MODEL_ID_PREFIX = "claude-3-freecc-no-thinking"

# Claude Desktop's third-party (3P) mode only lists model ids matching
# ``claude-*`` / ``anthropic/claude-*``. Desktop-compatible ids join the
# provider and model with a single hyphen into one path segment. Provider ids
# never contain hyphens today, so the longest known-provider prefix decode is
# exact; a future hyphenated provider id would be resolved longest-first.
CLAUDE_DESKTOP_MODEL_ID_PREFIX = "claude-"
_DESKTOP_REF_SEPARATOR = "-"


@dataclass(frozen=True, slots=True)
class DecodedGatewayModelId:
    provider_id: str
    provider_model: str
    force_reasoning_off: bool = False


def gateway_model_id(provider_model_ref: str) -> str:
    """Return the normal Claude Code-discoverable id for a provider/model ref."""
    return f"{GATEWAY_MODEL_ID_PREFIX}/{provider_model_ref}"


def no_thinking_gateway_model_id(provider_model_ref: str) -> str:
    """Return a Claude Code-discoverable id that disables client thinking."""
    return f"{NO_THINKING_GATEWAY_MODEL_ID_PREFIX}/{provider_model_ref}"


def decode_gateway_model_id(model_name: str) -> DecodedGatewayModelId | None:
    """Decode a model id advertised by this gateway, if it is one."""
    prefix, separator, remainder = model_name.partition("/")
    if not separator:
        return None

    if prefix == GATEWAY_MODEL_ID_PREFIX:
        force_reasoning_off = False
    elif prefix == NO_THINKING_GATEWAY_MODEL_ID_PREFIX:
        force_reasoning_off = True
    else:
        return None

    provider_id, provider_separator, provider_model = remainder.partition("/")
    if not provider_separator or not provider_model:
        return None

    return DecodedGatewayModelId(
        provider_id=provider_id,
        provider_model=provider_model,
        force_reasoning_off=force_reasoning_off,
    )


def claude_desktop_model_id(provider_model_ref: str) -> str:
    """Encode a ``provider/model`` ref as a single-segment Claude Desktop 3P id."""
    provider_id, separator, provider_model = provider_model_ref.partition("/")
    if not separator or not provider_id or not provider_model:
        raise ValueError("Model reference must contain provider and model names.")
    return (
        f"{CLAUDE_DESKTOP_MODEL_ID_PREFIX}{provider_id}"
        f"{_DESKTOP_REF_SEPARATOR}{provider_model}"
    )


def decode_claude_desktop_model_id(
    model_name: str, known_provider_ids: Iterable[str]
) -> DecodedGatewayModelId | None:
    """Decode a Claude Desktop 3P id, if ``model_name`` is one.

    Matches the longest known provider id that the encoded segment starts with
    (so hyphenated provider ids would resolve longest-first). Returns ``None``
    for plain ``claude-*`` ids that do not carry a provider segment.
    """
    if not model_name.startswith(CLAUDE_DESKTOP_MODEL_ID_PREFIX):
        return None
    remainder = model_name[len(CLAUDE_DESKTOP_MODEL_ID_PREFIX) :]

    for provider_id in sorted(known_provider_ids, key=len, reverse=True):
        separator = f"{provider_id}{_DESKTOP_REF_SEPARATOR}"
        if remainder.startswith(separator) and len(remainder) > len(separator):
            return DecodedGatewayModelId(
                provider_id=provider_id,
                provider_model=remainder[len(separator) :],
                force_reasoning_off=False,
            )
    return None
