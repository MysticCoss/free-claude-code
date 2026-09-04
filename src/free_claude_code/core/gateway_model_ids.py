"""Gateway-safe model ID encoding shared by API and CLI adapters."""

import re
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

# Claude Desktop 3P discovery drops any advertised id whose lowercase form
# contains a third-party model-vendor token, even when the id starts with
# "claude-". The pattern below replicates the blacklist shipped in Claude
# Desktop 1.46388.2.0; a Desktop update may extend it. Matching is
# case-insensitive because ids are upper-cased by some catalogs while the
# client filters the lowercased form.
_DESKTOP_VENDOR_TOKENS = re.compile(
    r"ark-code|astron|command-r|deepseek|doubao|gemini|gemma|glm|gpt|grok|hermes|hy3"
    r"|kimi|lfm|\bling\b|llama|longcat|mimo|minimax|mistral|mixtral|moonshot|nemotron"
    r"|openai|phi-|qianfan|qwen|tc-code|\bunic\b|yi-|stepfun|step-3|seed-|bytedance"
    r"|hunyuan|granite|amazon\.nova|nova-|devstral|ministral|ernie|codex|arcee|trinity"
    r"|abab|phi\d|\bk2\.|\bm2\.|jamba|arctic|solar|mercury|zamba|kat-coder|\bds-|dpsk",
    re.IGNORECASE,
)

# Obfuscation replaces the first vowel of a matched vendor token with a hyphen
# (``deepseek`` -> ``d-epseek``); vowelless tokens get a hyphen inserted after
# their first character (``glm`` -> ``g-lm``). The marker never appears at a
# token's first position, so every obfuscated id is token-free and re-scan
# converges after one pass.
_DESKTOP_OBFUSCATION_MARKER = "-"
_DESKTOP_VOWELS = "aeiouAEIOU"
_DESKTOP_RESTORE_CANDIDATES = ("", "a", "e", "i", "o", "u", "A", "E", "I", "O", "U")
_MAX_DESKTOP_OBFUSCATION_ROUNDS = 8


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


def _obfuscate_vendor_token(token: str) -> str:
    for index, char in enumerate(token):
        if char in _DESKTOP_VOWELS:
            return f"{token[:index]}{_DESKTOP_OBFUSCATION_MARKER}{token[index + 1 :]}"
    return f"{token[0]}{_DESKTOP_OBFUSCATION_MARKER}{token[1:]}"


def _obfuscate_desktop_id(model_id: str) -> str:
    """Break every blacklisted vendor token in ``model_id`` out of the filter.

    The substitution inserts a hyphen inside each matched token, which never
    recreates a token at a new position; the fixed-point re-scan is a
    defensive guard against token-list interactions, not a normal path.
    """
    current = model_id
    for _ in range(_MAX_DESKTOP_OBFUSCATION_ROUNDS):
        candidate = _DESKTOP_VENDOR_TOKENS.sub(
            lambda match: _obfuscate_vendor_token(match.group()),
            current,
        )
        if candidate == current:
            return current
        current = candidate
    return current


def _restore_site_match(text: str, site: int) -> re.Match[str] | None:
    for match in _DESKTOP_VENDOR_TOKENS.finditer(text):
        if match.start() <= site < match.end():
            return match
    return None


def _deobfuscate_desktop_id(model_id: str) -> str:
    """Undo :func:`_obfuscate_desktop_id`, restoring blacklisted tokens.

    Scans left to right for marker hyphens. A hyphen is only reverted when
    deleting it or replacing it with a vowel re-creates a vendor-token match
    *covering that position*; every other hyphen is a real separator in the
    underlying name and is left untouched. The caller re-encodes the result
    and compares against the original id, so a wrong restore never routes.
    """
    result = model_id
    position = 0
    while (index := result.find(_DESKTOP_OBFUSCATION_MARKER, position)) >= 0:
        for restore in _DESKTOP_RESTORE_CANDIDATES:
            trial = f"{result[:index]}{restore}{result[index + 1 :]}"
            match = _restore_site_match(trial, index)
            if match is not None:
                result = trial
                position = match.end()
                break
        else:
            position = index + 1
    return result


def claude_desktop_model_id(provider_model_ref: str) -> str:
    """Encode a ``provider/model`` ref as a single-segment Claude Desktop 3P id.

    The id is obfuscated so it passes Desktop's discovery name filter: any
    third-party vendor token in the provider or model name gets a hyphen
    injected, while the id keeps its ``claude-`` prefix and stays exactly
    decodable via :func:`decode_claude_desktop_model_id`.
    """
    provider_id, separator, provider_model = provider_model_ref.partition("/")
    if not separator or not provider_id or not provider_model:
        raise ValueError("Model reference must contain provider and model names.")
    return _obfuscate_desktop_id(
        f"{CLAUDE_DESKTOP_MODEL_ID_PREFIX}{provider_id}"
        f"{_DESKTOP_REF_SEPARATOR}{provider_model}"
    )


def claude_desktop_no_thinking_model_id(provider_model_ref: str) -> str:
    """Desktop-advertised no-thinking id for a ``provider/model`` ref.

    Shares the gateway no-thinking prefix (so thinking stays off client-side)
    but carries the same vendor-token obfuscation as
    :func:`claude_desktop_model_id`; the ref keeps its ``/`` segments.
    """
    provider_id, separator, provider_model = provider_model_ref.partition("/")
    if not separator or not provider_id or not provider_model:
        raise ValueError("Model reference must contain provider and model names.")
    return _obfuscate_desktop_id(
        f"{NO_THINKING_GATEWAY_MODEL_ID_PREFIX}/{provider_model_ref}"
    )


def _match_desktop_provider(
    remainder: str, known_provider_ids: Iterable[str]
) -> DecodedGatewayModelId | None:
    for provider_id in sorted(known_provider_ids, key=len, reverse=True):
        separator = f"{provider_id}{_DESKTOP_REF_SEPARATOR}"
        if remainder.startswith(separator) and len(remainder) > len(separator):
            return DecodedGatewayModelId(
                provider_id=provider_id,
                provider_model=remainder[len(separator) :],
                force_reasoning_off=False,
            )
    return None


def decode_claude_desktop_model_id(
    model_name: str, known_provider_ids: Iterable[str]
) -> DecodedGatewayModelId | None:
    """Decode a Claude Desktop 3P id, if ``model_name`` is one.

    Obfuscated ids are token-free by construction, so their provider segment
    may be obfuscated too and must be restored before matching. Ids that
    still carry a raw vendor token predate obfuscation (or were entered by
    hand) and decode through the original longest-provider-prefix match.
    Decoding the obfuscated form requires the exact round-trip: re-encoding
    the decoded ref must reproduce ``model_name``. Returns ``None`` for plain
    ``claude-*`` ids that do not carry a provider segment.
    """
    if not model_name.startswith(CLAUDE_DESKTOP_MODEL_ID_PREFIX):
        return None
    remainder = model_name[len(CLAUDE_DESKTOP_MODEL_ID_PREFIX) :]

    if _DESKTOP_VENDOR_TOKENS.search(remainder):
        return _match_desktop_provider(remainder, known_provider_ids)

    decoded = _match_desktop_provider(
        _deobfuscate_desktop_id(remainder), known_provider_ids
    )
    if decoded is None:
        return None
    ref = f"{decoded.provider_id}/{decoded.provider_model}"
    if claude_desktop_model_id(ref) != model_name:
        return None
    return decoded


def decode_claude_desktop_no_thinking_model_id(
    model_name: str, known_provider_ids: Iterable[str]
) -> DecodedGatewayModelId | None:
    """Decode an obfuscated Claude Desktop no-thinking id, if it is one.

    Plain (raw) no-thinking ids are already handled by
    :func:`decode_gateway_model_id`; this covers the desktop-advertised form
    whose provider and model segments went through vendor-token
    obfuscation. The decoded ref must re-encode to ``model_name`` exactly.
    """
    if not model_name.startswith(f"{NO_THINKING_GATEWAY_MODEL_ID_PREFIX}/"):
        return None
    decoded = decode_gateway_model_id(model_name)
    if decoded is None:
        return None

    if _DESKTOP_VENDOR_TOKENS.search(model_name):
        if decoded.provider_id in known_provider_ids:
            return decoded
        return None

    ref = f"{decoded.provider_id}/{decoded.provider_model}"
    provider_id, separator, provider_model = _deobfuscate_desktop_id(ref).partition("/")
    if not separator or not provider_model or provider_id not in known_provider_ids:
        return None
    if claude_desktop_no_thinking_model_id(f"{provider_id}/{provider_model}") != (
        model_name
    ):
        return None
    return DecodedGatewayModelId(
        provider_id=provider_id,
        provider_model=provider_model,
        force_reasoning_off=True,
    )
