"""Gateway-safe model ID encoding shared by API and CLI adapters."""

import re
from dataclasses import dataclass

GATEWAY_MODEL_ID_PREFIX = "anthropic"

# Claude Code currently treats any model id containing ``claude-3-`` as not
# supporting thinking. This intentionally uses that client-side capability
# heuristic while keeping the real provider/model ref reversible for routing.
NO_THINKING_GATEWAY_MODEL_ID_PREFIX = "claude-3-freecc-no-thinking"
DESKTOP_MODEL_PREFIX = "claude-fcc"
DESKTOP_NO_THINKING_PREFIX = "claude-3-fcc"

# Split a provider/model ref into words, keeping the separators. Every word
# is munged independently so a vendor token after a separator (the
# ``openai`` in ``azure_openai``) is broken just like a leading one. The
# hyphen is deliberately NOT a separator: it is the munging mark itself,
# so splitting on it would shred inserted hyphens on decode (``a-b`` back
# into lone ``a``/``b``) and make the codec non-injective. A hyphen inside
# a word is just another character; munging still breaks hyphenated tokens
# (``command-r`` becomes ``c-ummand-r``).
_DESKTOP_WORD_PATTERN = re.compile(r"([/_.: ])")

_VOWELS = frozenset("aeiouAEIOU")
_VOWEL_ROTATION = {
    "a": "e",
    "e": "i",
    "i": "o",
    "o": "u",
    "u": "a",
    "A": "E",
    "E": "I",
    "I": "O",
    "O": "U",
    "U": "A",
}
_VOWEL_ROTATION_INVERSE = {vowel: plain for plain, vowel in _VOWEL_ROTATION.items()}


def _rotate_first_vowel(text: str, table: dict[str, str]) -> str:
    """Return ``text`` with only its first vowel remapped through ``table``."""
    for index, char in enumerate(text):
        if char in _VOWELS:
            return text[:index] + table[char] + text[index + 1 :]
    return text


def _munge_desktop_word(word: str) -> str:
    """Munge one word so Desktop's vendor blacklist no longer matches it.

    A hyphen after the first character breaks vendor tokens spanning the
    word start (``gpt`` in ``gpt-5``), while rotating the next vowel breaks
    tokens starting later (``llama`` in ``ollama``). Both steps are exactly
    reversible, unlike dropping the vowel outright: ``d-ipseek`` could be
    ``deepseek`` or ``daepseek`` once the vowel is gone, and the router
    decodes these ids statelessly.

    A lone character (``5`` in ``gpt-5``) is a fixed point under
    hyphenation, which would collide with the hyphen-split twin: ``ab``
    and ``a-b`` would both munge to ``a-b``. Escaping it with a ``+``
    suffix (never a separator, never emitted otherwise) keeps the codec
    injective.
    """
    if not word:
        return word
    if len(word) < 2:
        return f"{word}+"
    return f"{word[0]}-{_rotate_first_vowel(word[1:], _VOWEL_ROTATION)}"


def _unmunge_desktop_word(word: str) -> str:
    """Invert :func:`_munge_desktop_word`, rejecting anything it never emits."""
    if not word:
        return word
    if len(word) < 2:
        raise ValueError("Invalid Desktop model ID")
    if word[1] != "-":
        if word.endswith("+"):
            return word[:-1]
        raise ValueError("Invalid Desktop model ID")
    if word.endswith("+"):
        raise ValueError("Invalid Desktop model ID")
    return word[0] + _rotate_first_vowel(word[2:], _VOWEL_ROTATION_INVERSE)


def munge_desktop_ref(provider_model_ref: str) -> str:
    """Return the readable Desktop-safe form of a provider/model ref."""
    return "".join(
        _munge_desktop_word(part) if index % 2 == 0 else part
        for index, part in enumerate(_DESKTOP_WORD_PATTERN.split(provider_model_ref))
    )


def unmunge_desktop_ref(munged_ref: str) -> str:
    """Return the provider/model ref behind a munged Desktop id."""
    if not munged_ref:
        raise ValueError("Invalid Desktop model ID")
    return "".join(
        _unmunge_desktop_word(part) if index % 2 == 0 else part
        for index, part in enumerate(_DESKTOP_WORD_PATTERN.split(munged_ref))
    )


def desktop_model_id(provider_model_ref: str, *, no_thinking: bool = False) -> str:
    """Return the readable Desktop-safe id for a provider/model ref."""
    prefix = DESKTOP_NO_THINKING_PREFIX if no_thinking else DESKTOP_MODEL_PREFIX
    return f"{prefix}/{munge_desktop_ref(provider_model_ref)}"


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
        if prefix in {DESKTOP_MODEL_PREFIX, DESKTOP_NO_THINKING_PREFIX}:
            raise ValueError("Invalid Desktop model ID")
        return None

    if prefix in {DESKTOP_MODEL_PREFIX, DESKTOP_NO_THINKING_PREFIX}:
        # Desktop synthesizes its `[1m]` picker row by appending a literal
        # suffix to the discovered base id (see api/model_catalog), so strip
        # it before unmunging; the router's own suffix strip then sees the
        # bare ref. A `[` byte is never emitted by the munging, so this only
        # ever strips a genuine Desktop-appended suffix.
        remainder = remainder.removesuffix("[1m]")
        decoded = unmunge_desktop_ref(remainder)
        if not remainder or munge_desktop_ref(decoded) != remainder:
            raise ValueError("Invalid Desktop model ID")
        provider_id, separator, provider_model = decoded.partition("/")
        if not provider_id or not separator or not provider_model:
            raise ValueError("Invalid Desktop model ID")
        return DecodedGatewayModelId(
            provider_id, provider_model, prefix == DESKTOP_NO_THINKING_PREFIX
        )
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
