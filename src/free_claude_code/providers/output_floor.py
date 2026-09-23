"""Recover from upstream ``max_*tokens`` too-small 400 rejections.

Some providers serve certain models through endpoints that require a minimum
output-token budget and reject smaller requests with an HTTP 400 naming the
minimum, e.g.::

    `max_output_tokens` The number must be `>= 16`.

This module parses that minimum and raises the request body so the provider
can retry once and succeed. Only requests that would otherwise fail are
touched; compliant bodies pass through unchanged. It is the mirror image of
``free_claude_code.providers.openai_chat.output_cap`` (which clamps down to
an upstream maximum).
"""

import re
from typing import Any

import openai

# Body keys that carry the output-token budget across OpenAI-compatible policies.
_OUTPUT_TOKEN_FIELDS = ("max_output_tokens", "max_completion_tokens", "max_tokens")
_STRUCTURED_TEXT_FIELDS = ("message", "detail", "error")
_STRUCTURED_CONTAINER_FIELDS = ("error", "errors", "detail", "details")

_FLOOR_VALUE_PATTERN = r"[`'\"]?(\d+)[`'\"]?"
_OUTPUT_TOKEN_FIELD_PATTERN = (
    r"(?<!\w)[`'\"]?(?:max_output_tokens|max_completion_tokens|max_tokens)[`'\"]?(?!\w)"
)

# An accepted grammar must bind the output-token field and its numeric limit.
# A field mentioned elsewhere in the response must never authorize an unrelated
# comparator.
# Some gateways phrase the requirement as "`max_output_tokens` The number must
# be `>= 16`", with a short noun phrase between the field and the comparator.
_NOUN_PHRASE = r"(?:the\s+(?:number|value|amount)\s+)?"
_FIELD_BOUND_FLOOR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        rf"{_OUTPUT_TOKEN_FIELD_PATTERN}\s*(?::\s*)?{_NOUN_PHRASE}"
        rf"(?:"
        rf"(?:(?:must|should)\s+be\s+)?[`'\"]?>="
        rf"|(?:(?:must|should)\s+be\s+)?greater\s+than\s+or\s+equal\s+to"
        rf"|(?:(?:must|should)\s+be\s+)?at\s+least"
        rf")\s*{_FLOOR_VALUE_PATTERN}"
    ),
    re.compile(
        rf"minimum(?:\s+allowed)?(?:\s+value)?\s+(?:is|of)\s+"
        rf"{_FLOOR_VALUE_PATTERN}\s+for\s+{_OUTPUT_TOKEN_FIELD_PATTERN}"
    ),
    re.compile(
        rf"minimum(?:\s+allowed)?(?:\s+value)?\s+of\s+"
        rf"{_FLOOR_VALUE_PATTERN}\s+for\s+{_OUTPUT_TOKEN_FIELD_PATTERN}"
    ),
)


def _is_bad_request(error: Exception) -> bool:
    return isinstance(error, openai.BadRequestError) or (
        getattr(error, "status_code", None) == 400
    )


def _normalized_output_field(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    field = value.strip("`'\" ").lower()
    return field if field in _OUTPUT_TOKEN_FIELDS else None


def _structured_error_texts(body: dict[str, Any] | list[Any]) -> tuple[str, ...]:
    """Extract messages without discarding their structured parameter scope."""
    texts: list[str] = []
    pending: list[dict[str, Any] | list[Any]] = [body]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    texts.append(item)
                elif isinstance(item, (dict, list)):
                    pending.append(item)
            continue

        param = value.get("param")
        output_field = _normalized_output_field(param)
        messages = tuple(
            message
            for field in _STRUCTURED_TEXT_FIELDS
            if isinstance(message := value.get(field), str)
        )
        if output_field is not None:
            # Structured parameter metadata is authoritative. Prefixing the
            # paired field supports comparator-only messages such as ">= 16".
            texts.extend(f"{output_field} {message}" for message in messages)
        if param is not None:
            # Do not let text nested below an explicitly scoped validation error
            # escape that scope and become a new unscoped candidate.
            continue

        texts.extend(messages)
        for field in _STRUCTURED_CONTAINER_FIELDS:
            nested = value.get(field)
            if isinstance(nested, (dict, list)):
                pending.append(nested)
    return tuple(texts)


def _error_texts(error: Exception) -> tuple[str, ...]:
    """Keep unstructured text separate from structured validation messages."""
    body = getattr(error, "body", None)
    if body is None:
        texts = (str(error),)
    elif isinstance(body, str):
        texts = (body,)
    elif isinstance(body, (dict, list)):
        # The OpenAI SDK embeds a representation of this body in ``str(error)``.
        # Parsing both would flatten the parameter scopes restored above.
        texts = _structured_error_texts(body)
    else:
        texts = ()
    return tuple(text.lower() for text in texts)


def _parse_floor(text: str) -> int | None:
    for pattern in _FIELD_BOUND_FLOOR_PATTERNS:
        match = pattern.search(text)
        if match:
            floor = int(match.group(1))
            if floor > 0:
                return floor
    return None


def parse_output_token_floor(error: Exception) -> int | None:
    """Return the required output-token minimum named in a 400 rejection, if any."""
    if not _is_bad_request(error):
        return None

    for text in _error_texts(error):
        floor = _parse_floor(text)
        if floor is not None:
            return floor
    return None


def raise_output_tokens(body: dict[str, Any], floor: int) -> dict[str, Any] | None:
    """Return a shallow clone with output-token fields raised to ``floor``.

    Returns ``None`` when nothing needs raising (no output field is below the
    floor), so callers can avoid a pointless identical retry.
    """
    raised: dict[str, Any] | None = None
    for field in _OUTPUT_TOKEN_FIELDS:
        value = body.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value < floor:
            if raised is None:
                raised = dict(body)
            raised[field] = floor
    return raised
