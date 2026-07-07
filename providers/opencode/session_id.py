"""Deterministic Claude/Anthropic session id → opencode session id converter.

The opencode CLI sends a session id of the form ``ses_<12hex><14base62>`` (30
chars) verbatim as the ``x-opencode-session`` header. The OpenCode Zen gateway
uses that string to group requests in its billing dashboard.

Claude Code (the Anthropic SDK) sends a different shape of session id over
HTTP headers (e.g. ``sess_<uuid>``). This proxy forwards those ids to the
upstream provider, but raw values don't conform to the opencode ``Identifier``
schema (which validates ``startsWith("ses")``) and they cluster oddly in the
dashboard next to native opencode sessions.

This module provides a **pure deterministic** mapping so the same Claude
session id always produces the same opencode-shape id — across processes,
restarts, and machines — with no cache, no RNG, no clock dependency.

Algorithm:
    digest   = sha256(input.encode("utf-8"))       # 32 bytes
    hex_part  = digest[:6].hex()                   # 12 lowercase hex chars
    b62_part  = "".join(BASE62[b % 62] for b in digest[6:20])  # 14 base62 chars
    return f"ses_{hex_part}{b62_part}"             # 30 chars, opencode-shaped

If the input is ``None`` or empty, the function returns the empty string.
Callers should pass that through as the ``x-opencode-session`` header value as
an explicit "no session" sentinel; the previous behavior of falling back to a
synthesized request id conflated two distinct identifiers.
"""

from __future__ import annotations

import hashlib

_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_PREFIX = "ses_"
_HEX_LEN = 6  # 6 bytes → 12 hex chars (time-shape portion of opencode id)
_RANDOM_LEN = 14  # 14 base62 chars (random-shape suffix of opencode id)


def claude_to_opencode_session_id(claude_session_id: str | None) -> str:
    """Map a Claude/Anthropic session id to an opencode-shaped session id.

    Pure function: same input always yields the same output, no I/O, no RNG,
    no module state, no cache. Safe to call on every request.

    Args:
        claude_session_id: The session id forwarded by Claude Code / SDK
            via an HTTP header (e.g. ``sess_<uuid>``). May be ``None`` or
            empty if no session header was provided.

    Returns:
        A 30-character ``ses_<12hex><14base62>`` string when the input is
        non-empty, or the empty string when the input is ``None`` or empty.
    """
    if not claude_session_id:
        return ""
    digest = hashlib.sha256(claude_session_id.encode("utf-8")).digest()
    hex_part = digest[:_HEX_LEN].hex()
    base62_part = "".join(_BASE62[b % 62] for b in digest[_HEX_LEN:_HEX_LEN + _RANDOM_LEN])
    return f"{_PREFIX}{hex_part}{base62_part}"
