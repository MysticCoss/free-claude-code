"""Tests for the deterministic Claude → opencode session id converter.

The converter must be a pure function: same input always yields the same
output, no I/O, no RNG, no module state. The output must conform to the
opencode ``Identifier`` schema (``startswith("ses")``, 30 chars total, body
of 12 lowercase hex chars followed by 14 base62 chars).
"""

import re

from providers.opencode.session_id import (
    _BASE62,
    _HEX_LEN,
    _PREFIX,
    _RANDOM_LEN,
    claude_to_opencode_session_id,
)

_HEX_RE = re.compile(r"^[0-9a-f]{12}$")
_BASE62_RE = re.compile(r"^[0-9A-Za-z]{14}$")


def test_returns_empty_string_for_none():
    """``None`` input maps to empty string (the no-session sentinel)."""
    assert claude_to_opencode_session_id(None) == ""


def test_returns_empty_string_for_empty_string():
    """Empty string input also maps to empty string."""
    assert claude_to_opencode_session_id("") == ""


def test_starts_with_ses_prefix():
    """Any non-empty input produces a string that starts with ``ses_``."""
    for value in (
        "sess_abc123",
        "sess_019b4f3a-2c1d-8a7b-c3de9fg2hj5k",
        "x",
        "anthropic-session-id-1234",
    ):
        result = claude_to_opencode_session_id(value)
        assert result.startswith(_PREFIX), (value, result)


def test_total_length_is_30():
    """Output is always exactly 30 chars when the input is non-empty."""
    for value in ("a", "sess_abc123", "x" * 1000):
        assert len(claude_to_opencode_session_id(value)) == 30, value


def test_body_length_is_26():
    """After the ``ses_`` prefix, the body must be exactly 26 chars."""
    result = claude_to_opencode_session_id("sess_abc123")
    assert len(result) == len(_PREFIX) + _HEX_LEN * 2 + _RANDOM_LEN
    assert len(result) == 30


def test_hex_part_is_12_lowercase_hex():
    """Chars 4..16 are 12 lowercase hex characters."""
    result = claude_to_opencode_session_id("sess_abc123")
    assert _HEX_RE.match(result[4:16]), result


def test_base62_part_is_14_alphanumeric():
    """Chars 16..30 are 14 characters from the base62 alphabet."""
    result = claude_to_opencode_session_id("sess_abc123")
    assert _BASE62_RE.match(result[16:30]), result
    for ch in result[16:30]:
        assert ch in _BASE62, ch


def test_deterministic_same_input_same_output():
    """Calling the function twice on the same input returns identical output."""
    a = claude_to_opencode_session_id("sess_abc123")
    b = claude_to_opencode_session_id("sess_abc123")
    assert a == b
    # And again, to exercise the no-cache claim.
    c = claude_to_opencode_session_id("sess_abc123")
    assert a == c


def test_distinct_inputs_distinct_outputs():
    """Different inputs map to different outputs (SHA-256 collision resistance)."""
    a = claude_to_opencode_session_id("sess_abc123")
    b = claude_to_opencode_session_id("sess_xyz")
    c = claude_to_opencode_session_id("sess_abc124")
    d = claude_to_opencode_session_id("sess_abc1234")
    assert len({a, b, c, d}) == 4


def test_known_value_snapshot():
    """Snapshot test derived from the documented algorithm, not the implementation.

    Derivation (re-runnable, see ``_derive`` docstring):
        sha256(b"sess_abc123").hex() =
            "61561039cbe7ae58aa51dbaa9403eb7a67e2261810262090fe57f5cefb60edf4"
        hex_part   = first 12 hex chars = "61561039cbe7"
        b62_part   = map bytes 6..20 (% 62) through BASE62 = "oQkJXkO3nyfecO"
        expected   = "ses_" + hex_part + b62_part = "ses_61561039cbe7oQkJXkO3nyfecO"

    Derivation for "sess_xyz":
        sha256(b"sess_xyz").hex() =
            "1e2e235bdeb43605c7c2572d220a67aeb18b73421120f875c3fa4f327f9a4ba7"
        hex_part   = "1e2e235bdeb4"
        b62_part   = "s5D8PjYAforFr4"
        expected   = "ses_1e2e235bdeb4s5D8PjYAforFr4"

    If the algorithm or constants (_HEX_LEN / _RANDOM_LEN / _PREFIX / _BASE62)
    change intentionally, regenerate these constants with the helper below and
    update the assertions.
    """
    expected_abc = _derive("sess_abc123")
    expected_xyz = _derive("sess_xyz")
    assert expected_abc == "ses_61561039cbe7oQkJXkO3nyfecO"
    assert expected_xyz == "ses_1e2e235bdeb4s5D8PjYAforFr4"
    assert claude_to_opencode_session_id("sess_abc123") == expected_abc
    assert claude_to_opencode_session_id("sess_xyz") == expected_xyz


def _derive(input_str: str) -> str:
    """Independent re-implementation of the converter's algorithm.

    Used by ``test_known_value_snapshot`` to derive the expected value from
    the spec rather than from the production code, so the test fails loudly
    if the implementation drifts from the documented algorithm.
    """
    import hashlib as _hashlib

    digest = _hashlib.sha256(input_str.encode("utf-8")).digest()
    hex_part = digest[:_HEX_LEN].hex()
    b62_part = "".join(_BASE62[b % 62] for b in digest[_HEX_LEN:_HEX_LEN + _RANDOM_LEN])
    return f"{_PREFIX}{hex_part}{b62_part}"


def test_unicode_input_is_deterministic():
    """Unicode inputs hash deterministically via UTF-8 encoding."""
    a1 = claude_to_opencode_session_id("phiên-vi-🚀")
    a2 = claude_to_opencode_session_id("phiên-vi-🚀")
    b = claude_to_opencode_session_id("phiên-vi-🌙")
    assert a1 == a2
    assert a1 != b
    assert a1.startswith("ses_")
    assert b.startswith("ses_")


def test_very_long_input_is_stable():
    """A 10 KB input still produces a stable, well-formed output."""
    long_input = "x" * 10_000
    a = claude_to_opencode_session_id(long_input)
    b = claude_to_opencode_session_id(long_input)
    assert a == b
    assert len(a) == 30
    assert a.startswith("ses_")
    assert _HEX_RE.match(a[4:16])
    assert _BASE62_RE.match(a[16:30])
