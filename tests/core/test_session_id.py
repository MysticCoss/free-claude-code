"""Deterministic Claude session → opencode session conversion."""

import re

from free_claude_code.core.anthropic.models import MessagesRequest
from free_claude_code.core.session_id import (
    claude_to_opencode_session_id,
    conversation_seed,
    opencode_request_headers,
)

_OPENCODE_SESSION_RE = re.compile(r"ses_[0-9a-f]{12}[0-9A-Za-z]{14}\Z")
GOLDEN = {
    "sess_abc-123": "ses_98cd96b77333egPHPVV7mm2Css",
    "claude-session-1": "ses_08d21ca82c5bRbm5LMECoMGALn",
}
VALID_SESSION = "ses_98cd96b77333egPHPVV7mm2Css"


def test_empty_input_yields_no_identity() -> None:
    assert claude_to_opencode_session_id(None) == ""
    assert claude_to_opencode_session_id("") == ""


def test_golden_values_are_stable() -> None:
    for source, expected in GOLDEN.items():
        assert claude_to_opencode_session_id(source) == expected


def test_output_matches_opencode_identifier_shape() -> None:
    for source in ("sess_abc-123", "conversation-a", "x" * 500):
        converted = claude_to_opencode_session_id(source)
        assert _OPENCODE_SESSION_RE.match(converted), converted
        assert len(converted) == 30


def test_short_and_long_inputs_are_deterministic() -> None:
    short = "sess_abc-123"
    long = "seed-" + "y" * 500
    assert claude_to_opencode_session_id(short) == claude_to_opencode_session_id(short)
    assert claude_to_opencode_session_id(long) == claude_to_opencode_session_id(long)
    assert claude_to_opencode_session_id(short) == GOLDEN[short]


def test_verbatim_passes_valid_identifier_unchanged() -> None:
    headers = opencode_request_headers(VALID_SESSION, verbatim_session=True)
    assert headers["x-opencode-session"] == VALID_SESSION


def test_verbatim_maps_non_conforming_values() -> None:
    for source in ("native-session", "conversation-a", "explicit-session", "SES_"):
        headers = opencode_request_headers(source, verbatim_session=True)
        assert headers["x-opencode-session"] == claude_to_opencode_session_id(source)
        assert _OPENCODE_SESSION_RE.match(headers["x-opencode-session"])


def test_non_verbatim_always_maps() -> None:
    headers = opencode_request_headers(VALID_SESSION)
    assert headers["x-opencode-session"] == claude_to_opencode_session_id(VALID_SESSION)


def test_missing_session_omits_header_and_keeps_request_id() -> None:
    headers = opencode_request_headers(None, request_id="req_1")
    assert "x-opencode-session" not in headers
    assert headers == {
        "x-opencode-client": "desktop",
        "x-opencode-request": "req_1",
    }


def test_fallback_seed_used_only_without_session() -> None:
    seed_headers = opencode_request_headers(None, fallback_seed="system+hello")
    assert seed_headers["x-opencode-session"] == claude_to_opencode_session_id(
        "system+hello"
    )
    session_headers = opencode_request_headers(
        "sess_abc-123",
        fallback_seed="system+hello",
    )
    assert session_headers["x-opencode-session"] == GOLDEN["sess_abc-123"]


def test_conversation_seed_joins_system_and_first_message() -> None:
    request = MessagesRequest.model_validate(
        {
            "model": "m",
            "system": "be brief",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        }
    )
    assert conversation_seed(request) == "be brief\nhello"


def test_conversation_seed_is_stable_across_later_turns() -> None:
    opening = MessagesRequest.model_validate(
        {
            "model": "m",
            "system": "sys",
            "messages": [{"role": "user", "content": "first"}],
            "max_tokens": 16,
        }
    )
    continued = opening.model_copy(
        update={
            "messages": [
                *opening.messages,
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "second"},
            ]
        }
    )
    assert conversation_seed(opening) == conversation_seed(continued)
    assert conversation_seed(opening) == "sys\nfirst"
