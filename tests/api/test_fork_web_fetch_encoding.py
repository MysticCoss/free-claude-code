"""Fork regression coverage for web_fetch response charset handling.

Kept in its own file so upstream rewrites of ``test_web_server_tools.py``
cannot drop it. Upstream ``_run_web_fetch`` called ``response.get_encoding()``
on a never-fully-read aiohttp response, which raises ``RuntimeError`` for any
site that omits the ``charset`` Content-Type parameter (e.g. docs.python.org);
the fix parses the charset from the header text instead.
"""

import codecs
from collections.abc import AsyncIterator
from unittest.mock import patch

import pytest

from free_claude_code.api.web_tools.egress import WebFetchEgressPolicy
from free_claude_code.api.web_tools.outbound import (
    _content_type_charset,
    _run_web_fetch,
)

_STRICT_EGRESS = WebFetchEgressPolicy(
    allow_private_network_targets=False,
    allowed_schemes=frozenset({"http", "https"}),
)


class _NotReadRuntimeError(RuntimeError):
    """Signal that code still routes through aiohttp's read-state check."""


class _FakeContent:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def iter_chunked(self, n: int) -> AsyncIterator[bytes]:
        for start in range(0, len(self._body), n):
            yield self._body[start : start + n]


class _FakeResponse:
    def __init__(
        self, url: str, status: int, headers: dict[str, str], body: bytes
    ) -> None:
        self.url = url
        self.status = status
        self.headers = headers
        self.content = _FakeContent(body)

    def raise_for_status(self) -> None:
        return None

    def get_encoding(self) -> str:
        raise _NotReadRuntimeError(
            "Cannot compute fallback encoding of a not yet read body"
        )


class _FakeResponseCM:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _patch_client_session(response: _FakeResponse):
    class _FakeSession:
        """Stands in for ``aiohttp.ClientSession``; always returns one response."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def __aenter__(self) -> _FakeSession:
            return self

        async def __aexit__(self, *exc: object) -> bool:
            return False

        def get(self, url: str, **kwargs: object) -> _FakeResponseCM:
            return _FakeResponseCM(response)

    return patch("free_claude_code.api.web_tools.outbound.ClientSession", _FakeSession)


@pytest.mark.parametrize(
    ("header", "raw", "expected"),
    [
        ("text/html", "café — ok".encode(), "café — ok"),
        ("text/plain; charset=iso-8859-1", "café".encode("iso-8859-1"), "café"),
        ('application/json; charset="UTF-8"', b'{"k": "\xc3\xa9"}', '{"k": "é"}'),
        ("text/plain; charset=bogus-codec-x", b"ok", "ok"),
    ],
)
@pytest.mark.asyncio
async def test_run_web_fetch_decodes_without_aiohttp_get_encoding(
    header: str, raw: bytes, expected: str
) -> None:
    response = _FakeResponse(
        url="http://8.8.8.8/page",
        status=200,
        headers={"content-type": header},
        body=raw,
    )
    with _patch_client_session(response):
        out = await _run_web_fetch("http://8.8.8.8/page", _STRICT_EGRESS)

    assert out["data"] == expected
    assert out["url"] == "http://8.8.8.8/page"


@pytest.mark.parametrize(
    ("header", "charset"),
    [
        ("text/html; charset=utf-8", "utf-8"),
        ("text/html; charset=ISO-8859-1", "iso-8859-1"),
        ('text/html; charset="utf-8"', "utf-8"),
        ("text/html", None),
        ("", None),
        ("text/html; charset", None),
        ("text/html; charset=not-a-real-codec", None),
    ],
)
def test_content_type_charset(header: str, charset: str | None) -> None:
    found = _content_type_charset(header)
    if charset is None:
        assert found is None
    else:
        assert found == codecs.lookup(charset).name
