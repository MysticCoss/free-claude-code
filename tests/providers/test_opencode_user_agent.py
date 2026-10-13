"""Versioned OpenCode User-Agent cache."""

import time

import pytest

from free_claude_code.providers.opencode import user_agent as ua


@pytest.fixture(autouse=True)
def _stale_version_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ua, "_version", ua.FALLBACK_VERSION)
    # Seed a guaranteed-stale stamp, not 0.0: time.monotonic() counts from
    # boot, so on machines with less uptime than _TTL_S (every CI runner)
    # 0.0 looks fresh and ensure_opencode_version() returns the fallback
    # without ever calling the mocked fetchers.
    monkeypatch.setattr(ua, "_refreshed_at", time.monotonic() - ua._TTL_S - 1)


def test_user_agent_is_versioned_product_token() -> None:
    assert ua.opencode_user_agent() == "opencode/1.18.32"
    assert ua.FALLBACK_VERSION == "1.18.32"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("v1.18.32", "1.18.32"),
        ("1.18.32", "1.18.32"),
        ("1.18", "1.18"),
        ("  2.0.1-beta  ", "2.0.1-beta"),
        ("1", None),
        ("opencode", None),
        ("", None),
        (None, None),
        (118, None),
    ],
)
def test_normalize_accepts_only_semver_looking_strings(
    raw: object,
    expected: str | None,
) -> None:
    assert ua._normalize(raw) == expected


def _patch_fetch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    github: str | Exception | None = None,
    npm: str | Exception | None = None,
) -> list[str]:
    calls: list[str] = []

    async def fake_github(_client: object) -> str | None:
        calls.append("github")
        if isinstance(github, Exception):
            raise github
        return github

    async def fake_npm(_client: object) -> str | None:
        calls.append("npm")
        if isinstance(npm, Exception):
            raise npm
        return npm

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(ua.httpx, "AsyncClient", lambda **_kwargs: _Client())
    monkeypatch.setattr(ua, "_github_version", fake_github)
    monkeypatch.setattr(ua, "_npm_version", fake_npm)
    return calls


@pytest.mark.asyncio
async def test_fresh_cache_skips_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(**_kwargs: object) -> None:
        raise AssertionError("fresh TTL must not construct a client")

    monkeypatch.setattr(ua.httpx, "AsyncClient", explode)
    monkeypatch.setattr(ua, "_refreshed_at", time.monotonic())
    assert await ua.ensure_opencode_version() == ua.FALLBACK_VERSION


@pytest.mark.asyncio
async def test_github_primary_updates_version(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_fetch(monkeypatch, github="9.9.9")
    assert await ua.ensure_opencode_version() == "9.9.9"
    assert ua.opencode_user_agent() == "opencode/9.9.9"
    assert calls == ["github"]


@pytest.mark.asyncio
async def test_github_failure_falls_back_to_npm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fetch(monkeypatch, github=None, npm="8.8.8")
    assert await ua.ensure_opencode_version() == "8.8.8"
    assert calls == ["github", "npm"]


@pytest.mark.asyncio
async def test_both_sources_fail_keep_version_and_stamp_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fetch(monkeypatch, github=RuntimeError("down"), npm=None)
    assert await ua.ensure_opencode_version() == ua.FALLBACK_VERSION
    assert ua._refreshed_at > 0.0
    assert await ua.ensure_opencode_version() == ua.FALLBACK_VERSION


@pytest.mark.asyncio
async def test_invalid_payloads_fall_back_to_fallback_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _patch_fetch(monkeypatch, github=None, npm=None)
    assert await ua.ensure_opencode_version() == ua.FALLBACK_VERSION
    assert calls == ["github", "npm"]


@pytest.mark.asyncio
async def test_client_construction_error_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(**_kwargs: object) -> None:
        raise RuntimeError("cannot connect")

    monkeypatch.setattr(ua.httpx, "AsyncClient", boom)
    assert await ua.ensure_opencode_version() == ua.FALLBACK_VERSION
    assert ua._refreshed_at > 0.0


@pytest.mark.asyncio
async def test_fetch_helper_exception_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def explode(_client: object) -> str:
        raise ValueError("bad json")

    monkeypatch.setattr(ua, "_github_version", explode)

    class _Client:
        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(ua.httpx, "AsyncClient", lambda **_kwargs: _Client())
    assert await ua.ensure_opencode_version() == ua.FALLBACK_VERSION


@pytest.mark.asyncio
async def test_github_tag_name_field_is_parsed() -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return {"tag_name": "v1.18.32"}

    class _Client:
        async def get(self, url: str, /, **_kwargs: object) -> _Response:
            return _Response()

    assert await ua._github_version(_Client()) == "1.18.32"


@pytest.mark.asyncio
async def test_npm_version_field_is_parsed() -> None:
    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return {"version": "1.19.0"}

    class _Client:
        async def get(self, url: str, /, **_kwargs: object) -> _Response:
            return _Response()

    assert await ua._npm_version(_Client()) == "1.19.0"


@pytest.mark.asyncio
async def test_get_json_swallows_http_errors() -> None:
    class _Response:
        def raise_for_status(self) -> None:
            raise RuntimeError("403")

        def json(self) -> object:
            return {"tag_name": "v1.0.0"}

    class _Client:
        async def get(self, url: str, /, **_kwargs: object) -> _Response:
            return _Response()

    assert await ua._get_json(_Client(), "https://example.invalid") is None


def test_field_version_rejects_non_objects() -> None:
    assert ua._field_version(["nope"], "tag_name") is None
    assert ua._field_version({"tag_name": "v1.2.3"}, "tag_name") == "1.2.3"
    assert ua._field_version({"tag_name": "nightly"}, "tag_name") is None
