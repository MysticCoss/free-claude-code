"""Versioned OpenCode User-Agent for first-party upstream requests.

The OpenCode Zen free-tier gateway rejects any client that does not identify
as ``opencode/<major.minor[.patch...]>`` (HTTP 403 FreeTierError). The
version is refreshed from the GitHub releases API (primary) or the npm
registry (fallback) on a TTL so the proxy tracks new CLI releases without a
code change; on any failure the last known version is kept.

``opencode_user_agent()`` is a pure sync read of the module cache and is safe
on the hot path. ``ensure_opencode_version()`` is awaited once per dispatch
when the TTL has expired and never raises.
"""

import re
import time
from typing import Any

import httpx
from loguru import logger

FALLBACK_VERSION = "1.18.32"
_TTL_S = 6 * 60 * 60
_FETCH_TIMEOUT_S = 5.0
_GITHUB_URL = "https://api.github.com/repos/anomalyco/opencode/releases/latest"
_NPM_URL = "https://registry.npmjs.org/opencode-ai/latest"
_VERSION_RE = re.compile(r"\d+\.\d+")

_version = FALLBACK_VERSION
# Stale at import so the first dispatch refreshes; tests re-seed freshness.
# This must not be 0.0: time.monotonic() counts from boot, so on a machine
# with less uptime than _TTL_S (every fresh CI runner, any rebooted host)
# 0.0 reads as "refreshed just now" and the first dispatch would serve the
# fallback version without ever trying the network. -inf is stale on every
# clock.
_refreshed_at = float("-inf")


def opencode_user_agent() -> str:
    """Return the current ``opencode/<version>`` User-Agent string."""
    return f"opencode/{_version}"


async def ensure_opencode_version(*, proxy: str | None = None) -> str:
    """Refresh the cached version when the TTL has expired.

    GitHub releases first, npm registry second; any network or parse failure
    keeps the previous version and still stamps the TTL so a flaky upstream
    cannot stampede. Never raises.
    """
    global _version, _refreshed_at
    if time.monotonic() - _refreshed_at < _TTL_S:
        return _version
    try:
        async with httpx.AsyncClient(
            proxy=proxy,
            timeout=_FETCH_TIMEOUT_S,
            follow_redirects=True,
        ) as client:
            fetched = await _github_version(client) or await _npm_version(client)
        if fetched is not None:
            _version = fetched
    except Exception as exc:
        logger.debug("opencode version refresh failed: {}", exc)
    _refreshed_at = time.monotonic()
    return _version


def _normalize(raw: object) -> str | None:
    if not isinstance(raw, str):
        return None
    version = raw.strip()
    if version.startswith("v"):
        version = version[1:]
    if _VERSION_RE.match(version):
        return version
    return None


async def _get_json(client: Any, url: str) -> object | None:
    try:
        response = await client.get(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "free-claude-code",
            },
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def _field_version(payload: object, key: str) -> str | None:
    if not isinstance(payload, dict):
        return None
    return _normalize(payload.get(key))


async def _github_version(client: Any) -> str | None:
    return _field_version(await _get_json(client, _GITHUB_URL), "tag_name")


async def _npm_version(client: Any) -> str | None:
    return _field_version(await _get_json(client, _NPM_URL), "version")
