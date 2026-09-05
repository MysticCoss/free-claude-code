"""Canonical installed Free Claude Code package version."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

DISTRIBUTION_NAME = "free-claude-code"
UNKNOWN_PACKAGE_VERSION = "0+unknown"


def package_version() -> str:
    """Return installed metadata, or an explicit source-only fallback."""
    try:
        return distribution_version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return UNKNOWN_PACKAGE_VERSION
