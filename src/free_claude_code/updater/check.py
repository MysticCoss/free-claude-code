"""Standalone version preflight for the native update launchers.

The fork updates from its own GitHub branch (``FCC_UPDATE_REPO`` /
``FCC_UPDATE_BRANCH``), not from PyPI. The ``fcc-update`` wrapper passes a
launcher path; when the branch is newer this stages the same guardian the
Admin UI uses into that launcher, and the wrapper runs it after this
interpreter exits, because ``uv tool install`` replaces it.
"""

import asyncio
import sys
from pathlib import Path

from pydantic import ValidationError

from free_claude_code.application.updater import (
    UpdateCheckFailedError,
    UpdateDisabledError,
    UpdateInProgressError,
    UpdateService,
)
from free_claude_code.config.loader import get_settings

UPDATE_AVAILABLE = 10


def main(argv: list[str] | None = None) -> int:
    """Return 10 after staging the launcher, 0 to skip, or 1 on failure."""
    args = sys.argv[1:] if argv is None else argv
    # Legacy console codepages cannot encode every path in error messages.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    if len(args) != 1:
        print("usage: _fcc-update-check <launcher-path>", file=sys.stderr)
        return 1
    launcher = Path(args[0])
    try:
        settings = get_settings()
        snapshot, available = asyncio.run(
            UpdateService().prepare_cli_update(settings, launcher)
        )
    except (
        UpdateCheckFailedError,
        UpdateDisabledError,
        UpdateInProgressError,
        ValidationError,
        OSError,
    ) as exc:
        print(f"Could not check for FCC updates: {exc} Please retry.")
        return 1

    source = f"{settings.fcc_update_repo}@{settings.fcc_update_branch}"
    installed = snapshot["current_version"]
    latest = snapshot["latest_version"]
    if not available:
        print(f"FCC {installed} is up to date with {source} ({latest}).")
        return 0
    print(
        f"FCC update available from {source}: {installed} -> {latest}. "
        "Running the test gate and installer."
    )
    return UPDATE_AVAILABLE
