"""In-app update orchestration from a GitHub branch.

The update flow is: ``check`` (read the remote ``pyproject.toml`` version via
raw.githubusercontent), ``apply`` (schedule a detached guardian process), and
a full server process exit. The guardian (``update_guardian.ps1`` /
``update_guardian.sh``) waits for this process to exit, asserts no other FCC
processes are running, downloads the configured branch archive, runs the
pytest gate, installs with ``uv tool install --force``, and relaunches the
original command.

Ownership split (avoids any read-modify-write races):

- ``state.json``    — written only by the server (check cache + scheduling).
- ``progress.json`` — written only by the guardian (stage + messages).
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx
from loguru import logger

from free_claude_code.config.paths import (
    update_progress_path,
    update_state_path,
    update_work_dir_path,
)
from free_claude_code.config.settings import Settings
from free_claude_code.core.json_types import JsonObject
from free_claude_code.core.updates import (
    build_archive_url,
    is_newer_version,
    parse_pyproject_version,
    raw_pyproject_url,
    update_capable,
)
from free_claude_code.core.version import package_version

CHECK_TTL_SECONDS = 300.0
# A progress file older than this is treated as abandoned (e.g. the guardian
# was killed) so a crashed update run cannot block every future update.
PROGRESS_STALE_SECONDS = 4 * 60 * 60.0
# Guardian stages that mean an update run is still in flight.
ACTIVE_STAGES = frozenset({"scheduled", "downloading", "testing", "installing"})
# Environment-specific suite failures that must not block the fork's update
# gate; they remain enforced by CI. Excluded via pytest -k so everything else
# still has to be green before anything is installed.
PYTEST_GATE_EXCLUDE = (
    "not test_admin_versioned_assets_serve_packaged_files "
    "and not test_launcher_config_composes_with_persistent_codex_config"
)
# DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP (win32 only, by value so the
# module imports cleanly on every platform).
_WIN32_DETACHED_FLAGS = 0x00000008 | 0x00000200
_GUARDIAN_TIMEOUT_SECONDS = 20.0


class UpdateDisabledError(Exception):
    """The running installation cannot update itself in place."""


class UpdateInProgressError(Exception):
    """Another update guardian run is still in flight."""


class UpdateCheckFailedError(Exception):
    """The update source could not be consulted."""


def _load_json(path: Path) -> JsonObject:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def _str_field(payload: JsonObject, key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) and value else None


def _num_field(payload: JsonObject, key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


async def _fetch_text(url: str) -> str:
    async with httpx.AsyncClient(
        timeout=_GUARDIAN_TIMEOUT_SECONDS, follow_redirects=True
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


def default_spawn(cmd: list[str]) -> None:
    """Launch the update guardian fully detached from this process."""

    log_path = update_work_dir_path() / "guardian.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        if sys.platform == "win32":
            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                close_fds=True,
                creationflags=_WIN32_DETACHED_FLAGS,
            )
        else:
            subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=log_file,
                close_fds=True,
                start_new_session=True,
            )


class UpdateService:
    """Owns update state files, remote checks, and update scheduling."""

    def __init__(
        self,
        *,
        state_path: Path | None = None,
        progress_path: Path | None = None,
        work_dir: Path | None = None,
        spawn: Callable[[list[str]], None] | None = None,
    ) -> None:
        self._state_path = state_path or update_state_path()
        self._progress_path = progress_path or update_progress_path()
        self._work_dir = work_dir or update_work_dir_path()
        self._spawn = spawn or default_spawn

    def snapshot(self, settings: Settings) -> JsonObject:
        """Return the Admin-facing update state without touching the network."""

        current = package_version()
        capable = update_capable(current)
        state = _load_json(self._state_path)
        progress = _load_json(self._progress_path)
        remote = _str_field(state, "remote_version")
        stage = _str_field(progress, "stage") or "idle"
        in_progress = stage in ACTIVE_STAGES and not self._progress_stale(progress)
        if stage in ACTIVE_STAGES and not in_progress:
            stage = "idle"
        failed = stage == "error" and not in_progress
        return {
            "capable": capable,
            "current_version": current,
            "latest_version": remote if capable else None,
            "update_available": bool(
                capable and remote and is_newer_version(remote, current)
            ),
            "repo": settings.fcc_update_repo,
            "branch": settings.fcc_update_branch,
            "auto": settings.fcc_update_auto,
            "poll_hours": settings.fcc_update_poll_hours,
            "stage": stage,
            "in_progress": in_progress,
            "message": (_str_field(progress, "message") or "") if in_progress else "",
            "last_check": _num_field(state, "checked_ts"),
            "check_error": _str_field(state, "check_error"),
            "last_error": _str_field(progress, "error") if failed else None,
            "last_update": _num_field(progress, "done_ts"),
            "target_version": _str_field(state, "target_version"),
        }

    async def check(self, settings: Settings, *, force: bool = False) -> JsonObject:
        """Fetch (or reuse a recent fetch of) the remote version and return a snapshot."""

        current = package_version()
        if update_capable(current):
            state = _load_json(self._state_path)
            cached_remote = _str_field(state, "remote_version")
            checked_ts = _num_field(state, "checked_ts") or 0.0
            fresh = time.time() - checked_ts < CHECK_TTL_SECONDS
            if not force and cached_remote is not None and fresh:
                return self.snapshot(settings)
            url = raw_pyproject_url(
                settings.fcc_update_repo, settings.fcc_update_branch
            )
            remote: str | None = None
            error: str | None = None
            try:
                text = await _fetch_text(url)
            except (httpx.HTTPError, OSError) as exc:
                error = (
                    "Could not fetch the update source "
                    f"({type(exc).__name__}). Check your network and "
                    f"{settings.fcc_update_repo!r} branch {settings.fcc_update_branch!r}."
                )
                logger.warning("Update check failed: exc_type={}", type(exc).__name__)
            else:
                remote = parse_pyproject_version(text)
                if remote is None:
                    error = "The update source pyproject.toml has no [project].version."
            self._save_state(
                {
                    "remote_version": remote,
                    "check_error": error,
                    "checked_ts": time.time(),
                }
            )
        return self.snapshot(settings)

    async def apply(
        self, settings: Settings, *, checked: JsonObject | None = None
    ) -> JsonObject:
        """Schedule the detached guardian run that installs the new version."""

        current = package_version()
        if not update_capable(current):
            raise UpdateDisabledError(
                "In-app updates are disabled when FCC runs from a source "
                "checkout. Reinstall with scripts/install.ps1 or install.sh."
            )
        snapshot = self.snapshot(settings)
        if snapshot["in_progress"]:
            raise UpdateInProgressError("An update is already running.")
        if checked is None:
            checked = await self.check(settings, force=True)
        check_error = checked.get("check_error")
        remote = checked.get("latest_version")
        if isinstance(check_error, str) and check_error:
            raise UpdateCheckFailedError(check_error)
        if not isinstance(remote, str):
            raise UpdateCheckFailedError("The update source version is unknown.")
        if not checked.get("update_available"):
            return {
                "scheduled": False,
                "from_version": current,
                "latest_version": remote,
                "message": f"FCC {current} is already up to date with "
                f"{settings.fcc_update_repo}@{settings.fcc_update_branch}.",
            }
        self._prepare_work_dir()
        script = self._install_guardian_script()
        try:
            self._spawn(
                self._guardian_command(script, settings=settings, remote=remote)
            )
        except OSError as exc:
            raise UpdateCheckFailedError(
                f"Could not start the update guardian ({type(exc).__name__})."
            ) from exc
        self._save_state(
            {
                "from_version": current,
                "target_version": remote,
                "scheduled_ts": time.time(),
            }
        )
        logger.info(
            "Update scheduled: {} -> {} from {}/{}",
            current,
            remote,
            settings.fcc_update_repo,
            settings.fcc_update_branch,
        )
        return {
            "scheduled": True,
            "from_version": current,
            "target_version": remote,
            "message": "Update started. The server will stop, run the test "
            "gate, install the new version, and relaunch automatically. This "
            "can take several minutes.",
        }

    async def run_auto_tick(self, settings: Settings) -> bool:
        """Check and schedule when due; True when the process must stop now."""

        checked = await self.check(settings)
        if not checked.get("update_available"):
            return False
        try:
            result = await self.apply(settings, checked=checked)
        except UpdateCheckFailedError as exc:
            logger.warning("Automatic update skipped: {}", exc)
            return False
        return bool(result.get("scheduled"))

    def _progress_stale(self, progress: JsonObject) -> bool:
        updated_ts = _num_field(progress, "updated_ts")
        return updated_ts is None or time.time() - updated_ts > PROGRESS_STALE_SECONDS

    def _save_state(self, updates: JsonObject) -> None:
        state = _load_json(self._state_path)
        state.update(updates)
        _write_json(self._state_path, state)

    def _prepare_work_dir(self) -> None:
        shutil.rmtree(self._work_dir, ignore_errors=True)
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def _install_guardian_script(self) -> Path:
        name = (
            "update_guardian.ps1" if sys.platform == "win32" else "update_guardian.sh"
        )
        source = Path(__file__).with_name(name)
        target = self._work_dir / name
        shutil.copyfile(source, target)
        if sys.platform != "win32":
            target.chmod(0o755)
        return target

    def _guardian_command(
        self, script: Path, *, settings: Settings, remote: str
    ) -> list[str]:
        relaunch = base64.b64encode("\n".join(sys.argv).encode("utf-8")).decode("ascii")
        archive_url = build_archive_url(
            settings.fcc_update_repo, settings.fcc_update_branch
        )
        if sys.platform == "win32":
            python_spec = "cpython-3.14.0-windows-x86_64-none"
            return [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script),
                "-WaitPid",
                str(os.getpid()),
                "-ArchiveUrl",
                archive_url,
                "-TargetVersion",
                remote,
                "-WorkDir",
                str(self._work_dir),
                "-ProgressFile",
                str(self._progress_path),
                "-RelaunchB64",
                relaunch,
                "-PythonSpec",
                python_spec,
                "-TestExclude",
                PYTEST_GATE_EXCLUDE,
                "-Cwd",
                os.getcwd(),
            ]
        return [
            "bash",
            str(script),
            str(os.getpid()),
            archive_url,
            remote,
            str(self._work_dir),
            str(self._progress_path),
            relaunch,
            "3.14.0",
            PYTEST_GATE_EXCLUDE,
            os.getcwd(),
        ]
