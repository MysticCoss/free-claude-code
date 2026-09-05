import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from free_claude_code.application import updater as updater_module
from free_claude_code.application.updater import (
    ACTIVE_STAGES,
    PROGRESS_STALE_SECONDS,
    PYTEST_GATE_EXCLUDE,
    UpdateCheckFailedError,
    UpdateDisabledError,
    UpdateInProgressError,
    UpdateService,
)
from free_claude_code.config.settings import Settings


class SpawnRecorder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> None:
        self.calls.append(cmd)


def _service(tmp_path: Path, spawn: Callable[[list[str]], None]) -> UpdateService:
    return UpdateService(
        state_path=tmp_path / "state.json",
        progress_path=tmp_path / "progress.json",
        work_dir=tmp_path / "work",
        spawn=spawn,
    )


def _settings() -> Settings:
    """Pin the update fields so ambient process env cannot skew tests."""

    return Settings(
        fcc_update_repo="MysticCoss/free-claude-code",
        fcc_update_branch="main",
        fcc_update_auto=False,
        fcc_update_poll_hours=6.0,
    )


@pytest.fixture
def installed_version(monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(updater_module, "package_version", lambda: "5.22.4")
    return "5.22.4"


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    async def fake_fetch(url: str) -> str:
        calls.append(url)
        return '[project]\nversion = "5.23.0"\n'

    monkeypatch.setattr(updater_module, "_fetch_text", fake_fetch)
    return calls


def test_snapshot_defaults_to_idle_when_no_state_exists(
    tmp_path: Path, installed_version: str
) -> None:
    service = _service(tmp_path, SpawnRecorder())

    snapshot = service.snapshot(_settings())

    assert snapshot["capable"] is True
    assert snapshot["current_version"] == "5.22.4"
    assert snapshot["latest_version"] is None
    assert snapshot["update_available"] is False
    assert snapshot["stage"] == "idle"
    assert snapshot["in_progress"] is False
    assert snapshot["repo"] == "MysticCoss/free-claude-code"
    assert snapshot["branch"] == "main"


def test_snapshot_marks_source_checkouts_incapable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(updater_module, "package_version", lambda: "0+unknown")
    service = _service(tmp_path, SpawnRecorder())

    snapshot = service.snapshot(_settings())

    assert snapshot["capable"] is False
    assert snapshot["latest_version"] is None
    assert snapshot["update_available"] is False


def test_snapshot_reads_guardian_progress(
    tmp_path: Path, installed_version: str
) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "stage": "testing",
                "message": "Running the pytest gate",
                "updated_ts": time.time(),
            }
        ),
        encoding="utf-8",
    )
    service = _service(tmp_path, SpawnRecorder())

    snapshot = service.snapshot(_settings())

    assert snapshot["stage"] == "testing"
    assert snapshot["in_progress"] is True
    assert snapshot["message"] == "Running the pytest gate"


def test_snapshot_ignores_stale_progress(
    tmp_path: Path, installed_version: str
) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "stage": "installing",
                "updated_ts": time.time() - PROGRESS_STALE_SECONDS - 1,
            }
        ),
        encoding="utf-8",
    )
    service = _service(tmp_path, SpawnRecorder())

    snapshot = service.snapshot(_settings())

    assert snapshot["stage"] == "idle"
    assert snapshot["in_progress"] is False


def test_snapshot_surfaces_last_error(tmp_path: Path, installed_version: str) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps({"stage": "error", "error": "pytest gate failed"}),
        encoding="utf-8",
    )
    service = _service(tmp_path, SpawnRecorder())

    snapshot = service.snapshot(_settings())

    assert snapshot["in_progress"] is False
    assert snapshot["last_error"] == "pytest gate failed"


@pytest.mark.asyncio
async def test_check_fetches_remote_version_and_caches_it(
    tmp_path: Path, installed_version: str, no_network: list[str]
) -> None:
    service = _service(tmp_path, SpawnRecorder())

    first = await service.check(_settings())
    assert first["latest_version"] == "5.23.0"
    assert first["update_available"] is True
    assert no_network == [
        "https://raw.githubusercontent.com/MysticCoss/free-claude-code/main/pyproject.toml"
    ]

    second = await service.check(_settings())
    assert second["latest_version"] == "5.23.0"
    assert len(no_network) == 1, "the TTL cache must avoid a second fetch"

    forced = await service.check(_settings(), force=True)
    assert forced["latest_version"] == "5.23.0"
    assert len(no_network) == 2


@pytest.mark.asyncio
async def test_check_records_network_errors_without_raising(
    tmp_path: Path, installed_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_fetch(url: str) -> str:
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(updater_module, "_fetch_text", failing_fetch)
    service = _service(tmp_path, SpawnRecorder())

    snapshot = await service.check(_settings())

    assert snapshot["latest_version"] is None
    check_error = snapshot["check_error"]
    assert isinstance(check_error, str)
    assert "Could not fetch the update source" in check_error


@pytest.mark.asyncio
async def test_check_reports_unparseable_source(
    tmp_path: Path, installed_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def bad_toml(url: str) -> str:
        return "version = 5"

    monkeypatch.setattr(updater_module, "_fetch_text", bad_toml)
    service = _service(tmp_path, SpawnRecorder())

    snapshot = await service.check(_settings())

    check_error = snapshot["check_error"]
    assert isinstance(check_error, str)
    assert "pyproject.toml has no [project].version" in check_error


@pytest.mark.asyncio
async def test_check_skips_network_for_source_checkouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(updater_module, "package_version", lambda: "0+unknown")
    calls: list[str] = []

    async def spy_fetch(url: str) -> str:
        calls.append(url)
        return ""

    monkeypatch.setattr(updater_module, "_fetch_text", spy_fetch)
    service = _service(tmp_path, SpawnRecorder())

    snapshot = await service.check(_settings())

    assert calls == []
    assert snapshot["capable"] is False


@pytest.mark.asyncio
async def test_apply_refuses_source_checkouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(updater_module, "package_version", lambda: "0+unknown")
    spawn = SpawnRecorder()
    service = _service(tmp_path, spawn)

    with pytest.raises(UpdateDisabledError):
        await service.apply(_settings())
    assert spawn.calls == []


@pytest.mark.asyncio
async def test_apply_schedules_guardian_when_update_available(
    tmp_path: Path, installed_version: str, no_network: list[str]
) -> None:
    spawn = SpawnRecorder()
    service = _service(tmp_path, spawn)

    result = await service.apply(_settings())

    assert result["scheduled"] is True
    assert result["from_version"] == "5.22.4"
    assert result["target_version"] == "5.23.0"
    assert len(spawn.calls) == 1
    command = spawn.calls[0]
    win32 = sys.platform == "win32"
    assert command[0] == ("powershell" if win32 else "bash")
    script = Path(command[1] if not win32 else command[command.index("-File") + 1])
    assert script.name == ("update_guardian.ps1" if win32 else "update_guardian.sh")
    assert script.read_text(encoding="utf-8").strip()
    assert PYTEST_GATE_EXCLUDE in command
    assert "5.23.0" in command
    progress = service.snapshot(_settings())
    assert progress["target_version"] == "5.23.0"


@pytest.mark.asyncio
async def test_apply_is_noop_when_already_up_to_date(
    tmp_path: Path,
    installed_version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def same_version(url: str) -> str:
        return '[project]\nversion = "5.22.4"\n'

    monkeypatch.setattr(updater_module, "_fetch_text", same_version)
    spawn = SpawnRecorder()
    service = _service(tmp_path, spawn)

    result = await service.apply(_settings())

    assert result["scheduled"] is False
    message = result["message"]
    assert isinstance(message, str)
    assert "up to date" in message
    assert spawn.calls == []


@pytest.mark.asyncio
async def test_apply_fails_when_check_is_unreachable(
    tmp_path: Path, installed_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_fetch(url: str) -> str:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(updater_module, "_fetch_text", failing_fetch)
    service = _service(tmp_path, SpawnRecorder())

    with pytest.raises(UpdateCheckFailedError):
        await service.apply(_settings())


@pytest.mark.asyncio
async def test_apply_refuses_while_guardian_runs(
    tmp_path: Path, installed_version: str, no_network: list[str]
) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps({"stage": "downloading", "updated_ts": time.time()}),
        encoding="utf-8",
    )
    service = _service(tmp_path, SpawnRecorder())

    with pytest.raises(UpdateInProgressError):
        await service.apply(_settings())


@pytest.mark.asyncio
async def test_apply_ignores_stale_progress_after_dead_guardian(
    tmp_path: Path, installed_version: str, no_network: list[str]
) -> None:
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps({"stage": "testing", "updated_ts": time.time() - 5 * 3600}),
        encoding="utf-8",
    )
    spawn = SpawnRecorder()
    service = _service(tmp_path, spawn)

    result = await service.apply(_settings())

    assert result["scheduled"] is True
    assert len(spawn.calls) == 1


@pytest.mark.asyncio
async def test_apply_maps_spawn_oserror_to_check_failed(
    tmp_path: Path, installed_version: str, no_network: list[str]
) -> None:
    def refusing_spawn(cmd: list[str]) -> None:
        raise OSError("cannot exec")

    service = _service(tmp_path, refusing_spawn)

    with pytest.raises(UpdateCheckFailedError, match="guardian"):
        await service.apply(_settings())


@pytest.mark.asyncio
async def test_run_auto_tick_applies_only_when_update_available(
    tmp_path: Path, installed_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def older(url: str) -> str:
        return '[project]\nversion = "5.0.0"\n'

    monkeypatch.setattr(updater_module, "_fetch_text", older)
    spawn = SpawnRecorder()
    service = _service(tmp_path, spawn)

    assert await service.run_auto_tick(_settings()) is False
    assert spawn.calls == []

    async def newer(url: str) -> str:
        return '[project]\nversion = "5.23.0"\n'

    monkeypatch.setattr(updater_module, "_fetch_text", newer)
    (tmp_path / "state.json").unlink()  # expire the TTL cache between ticks

    assert await service.run_auto_tick(_settings()) is True
    assert len(spawn.calls) == 1


@pytest.mark.asyncio
async def test_run_auto_tick_swallows_check_failures(
    tmp_path: Path, installed_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_fetch(url: str) -> str:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(updater_module, "_fetch_text", failing_fetch)
    spawn = SpawnRecorder()
    service = _service(tmp_path, spawn)

    assert await service.run_auto_tick(_settings()) is False
    assert spawn.calls == []


@pytest.mark.asyncio
async def test_active_stages_cover_guardian_progress_windows() -> None:
    assert {"scheduled", "downloading", "testing", "installing"} == set(ACTIVE_STAGES)


def test_guardian_scripts_ship_beside_the_updater_module() -> None:
    ps1 = (Path(updater_module.__file__).with_name("update_guardian.ps1")).read_text(
        encoding="utf-8"
    )
    sh = (Path(updater_module.__file__).with_name("update_guardian.sh")).read_text(
        encoding="utf-8"
    )
    for script in (ps1, sh):
        assert "uv tool install" in script
    # pytest collects from the working directory, so both guardians must run
    # the gate from inside the extracted source tree, never the server's CWD.
    assert 'cd "$src_dir"' in sh
    assert "-WorkingDirectory $srcDir.FullName" in ps1
