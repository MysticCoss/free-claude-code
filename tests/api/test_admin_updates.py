import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from free_claude_code.application import updater as updater_module
from free_claude_code.application.updater import UpdateService
from free_claude_code.config.settings import Settings
from free_claude_code.core.version import package_version
from free_claude_code.runtime import application as runtime_application
from tests.api.support import create_test_app, runtime_for_app


def _local_client(app: FastAPI) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))


class Calls:
    def __init__(self) -> None:
        self.spawn: list[list[str]] = []
        self.stop = 0

    def record_spawn(self, cmd: list[str]) -> None:
        self.spawn.append(cmd)

    def record_stop(self) -> None:
        self.stop += 1


def _app(tmp_path: Path, calls: Calls, *, stop_callback: bool = True) -> FastAPI:
    updates = UpdateService(
        state_path=tmp_path / "update" / "state.json",
        progress_path=tmp_path / "update" / "progress.json",
        work_dir=tmp_path / "update" / "work",
        spawn=calls.record_spawn,
    )
    return create_test_app(
        process_stop_callback=calls.record_stop if stop_callback else None,
        updates=updates,
    )


def _serve_newer(monkeypatch: pytest.MonkeyPatch, version: str = "99.0.0") -> None:
    async def fake_fetch(url: str) -> str:
        return f'[project]\nversion = "{version}"\n'

    monkeypatch.setattr(updater_module, "_fetch_text", fake_fetch)


def _serve_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    async def failing_fetch(url: str) -> str:
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(updater_module, "_fetch_text", failing_fetch)


def test_update_status_reports_snapshot_fields(tmp_path: Path) -> None:
    app = _app(tmp_path, Calls())

    body = _local_client(app).get("/admin/api/update").json()

    assert body["capable"] is True
    assert body["current_version"] == package_version()
    assert body["repo"] == "MysticCoss/free-claude-code"
    assert body["branch"] == "main"
    assert body["stage"] == "idle"
    assert body["in_progress"] is False


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/admin/api/update"),
        ("POST", "/admin/api/update/check"),
        ("POST", "/admin/api/update/apply"),
    ],
)
def test_update_routes_are_loopback_only(
    tmp_path: Path, method: str, path: str
) -> None:
    app = _app(tmp_path, Calls())
    remote = TestClient(app, base_url="http://127.0.0.1", client=("10.9.8.7", 50000))

    assert remote.request(method, path).status_code == 403


def test_update_check_fetches_and_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_newer(monkeypatch)
    app = _app(tmp_path, Calls())
    client = _local_client(app)

    body = client.post("/admin/api/update/check").json()

    assert body["latest_version"] == "99.0.0"
    assert body["update_available"] is True

    (tmp_path / "update" / "state.json").read_text(encoding="utf-8")
    assert client.get("/admin/api/update").json()["latest_version"] == "99.0.0"


def test_update_check_surfaces_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_unreachable(monkeypatch)
    app = _app(tmp_path, Calls())

    body = _local_client(app).post("/admin/api/update/check").json()

    assert body["latest_version"] is None
    assert "Could not fetch the update source" in body["check_error"]


def test_update_apply_schedules_guardian_and_stops_after_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_newer(monkeypatch)
    calls = Calls()
    app = _app(tmp_path, calls)

    response = _local_client(app).post("/admin/api/update/apply")

    assert response.status_code == 200
    body = response.json()
    assert body["scheduled"] is True
    assert body["from_version"] == package_version()
    assert body["target_version"] == "99.0.0"
    assert len(calls.spawn) == 1, "no real process may be spawned"
    assert calls.stop == 1, "the full stop runs as a background task"
    state = json.loads((tmp_path / "update" / "state.json").read_text(encoding="utf-8"))
    assert state["target_version"] == "99.0.0"


def test_update_apply_noop_when_up_to_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_newer(monkeypatch, version=package_version())
    calls = Calls()
    app = _app(tmp_path, calls)

    body = _local_client(app).post("/admin/api/update/apply").json()

    assert body["scheduled"] is False
    assert calls.spawn == []
    assert calls.stop == 0


def test_update_apply_requires_a_process_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_newer(monkeypatch)
    calls = Calls()
    app = _app(tmp_path, calls, stop_callback=False)

    response = _local_client(app).post("/admin/api/update/apply")

    assert response.status_code == 409
    assert "install" in response.json()["detail"]
    assert calls.spawn == []


def test_update_apply_conflicts_while_guardian_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_newer(monkeypatch)
    progress_dir = tmp_path / "update"
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / "progress.json").write_text(
        json.dumps({"stage": "downloading", "updated_ts": time.time()}),
        encoding="utf-8",
    )
    app = _app(tmp_path, Calls())

    response = _local_client(app).post("/admin/api/update/apply")

    assert response.status_code == 409


def test_update_apply_fails_when_source_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _serve_unreachable(monkeypatch)
    app = _app(tmp_path, Calls())

    response = _local_client(app).post("/admin/api/update/apply")

    assert response.status_code == 502


def test_update_disabled_for_source_checkouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(updater_module, "package_version", lambda: "0+unknown")
    calls = Calls()
    app = _app(tmp_path, calls)

    status = _local_client(app).get("/admin/api/update").json()
    assert status["capable"] is False

    apply_response = _local_client(app).post("/admin/api/update/apply")
    assert apply_response.status_code == 409
    assert "source checkout" in apply_response.json()["detail"]
    assert calls.spawn == []


def test_runtime_exposes_update_port(tmp_path: Path) -> None:
    calls = Calls()
    app = _app(tmp_path, calls)

    runtime = runtime_for_app(app)

    assert runtime.update_status()["stage"] == "idle"


class _AlwaysAppliesService(UpdateService):
    def __init__(self, tmp_path: Path, spawn: Callable[[list[str]], None]) -> None:
        super().__init__(
            state_path=tmp_path / "update" / "state.json",
            progress_path=tmp_path / "update" / "progress.json",
            work_dir=tmp_path / "update" / "work",
            spawn=spawn,
        )
        self.ticks = 0

    async def run_auto_tick(self, settings: Settings) -> bool:
        self.ticks += 1
        return True


@pytest.mark.asyncio
async def test_auto_update_loop_stops_process_after_applying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runtime_application, "AUTO_UPDATE_MIN_INTERVAL_SECONDS", 0.0)
    calls = Calls()
    service = _AlwaysAppliesService(tmp_path, calls.record_spawn)
    app = create_test_app(
        Settings(fcc_update_auto=True, fcc_update_poll_hours=1e-6),
        process_stop_callback=calls.record_stop,
        updates=service,
    )
    runtime = runtime_for_app(app)

    await runtime.start()
    try:
        for _ in range(200):
            if calls.stop:
                break
            await asyncio.sleep(0.01)
    finally:
        await runtime.close()

    assert service.ticks >= 1
    assert calls.stop == 1
    assert calls.spawn == [], "the stub service decides; nothing real spawns"


@pytest.mark.asyncio
async def test_request_full_stop_awaits_async_callback(tmp_path: Path) -> None:
    calls: list[int] = []

    async def stop() -> None:
        calls.append(1)

    app = create_test_app(
        process_stop_callback=stop,
        updates=UpdateService(
            state_path=tmp_path / "update" / "state.json",
            progress_path=tmp_path / "update" / "progress.json",
            work_dir=tmp_path / "update" / "work",
            spawn=Calls().record_spawn,
        ),
    )

    await runtime_for_app(app).request_full_stop()

    assert calls == [1]


@pytest.mark.asyncio
async def test_request_full_stop_without_callback_is_a_noop() -> None:
    await runtime_for_app(create_test_app()).request_full_stop()


def test_supervisor_wires_process_stop_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from free_claude_code.cli import commands as commands_module
    from free_claude_code.cli.commands import ServerSupervisor

    recorded: dict[str, object] = {}

    def fake_build_asgi_app(
        settings, *, restart_callback=None, process_stop_callback=None
    ):
        recorded["stop"] = process_stop_callback
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(commands_module, "build_asgi_app", fake_build_asgi_app)
    supervisor = ServerSupervisor()

    with pytest.raises(RuntimeError):
        supervisor._run_once(Settings(), open_admin_browser=False, restart_generation=0)

    assert recorded["stop"] == supervisor.request_stop

    custom = Calls().record_stop
    supervisor.process_stop_callback = custom
    with pytest.raises(RuntimeError):
        supervisor._run_once(Settings(), open_admin_browser=False, restart_generation=1)

    assert recorded["stop"] is custom


@pytest.mark.asyncio
async def test_auto_update_task_does_not_start_when_disabled(
    tmp_path: Path,
) -> None:
    service = UpdateService(
        state_path=tmp_path / "update" / "state.json",
        progress_path=tmp_path / "update" / "progress.json",
        work_dir=tmp_path / "update" / "work",
        spawn=Calls().record_spawn,
    )
    app = create_test_app(Settings(fcc_update_auto=False), updates=service)
    runtime = runtime_for_app(app)

    await runtime.start()
    try:
        task_names = {task.get_name() for task in asyncio.all_tasks()}
        assert "fcc-auto-update" not in task_names
    finally:
        await runtime.close()
