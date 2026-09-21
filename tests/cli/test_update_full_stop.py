"""Update-apply full stop must exit the server process.

Regression coverage for a failed in-app update whose guardian reported
"The previous FCC process did not exit within 120 seconds": with the
Claude Desktop 3P listener enabled, the supervisor generation never
exited after ``request_stop``. The listener ran a second lifespan on the
shared ``RuntimeASGIApp``, driving ``runtime.start()``/``close()`` from a
second event loop and fouling the runtime's loop-bound asyncio locks, so
startup or shutdown wedged while the port looked alive. The listener now
serves HTTP only (``lifespan="off"``); the main server owns the lifecycle.

These tests run a real ``ServerSupervisor`` generation against the real
application runtime on ephemeral ports and require it to exit promptly
after a full stop, with and without the desktop listener.
"""

import http.client
import socket
import threading
import time
from unittest.mock import patch

from free_claude_code.cli.commands import ServerSupervisor
from free_claude_code.config.settings import Settings
from free_claude_code.runtime.asgi import RuntimeASGIApp
from tests.api.support import create_test_app, runtime_for_app


def _ephemeral_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _settings(*, desktop_port: int | None = None) -> Settings:
    return Settings(
        host="127.0.0.1",
        port=_ephemeral_port(),
        messaging_platform="none",
        fcc_update_auto=False,
        enable_claude_desktop_3p=desktop_port is not None,
        claude_desktop_port=desktop_port if desktop_port is not None else 8083,
    )


def _await_http(port: int, timeout_s: float = 15.0) -> None:
    """Poll until uvicorn serves HTTP (not just a reserved-port backlog)."""

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            try:
                connection.request("GET", "/admin/api/status")
                connection.getresponse().read()
            finally:
                connection.close()
            return
        except Exception:
            time.sleep(0.05)
    raise AssertionError(f"server never served HTTP on {port}")


def _run_generation_until_full_stop(settings: Settings) -> None:
    app = create_test_app(settings)
    wrapped = RuntimeASGIApp(app, runtime_for_app(app))
    supervisor = ServerSupervisor(console_logging=False)
    errors: list[BaseException] = []

    def _serve() -> None:
        try:
            supervisor.run(open_admin_browser=False)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=_serve, name="fcc-test-supervisor")
    worker.daemon = True
    with (
        patch(
            "free_claude_code.cli.commands.load_server_settings",
            return_value=settings,
        ),
        patch(
            "free_claude_code.runtime.bootstrap.build_asgi_app",
            return_value=wrapped,
        ),
    ):
        worker.start()
        try:
            _await_http(settings.port)
            if settings.enable_claude_desktop_3p:
                _await_http(settings.claude_desktop_port)
            supervisor.request_stop()
            worker.join(20.0)
            assert not worker.is_alive(), "supervisor did not exit after full stop"
        finally:
            supervisor.request_stop()
            worker.join(5.0)
    assert errors == [], f"supervisor raised during shutdown: {errors!r}"


def test_full_stop_exits_without_desktop() -> None:
    _run_generation_until_full_stop(_settings())


def test_full_stop_exits_with_desktop_listener() -> None:
    _run_generation_until_full_stop(_settings(desktop_port=_ephemeral_port()))
