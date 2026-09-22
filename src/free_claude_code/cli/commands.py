"""Implementations for installed Free Claude Code commands."""

import errno
import json
import socket
import threading
import time
import webbrowser
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.request import Request

from loguru import logger
from starlette.types import ASGIApp

from free_claude_code.cli.local_http import open_local_request
from free_claude_code.cli.process_registry import kill_all_best_effort
from free_claude_code.config.loader import (
    ManagedConfigStore,
    clear_settings_cache,
    get_settings,
)
from free_claude_code.config.paths import managed_env_path
from free_claude_code.config.server_urls import (
    local_admin_url,
    local_claude_desktop_url,
    local_proxy_root_url,
)
from free_claude_code.config.settings import Settings

from .server_socket import ServerSockets

if TYPE_CHECKING:
    import uvicorn

    from free_claude_code.runtime.bootstrap import RuntimeASGIApp

SERVER_GRACEFUL_SHUTDOWN_SECONDS = 5
DESKTOP_LISTENER_JOIN_GRACE_SECONDS = 2
DESKTOP_LISTENER_BIND_ATTEMPTS = 10
DESKTOP_LISTENER_BIND_RETRY_SECONDS = 0.5
_BROWSER_HANDOFF_SECONDS = 5.0


def desktop_listener_port(settings: Settings) -> int | None:
    """Return the dedicated Claude Desktop 3P listener port, or None if unneeded.

    The listener is only planned when the feature is on and its port differs
    from the main port; equality is rejected by settings validation but is
    also refused here so a bad snapshot can never produce a duplicate bind.
    """

    if not settings.enable_claude_desktop_3p:
        return None
    if settings.claude_desktop_port == settings.port:
        return None
    return settings.claude_desktop_port


def _start_admin_browser(
    settings: Settings, eligible: Callable[[], bool]
) -> threading.Event:
    """Hand off an optional browser action without keeping FCC alive."""
    completed = threading.Event()
    url = local_admin_url(settings)

    def open_browser() -> None:
        try:
            if eligible() and not webbrowser.open(url):
                logger.warning(
                    "Could not open Admin in a browser. Open {} manually.", url
                )
        except Exception as exc:
            logger.warning("Could not open Admin: {}. Open {} manually.", exc, url)
        finally:
            completed.set()

    try:
        threading.Thread(
            target=open_browser, name="fcc-open-admin-browser", daemon=True
        ).start()
    except Exception as exc:
        logger.warning(
            "Could not start the Admin browser: {}. Open {} manually.", exc, url
        )
        completed.set()
    return completed


def serve() -> None:
    """Start and supervise the FastAPI server."""
    try:
        ServerSupervisor().run()
    except OSError as exc:
        logger.error("Could not start FCC: {}", exc)
        raise SystemExit(1) from None


class ServerStatus(StrEnum):
    """Observable state of the server owned by a supervisor."""

    STARTING = "Starting"
    RUNNING = "Running"
    STOPPING = "Stopping"
    STOPPED = "Stopped"


class ServerSupervisor:
    """Own one FCC server lifecycle, including config-driven restarts."""

    def __init__(self, *, console_logging: bool = True) -> None:
        self._console_logging = console_logging
        self._lock = threading.Lock()
        self._server: uvicorn.Server | None = None
        self._run_scheduled = False
        self._running = False
        self.stop_event = threading.Event()
        self._ready_settings: Settings | None = None
        self._pending_admin = False
        self._auto_browser_opened = False
        self._owned_server = False
        self._restart_generation = 0
        # In-app updates ask the process owner to exit so the guardian can
        # replace the binary; desktop installs wire their quit controller.
        self.process_stop_callback: Callable[[], None] | None = None

    @property
    def status(self) -> ServerStatus:
        with self._lock:
            if self._run_scheduled:
                return ServerStatus.STARTING
            if not self._running:
                return ServerStatus.STOPPED
            if self._server is None:
                return ServerStatus.STARTING
            if self._server.should_exit:
                return ServerStatus.STOPPING
            if self._server.started:
                return ServerStatus.RUNNING
            return ServerStatus.STARTING

    def schedule_run(self) -> bool:
        """Reserve a worker run before its thread starts."""

        with self._lock:
            if self.stop_event.is_set() or self._run_scheduled or self._running:
                return False
            self._run_scheduled = True
            return True

    def run(
        self,
        *,
        open_admin_browser: bool | None = None,
        existing_server: Callable[[Settings], bool] | None = None,
    ) -> None:
        """Block until stopped, applying only fully closed Admin restarts."""

        with self._lock:
            self._run_scheduled = False
            if self._running:
                raise RuntimeError("The FCC server supervisor is already running.")
            if self.stop_event.is_set():
                return
            self._running = True

        self._auto_browser_opened = False
        try:
            try:
                while not self._is_stop_requested():
                    with self._lock:
                        restart_generation = self._restart_generation
                    settings = load_server_settings()
                    should_open_admin = (
                        settings.open_admin_browser
                        if open_admin_browser is None
                        else open_admin_browser
                    ) and not self._auto_browser_opened
                    try:
                        restart = self._run_once(
                            settings,
                            open_admin_browser=should_open_admin,
                            restart_generation=restart_generation,
                        )
                    except OSError as exc:
                        if (
                            exc.errno == errno.EADDRINUSE
                            and existing_server
                            and existing_server(settings)
                        ):
                            return
                        raise
                    if not restart:
                        return
                    clear_settings_cache()
            except KeyboardInterrupt:
                return
        finally:
            with self._lock:
                self._server = None
                self._running = False
            if self._owned_server:
                kill_all_best_effort()

    def request_restart(self) -> bool:
        """Reload an active generation or coalesce into a scheduled fresh run."""

        with self._lock:
            if self.stop_event.is_set():
                return False
            if self._run_scheduled:
                self._restart_generation += 1
                return True
            if not self._running:
                return False
            self._restart_generation += 1
            if self._server is not None:
                self._server.should_exit = True
            return True

    def request_stop(self) -> None:
        """Permanently stop this supervisor after graceful runtime cleanup."""

        with self._lock:
            self.stop_event.set()
            self._run_scheduled = False
            if self._server is not None:
                self._server.should_exit = True

    def _is_stop_requested(self) -> bool:
        with self._lock:
            return self.stop_event.is_set()

    def request_open_admin(self) -> None:
        with self._lock:
            if self.stop_event.is_set():
                return
            self._pending_admin = True
            settings = self._ready_settings
            generation = self._restart_generation
        if settings is not None:
            self._open_admin(settings, generation)

    def _open_admin(self, settings: Settings, generation: int) -> None:
        def eligible() -> bool:
            with self._lock:
                if (
                    self.stop_event.is_set()
                    or self._restart_generation != generation
                    or self._ready_settings is not settings
                ):
                    return False
                self._pending_admin = False
                return True

        _start_admin_browser(settings, eligible)

    def _run_once(
        self,
        settings: Settings,
        *,
        open_admin_browser: bool,
        restart_generation: int,
    ) -> bool:
        with ServerSockets.reserve(settings.host, settings.port) as listeners:
            self._owned_server = True
            if self.stop_event.is_set():
                return False
            return self._run_bound(
                settings,
                listeners.sockets,
                open_admin_browser=open_admin_browser,
                restart_generation=restart_generation,
            )

    def _run_bound(
        self,
        settings: Settings,
        sockets: list[socket.socket],
        *,
        open_admin_browser: bool,
        restart_generation: int,
    ) -> bool:
        import uvicorn

        from free_claude_code.runtime.bootstrap import build_asgi_app

        from .uvicorn_server import RuntimeServer

        asgi_app = build_asgi_app(
            settings,
            restart_callback=self._request_runtime_restart,
            process_stop_callback=self.process_stop_callback or self.request_stop,
        )
        config = uvicorn.Config(
            asgi_app,
            host=settings.host,
            port=settings.port,
            log_level="debug",
            log_config=(
                uvicorn.config.LOGGING_CONFIG if self._console_logging else None
            ),
            timeout_graceful_shutdown=SERVER_GRACEFUL_SHUTDOWN_SECONDS,
        )

        def on_started() -> None:
            with self._lock:
                if (
                    self._server is not server
                    or self.stop_event.is_set()
                    or self._restart_generation != restart_generation
                ):
                    return
                self._ready_settings = settings
                should_open = open_admin_browser or self._pending_admin
                if open_admin_browser:
                    self._auto_browser_opened = True
            asgi_app.runtime.http_started()
            if should_open:
                self._open_admin(settings, restart_generation)

        server = RuntimeServer(
            config,
            begin_shutdown=asgi_app.runtime.begin_shutdown,
            on_started=on_started,
            close_runtime=asgi_app.runtime.close,
        )
        with self._lock:
            self._server = server
            if (
                self.stop_event.is_set()
                or self._restart_generation != restart_generation
            ):
                server.should_exit = True

        desktop_server, desktop_thread = self._start_desktop_listener(
            asgi_app, settings
        )
        try:
            server.run(sockets=sockets)
        finally:
            with self._lock:
                if self._server is server:
                    self._server = None
                    self._ready_settings = None
            self._stop_desktop_listener(desktop_server, desktop_thread)

        with self._lock:
            restart_requested = self._restart_generation != restart_generation
            stop_requested = self.stop_event.is_set()
        return restart_requested and not stop_requested and asgi_app.runtime.is_closed

    def _start_desktop_listener(
        self,
        asgi_app: RuntimeASGIApp | ASGIApp,
        settings: Settings,
    ) -> tuple[uvicorn.Server | None, threading.Thread | None]:
        """Run the optional Claude Desktop 3P listener beside the main server.

        The listener shares the main ASGI app and runtime; requests are told
        apart by the accepting socket port (``scope["server"]``).
        """

        import uvicorn

        port = desktop_listener_port(settings)
        if port is None:
            return None, None
        # Reserve the port synchronously: the previous generation's listener
        # thread (or a still-exiting process) may hold it briefly after a
        # restart, and a blind spawn would die on EADDRINUSE while the "port
        # starting" log already claimed success. Only advertise once bound.
        reserved: ServerSockets | None = None
        for attempt in range(1, DESKTOP_LISTENER_BIND_ATTEMPTS + 1):
            try:
                reserved = ServerSockets.reserve(settings.host, port)
                break
            except OSError as exc:
                if (
                    exc.errno != errno.EADDRINUSE
                    or attempt == DESKTOP_LISTENER_BIND_ATTEMPTS
                ):
                    logger.error(
                        "Claude Desktop 3P listener cannot bind {}:{} ({}). "
                        "The main FCC server is unaffected; the listener "
                        "will be retried on the next restart.",
                        settings.host,
                        port,
                        exc,
                    )
                    return None, None
                logger.warning(
                    "Claude Desktop 3P listener port {} in use "
                    "(attempt {}/{}); retrying while the previous "
                    "listener shuts down.",
                    port,
                    attempt,
                    DESKTOP_LISTENER_BIND_ATTEMPTS,
                )
                time.sleep(DESKTOP_LISTENER_BIND_RETRY_SECONDS)
        if reserved is None:
            return None, None
        server = uvicorn.Server(
            uvicorn.Config(
                asgi_app,
                host=settings.host,
                port=port,
                # The listener shares the main server's runtime, which owns
                # asyncio locks bound to the main event loop. Running a second
                # lifespan here would drive runtime.start()/close() from a
                # second loop, fouling those locks and wedging startup or
                # shutdown (update-apply full stop never completes and the
                # guardian rolls back). Serve HTTP only; the main server owns
                # the whole lifecycle.
                lifespan="off",
                log_level="debug",
                log_config=(
                    uvicorn.config.LOGGING_CONFIG if self._console_logging else None
                ),
                timeout_graceful_shutdown=SERVER_GRACEFUL_SHUTDOWN_SECONDS,
            )
        )
        thread = threading.Thread(
            target=self._serve_desktop_listener,
            args=(server, port, reserved.sockets),
            name="fcc-claude-desktop-3p",
            daemon=True,
        )
        try:
            thread.start()
        except BaseException:
            reserved.close()
            raise
        logger.info(
            "Claude Desktop 3P listener starting on {}",
            local_claude_desktop_url(settings),
        )
        return server, thread

    @staticmethod
    def _serve_desktop_listener(
        server: uvicorn.Server, port: int, sockets: list[socket.socket]
    ) -> None:
        """Serve the desktop listener; its failure must never kill the main server."""

        try:
            server.run(sockets=sockets)
        except (SystemExit, Exception) as error:
            logger.error(
                "Claude Desktop 3P listener on port {} failed or stopped: {}. "
                "The main FCC server is unaffected.",
                port,
                error,
            )

    @staticmethod
    def _stop_desktop_listener(
        server: uvicorn.Server | None,
        thread: threading.Thread | None,
    ) -> None:
        """Drain the desktop listener before the generation returns."""

        if server is None or thread is None:
            return
        server.should_exit = True
        thread.join(
            timeout=SERVER_GRACEFUL_SHUTDOWN_SECONDS
            + DESKTOP_LISTENER_JOIN_GRACE_SECONDS
        )
        if thread.is_alive():
            logger.warning(
                "Claude Desktop 3P listener did not stop within the join grace; "
                "leaving the daemon thread to exit on its own."
            )

    def _request_runtime_restart(self) -> None:
        self.request_restart()


def load_server_settings() -> Settings:
    """Return canonical settings after repairing invalid managed proxies."""

    settings = get_settings()
    removed = ManagedConfigStore().repair_invalid_provider_proxies()
    if not removed:
        return settings

    logger.warning(
        "Removed invalid managed provider proxy settings from {}: {}. "
        "Configure valid proxy URLs in Admin if needed.",
        managed_env_path(),
        ", ".join(removed),
    )
    clear_settings_cache()
    return get_settings()


def open_admin_when_ready(
    settings: Settings, *, stop_event: threading.Event | None = None
) -> bool:
    """Recognize an external FCC instance and attempt to open its local Admin page."""
    stop = stop_event or threading.Event()
    deadline = time.monotonic() + 30.0
    url = f"{local_proxy_root_url(settings)}/admin/api/status"
    while not stop.is_set() and time.monotonic() < deadline:
        try:
            with open_local_request(Request(url), timeout=1.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict) or not (
                isinstance(payload.get("instance_id"), str)
                and len(payload["instance_id"]) == 32
                and payload.get("status") in {"running", "stopping"}
                and isinstance(payload.get("host"), str)
                and payload.get("port") == settings.port
                and isinstance(payload.get("provider_status"), list)
                and isinstance(payload.get("cached_models"), dict)
            ):
                return False
            if payload["status"] == "running" and not stop.is_set():
                completed = _start_admin_browser(settings, lambda: not stop.is_set())
                # This extra launcher is about to exit: allow a brief URL handoff.
                handoff_deadline = time.monotonic() + _BROWSER_HANDOFF_SECONDS
                while not stop.is_set():
                    remaining = handoff_deadline - time.monotonic()
                    if remaining <= 0 or completed.wait(min(0.05, remaining)):
                        break
                return True
        except HTTPError, ValueError, UnicodeError:
            return False
        except URLError, OSError:
            pass
        stop.wait(0.15)
    return False
