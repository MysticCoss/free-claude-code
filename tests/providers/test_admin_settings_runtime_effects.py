"""Admin UI settings take effect on provider runtime behavior.

Each test starts from a valid ``Settings`` object (the Admin UI writes
Settings) and proves the value drives real runtime behavior end to end:
timeouts reach ``ProviderConfig``, admission limits throttle live calls,
and host/port values bind.
"""

import asyncio
import socket
from unittest.mock import patch

import pytest

from free_claude_code.cli.server_socket import ServerSockets
from free_claude_code.config.nim import NimSettings
from free_claude_code.config.provider_catalog import PROVIDER_CATALOG
from free_claude_code.config.settings import Settings
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.nvidia_nim import NvidiaNimProvider
from free_claude_code.providers.runtime import build_provider_config
from tests.providers.support import immediate_admission


def _admission_from_settings(settings: Settings) -> ProviderAdmissionController:
    """Build the real admission controller through the production factory path."""
    config = build_provider_config(PROVIDER_CATALOG["nvidia_nim"], settings)
    return ProviderAdmissionController(
        provider_name="nvidia_nim",
        rate_limit=config.rate_limit,
        rate_window=config.rate_window,
        max_concurrency=config.max_concurrency,
    )


def test_settings_http_timeouts_reach_provider_config() -> None:
    """Distinct Settings timeouts arrive intact on the built ProviderConfig."""
    settings = Settings(
        nvidia_nim_api_key="test-key",
        http_connect_timeout=5.0,
        http_read_timeout=600.0,
        http_write_timeout=15.0,
    )

    config = build_provider_config(PROVIDER_CATALOG["nvidia_nim"], settings)

    assert config.http_connect_timeout == 5.0
    assert config.http_read_timeout == 600.0
    assert config.http_write_timeout == 15.0


def test_settings_http_timeouts_reach_provider_client() -> None:
    """Settings timeouts flow through ProviderConfig into the HTTP client."""
    settings = Settings(
        nvidia_nim_api_key="test-key",
        http_connect_timeout=5.0,
        http_read_timeout=600.0,
        http_write_timeout=15.0,
    )
    config = build_provider_config(PROVIDER_CATALOG["nvidia_nim"], settings)

    with patch(
        "free_claude_code.providers.openai_chat.client.AsyncOpenAI"
    ) as mock_openai:
        NvidiaNimProvider(
            config, nim_settings=NimSettings(), admission=immediate_admission()
        )
        timeout = mock_openai.call_args[1]["timeout"]

    assert timeout.connect == 5.0
    assert timeout.read == 600.0
    assert timeout.write == 15.0


@pytest.mark.asyncio
async def test_settings_max_concurrency_limits_in_flight_provider_calls() -> None:
    """Settings(provider_max_concurrency=2) caps concurrent provider calls at 2."""
    settings = Settings(
        nvidia_nim_api_key="test-key",
        provider_rate_limit=1_000_000,
        provider_rate_window=1,
        provider_max_concurrency=2,
    )
    controller = _admission_from_settings(settings)
    release = asyncio.Event()
    entered = asyncio.Event()
    in_flight = 0
    max_in_flight = 0
    entered_count = 0

    async def fake_provider_call() -> str:
        nonlocal in_flight, max_in_flight, entered_count
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        entered_count += 1
        if entered_count == 2:
            entered.set()
        try:
            await release.wait()
        finally:
            in_flight -= 1
        return "ok"

    tasks = [
        asyncio.create_task(
            controller.start_execution().run_call(
                fake_provider_call,
                operation_kind=ProviderOperationKind.GENERATION,
            )
        )
        for _ in range(4)
    ]
    try:
        await asyncio.wait_for(entered.wait(), timeout=2)
        await asyncio.sleep(0.05)
        assert max_in_flight == 2
        assert all(not task.done() for task in tasks)
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
        assert results == ["ok", "ok", "ok", "ok"]
        assert max_in_flight == 2
    finally:
        release.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_settings_rate_limit_holds_excess_calls_in_window() -> None:
    """Settings(provider_rate_limit=1) parks the second call for the window."""
    settings = Settings(
        nvidia_nim_api_key="test-key",
        provider_rate_limit=1,
        provider_rate_window=60,
        provider_max_concurrency=1_000,
    )
    controller = _admission_from_settings(settings)

    first = await controller.start_execution().open_attempt(
        ProviderOperationKind.GENERATION
    )
    try:
        await first.accept()
    finally:
        await first.aclose()

    second_task = asyncio.create_task(
        controller.start_execution().open_attempt(ProviderOperationKind.GENERATION)
    )
    try:
        await asyncio.sleep(0.1)
        assert not second_task.done()
    finally:
        second_task.cancel()
        await asyncio.gather(second_task, return_exceptions=True)


def test_settings_host_port_bind_configured_loopback() -> None:
    """Settings host/port values are the address actually bound and listening."""
    settings = Settings(host="127.0.0.1", port=0)
    assert (settings.host, settings.port) != ("0.0.0.0", 8082)

    with ServerSockets.reserve(settings.host, settings.port) as reserved:
        assert reserved.sockets
        bound_port = reserved.sockets[0].getsockname()[1]
        assert bound_port not in (0, 8082, 8083)
        with socket.create_connection((settings.host, bound_port), timeout=2) as client:
            assert client.getpeername()[1] == bound_port


def test_settings_explicit_ephemeral_port_is_honored() -> None:
    """A concrete configured port number is the port the listener takes."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", 0))
        free_port: int = probe.getsockname()[1]
    finally:
        probe.close()
    assert free_port not in (8082, 8083)

    settings = Settings(host="127.0.0.1", port=free_port)

    with ServerSockets.reserve(settings.host, settings.port) as reserved:
        assert reserved.sockets
        assert {listener.getsockname()[1] for listener in reserved.sockets} == {
            free_port
        }
