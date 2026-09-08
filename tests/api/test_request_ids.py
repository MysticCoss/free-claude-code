"""Tests for the pure-ASGI ingress correlation owner."""

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
from fastapi import Request
from loguru import logger
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from free_claude_code.api.request_ids import (
    RequestCorrelationMiddleware,
    get_request_id,
)
from free_claude_code.api.request_lifetime import InferenceRequestLifetimeMiddleware
from free_claude_code.config.logging_config import configure_logging
from tests.api.support import create_test_app


def _http_scope(path: str) -> Scope:
    return cast(
        Scope,
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [(b"anthropic-session-id", b"session_test")],
            "client": None,
            "server": None,
        },
    )


def test_application_uses_the_pure_asgi_correlation_owner() -> None:
    app = create_test_app()
    middleware_classes = [
        cast(type[object], middleware.cls) for middleware in app.user_middleware
    ]

    assert sum(cls is RequestCorrelationMiddleware for cls in middleware_classes) == 1
    assert middleware_classes.index(
        RequestCorrelationMiddleware
    ) < middleware_classes.index(InferenceRequestLifetimeMiddleware)
    assert all(cls is not BaseHTTPMiddleware for cls in middleware_classes)


@pytest.mark.asyncio
async def test_correlation_context_and_headers_span_the_complete_stream() -> None:
    response_started = asyncio.Event()
    allow_body = asyncio.Event()
    sent: list[Message] = []
    context_entries: list[dict[str, object]] = []
    context_exits: list[dict[str, object]] = []
    app_request_id: str | None = None

    async def app(scope: Scope, _receive, send) -> None:
        nonlocal app_request_id
        app_request_id = get_request_id(Request(scope))
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        response_started.set()
        await allow_body.wait()
        await send(
            {
                "type": "http.response.body",
                "body": b"done",
                "more_body": False,
            }
        )

    @contextmanager
    def contextualize(**fields: object) -> Iterator[None]:
        context_entries.append(fields)
        try:
            yield
        finally:
            context_exits.append(fields)

    async def receive() -> Message:
        raise AssertionError("Test application does not receive messages")

    async def send(message: Message) -> None:
        sent.append(message)

    middleware = RequestCorrelationMiddleware(cast(ASGIApp, app))
    with patch(
        "free_claude_code.api.request_ids.logger.contextualize",
        side_effect=contextualize,
    ):
        request = asyncio.create_task(
            middleware(_http_scope("/v1/responses"), receive, send)
        )
        await response_started.wait()

        assert context_exits == []
        assert app_request_id is not None
        headers = Headers(raw=sent[0]["headers"])
        assert headers["request-id"] == app_request_id
        assert headers["x-request-id"] == app_request_id
        assert context_entries == [
            {
                "http_method": "POST",
                "http_path": "/v1/responses",
                "claude_session_id": "session_test",
                "request_id": app_request_id,
            }
        ]

        allow_body.set()
        await request

    assert context_exits == context_entries


async def _run_middleware(scope: Scope) -> None:
    async def app(_scope: Scope, _receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok", "more_body": False})

    async def receive() -> Message:
        raise AssertionError("Test application has no request body")

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    await RequestCorrelationMiddleware(cast(ASGIApp, app))(scope, receive, send)


@pytest.mark.asyncio
async def test_ingress_headers_trace_reports_matched_session() -> None:
    with patch(
        "free_claude_code.api.request_ids.trace_event",
    ) as trace_mock:
        await _run_middleware(_http_scope("/v1/messages"))

    assert trace_mock.call_count == 1
    kwargs = trace_mock.call_args.kwargs
    assert kwargs["stage"] == "ingress"
    assert kwargs["event"] == "ingress.headers"
    assert kwargs["source"] == "api"
    assert kwargs["method"] == "POST"
    assert kwargs["path"] == "/v1/messages"
    assert kwargs["session_header_matched"] is True
    assert kwargs["headers"] == {"anthropic-session-id": "session_test"}
    assert isinstance(kwargs["request_id"], str)


@pytest.mark.asyncio
async def test_ingress_headers_trace_flags_headerless_client() -> None:
    scope = _http_scope("/v1/messages")
    scope["headers"] = [(b"x-custom-client", b"desktop-probe")]
    with patch(
        "free_claude_code.api.request_ids.trace_event",
    ) as trace_mock:
        await _run_middleware(scope)

    kwargs = trace_mock.call_args.kwargs
    assert kwargs["session_header_matched"] is False
    assert kwargs["headers"] == {"x-custom-client": "desktop-probe"}


@pytest.mark.asyncio
async def test_ingress_headers_trace_redacts_credentials_in_sink(
    tmp_path: Path,
) -> None:
    log_file = str(tmp_path / "ingress.log")
    configure_logging(log_file, force=True, level="DEBUG")
    scope = _http_scope("/v1/messages")
    scope["headers"] = [
        (b"anthropic-session-id", b"session_test"),
        (b"authorization", b"Bearer SECRET"),
        (b"cookie", b"token=SECRET"),
        (b"x-custom-client", b"desktop-probe"),
    ]
    await _run_middleware(scope)

    logger.complete()
    rows = [
        json.loads(line)
        for line in Path(log_file).read_text(encoding="utf-8").strip().split("\n")
    ]
    row = next(row for row in rows if row.get("event") == "ingress.headers")
    assert row["headers"]["authorization"] == "<redacted>"
    assert row["headers"]["cookie"] == "<redacted>"
    assert row["headers"]["x-custom-client"] == "desktop-probe"
    assert row["session_header_matched"] is True


@pytest.mark.asyncio
async def test_correlation_middleware_passes_non_http_scopes_unchanged() -> None:
    observed_scope: Scope | None = None

    async def app(scope: Scope, _receive, _send) -> None:
        nonlocal observed_scope
        observed_scope = scope

    async def receive() -> Message:
        return {"type": "lifespan.startup"}

    async def send(_message: Message) -> None:
        return None

    scope = cast(Scope, {"type": "lifespan", "asgi": {"version": "3.0"}})
    await RequestCorrelationMiddleware(cast(ASGIApp, app))(scope, receive, send)

    assert observed_scope is scope
    assert "state" not in scope
