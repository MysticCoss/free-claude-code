"""Error forensics: 4xx/5xx responses emit correlated, redacted error logs."""

from contextlib import suppress
from unittest.mock import patch

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from free_claude_code.application.errors import InvalidRequestError
from free_claude_code.config.settings import Settings
from free_claude_code.core.diagnostics import attach_upstream_error_body
from tests.api.support import create_test_app


class _UpstreamLikeError(RuntimeError):
    status_code = 503


def _settings(**updates: object) -> Settings:
    return Settings().model_copy(update=updates)


@pytest.fixture(autouse=True)
def _redirect_fcc_home(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _logged_blob(log_error) -> str:
    """Render captured loguru-style calls (template + args) into plain text."""
    lines: list[str] = []
    for call in log_error.call_args_list:
        template, *rest = call.args
        values = [str(value) for value in rest]
        lines.append(str(template))
        lines.extend(values)
        with suppress(Exception):
            lines.append(str(template).format(*values))
    return "\n".join(lines)


def test_application_error_logs_one_line_without_message_text():
    app = create_test_app(_settings(log_api_error_tracebacks=False))
    secret = "application-secret-detail-marker"

    @app.get("/raise_application_forensics")
    async def _raise_application_forensics():
        raise InvalidRequestError(secret)

    with patch("free_claude_code.api.request_errors.logger.error") as log_error:
        response = TestClient(app).get("/raise_application_forensics")

    assert response.status_code == 400
    blob = _logged_blob(log_error)
    assert secret not in blob
    assert "status=400" in blob
    assert "InvalidRequestError" in blob
    assert "request_id=req_" in blob


def test_quiet_500_logs_one_line_without_body_or_traceback():
    app = create_test_app(_settings(log_api_error_tracebacks=False))

    @app.post("/raise_quiet")
    async def _raise_quiet():
        raise RuntimeError("quiet-boom-marker")

    with patch("free_claude_code.api.request_errors.logger.error") as log_error:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/raise_quiet", json={"note": "quiet-body-marker"}
        )

    assert response.status_code == 500
    blob = _logged_blob(log_error)
    assert "status=500" in blob
    assert "RuntimeError" in blob
    assert "quiet-boom-marker" not in blob
    assert "quiet-body-marker" not in blob
    assert "Traceback" not in blob


def test_verbose_500_includes_traceback_and_redacted_body():
    app = create_test_app(_settings(log_api_error_tracebacks=True))
    raw_secret = "sk-ant-verboseforensics12345678"

    @app.post("/raise_verbose")
    async def _raise_verbose(request: Request):
        # Real endpoints parse the body before failing; reading it here lets
        # the correlation middleware observe the bytes for forensics.
        await request.body()
        raise RuntimeError("verbose-boom-marker")

    with patch("free_claude_code.api.request_errors.logger.error") as log_error:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/raise_verbose",
            json={"note": "verbose-body-marker", "api_key": raw_secret},
        )

    assert response.status_code == 500
    blob = _logged_blob(log_error)
    assert "verbose-boom-marker" in blob
    assert "Traceback" in blob
    assert "verbose-body-marker" in blob
    assert raw_secret not in blob
    assert "<redacted>" in blob


def test_verbose_500_truncates_huge_body():
    app = create_test_app(_settings(log_api_error_tracebacks=True))

    @app.post("/raise_huge")
    async def _raise_huge(request: Request):
        await request.body()
        raise RuntimeError("huge-boom-marker")

    with patch("free_claude_code.api.request_errors.logger.error") as log_error:
        response = TestClient(app, raise_server_exceptions=False).post(
            "/raise_huge", json={"blob": "x" * 20_000}
        )

    assert response.status_code == 500
    assert "truncated after" in _logged_blob(log_error)


def test_verbose_500_includes_upstream_detail():
    app = create_test_app(_settings(log_api_error_tracebacks=True))

    @app.get("/raise_upstream")
    async def _raise_upstream():
        exc = _UpstreamLikeError("provider blew up")
        attach_upstream_error_body(exc, '{"error": "upstream-marker"}')
        raise exc

    with patch("free_claude_code.api.request_errors.logger.error") as log_error:
        response = TestClient(app, raise_server_exceptions=False).get("/raise_upstream")

    assert response.status_code == 500
    blob = _logged_blob(log_error)
    assert "upstream_status=503" in blob
    assert "upstream-marker" in blob


def test_http_error_logs_one_line_and_preserves_response_shape():
    app = create_test_app(_settings(log_api_error_tracebacks=False))

    with patch("free_claude_code.api.request_errors.logger.error") as log_error:
        response = TestClient(app).get("/no_such_route_xyz")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
    blob = _logged_blob(log_error)
    assert "status=404" in blob
    assert "Traceback" not in blob
