"""Observability tests proving Admin UI settings take effect at runtime.

Each test starts from a valid non-default ``Settings`` value and asserts the
wired runtime effect (not just schema acceptance). Default-only behavior is
covered elsewhere; these tests pin the opt-in paths.
"""

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from free_claude_code.application import updater as updater_module
from free_claude_code.application.updater import UpdateService
from free_claude_code.config.settings import Settings
from free_claude_code.core.anthropic.streaming.emitter import AnthropicSseEmitter
from free_claude_code.messaging.event_parser import parse_cli_event
from free_claude_code.messaging.node_runner import MessagingNodeRunner
from free_claude_code.messaging.platforms.discord_inbound import (
    discord_text_message_from_event,
)
from free_claude_code.messaging.transcript import RenderCtx, TranscriptBuffer
from free_claude_code.messaging.ui_updates import ThrottledTranscriptEditor
from free_claude_code.runtime.application import (
    AUTO_UPDATE_MIN_INTERVAL_SECONDS,
    ApplicationRuntime,
)
from free_claude_code.runtime.bootstrap import build_asgi_app
from free_claude_code.runtime.configuration import ConfigurationService
from free_claude_code.runtime.provider_manager import ProviderRuntimeManager


@pytest.fixture(autouse=True)
def _redirect_fcc_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _identity(text: str) -> str:
    return text


def _render_ctx() -> RenderCtx:
    return RenderCtx(
        bold=_identity,
        code_inline=_identity,
        escape_code=_identity,
        escape_text=_identity,
        render_markdown=_identity,
    )


def _make_runner(**flags: bool) -> MessagingNodeRunner:
    return MessagingNodeRunner(
        platform_name="test",
        outbound=MagicMock(),
        cli_manager=MagicMock(),
        session_store=MagicMock(),
        get_tree_queue=MagicMock(),
        format_status=lambda emoji, title, _detail: f"{emoji} {title}",
        get_parse_mode=lambda: None,
        get_render_ctx=_render_ctx,
        get_limit_chars=lambda: 4000,
        **flags,
    )


def _caplog_blob(caplog: pytest.LogCaptureFixture) -> str:
    return " | ".join(record.getMessage() for record in caplog.records)


def test_log_level_debug_reaches_configure_logging(tmp_path: Path) -> None:
    """A non-default LOG_LEVEL must reach configure_logging from Settings."""
    settings = Settings(log_level="DEBUG")
    assert settings.log_level == "DEBUG"

    log_path = tmp_path / "server.log"
    with (
        patch(
            "free_claude_code.runtime.bootstrap.server_log_path",
            return_value=log_path,
        ),
        patch("free_claude_code.runtime.bootstrap.configure_logging") as configure,
    ):
        build_asgi_app(settings)

    configure.assert_called_once_with(
        Path(log_path),
        level="DEBUG",
        verbose_third_party=settings.log_raw_api_payloads,
    )


def test_log_raw_cli_diagnostics_wires_from_settings_into_parser() -> None:
    """Settings.log_raw_cli_diagnostics=True must reach the CLI event parser."""
    settings = Settings().model_copy(update={"log_raw_cli_diagnostics": True})
    assert settings.log_raw_cli_diagnostics is True
    runner = _make_runner(log_raw_cli_diagnostics=settings.log_raw_cli_diagnostics)
    assert runner._log_raw_cli_diagnostics is True

    secret = "wired-cli-secret-abc123"
    with patch("free_claude_code.messaging.event_parser.logger.info") as log_info:
        parse_cli_event(
            {"type": "error", "error": {"message": secret}},
            log_raw_cli=runner._log_raw_cli_diagnostics,
        )
    flat = " ".join(str(call) for call in log_info.call_args_list)
    assert secret in flat


def test_log_raw_messaging_content_true_logs_text_preview(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Settings.log_raw_messaging_content=True must log raw message text."""
    settings = Settings().model_copy(update={"log_raw_messaging_content": True})
    assert settings.log_raw_messaging_content is True
    secret = "raw-discord-text-secret-456"

    def make_message(content: str) -> object:
        return SimpleNamespace(
            channel=SimpleNamespace(id="chan_9"),
            id="msg_7",
            reference=None,
            content=content,
            author=SimpleNamespace(id="user_3", display_name="tester"),
        )

    with caplog.at_level(logging.INFO):
        discord_text_message_from_event(
            make_message(secret),
            log_raw_messaging_content=settings.log_raw_messaging_content,
        )
    blob = _caplog_blob(caplog)
    assert secret in blob
    assert "text_preview" in blob

    caplog.clear()
    with caplog.at_level(logging.INFO):
        discord_text_message_from_event(
            make_message(secret),
            log_raw_messaging_content=False,
        )
    blob = _caplog_blob(caplog)
    assert secret not in blob
    assert "text_len" in blob


def test_log_raw_sse_events_true_logs_event_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Settings.log_raw_sse_events=True must log raw SSE event bodies."""
    settings = Settings().model_copy(update={"log_raw_sse_events": True})
    assert settings.log_raw_sse_events is True

    emitter = AnthropicSseEmitter(log_raw_events=settings.log_raw_sse_events)
    secret = "sse-secret-body-789"
    with caplog.at_level(logging.DEBUG):
        frame = emitter.event("message_start", {"secret_body": secret})
    assert secret in frame
    sse_records = [
        record for record in caplog.records if "SSE_EVENT" in record.getMessage()
    ]
    assert sse_records
    assert secret in " | ".join(record.getMessage() for record in sse_records)

    caplog.clear()
    quiet = AnthropicSseEmitter(log_raw_events=False)
    with caplog.at_level(logging.DEBUG):
        quiet.event("message_start", {"secret_body": secret})
    assert not [
        record for record in caplog.records if "SSE_EVENT" in record.getMessage()
    ]


@pytest.mark.asyncio
async def test_debug_platform_edits_true_logs_edit_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Settings.debug_platform_edits=True must log full platform edit text."""
    settings = Settings().model_copy(update={"debug_platform_edits": True})
    assert settings.debug_platform_edits is True
    runner = _make_runner(debug_platform_edits=settings.debug_platform_edits)
    assert runner._debug_platform_edits is True

    body = "visible-edit-body-123"

    async def run_update(*, debug: bool) -> None:
        transcript = TranscriptBuffer()
        transcript.apply({"type": "error", "message": body})
        outbound = MagicMock()
        outbound.queue_edit_message = AsyncMock()
        editor = ThrottledTranscriptEditor(
            outbound=outbound,
            parse_mode=None,
            get_limit_chars=lambda: 4000,
            transcript=transcript,
            render_ctx=_render_ctx(),
            node_id="n1",
            chat_id="c1",
            status_msg_id="m1",
            debug_platform_edits=debug,
        )
        await editor.update("status-line", force=True)

    with caplog.at_level(logging.DEBUG):
        await run_update(debug=True)
    blob = _caplog_blob(caplog)
    assert "PLATFORM_EDIT_TEXT" in blob
    assert body in blob

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        await run_update(debug=False)
    blob = _caplog_blob(caplog)
    assert "PLATFORM_EDIT_TEXT" not in blob
    assert "PLATFORM_EDIT" in blob


def test_debug_subagent_stack_true_logs_stack_events() -> None:
    """Settings.debug_subagent_stack=True must log subagent push/pop events."""
    settings = Settings().model_copy(update={"debug_subagent_stack": True})
    assert settings.debug_subagent_stack is True

    noisy, _ = _make_runner(
        debug_subagent_stack=settings.debug_subagent_stack
    )._create_transcript_and_render_ctx()
    with patch(
        "free_claude_code.messaging.transcript.subagents.logger.debug"
    ) as mock_debug:
        noisy.apply(
            {
                "type": "tool_use",
                "id": "task_42",
                "name": "Task",
                "input": {"description": "Deep research"},
            }
        )
    flat = " ".join(str(call) for call in mock_debug.call_args_list)
    assert "SUBAGENT_STACK" in flat
    assert "task_42" in flat

    quiet, _ = _make_runner(
        debug_subagent_stack=False
    )._create_transcript_and_render_ctx()
    with patch(
        "free_claude_code.messaging.transcript.subagents.logger.debug"
    ) as mock_debug:
        quiet.apply(
            {
                "type": "tool_use",
                "id": "task_42",
                "name": "Task",
                "input": {"description": "Deep research"},
            }
        )
    assert mock_debug.call_count == 0


def _auto_runtime(settings: Settings) -> ApplicationRuntime:
    manager = ProviderRuntimeManager(settings)
    return ApplicationRuntime(
        manager,
        configuration=AsyncMock(spec=ConfigurationService),
        transcriber=None,
    )


@pytest.mark.asyncio
async def test_fcc_update_poll_hours_derives_sleep_interval() -> None:
    """The auto-update loop must sleep poll_hours converted to seconds."""
    settings = Settings().model_copy(update={"fcc_update_poll_hours": 2.5})
    runtime = _auto_runtime(settings)
    with (
        patch(
            "free_claude_code.runtime.application.asyncio.sleep",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ) as mock_sleep,
        pytest.raises(asyncio.CancelledError),
    ):
        await runtime._run_update_auto_loop()
    mock_sleep.assert_awaited_once_with(2.5 * 3600.0)


@pytest.mark.asyncio
async def test_fcc_update_poll_hours_floor_guards_tiny_intervals() -> None:
    """Tiny poll intervals must clamp to the minimum update interval."""
    settings = Settings().model_copy(update={"fcc_update_poll_hours": 0.001})
    runtime = _auto_runtime(settings)
    with (
        patch(
            "free_claude_code.runtime.application.asyncio.sleep",
            new=AsyncMock(side_effect=asyncio.CancelledError()),
        ) as mock_sleep,
        pytest.raises(asyncio.CancelledError),
    ):
        await runtime._run_update_auto_loop()
    mock_sleep.assert_awaited_once_with(AUTO_UPDATE_MIN_INTERVAL_SECONDS)


@pytest.mark.asyncio
async def test_fcc_update_custom_repo_branch_changes_fetch_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom repo/branch Settings must change the version-check fetch URL."""
    monkeypatch.setattr(updater_module, "package_version", lambda: "5.22.4")
    calls: list[str] = []

    async def fake_fetch(url: str) -> str:
        calls.append(url)
        return '[project]\nversion = "5.23.0"\n'

    monkeypatch.setattr(updater_module, "_fetch_text", fake_fetch)
    service = UpdateService(
        state_path=tmp_path / "state.json",
        progress_path=tmp_path / "progress.json",
        work_dir=tmp_path / "work",
    )
    settings = Settings(
        fcc_update_repo="octo/custom-fork",
        fcc_update_branch="release-x",
        fcc_update_auto=False,
        fcc_update_poll_hours=6.0,
    )

    snapshot = await service.check(settings, force=True)

    assert calls == [
        "https://raw.githubusercontent.com/octo/custom-fork/release-x/pyproject.toml"
    ]
    assert snapshot["repo"] == "octo/custom-fork"
    assert snapshot["branch"] == "release-x"
    assert snapshot["latest_version"] == "5.23.0"
    assert snapshot["update_available"] is True
