"""The fork updater checks its own branch before authorizing installation."""

from pathlib import Path

import pytest

from free_claude_code.application.updater import (
    UpdateCheckFailedError,
    UpdateDisabledError,
    UpdateInProgressError,
)
from free_claude_code.config.settings import Settings
from free_claude_code.updater import check as update_check


def _settings() -> Settings:
    return Settings(
        fcc_update_repo="MysticCoss/free-claude-code", fcc_update_branch="main"
    )


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch):
    def run(*, available: bool = False, error: Exception | None = None) -> int:
        monkeypatch.setattr(update_check, "get_settings", _settings)

        async def prepare(self, settings, launcher):
            assert settings.fcc_update_repo == "MysticCoss/free-claude-code"
            assert launcher == Path("staged-launcher")
            if error is not None:
                raise error
            snapshot = {
                "current_version": "6.16.3+f1.0",
                "latest_version": "6.16.3+f1.1" if available else "6.16.3+f1.0",
            }
            return snapshot, available

        monkeypatch.setattr(update_check.UpdateService, "prepare_cli_update", prepare)
        return update_check.main(["staged-launcher"])

    return run


def test_update_available_authorizes_the_staged_launcher(run, capsys) -> None:
    assert run(available=True) == update_check.UPDATE_AVAILABLE

    output = capsys.readouterr().out
    assert "6.16.3+f1.0 -> 6.16.3+f1.1" in output
    assert "MysticCoss/free-claude-code@main" in output


def test_up_to_date_skips_installation(run, capsys) -> None:
    assert run() == 0

    assert "up to date" in capsys.readouterr().out


@pytest.mark.parametrize(
    "error",
    [
        UpdateCheckFailedError("Could not fetch the update source."),
        UpdateDisabledError("source checkout"),
        UpdateInProgressError("An update is already running."),
        OSError("disk"),
    ],
)
def test_failures_never_authorize_installation(run, capsys, error) -> None:
    assert run(error=error) == 1

    assert "Could not check for FCC updates" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [[], ["a", "b"]])
def test_launcher_path_argument_is_required(capsys, argv) -> None:
    assert update_check.main(argv) == 1
    assert "usage" in capsys.readouterr().err
