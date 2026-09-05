from pathlib import Path

import pytest
from pydantic import ValidationError

from free_claude_code.config.admin.manifest import FIELD_BY_KEY, SECTIONS
from free_claude_code.config.loader import compose_settings_snapshot
from free_claude_code.config.settings import Settings

ENV_KEYS = (
    "FCC_UPDATE_REPO",
    "FCC_UPDATE_BRANCH",
    "FCC_UPDATE_AUTO",
    "FCC_UPDATE_POLL_HOURS",
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_update_settings_defaults() -> None:
    settings = Settings()

    assert settings.fcc_update_repo == "MysticCoss/free-claude-code"
    assert settings.fcc_update_branch == "main"
    assert settings.fcc_update_auto is False
    assert settings.fcc_update_poll_hours == 6.0


def test_update_settings_compose_from_process_env() -> None:
    snapshot = compose_settings_snapshot(
        {},
        {
            "FCC_UPDATE_REPO": "some/other",
            "FCC_UPDATE_BRANCH": "dev/x",
            "FCC_UPDATE_AUTO": "true",
            "FCC_UPDATE_POLL_HOURS": "12.5",
        },
    )

    assert snapshot.settings.fcc_update_repo == "some/other"
    assert snapshot.settings.fcc_update_branch == "dev/x"
    assert snapshot.settings.fcc_update_auto is True
    assert snapshot.settings.fcc_update_poll_hours == 12.5


@pytest.mark.parametrize(
    ("attr", "value"),
    [
        ("fcc_update_repo", "not-a-slug"),
        ("fcc_update_repo", "owner//repo"),
        ("fcc_update_branch", "bad..branch"),
        ("fcc_update_branch", "has space"),
        ("fcc_update_branch", ""),
        ("fcc_update_poll_hours", 0),
        ("fcc_update_poll_hours", -1.0),
    ],
)
def test_direct_settings_reject_invalid_update_values(attr: str, value: object) -> None:
    with pytest.raises(ValidationError, match=attr):
        Settings.model_validate({attr: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [("FCC_UPDATE_REPO", "not-a-slug"), ("FCC_UPDATE_BRANCH", "bad..branch")],
)
def test_loader_rejects_invalid_update_values(key: str, value: str) -> None:
    with pytest.raises(ValidationError):
        compose_settings_snapshot({}, {key: value})


def test_update_manifest_exposes_four_fields_in_updates_section() -> None:
    section_ids = [section.section_id for section in SECTIONS]
    assert "updates" in section_ids

    repo = FIELD_BY_KEY["FCC_UPDATE_REPO"]
    assert repo.settings_attr == "fcc_update_repo"
    assert repo.section_id == "updates"
    assert repo.restart_required is True
    branch = FIELD_BY_KEY["FCC_UPDATE_BRANCH"]
    assert branch.settings_attr == "fcc_update_branch"
    auto = FIELD_BY_KEY["FCC_UPDATE_AUTO"]
    assert auto.field_type == "boolean"
    poll = FIELD_BY_KEY["FCC_UPDATE_POLL_HOURS"]
    assert poll.field_type == "number"
    assert poll.advanced is True


def test_env_example_documents_update_keys() -> None:
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")

    for key in ENV_KEYS:
        assert f"{key}=" in text
