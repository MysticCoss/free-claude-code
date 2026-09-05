import pytest

from free_claude_code.core.updates import (
    build_archive_url,
    is_newer_version,
    parse_pyproject_version,
    raw_pyproject_url,
    update_capable,
    validate_update_branch,
    validate_update_repo,
    version_tuple,
)


def test_update_capable_rejects_source_checkout_version() -> None:
    assert update_capable("5.22.4") is True
    assert update_capable("0+unknown") is False


def test_version_tuple_ignores_local_labels_and_prerelease_noise() -> None:
    assert version_tuple("5.22.4") == (5, 22, 4)
    assert version_tuple("5.22.4+local") == (5, 22, 4)
    assert version_tuple(" 1.2 ") == (1, 2)
    assert version_tuple("0+unknown") == (0,)


def test_version_tuple_returns_none_for_non_numeric_text() -> None:
    assert version_tuple("main") is None
    assert version_tuple("") is None
    assert version_tuple("+meta") is None


def test_is_newer_version_compares_numerically_not_lexically() -> None:
    assert is_newer_version("5.23.0", "5.22.4") is True
    assert is_newer_version("1.10.0", "1.9.0") is True
    assert is_newer_version("1.9.0", "1.10.0") is False
    assert is_newer_version("5.22.4", "5.22.4") is False
    assert is_newer_version("5.22.4", "5.22.4+build") is False
    assert is_newer_version("5.23", "5.22.4") is True
    assert is_newer_version("v5.23", "5.22") is False


def test_is_newer_version_is_false_when_either_side_is_unparsable() -> None:
    assert is_newer_version("garbage", "5.22.4") is False
    assert is_newer_version("5.22.4", "garbage") is False


def test_parse_pyproject_version_extracts_project_version() -> None:
    assert parse_pyproject_version('[project]\nversion = "5.23.0"\n') == "5.23.0"
    assert parse_pyproject_version("[tool.uv]\nx = 1\n") is None
    assert parse_pyproject_version("not toml [[[") is None
    assert parse_pyproject_version("[project]\nversion = 5\n") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("MysticCoss/free-claude-code", "MysticCoss/free-claude-code"),
        ("a/b", "a/b"),
        ("Some.Owner/some-repo_name.v2", "Some.Owner/some-repo_name.v2"),
        (" owner/repo ", "owner/repo"),
    ],
)
def test_validate_update_repo_accepts_and_normalizes(value: str, expected: str) -> None:
    assert validate_update_repo(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "repo",
        "owner/repo/extra",
        "owner/",
        "/repo",
        "own er/repo",
        ".owner/repo",
        "owner/.repo",
        "owner/../repo",
        "owner/repo.",
        "owner/-repo",
    ],
)
def test_validate_update_repo_rejects_malformed_slugs(value: str) -> None:
    with pytest.raises(ValueError, match="FCC_UPDATE_REPO"):
        validate_update_repo(value)


@pytest.mark.parametrize("value", ["main", "feature/x", "dev-1.2", "a/b/c", " main "])
def test_validate_update_branch_accepts_and_normalizes(value: str) -> None:
    assert validate_update_branch(value) == value.strip()


@pytest.mark.parametrize(
    "value",
    ["", "..", "a..b", "no spaces", "refs/heads/main?", "bad*branch", "a//b"],
)
def test_validate_update_branch_rejects_malformed_names(value: str) -> None:
    with pytest.raises(ValueError, match="FCC_UPDATE_BRANCH"):
        validate_update_branch(value)


def test_urls_point_at_github_for_repo_and_branch() -> None:
    assert build_archive_url("owner/repo", "main") == (
        "https://github.com/owner/repo/archive/refs/heads/main.zip"
    )
    assert raw_pyproject_url("owner/repo", "dev/x") == (
        "https://raw.githubusercontent.com/owner/repo/dev/x/pyproject.toml"
    )
