import pytest

from free_claude_code.core.updates import (
    fork_revision_tuple,
    is_newer_version,
    parse_pyproject_python,
    version_tuple,
)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("6.16.3+f1.0", (1, 0)),
        ("6.16.3+f12.4", (12, 4)),
        ("6.16.3+f2", (2,)),
        ("6.16.3", ()),
        ("6.16.3+cu130", ()),
        ("6.16.3+f1.x", ()),
        (" 6.16.3+f1.0 ", (1, 0)),
    ],
)
def test_fork_revision_tuple(version: str, expected: tuple[int, ...]) -> None:
    assert fork_revision_tuple(version) == expected


def test_version_tuple_ignores_fork_label() -> None:
    assert version_tuple("6.16.3+f1.0") == (6, 16, 3)


@pytest.mark.parametrize(
    ("candidate", "current", "expected"),
    [
        # Fork revision decides when the upstream part ties.
        ("6.16.3+f1.1", "6.16.3+f1.0", True),
        ("6.16.3+f2.0", "6.16.3+f1.9", True),
        ("6.16.3+f1.10", "6.16.3+f1.9", True),
        ("6.16.3+f1.0", "6.16.3+f1.1", False),
        ("6.16.3+f1.0", "6.16.3+f1.0", False),
        ("6.16.3+f1", "6.16.3+f1.0", False),
        # Upstream part decides first, whatever the fork revision.
        ("6.16.4+f1.0", "6.16.3+f9.9", True),
        ("6.16.3+f9.9", "6.16.4+f1.0", False),
        # Pre-convention fork installs carried plain semver.
        ("6.16.3+f1.0", "6.4.5", True),
        ("6.16.3+f1.0", "6.16.3", True),
        ("6.16.3", "6.16.3+f1.0", False),
        ("6.16.3", "6.16.3", False),
        ("garbage", "6.16.3+f1.0", False),
    ],
)
def test_is_newer_version_with_fork_revision(
    candidate: str, current: str, expected: bool
) -> None:
    assert is_newer_version(candidate, current) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('[project]\nrequires-python = "==3.14.7"\n', "3.14.7"),
        ('[project]\nrequires-python = "== 3.14.7"\n', "3.14.7"),
        ('[project]\nrequires-python = ">=3.14.0"\n', None),
        ('[project]\nrequires-python = "==3.14"\n', None),
        ("[project]\n", None),
        ("not toml [", None),
    ],
)
def test_parse_pyproject_python(text: str, expected: str | None) -> None:
    assert parse_pyproject_python(text) == expected
