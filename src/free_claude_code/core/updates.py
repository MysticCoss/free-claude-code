"""Pure helpers for the in-app update flow (version compare + GitHub URLs).

Kept SDK-free and I/O-free per architecture rules: providers alone talk to
the network; this module only computes values and validates config strings.
"""

import re
import tomllib

from .version import _UNKNOWN_VERSION

# GitHub repository slugs allow alphanumerics plus . _ - and may not start
# or end with a separator, so reject ".." traversal implicitly.
_REPO_SEGMENT_PATTERN = re.compile(r"^(?![-.])[A-Za-z0-9._-]+(?<![-.])$")
# Git branch names allow alphanumerics plus . _ - / and may not contain "..".
_BRANCH_SEGMENT_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
# Fork revision local label: ``+f1.0`` in ``6.16.3+f1.0``.
_FORK_LABEL_PATTERN = re.compile(r"f(\d+(?:\.\d+)*)")


def update_capable(version: str) -> bool:
    """Return whether in-app updates can run for an installed version.

    Source checkouts report ``0+unknown``; installing over them in-place
    would fight the editable tree, so in-app updates are disabled there.
    """

    return version != _UNKNOWN_VERSION


def version_tuple(version: str) -> tuple[int, ...] | None:
    """Return the numeric release segments of a version, or None if invalid.

    Local labels (``+cu130``, ``+f1.0``) and prerelease suffixes are ignored;
    a plain ``5.22.4`` and a ``5.22.4`` with build metadata compare equal.
    """

    stripped = version.strip().split("+", 1)[0]
    match = re.match(r"^(\d+(?:\.\d+)*)", stripped)
    if match is None:
        return None
    return tuple(int(part) for part in match.group(1).split("."))


def fork_revision_tuple(version: str) -> tuple[int, ...]:
    """Return the fork revision from a ``<upstream>+f<major>.<minor>`` version.

    Versions without a fork label (plain upstream releases, other local
    labels) return ``()``, which sorts below every fork revision.
    """

    _, separator, local = version.strip().partition("+")
    if not separator:
        return ()
    match = _FORK_LABEL_PATTERN.fullmatch(local)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _padded(parts: tuple[int, ...], width: int) -> tuple[int, ...]:
    return parts + (0,) * (width - len(parts))


def is_newer_version(candidate: str, current: str) -> bool:
    """Return True when ``candidate`` is a strictly newer release of ``current``.

    The upstream release segments decide first; on a tie the fork revision
    (``+f<major>.<minor>``) decides.
    """

    candidate_parts = version_tuple(candidate)
    current_parts = version_tuple(current)
    if candidate_parts is None or current_parts is None:
        return False
    width = max(len(candidate_parts), len(current_parts))
    padded_candidate = _padded(candidate_parts, width)
    padded_current = _padded(current_parts, width)
    if padded_candidate != padded_current:
        return padded_candidate > padded_current
    candidate_fork = fork_revision_tuple(candidate)
    current_fork = fork_revision_tuple(current)
    if not candidate_fork or not current_fork:
        return bool(candidate_fork) and not current_fork
    fork_width = max(len(candidate_fork), len(current_fork))
    return _padded(candidate_fork, fork_width) > _padded(current_fork, fork_width)


def parse_pyproject_version(text: str) -> str | None:
    """Return ``[project].version`` from pyproject.toml text, if parseable."""

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return None
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    version = project.get("version")
    return version if isinstance(version, str) else None


def validate_update_repo(value: str) -> str:
    """Return a normalized ``owner/repo`` value or raise ValueError."""

    owner, separator, repository = value.strip().partition("/")
    if (
        not separator
        or not _REPO_SEGMENT_PATTERN.fullmatch(owner)
        or not _REPO_SEGMENT_PATTERN.fullmatch(repository)
    ):
        raise ValueError(
            "FCC_UPDATE_REPO must be a GitHub 'owner/repo' slug "
            f"(letters, digits, '.', '_', '-'), got {value!r}"
        )
    return f"{owner}/{repository}"


def validate_update_branch(value: str) -> str:
    """Return a normalized branch name or raise ValueError."""

    branch = value.strip()
    segments = branch.split("/")
    if (
        not branch
        or ".." in branch
        or not all(_BRANCH_SEGMENT_PATTERN.fullmatch(segment) for segment in segments)
    ):
        raise ValueError(
            "FCC_UPDATE_BRANCH must be a git branch name made of "
            f"letters, digits, '.', '_', '-' or '/', got {value!r}"
        )
    return branch


def build_archive_url(repo: str, branch: str) -> str:
    """Return the GitHub codeload-style zip URL for one branch."""

    return f"https://github.com/{repo}/archive/refs/heads/{branch}.zip"


def raw_pyproject_url(repo: str, branch: str) -> str:
    """Return the raw.githubusercontent URL for one branch's pyproject.toml."""

    return f"https://raw.githubusercontent.com/{repo}/{branch}/pyproject.toml"
