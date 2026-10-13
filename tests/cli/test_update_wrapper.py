"""Native update commands run the staged fork launcher and survive replacement."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ("fcc-update", "fcc-update.cmd")

# Stands in for free_claude_code.updater.check: the wrapper contract is only
# "argv[1] is the launcher path; exit 0 / 1 / 10, and on 10 it is staged".
STUB_CHECK = """
import os
import sys
from pathlib import Path


def main():
    outcome = os.environ["FCC_TEST_OUTCOME"]
    print("checker ran", file=sys.stderr)
    if outcome == "current":
        return 0
    if outcome == "check-error":
        return 1
    launcher = Path(sys.argv[1])
    code = {"success": 0, "exit": 23}[outcome]
    launcher_target = os.environ["FCC_TEST_LAUNCHER"]
    checker_target = os.environ["FCC_TEST_CHECKER"]
    if os.name == "nt":
        launcher.write_text(
            "@echo off\\r\\n"
            "echo launcher ran\\r\\n"
            f'> "{launcher_target}" echo exit /b 99\\r\\n'
            f'> "{checker_target}" echo replaced after checker exited\\r\\n'
            f"exit /b {code}\\r\\n",
            encoding="utf-8",
            newline="",
        )
    else:
        launcher.write_text(
            "#!/bin/sh\\n"
            "echo launcher ran\\n"
            f"printf 'exit 99\\\\n' > '{launcher_target}'\\n"
            f"printf 'replaced' > '{checker_target}'\\n"
            f"exit {code}\\n",
            encoding="utf-8",
            newline="",
        )
    return 10
"""


def fixture_wheel(area: Path) -> Path:
    wheel = area / "free_claude_code-1.0.0-py3-none-any.whl"
    dist = "free_claude_code-1.0.0"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"{dist}.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: free-claude-code\nVersion: 1.0.0\n",
        )
        archive.writestr(
            f"{dist}.dist-info/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        for name in SCRIPTS:
            source = (ROOT / "scripts/update" / name).read_text(encoding="utf-8")
            archive.writestr(f"{dist}.data/scripts/{name}", source)
        archive.writestr(
            f"{dist}.dist-info/entry_points.txt",
            "[console_scripts]\n_fcc-update-check = free_claude_code.updater.check:main\n",
        )
        archive.writestr("free_claude_code/__init__.py", "")
        archive.writestr("free_claude_code/updater/__init__.py", "")
        archive.writestr("free_claude_code/updater/check.py", STUB_CHECK)
        archive.writestr(
            f"{dist}.dist-info/RECORD",
            "".join(f"{name},,\n" for name in archive.namelist())
            + f"{dist}.dist-info/RECORD,,\n",
        )
    return wheel


@pytest.mark.parametrize(
    "outcome", ["success", "exit", "current", "check-error", "options"]
)
def test_installed_update_runs_launcher_and_survives_replacement(
    tmp_path: Path, outcome: str
) -> None:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is required for native script lifecycle coverage")
    area = tmp_path / "José 测试 (spaces) apostrophe'"
    area.mkdir()
    env = dict(os.environ)
    env.update(
        UV_TOOL_DIR=str(area / "tools"),
        UV_TOOL_BIN_DIR=str(area / "bin"),
        UV_NO_CONFIG="1",
        UV_PYTHON_DOWNLOADS="never",
        FCC_TEST_OUTCOME=outcome,
        FCC_TEST_LAUNCHER=str(
            area / "bin" / ("fcc-update.cmd" if os.name == "nt" else "fcc-update")
        ),
        FCC_TEST_CHECKER=str(
            area
            / "bin"
            / ("_fcc-update-check.exe" if os.name == "nt" else "_fcc-update-check")
        ),
    )
    result = subprocess.run(
        [
            uv,
            "tool",
            "install",
            "--python",
            getattr(sys, "_base_executable", sys.executable),
            str(fixture_wheel(area)),
        ],
        env=env,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    launcher = env["FCC_TEST_LAUNCHER"]
    arguments = "--voice-local" if outcome == "options" else ""
    command = (
        f'"{os.environ.get("COMSPEC", "cmd.exe")}" /d /s /c ""{launcher}" {arguments}"'
        if os.name == "nt"
        else [launcher, *([arguments] if arguments else [])]
    )
    result = subprocess.run(
        command,
        env=env,
        cwd=area,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout + result.stderr
    expected_code = {
        "success": 0,
        "exit": 23,
        "current": 0,
        "check-error": 1,
        "options": 2,
    }[outcome]
    assert result.returncode == expected_code, output
    ran = "launcher ran" in result.stdout
    assert ran is (outcome in {"success", "exit"}), output
    assert ("checker ran" in result.stderr) is (outcome != "options"), output
    if outcome == "options":
        assert "takes no options" in result.stderr
    # The wrapper removes the staged launcher on every path.
    temp = Path(env.get("TMPDIR") or env.get("TEMP") or "/tmp")
    assert not list(temp.glob("fcc-update-*"))
    result = subprocess.run(
        [uv, "tool", "uninstall", "free-claude-code"],
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
