#!/usr/bin/env bash
# Detached FCC update guardian (Linux/macOS twin of update_guardian.ps1).
#
# Spawned by the Admin "Update" flow (free_claude_code.application.updater).
# It waits for the old server process to exit, asserts that no FCC process
# is running (like scripts/install.sh), downloads the configured GitHub
# archive, runs the pytest update gate, installs with `uv tool install
# --force`, and relaunches the original command. Progress is written to the
# guardian-owned JSON file consumed by the Admin UI.
set -u

if [ "$#" -lt 9 ]; then
    echo "usage: $0 <wait-pid> <archive-url> <target-version> <work-dir> <progress-file> <relaunch-b64> <python-spec> <test-exclude> <cwd> [test-timeout] [wait-timeout]" >&2
    exit 2
fi

WAIT_PID="$1"
ARCHIVE_URL="$2"
TARGET_VERSION="$3"
WORK_DIR="$4"
PROGRESS_FILE="$5"
RELAUNCH_B64="$6"
PYTHON_SPEC="$7"
TEST_EXCLUDE="$8"
CWD="$9"
TEST_TIMEOUT="${10:-3600}"
WAIT_PID_TIMEOUT="${11:-120}"

# Mirrors scripts/install.ps1 $FccCommands (retired entry points included).
FCC_COMMANDS=(
    fcc-desktop fcc-server fcc-claude fcc-codex fcc-pi fcc-opencode
    fcc-cline fcc-hermes fcc-dsh fcc-grok fcc-muse fcc-aider fcc-init
    free-claude-code
)

DONE_TS=""

epoch() { date +%s; }

write_state() { # stage message error
    local stage="$1" message="$2" error="$3"
    local json="{\"stage\":\"${stage}\",\"message\":\"${message}\",\"error\":\"${error}\",\"target_version\":\"${TARGET_VERSION}\",\"updated_ts\":$(epoch)}"
    if [ -n "$DONE_TS" ]; then
        json="${json%\}},\"done_ts\":${DONE_TS}}"
    fi
    mkdir -p "$(dirname "$PROGRESS_FILE")"
    printf '%s' "$json" > "$PROGRESS_FILE"
    echo "[guardian] ${stage}: ${message}${error}"
}

relaunch() {
    local argv exe line
    argv="$(printf '%s' "$RELAUNCH_B64" | base64 -d 2>/dev/null)" || argv=""
    if [ -z "$argv" ]; then return 0; fi
    exe=""
    local rest=()
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        if [ -z "$exe" ]; then
            exe="$line"
        else
            rest+=("$line")
        fi
    done << EOF
$argv
EOF
    [ -n "$exe" ] || return 0
    (
        cd "$CWD" 2>/dev/null || cd "$HOME" || exit 0
        # shellcheck disable=SC2086
        nohup "$exe" ${rest[@]+"${rest[@]}"} >/dev/null 2>&1 &
        disown 2>/dev/null || true
    )
    echo "[guardian] relaunched: $exe"
}

fail() {
    write_state "error" "" "$1"
    relaunch
    exit 1
}

write_state "scheduled" "Waiting for the FCC server to stop..." ""

wait_deadline=$(( $(epoch) + WAIT_PID_TIMEOUT ))
while kill -0 "$WAIT_PID" 2>/dev/null; do
    if [ "$(epoch)" -ge "$wait_deadline" ]; then
        write_state "error" "" "The previous FCC process (PID $WAIT_PID) did not exit within $WAIT_PID_TIMEOUT seconds. Nothing was installed."
        # The old server is still the live process: do not relaunch.
        exit 1
    fi
    sleep 0.5
done

for name in "${FCC_COMMANDS[@]}"; do
    for pid in $(pgrep -x "$name" 2>/dev/null) $(pgrep -f "(^|/)${name}([[:space:]]|$)" 2>/dev/null); do
        if [ "$pid" != "$$" ] && kill -0 "$pid" 2>/dev/null; then
            fail "FCC process $name (PID $pid) is still running. Stop it and retry the update."
        fi
    done
done

command -v uv >/dev/null 2>&1 || fail "uv was not found on PATH. Install uv or update with scripts/install.sh."

write_state "downloading" "Downloading ${ARCHIVE_URL}" ""
rm -rf "$WORK_DIR/source" "$WORK_DIR/source.zip"
mkdir -p "$WORK_DIR/source"
if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$ARCHIVE_URL" -o "$WORK_DIR/source.zip" || fail "Could not download the update archive."
elif command -v wget >/dev/null 2>&1; then
    wget -q "$ARCHIVE_URL" -O "$WORK_DIR/source.zip" || fail "Could not download the update archive."
else
    fail "Neither curl nor wget is available to download the update archive."
fi
if command -v unzip >/dev/null 2>&1; then
    unzip -q "$WORK_DIR/source.zip" -d "$WORK_DIR/source" || fail "Could not extract the update archive."
elif command -v python3 >/dev/null 2>&1; then
    python3 -m zipfile -e "$WORK_DIR/source.zip" "$WORK_DIR/source" || fail "Could not extract the update archive."
else
    fail "Install unzip (or python3) to extract the update archive."
fi
src_dir="$(find "$WORK_DIR/source" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
[ -n "$src_dir" ] || fail "The update archive did not contain a source directory."

write_state "testing" "Running the update test gate; this can take several minutes..." ""
# cd into the extracted tree first: pytest collects from the *working directory*,
# not from --project, so inheriting the server's CWD could test the wrong tree.
if command -v timeout >/dev/null 2>&1; then
    ( cd "$src_dir" && exec timeout "$TEST_TIMEOUT" \
        uv run --project "$src_dir" pytest -q --tb=short -k "$TEST_EXCLUDE" ) \
        >"$WORK_DIR/pytest.out" 2>"$WORK_DIR/pytest.err"
else
    ( cd "$src_dir" && exec uv run --project "$src_dir" pytest -q --tb=short -k "$TEST_EXCLUDE" ) \
        >"$WORK_DIR/pytest.out" 2>"$WORK_DIR/pytest.err"
fi
test_exit=$?
if [ "$test_exit" -ne 0 ]; then
    fail "The update gate is red (pytest exit $test_exit). Nothing was installed; see $WORK_DIR/pytest.out"
fi

write_state "installing" "Installing free-claude-code from the tested archive..." ""
uv tool install --force --refresh-package free-claude-code --python "$PYTHON_SPEC" "free-claude-code@${src_dir}" \
    >"$WORK_DIR/install.out" 2>"$WORK_DIR/install.err"
install_exit=$?
if [ "$install_exit" -ne 0 ]; then
    fail "uv tool install failed (exit $install_exit); the previous version is still installed. See $WORK_DIR/install.err"
fi

DONE_TS="$(epoch)"
write_state "done" "Installed version ${TARGET_VERSION}. Relaunching..." ""
relaunch
exit 0
