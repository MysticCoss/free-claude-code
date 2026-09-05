#Requires -Version 5.1
<#
    Detached FCC update guardian (Windows).

    Spawned by the Admin "Update" flow (free_claude_code.application.updater).
    It waits for the old server process to exit, asserts that no FCC process
    is running (like scripts/install.ps1 Assert-NoFccProcessesRunning),
    downloads the configured GitHub archive, runs the pytest update gate,
    installs with `uv tool install --force`, and relaunches the original
    command. Progress is written to -ProgressFile as JSON consumed by the
    Admin UI (the server never writes that file; only the guardian does).
#>
param(
    [Parameter(Mandatory = $true)][int]$WaitPid,
    [Parameter(Mandatory = $true)][string]$ArchiveUrl,
    [Parameter(Mandatory = $true)][string]$TargetVersion,
    [Parameter(Mandatory = $true)][string]$WorkDir,
    [Parameter(Mandatory = $true)][string]$ProgressFile,
    [Parameter(Mandatory = $true)][string]$RelaunchB64,
    [Parameter(Mandatory = $true)][string]$PythonSpec,
    [Parameter(Mandatory = $true)][string]$TestExclude,
    [Parameter(Mandatory = $true)][string]$Cwd,
    [int]$TestTimeoutSeconds = 3600,
    [int]$WaitPidTimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"

# Mirrors scripts/install.ps1 $FccCommands (retired entry points included).
$FccCommands = @(
    "fcc-desktop", "fcc-server", "fcc-claude", "fcc-codex", "fcc-pi",
    "fcc-opencode", "fcc-cline", "fcc-hermes", "fcc-dsh", "fcc-grok",
    "fcc-muse", "fcc-aider", "fcc-init", "free-claude-code"
)

$script:ProgressState = @{ target_version = $TargetVersion }

function Update-ProgressFile {
    param(
        [string]$Stage,
        [string]$Message = "",
        [string]$Failure = "",
        [switch]$Done
    )
    $script:ProgressState["stage"] = $Stage
    $script:ProgressState["message"] = $Message
    if ($Failure) { $script:ProgressState["error"] = $Failure }
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $script:ProgressState["updated_ts"] = $now
    if ($Done) { $script:ProgressState["done_ts"] = $now }
    $json = ConvertTo-Json -InputObject $script:ProgressState -Compress
    [System.IO.File]::WriteAllText(
        $ProgressFile,
        $json,
        (New-Object System.Text.UTF8Encoding($false))
    )
    Write-Output ("[guardian] {0}: {1} {2}" -f $Stage, $Message, $Failure)
}

function Start-Relaunch {
    try {
        $text = [System.Text.Encoding]::UTF8.GetString(
            [System.Convert]::FromBase64String($RelaunchB64)
        )
        $argv = @($text -split "`n")
        $exe = $argv[0]
        if (-not $exe) { return }
        $rest = @()
        if ($argv.Count -gt 1) { $rest = @($argv[1..($argv.Count - 1)]) }
        if ($rest.Count -gt 0) {
            Start-Process -FilePath $exe -ArgumentList $rest -WorkingDirectory $Cwd |
                Out-Null
        } else {
            Start-Process -FilePath $exe -WorkingDirectory $Cwd | Out-Null
        }
        Write-Output "[guardian] relaunched: $exe"
    } catch {
        Write-Output "[guardian] relaunch failed: $($_.Exception.Message)"
    }
}

function Invoke-LoggedProcess {
    param(
        [string]$FilePath,
        [string[]]$ProcessArguments,
        [string]$OutFile,
        [string]$ErrFile,
        [int]$TimeoutSeconds = 0,
        [string]$WorkingDirectory = ""
    )
    $startParams = @{
        FilePath             = $FilePath
        ArgumentList         = $ProcessArguments
        WorkingDirectory     = $(if ($WorkingDirectory) { $WorkingDirectory } else { (Get-Location).Path })
        NoNewWindow          = $true
        PassThru             = $true
        RedirectStandardOutput = $OutFile
        RedirectStandardError  = $ErrFile
    }
    $proc = Start-Process @startParams
    $null = $proc.Handle
    if ($TimeoutSeconds -gt 0) {
        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        while (-not $proc.WaitForExit(1000)) {
            if ((Get-Date) -gt $deadline) {
                try { $proc.Kill() } catch { }
                return -99
            }
        }
    } else {
        $proc.WaitForExit()
    }
    return $proc.ExitCode
}

try {
    Update-ProgressFile -Stage "scheduled" -Message "Waiting for the FCC server to stop..."

    $waitDeadline = (Get-Date).AddSeconds($WaitPidTimeoutSeconds)
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        if ((Get-Date) -gt $waitDeadline) {
            Update-ProgressFile -Stage "error" -Failure (
                "The previous FCC process (PID $WaitPid) did not exit within " +
                "$WaitPidTimeoutSeconds seconds. Nothing was installed."
            )
            # The old server is still the live process: do not relaunch.
            exit 1
        }
        Start-Sleep -Milliseconds 500
    }

    $others = @()
    foreach ($name in $FccCommands) {
        $others += @(Get-Process -Name $name -ErrorAction SilentlyContinue)
    }
    if ($others.Count -gt 0) {
        $list = ($others | ForEach-Object { "$($_.ProcessName) (PID $($_.Id))" }) -join ", "
        Update-ProgressFile -Stage "error" -Failure (
            "FCC processes are still running ($list). Stop them and retry the update."
        )
        Start-Relaunch
        exit 1
    }

    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCommand) {
        Update-ProgressFile -Stage "error" -Failure (
            "uv was not found on PATH. Install uv or update with scripts/install.ps1."
        )
        Start-Relaunch
        exit 1
    }
    $uv = $uvCommand.Source

    Update-ProgressFile -Stage "downloading" -Message "Downloading $ArchiveUrl"
    $zipPath = Join-Path $WorkDir "source.zip"
    $extractRoot = Join-Path $WorkDir "source"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    if (Test-Path $extractRoot) { Remove-Item $extractRoot -Recurse -Force }
    New-Item -ItemType Directory -Path $extractRoot | Out-Null
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch { }
    Invoke-WebRequest -UseBasicParsing -Uri $ArchiveUrl -OutFile $zipPath
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractRoot
    $srcDir = Get-ChildItem -LiteralPath $extractRoot -Directory | Select-Object -First 1
    if (-not $srcDir) {
        throw "The update archive did not contain a source directory."
    }

    Update-ProgressFile -Stage "testing" -Message (
        "Running the update test gate against {0}; this can take several minutes." -f $srcDir.Name
    )
    # Run the gate from inside the extracted tree: pytest collects from the
    # working directory, not from --project, so inheriting the server's CWD
    # could test the wrong tree.
    $testExit = Invoke-LoggedProcess -FilePath $uv -ProcessArguments @(
        "run", "--project", $srcDir.FullName, "pytest", "-q", "--tb=short", "-k", $TestExclude
    ) -OutFile (Join-Path $WorkDir "pytest.out") -ErrFile (Join-Path $WorkDir "pytest.err") -TimeoutSeconds $TestTimeoutSeconds -WorkingDirectory $srcDir.FullName
    if ($testExit -ne 0) {
        $reason = if ($testExit -eq -99) { "timed out" } else { "failed (exit $testExit)" }
        Update-ProgressFile -Stage "error" -Failure (
            "The update gate is red: pytest $reason. Nothing was installed; " +
            "see $WorkDir\pytest.out"
        )
        Start-Relaunch
        exit 1
    }

    Update-ProgressFile -Stage "installing" -Message "Installing free-claude-code from the tested archive..."
    $spec = "free-claude-code@$($srcDir.FullName.Replace('\', '/'))"
    $installExit = Invoke-LoggedProcess -FilePath $uv -ProcessArguments @(
        "tool", "install", "--force", "--refresh-package", "free-claude-code",
        "--python", $PythonSpec, $spec
    ) -OutFile (Join-Path $WorkDir "install.out") -ErrFile (Join-Path $WorkDir "install.err")
    if ($installExit -ne 0) {
        Update-ProgressFile -Stage "error" -Failure (
            "uv tool install failed (exit $installExit); the previous version is " +
            "still installed. See $WorkDir\install.err"
        )
        Start-Relaunch
        exit 1
    }

    Update-ProgressFile -Stage "done" -Message "Installed version $TargetVersion. Relaunching..." -Done
    Start-Relaunch
    exit 0
} catch {
    Update-ProgressFile -Stage "error" -Failure (
        "Update failed: $($_.Exception.Message). The previous version is still installed."
    )
    Start-Relaunch
    exit 1
}
