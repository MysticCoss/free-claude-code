@echo off
rem Update this fork from FCC_UPDATE_REPO/FCC_UPDATE_BRANCH (default
rem MysticCoss/free-claude-code@main) through the same tested guardian the
rem Admin UI uses. The checker stages that guardian into the launcher path.
setlocal
if not "%~1"=="" (
  echo fcc-update takes no options; it installs the fork branch set by FCC_UPDATE_REPO/FCC_UPDATE_BRANCH. 1>&2
  exit /b 2
)
set "FCC_UPDATE_LAUNCHER=%TEMP%\fcc-update-%RANDOM%%RANDOM%.cmd"
"%~dp0_fcc-update-check.exe" "%FCC_UPDATE_LAUNCHER%"
if %errorlevel%==0 (
  del "%FCC_UPDATE_LAUNCHER%" >nul 2>&1
  exit /b 0
)
if not %errorlevel%==10 (
  call set "FCC_UPDATE_STATUS=%%errorlevel%%"
  del "%FCC_UPDATE_LAUNCHER%" >nul 2>&1
  call exit /b %%FCC_UPDATE_STATUS%%
)
rem One parenthesized block: the install replaces this file while it runs.
(
  call "%FCC_UPDATE_LAUNCHER%"
  call set "FCC_UPDATE_STATUS=%%errorlevel%%"
  del "%FCC_UPDATE_LAUNCHER%" >nul 2>&1
  call exit /b %%FCC_UPDATE_STATUS%%
)
