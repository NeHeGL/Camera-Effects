@echo off
:: ============================================================
::  Camera Effects — Release Helper
::  Usage:  release.bat
::
::  What it does:
::    1. Asks for a version number  (e.g. v1.0.2)
::    2. Commits any uncommitted changes
::    3. Pushes to GitHub
::    4. Creates + pushes a git tag
::    5. GitHub Actions does the rest:
::         - Builds CameraEffects.exe via PyInstaller
::         - Zips it with README.md
::         - Creates a GitHub Release with the zip attached
:: ============================================================

setlocal

echo.
echo  Camera Effects - Release Helper
echo  ================================
echo.

:: ── Get version ─────────────────────────────────────────────
set /p VERSION="Enter version (e.g. v1.0.2): "
if "%VERSION%"=="" (
    echo ERROR: No version entered. Aborting.
    exit /b 1
)

:: Ensure it starts with 'v'
set FIRST=%VERSION:~0,1%
if not "%FIRST%"=="v" set VERSION=v%VERSION%

echo.
echo  Version : %VERSION%
echo.

:: ── Confirm ──────────────────────────────────────────────────
set /p CONFIRM="Commit, push, tag %VERSION% and trigger GitHub release? [y/N]: "
if /i not "%CONFIRM%"=="y" (
    echo Aborted.
    exit /b 0
)

echo.

:: ── Stage all changes ────────────────────────────────────────
echo [1/4] Staging changes...
git add -A
if errorlevel 1 goto :error

:: ── Commit (skip if nothing to commit) ───────────────────────
echo [2/4] Committing...
git diff --cached --quiet
if errorlevel 1 (
    git commit -m "Release %VERSION%"
    if errorlevel 1 goto :error
) else (
    echo        Nothing to commit, working tree clean.
)

:: ── Push master ──────────────────────────────────────────────
echo [3/4] Pushing to GitHub...
git push origin master
if errorlevel 1 goto :error

:: ── Tag + push ───────────────────────────────────────────────
echo [4/4] Tagging %VERSION% and pushing tag...
git tag -f %VERSION%
if errorlevel 1 goto :error
git push origin %VERSION% --force
if errorlevel 1 goto :error

echo.
echo  Done! GitHub Actions is now building the release.
echo  Check progress at:
echo    https://github.com/NeHeGL/Camera-Effects/actions
echo.
echo  The finished release will appear at:
echo    https://github.com/NeHeGL/Camera-Effects/releases/tag/%VERSION%
echo.
goto :eof

:error
echo.
echo  ERROR: Something went wrong (exit code %errorlevel%). Check output above.
exit /b 1
