@echo off
setlocal
title Camera Effects - Install

cd /d "%~dp0"

echo.
echo  ============================================================
echo   Camera Effects - Setup
echo  ============================================================
echo.

:: -- Create .venv if it doesn't exist --------------------------
if not exist ".venv\Scripts\python.exe" (
    echo  [INFO] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo  [ERROR] Failed to create .venv. Make sure Python 3.8+ is installed and on PATH.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
    echo.
) else (
    echo  [OK] Virtual environment already exists.
    echo.
)

:: -- Install / upgrade requirements ------------------------------
echo  [INFO] Installing requirements (first run may take a minute)...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo  [ERROR] Installation failed. See errors above.
    pause
    exit /b 1
)

echo.
echo  [OK] All packages installed successfully.
echo.
echo  You can now run the app with start_app.bat
echo.
if not defined CE_AUTO_INSTALL pause
