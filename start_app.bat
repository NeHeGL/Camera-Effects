@echo off
setlocal
title Camera Effects

cd /d "%~dp0"

:: -- Make sure the virtual environment exists --------------------
if not exist ".venv\Scripts\python.exe" (
    echo  [INFO] Virtual environment not found. Running installer...
    echo.
    set CE_AUTO_INSTALL=1
    call "%~dp0install.bat"
    set CE_AUTO_INSTALL=
    if errorlevel 1 (
        echo  [ERROR] Installation failed. Fix the errors above and try again.
        pause
        exit /b 1
    )
)

start "" ".venv\Scripts\pythonw.exe" "%~dp0camera_effects.py"
