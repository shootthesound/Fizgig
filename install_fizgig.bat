@echo off
title Fizgig Installer
cd /d "%~dp0"

echo ============================================================
echo   Fizgig Installer
echo   LoRA and Fine-tune Studio
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found on PATH.
    echo.
    echo Install Python 3.12 from https://www.python.org/downloads/ - 3.11 or newer works -
    echo and make sure "Add Python to PATH" is checked during install.
    echo.
    pause
    exit /b 1
)

python install_fizgig.py
echo.
pause
