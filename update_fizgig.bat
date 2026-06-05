@echo off
cd /d "%~dp0"
echo Updating Fizgig...
git pull
echo.
echo Installing/updating dependencies...
if not exist "requirements.txt" (
    echo WARNING: requirements.txt not found. Skipping dependency installation.
    echo.
    goto skip_deps
)

if not exist "venv\Scripts\python.exe" (
    echo WARNING: venv not found - run install_fizgig.bat to set it up.
    echo.
    goto skip_deps
)

"venv\Scripts\python.exe" -m uv --version >nul 2>&1
if errorlevel 1 (
    "venv\Scripts\python.exe" -m pip install --upgrade uv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install uv package.
        echo Aborting update.
        exit /b 1
    )
)

"venv\Scripts\python.exe" -m uv pip install --index-strategy unsafe-best-match -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install/update dependencies using uv. See output above.
    echo Aborting update.
    exit /b 1
)

:skip_deps
echo.
echo Update complete! Run Fizgig with run_fizgig.bat
pause
