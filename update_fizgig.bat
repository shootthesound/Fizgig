@echo off
cd /d "%~dp0"
echo Updating Fizgig...
git pull
echo.
echo Installing/updating dependencies...
if not exist "requirements.txt" (
    echo WARNING: requirements.txt not found. Skipping dependency installation.
    echo.
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -m pip install uv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install uv package.
        echo Aborting update.
        exit /b 1
    )
    "venv\Scripts\python.exe" -m uv pip install --index-strategy unsafe-best-match -r requirements.txt
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install/update dependencies using uv. See output above.
        echo Aborting update.
        exit /b 1
    )
) else (
    echo WARNING: venv not found - run install_fizgig.bat to set it up.
)
echo.
echo Update complete! Run Fizgig with run_fizgig.bat
pause
