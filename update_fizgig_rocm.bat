@echo off
REM Thin AMD ROCm launcher: all update logic lives in update_fizgig.py (--rocm). Pins such as
REM BNB_WHEEL stay in install_fizgig_rocm.bat. Do NOT use update_fizgig.bat on a ROCm venv.
REM Python reads the whole script before running it, so the git pull inside can rewrite any file
REM without breaking the running update (#157). The run line below is a single line: cmd parses it
REM whole, so a rewritten .bat is never read mid-run.
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (echo WARNING: venv not found - run install_fizgig_rocm.bat to set it up. & pause & exit /b 1)
"venv\Scripts\python.exe" "update_fizgig.py" --rocm & pause & exit /b
