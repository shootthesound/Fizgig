@echo off
REM Thin launcher: all update logic lives in update_fizgig.py. Python reads the whole script
REM before running it, so the git pull inside can rewrite any file - this one included - without
REM breaking the running update (#157: the old self-copy to TEMP looked like malware to antivirus).
REM Everything after the pull sits on single lines: cmd parses a whole line before running it,
REM so a rewritten .bat is never read mid-run.
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (git pull & echo. & echo WARNING: venv not found - run install_fizgig.bat to set it up. & pause & exit /b 1)
"venv\Scripts\python.exe" "update_fizgig.py" & pause & exit /b
