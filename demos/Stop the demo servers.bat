@echo off
cd /d "%~dp0"
"..\venv\Scripts\python.exe" -u launch.py stop
pause
