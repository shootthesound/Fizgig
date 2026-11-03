@echo off
cd /d "%~dp0"
"..\venv\Scripts\python.exe" -u amd_pod.py down
pause
