@echo off
cd /d "%~dp0.."
if not exist "venv\Scripts\python.exe" (
  echo Installing Fizgig first ...
  python install_fizgig.py --no-local-gpu || (echo Fizgig's installer stopped. & pause & exit /b 1)
)
"venv\Scripts\python.exe" -u demos\setup_demos.py --amd
pause
