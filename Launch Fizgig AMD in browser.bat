@echo off
REM Thin wrapper: same ROCm env as Launch Fizgig AMD.bat, web UI in a browser.
REM Plain "Launch Fizgig in browser.bat" is the NVIDIA path and intentionally skips this.
cd /d "%~dp0"
call "%~dp0Launch Fizgig AMD.bat" browser %*
