@echo off
REM Self-update guard: cmd reads a running .bat by BYTE OFFSET, so when git pull rewrites
REM this file mid-run, execution resumes at a garbage position in the new content (seen in
REM the wild as: 'installers' is not recognized as an internal or external command). Stage 1
REM copies this script to TEMP and re-launches the copy with the repo dir as an argument;
REM the copy is never rewritten by the pull, so it runs to completion untouched.
if "%~1"=="" (
    copy /y "%~f0" "%TEMP%\fizgig_update_stage2.bat" >nul
    call "%TEMP%\fizgig_update_stage2.bat" "%~dp0"
    exit /b %errorlevel%
)
cd /d "%~1"
echo Updating Fizgig...
REM Older installers overwrote the tracked run_fizgig.bat with a console-attached version.
REM Restore the repo's launcher so it never blocks a pull that touches the file — harmless
REM when the file is already clean.
git checkout -- run_fizgig.bat 2>nul
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

REM Guard: requirements.txt pulls CUDA torch/bitsandbytes. Running this on an AMD ROCm
REM venv replaces the ROCm stack and breaks training. ROCm users must use update_fizgig_rocm.bat.
"venv\Scripts\python.exe" -c "import torch; v=getattr(torch,'__version__','') or ''; r=getattr(getattr(torch,'version',None),'rocm',None); h=getattr(getattr(torch,'version',None),'hip',None); raise SystemExit(1 if (r or h or '+rocm' in v.lower()) else 0)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: This looks like an AMD ROCm install.
    echo.
    echo   update_fizgig.bat installs CUDA packages from requirements.txt and would
    echo   overwrite your ROCm PyTorch / bitsandbytes stack.
    echo.
    echo   Use instead:  update_fizgig_rocm.bat
    echo.
    pause
    exit /b 1
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

"venv\Scripts\python.exe" "uv_install_deps.py" "requirements.txt" "venv"
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install/update dependencies using uv. See output above.
    echo Aborting update.
    exit /b 1
)

:skip_deps

REM ---- Krea 2 Turbo LoRA (~470 MB) ----
REM Idempotent: exits instantly when the file is already present + linked in
REM Preferences, so this step is safe to keep across releases. Always downloads
REM when missing, whatever family is configured; failure never aborts the
REM update (the GUI fetches it at training start as a fallback).
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" "src\fizgig\scripts\fetch_turbo_lora.py"
)

REM ---- MSVC C++ Build Tools check (needed by torch.compile's inductor backend) ----
REM triton installs via requirements.txt, but inductor also needs a C++ compiler on
REM Windows. Detection via vswhere (ships with any VS/Build Tools install).
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set MSVC_FOUND=0
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set MSVC_FOUND=1
)
if "%MSVC_FOUND%"=="0" (
    echo.
    echo ------------------------------------------------------------------------
    echo  OPTIONAL: MSVC C++ Build Tools not detected.
    echo.
    echo  The torch.compile training speedup needs them. Training works fine
    echo  without - you just skip the compile speedup.
    echo.
    echo  Direct download ^(exact installer, no hunting on the MS site^):
    echo    https://aka.ms/vs/17/release/vs_BuildTools.exe
    echo  During install, tick the "Desktop development with C++" workload.
    echo.
    echo  Or install unattended from a terminal:
    echo    winget install Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --passive"
    echo.
    echo  Install it any time - no need to re-run this update. The next training
    echo  run detects it automatically.
    echo ------------------------------------------------------------------------
)

echo.
echo Update complete! Run Fizgig with run_fizgig.bat
pause
