@echo off
REM Thin AMD ROCm updater. Pins (BNB_WHEEL, …) live in install_fizgig_rocm.bat —
REM do not duplicate them here; this script reads them after git pull.
REM
REM Do NOT use update_fizgig.bat on a ROCm venv — it installs CUDA torch/bitsandbytes.
REM
REM Self-update guard: cmd reads a running .bat by BYTE OFFSET, so git pull rewriting
REM this file mid-run resumes at a garbage offset. Stage 1 copies to TEMP and re-launches.
if "%~1"=="" (
    copy /y "%~f0" "%TEMP%\fizgig_update_rocm_stage2.bat" >nul
    call "%TEMP%\fizgig_update_rocm_stage2.bat" "%~dp0"
    exit /b %errorlevel%
)
cd /d "%~1"
setlocal enabledelayedexpansion
echo Updating Fizgig ^(AMD ROCm^)...
git checkout -- run_fizgig_rocm.bat 2>nul
git checkout -- run_fizgig.bat 2>nul
git pull
echo.

if not exist "venv\Scripts\python.exe" (
    echo WARNING: venv not found - run install_fizgig_rocm.bat to set it up.
    pause
    exit /b 1
)
if not exist "requirements.txt" (
    echo WARNING: requirements.txt not found. Skipping dependency installation.
    goto skip_deps
)

REM Guard: this updater installs a ROCm bitsandbytes wheel and skips CUDA torch.
REM Running it on an NVIDIA venv would replace CUDA bitsandbytes with the AMD wheel.
"venv\Scripts\python.exe" -c "import torch; v=getattr(torch,'__version__','') or ''; r=getattr(getattr(torch,'version',None),'rocm',None); h=getattr(getattr(torch,'version',None),'hip',None); raise SystemExit(0 if (r or h or '+rocm' in v.lower()) else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo ERROR: This looks like an NVIDIA / CUDA install.
    echo.
    echo   update_fizgig_rocm.bat installs the AMD ROCm bitsandbytes wheel and would
    echo   overwrite your CUDA PyTorch / bitsandbytes stack.
    echo.
    echo   Use instead:  update_fizgig.bat
    echo.
    pause
    exit /b 1
)

"venv\Scripts\python.exe" -m uv --version >nul 2>&1
if errorlevel 1 (
    "venv\Scripts\python.exe" -m pip install --upgrade uv
    if errorlevel 1 (
        echo ERROR: Failed to install uv.
        pause
        exit /b 1
    )
)

REM Shared deps — same path as install_fizgig_rocm.bat ^(not uv_install_deps.py / CUDA^).
set "ROCM_REQS=%TEMP%\fizgig_rocm_shared_reqs.txt"
"venv\Scripts\python.exe" "filter_requirements_rocm.py" "requirements.txt" "!ROCM_REQS!"
if errorlevel 1 (
    echo ERROR: Failed to build ROCm-safe requirements. ROCm torch left untouched.
    pause
    exit /b 1
)
echo Installing shared dependencies ^(CUDA torch/bnb lines stripped^)...
REM hqq (4-bit HQQ base) builds an optional CUDA kernel from its sdist unless told not to.
set DISABLE_CUDA=1
"venv\Scripts\python.exe" -m uv pip install --index-strategy unsafe-best-match -r "!ROCM_REQS!"
if errorlevel 1 (
    echo ERROR: Failed to install shared dependencies.
    del "!ROCM_REQS!" >nul 2>&1
    pause
    exit /b 1
)
del "!ROCM_REQS!" >nul 2>&1

REM bitsandbytes URL from install_fizgig_rocm.bat ^(single set "BNB_WHEEL=…"^).
set "BNB_WHEEL="
for /f "usebackq delims=" %%L in (`findstr /I /C:"BNB_WHEEL=" "install_fizgig_rocm.bat"`) do (
    for /f "tokens=2 delims==" %%U in ("%%L") do set "BNB_WHEEL=%%~U"
)
set "BNB_WHEEL=!BNB_WHEEL:"=!"
if not defined BNB_WHEEL (
    echo WARNING: BNB_WHEEL not found in install_fizgig_rocm.bat — skipping bitsandbytes.
) else (
    echo Syncing bitsandbytes from installer pin:
    echo   !BNB_WHEEL!
    "venv\Scripts\python.exe" -m uv pip install "!BNB_WHEEL!"
    if errorlevel 1 (
        echo ERROR: Failed to install bitsandbytes wheel.
        pause
        exit /b 1
    )
)

REM Refresh launcher env. Preserve --experimental when no BNB_ROCM_VERSION= assignment
REM exists ^(REM text says "omitted"/"unset" without "=" — safe to findstr^).
set "ROCM_ENV_EXPERIMENTAL=1"
if exist "rocm_env.bat" (
    findstr /I /C:"BNB_ROCM_VERSION=" "rocm_env.bat" >nul 2>&1
    if not errorlevel 1 set "ROCM_ENV_EXPERIMENTAL=0"
)
if !ROCM_ENV_EXPERIMENTAL!==1 (
    echo Refreshing rocm_env.bat ^(experimental: BNB_ROCM_VERSION unset^)...
    "venv\Scripts\python.exe" "write_rocm_env.py" --experimental
) else (
    echo Refreshing rocm_env.bat...
    "venv\Scripts\python.exe" "write_rocm_env.py"
)

:skip_deps

"venv\Scripts\python.exe" "src\fizgig\scripts\fetch_turbo_lora.py"

echo.
echo Update complete! Launch with run_fizgig_rocm.bat
pause
