@echo off
REM Fizgig ROCm Windows installer.
REM Uses detect_gpu.py (GPL-3.0, from comfyui-rocm - see THIRD_PARTY_NOTICES.md)
REM and ROCm-wheel install patterns adapted from comfyui-rocm install.bat:
REM   https://github.com/patientx/comfyui-rocm
setlocal enabledelayedexpansion
title Fizgig ROCm Installer
cd /d "%~dp0"

set "Q=>nul 2>&1"
set "PY312="
set "PY312_SOURCE="
set "ROCM_EXPERIMENTAL=0"

REM Optional: install_fizgig_rocm.bat --experimental  -> floating multi-arch torch (no TORCH_PIN),
REM leave BNB_ROCM_VERSION unset so bitsandbytes auto-picks its highest matching lib.
REM Not the same as Linux ROCM_CHANNEL=nightly (constrained 7.14 / bnb 714 lane).
:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--experimental" set "ROCM_EXPERIMENTAL=1"
shift
goto :parse_args
:args_done

REM Pinned stack confirmed working on RDNA (multi-arch nightlies). Override if needed.
REM Ignored when --experimental is passed.
set "ROCM_INDEX=https://rocm.nightlies.amd.com/whl-multi-arch/"
if not defined TORCH_PIN set "TORCH_PIN=2.12.0+rocm7.15.0a20260728"
if not defined TORCHVISION_PIN set "TORCHVISION_PIN=0.27.0+rocm7.15.0a20260728"
if not defined ROCM_SDK_DEVEL_PIN set "ROCM_SDK_DEVEL_PIN=7.15.0a20260728"
set "BNB_WHEEL=https://github.com/0xDELUXA/bitsandbytes_win_rocm/releases/download/0.50.2.dev0-py3.12-rocm7.16-win_amd64_all/bitsandbytes-0.50.2.dev0-cp312-cp312-win_amd64.whl"

echo ============================================================
echo   Fizgig Installer - AMD ROCm (Windows)
echo   Klein 9B and Krea 2 LoRA Studio
if !ROCM_EXPERIMENTAL!==1 (
    echo   Mode: --experimental ^(floating multi-arch, no torch pin^)
) else (
    echo   Mode: pinned multi-arch ^(default^)
)
echo ============================================================
echo.
echo Requires Python 3.12 ^(the ROCm bitsandbytes wheel is cp312-only^).
echo Fizgig's GUI needs Tkinter, which ships with a full python.org / pymanager install.
echo.
echo PyTorch / ROCm wheels come from AMD nightlies - not built by Fizgig:
echo   Index:  !ROCM_INDEX!
if !ROCM_EXPERIMENTAL!==1 (
    echo   torch[device-ARCH] / torchvision[device-ARCH] / rocm-sdk-devel  ^(unpinned latest^)
    echo   BNB_ROCM_VERSION: unset - bitsandbytes auto-selects its highest matching DLL
) else (
    echo   torch==!TORCH_PIN!
    echo   torchvision==!TORCHVISION_PIN!
    echo   rocm-sdk-devel==!ROCM_SDK_DEVEL_PIN!
)
echo.
echo bitsandbytes is a community Windows ROCm wheel ^(0xDELUXA^) - neither AMD nor Fizgig:
echo   !BNB_WHEEL!
echo.

call :resolve_python312
if not defined PY312 (
    call :print_python312_help
    pause
    exit /b 1
)

echo Using Python 3.12: !PY312!
echo Source: !PY312_SOURCE!
"!PY312!" --version
echo.

"!PY312!" -c "import tkinter" %Q%
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.12 is installed but Tkinter is missing ^(Fizgig's GUI needs it^).
    echo Reinstall from python.org and tick "tcl/tk and IDLE", then re-run this script.
    echo.
    call :print_python312_help
    pause
    exit /b 1
)

if exist "venv" (
    echo Virtual environment already exists at venv\
    set /p "RECREATE=Delete and recreate? (y/N): "
    if /I "!RECREATE!"=="y" (
        echo Removing existing venv...
        rd /s /q "venv"
    )
)

if not exist "venv" (
    echo Creating virtual environment with Python 3.12...
    "!PY312!" -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create venv.
        echo Python 3.12 from python.org / pymanager includes the venv module.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate venv.
    pause
    exit /b 1
)

REM Sanity-check the venv is actually 3.12 (not whatever `python` on PATH defaults to).
python -c "import sys; v=sys.version_info; assert v.major==3 and v.minor==12, f'Expected 3.12, got {v.major}.{v.minor}'; print(f'Venv OK: Python {sys.version.split()[0]}')"
if errorlevel 1 (
    echo ERROR: venv is not Python 3.12 - delete venv\ and re-run.
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip
python -m pip install --upgrade uv

echo.
echo Detecting AMD GPU architecture...
set "arch="
if not exist "detect_gpu.py" (
    echo ERROR: detect_gpu.py not found in %~dp0
    pause
    exit /b 1
)

for /f "delims=" %%A in ('python detect_gpu.py 2^>"%~dp0gpu_detect_debug.log"') do (
    if not "%%A"=="" set "arch=%%A"
)

if "!arch!"=="" (
    echo ERROR: GPU detection failed or unsupported AMD GPU.
    type "%~dp0gpu_detect_debug.log" 2>nul
    pause
    exit /b 1
)

echo Detected GPU architecture: !arch!
echo.

set "USE_LEGACY_URL=0"
for %%G in (gfx942 gfx950) do (
    if /I "!arch!"=="%%G" set "USE_LEGACY_URL=1"
)

if !USE_LEGACY_URL!==1 (
    if /I "!arch!"=="gfx942" (
        set "LEGACY_INDEX=https://rocm.nightlies.amd.com/v2-staging/gfx942-dcgpu/"
        echo Installing ROCm PyTorch for MI300/MI325 ^(gfx942^)...
        echo Source: !LEGACY_INDEX!
        python -m uv pip install --index-strategy unsafe-best-match --index-url "!LEGACY_INDEX!" "rocm[devel,libraries]"
        if errorlevel 1 goto :install_failed
        rocm-sdk init
        python -m uv pip install --index-strategy unsafe-best-match --index-url "!LEGACY_INDEX!" torch torchvision
        if errorlevel 1 goto :install_failed
    )
    if /I "!arch!"=="gfx950" (
        set "LEGACY_INDEX=https://rocm.nightlies.amd.com/v2-staging/gfx950-dcgpu/"
        echo Installing ROCm PyTorch for MI350/MI355 ^(gfx950^)...
        echo Source: !LEGACY_INDEX!
        python -m uv pip install --index-strategy unsafe-best-match --index-url "!LEGACY_INDEX!" "rocm[devel,libraries]"
        if errorlevel 1 goto :install_failed
        rocm-sdk init
        python -m uv pip install --index-strategy unsafe-best-match --index-url "!LEGACY_INDEX!" torch torchvision
        if errorlevel 1 goto :install_failed
    )
    goto :install_shared
)

if !ROCM_EXPERIMENTAL!==1 (
    echo Installing floating ROCm experimental ^(multi-arch, unpinned^) for !arch!...
    echo Source: !ROCM_INDEX!
    echo   torch[device-!arch!]
    echo   torchvision[device-!arch!]
    echo   rocm-sdk-devel
    python -m uv pip install --index-strategy unsafe-best-match --index-url "!ROCM_INDEX!" ^
        "torch[device-!arch!]" ^
        "torchvision[device-!arch!]" ^
        "rocm-sdk-devel"
    if errorlevel 1 goto :install_failed
) else (
    echo Installing pinned ROCm PyTorch ^(multi-arch^) for !arch!...
    echo Source: !ROCM_INDEX!
    echo   torch[device-!arch!]==!TORCH_PIN!
    echo   torchvision[device-!arch!]==!TORCHVISION_PIN!
    echo   rocm-sdk-devel==!ROCM_SDK_DEVEL_PIN!
    python -m uv pip install --index-strategy unsafe-best-match --index-url "!ROCM_INDEX!" ^
        "torch[device-!arch!]==!TORCH_PIN!" ^
        "torchvision[device-!arch!]==!TORCHVISION_PIN!" ^
        "rocm-sdk-devel==!ROCM_SDK_DEVEL_PIN!"
    if errorlevel 1 goto :install_failed
)

:install_shared
echo.
echo Installing Fizgig dependencies from requirements.txt ^(CUDA torch/bnb lines stripped^)...
python filter_requirements_rocm.py requirements.txt "%TEMP%\fizgig_rocm_shared_reqs.txt"
if errorlevel 1 goto :install_failed
REM hqq (4-bit HQQ base) builds an optional CUDA kernel from its sdist unless told not to.
set DISABLE_CUDA=1
python -m uv pip install --index-strategy unsafe-best-match -r "%TEMP%\fizgig_rocm_shared_reqs.txt"
if errorlevel 1 goto :install_failed
del "%TEMP%\fizgig_rocm_shared_reqs.txt" %Q%

echo.
echo Installing bitsandbytes ^(community Windows ROCm wheel - neither AMD nor Fizgig^)...
echo Source: !BNB_WHEEL!
python -m uv pip install "!BNB_WHEEL!"
if errorlevel 1 goto :install_failed

echo.
echo Verifying ROCm / HIP...
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'GPU available: {torch.cuda.is_available()}'); print(f'HIP: {getattr(torch.version, \"hip\", \"n/a\")}'); print(f'ROCm: {getattr(torch.version, \"rocm\", \"n/a\")}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"
if errorlevel 1 goto :install_failed

echo.
if !ROCM_EXPERIMENTAL!==1 (
    echo Writing ROCm launcher config ^(BNB_ROCM_VERSION omitted - bitsandbytes auto^)...
    python write_rocm_env.py --experimental
) else (
    echo Writing ROCm launcher config ^(BNB_ROCM_VERSION for bitsandbytes^)...
    python write_rocm_env.py
)
if errorlevel 1 goto :install_failed

echo.
echo Downloading InsightFace models ^(CPU, ~300 MB^)...
python -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; from insightface.app import FaceAnalysis; app=FaceAnalysis(name='buffalo_l', allowed_modules=['detection','genderage','recognition'], providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1); print('Models ready.')"

echo.
echo ============================================================
echo   Installation complete!
echo.
echo   Launch with: run_fizgig_rocm.bat
echo   ^(sets ROCm tuning env vars before starting the GUI^)
echo ============================================================
goto :end

:install_failed
echo.
echo ERROR: Installation failed. See messages above.
pause
exit /b 1

:end
pause
exit /b 0


REM ---------------------------------------------------------------------------
REM Resolve Python 3.12 - never trust bare `python` when 3.14+ is the default.
REM Order: py -3.12  >  python3.12 on PATH  >  python if it is actually 3.12
REM ---------------------------------------------------------------------------
:resolve_python312

REM Python launcher / Install Manager (py list, py -3.12, py install 3.12).
where py >nul 2>&1
if not errorlevel 1 (
    py -3.12 -V %Q%
    if not errorlevel 1 (
        for /f "delims=" %%P in ('py -3.12 -c "import sys; print(sys.executable)" 2^>nul') do (
            call :verify_python312 "%%P" "py launcher (py -3.12)"
            if defined PY312 exit /b 0
        )
    )
)

REM Explicit python3.12 on PATH (OneTrainer install.custom.bat style).
for %%C in (python3.12 python3.12.exe) do (
    where %%C >nul 2>&1
    if not errorlevel 1 (
        for /f "delims=" %%P in ('where %%C 2^>nul') do (
            call :verify_python312 "%%P" "PATH (%%C)"
            if defined PY312 exit /b 0
        )
    )
)

REM Last resort: default `python` only if it is actually 3.12.x.
where python >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('where python 2^>nul') do (
        call :verify_python312 "%%P" "PATH (python)"
        if defined PY312 exit /b 0
    )
)
exit /b 0


:verify_python312
set "_CAND=%~1"
set "_SRC=%~2"
"%_CAND%" -c "import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)" %Q%
if errorlevel 1 exit /b 0
set "PY312=%_CAND%"
set "PY312_SOURCE=%_SRC%"
exit /b 0


:print_python312_help
echo.
echo ERROR: Python 3.12 not found ^(needed for the ROCm bitsandbytes wheel, cp312-only^).
echo `py -3.12` / `python3.12` must work, then re-run this script.
echo.
echo Windows downloads: https://www.python.org/downloads/windows/
echo.
echo Recommended ^(2026^) - Python Install Manager:
echo   Microsoft Store: https://apps.microsoft.com/detail/9NQ7512CXL7T
echo   Release page:    https://www.python.org/downloads/latest/pymanager
echo   Then in a new terminal:
echo     py install 3.12
echo.
echo Alternative - Python 3.12.10 installer:
echo   https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
echo   Tick "Add python.exe to PATH" and "tcl/tk and IDLE".
echo.
exit /b 0
