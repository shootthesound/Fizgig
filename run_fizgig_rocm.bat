@echo off
REM Fizgig launcher for AMD ROCm on Windows.
REM Sets ROCm/HIP tuning env vars (legacy RDNA1/2 vs newer), then starts the consoleless GUI.
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [AMD-ROCm] Setting environment variables...
set "MIOPEN_FIND_MODE=2"
set "FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE"
REM expandable_segments mirrors upstream train scripts' CUDA alloc policy (ROCm key).
set "PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8"
set "FIZGIG_GPU_BACKEND=rocm"

REM BNB_ROCM_VERSION / ROCM_PATH / HIP_PATH - written by
REM install_fizgig_rocm.bat / write_rocm_env.py (not on every launch; re-run write_rocm_env.py
REM after a pull if rocm_env.bat is stale).
if exist "%~dp0rocm_env.bat" (
    call "%~dp0rocm_env.bat"
) else if exist "%~dp0venv\Scripts\python.exe" (
    "%~dp0venv\Scripts\python.exe" "%~dp0write_rocm_env.py" >nul 2>&1
    if exist "%~dp0rocm_env.bat" call "%~dp0rocm_env.bat"
)
REM Pinned installs set BNB_ROCM_VERSION in rocm_env.bat. Floating --experimental leaves it
REM unset so bitsandbytes picks its highest matching DLL (override: set BNB_ROCM_VERSION).
if not defined ROCM_PATH set "ROCM_PATH=%~dp0venv\Lib\site-packages\_rocm_sdk_core"
if not defined HIP_PATH set "HIP_PATH=%ROCM_PATH%"

REM bitsandbytes cuda_specs runs "hipinfo" on Windows - lives under the pip ROCm SDK bin/ and venv\Scripts.
if defined ROCM_PATH if exist "%ROCM_PATH%\bin" set "PATH=%ROCM_PATH%\bin;%PATH%"
if exist "%~dp0venv\Lib\site-packages\_rocm_sdk_devel\bin" set "PATH=%~dp0venv\Lib\site-packages\_rocm_sdk_devel\bin;%PATH%"
if exist "%~dp0venv\Scripts" set "PATH=%~dp0venv\Scripts;%PATH%"

REM Detect gfx for legacy RDNA1/2 overrides and MIOpen/rocBLAS DB paths.
set "GPU_ARCH="
if exist "%~dp0venv\Scripts\python.exe" if exist "%~dp0detect_gpu.py" (
    for /f "delims=" %%A in ('"%~dp0venv\Scripts\python.exe" "%~dp0detect_gpu.py" 2^>nul') do set "GPU_ARCH=%%A"
)
set "IS_LEGACY_GPU=0"
if defined GPU_ARCH (
    if /I "!GPU_ARCH:~0,6!"=="gfx101" set "IS_LEGACY_GPU=1"
    if /I "!GPU_ARCH:~0,6!"=="gfx103" set "IS_LEGACY_GPU=1"
)

if "!IS_LEGACY_GPU!"=="1" (
    echo [AMD-ROCm] Legacy GPU !GPU_ARCH! - RDNA1/2 SDP overrides ^(no aotriton experimental^)
    set "TORCH_BACKENDS_CUDA_FLASH_SDP_ENABLED=0"
    set "TORCH_BACKENDS_CUDA_MEM_EFF_SDP_ENABLED=0"
    set "TORCH_BACKENDS_CUDA_MATH_SDP_ENABLED=1"
) else (
    if defined GPU_ARCH (
        echo [AMD-ROCm] GPU arch: !GPU_ARCH!
    ) else (
        echo [AMD-ROCm] GPU arch: unknown
    )
    set "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1"
)

REM MIOpen / rocBLAS tensile DBs live under _rocm_sdk_devel; skip on gfx1100
REM where those trees are often missing.
set "ROCM_DEVEL_BIN=%~dp0venv\Lib\site-packages\_rocm_sdk_devel\bin"
if /I not "!GPU_ARCH!"=="gfx1100" (
    if exist "!ROCM_DEVEL_BIN!" set "MIOPEN_SYSTEM_DB_PATH=!ROCM_DEVEL_BIN!"
    if exist "!ROCM_DEVEL_BIN!\rocblas" set "ROCBLAS_TENSILE_DB_PATH=!ROCM_DEVEL_BIN!\rocblas"
    if exist "!ROCM_DEVEL_BIN!\rocblas\library" set "ROCBLAS_TENSILE_LIBPATH=!ROCM_DEVEL_BIN!\rocblas\library"
)

set "ROCBLAS_USE_HIPBLASLT_BATCHED="

if defined BNB_ROCM_VERSION (
    echo [AMD-ROCm] BNB_ROCM_VERSION=%BNB_ROCM_VERSION%  ROCM_PATH=%ROCM_PATH%
) else (
    echo [AMD-ROCm] BNB_ROCM_VERSION unset ^(bitsandbytes selects DLL^)  ROCM_PATH=%ROCM_PATH%
)

start "" /b wscript //nologo //b "%~dp0run_silent.vbs"
