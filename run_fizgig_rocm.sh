#!/usr/bin/env bash
# Fizgig launcher for AMD ROCm on Linux - mirrors run_fizgig_rocm.bat env tuning.
# HIGHLY EXPERIMENTAL: Linux AMD training is best-effort only.
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source venv/bin/activate

export MIOPEN_FIND_MODE=2
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
# Allocator: upstream GUI honours FIZGIG_NO_EXPANDABLE=1 (A/B opt-out, lora_trainer_gui.py).
export FIZGIG_NO_EXPANDABLE=1
export PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
export FIZGIG_GPU_BACKEND=rocm
# Cache fast-exit: src/fizgig/rocm/cache_exit.py (Linux ROCm only). Opt out: FIZGIG_ROCM_NO_FAST_EXIT=1

# BNB_ROCM_VERSION / ROCM_PATH / HIP_PATH - write_rocm_env.py
# (install / first launch only; re-run write_rocm_env.py after a pull if rocm_env.sh is stale).
if [[ -f rocm_env.sh ]]; then
    # shellcheck source=rocm_env.sh
    source rocm_env.sh
elif [[ -x venv/bin/python ]]; then
    venv/bin/python write_rocm_env.py >/dev/null 2>&1 || true
    if [[ -f rocm_env.sh ]]; then
        # shellcheck source=rocm_env.sh
        source rocm_env.sh
    fi
fi

# Pinned installs set BNB_ROCM_VERSION in rocm_env.sh; otherwise leave unset for bitsandbytes.
if [[ -z "${ROCM_PATH:-}" ]]; then
    for _p in venv/lib/python*/site-packages/_rocm_sdk_core; do
        if [[ -d "$_p" ]]; then
            export ROCM_PATH="$(cd "$_p" && pwd)"
            break
        fi
    done
fi
if [[ -z "${HIP_PATH:-}" && -n "${ROCM_PATH:-}" ]]; then
    export HIP_PATH="$ROCM_PATH"
fi

for _d in /opt/rocm/core-*/bin /opt/rocm/bin; do
    [[ -d "$_d" ]] && PATH="$_d:$PATH"
done
export PATH

for _lib in venv/lib/python*/site-packages/_rocm_sdk_core/lib \
            venv/lib/python*/site-packages/_rocm_sdk/lib \
            venv/lib/python*/site-packages/_rocm_sdk_libraries/lib \
            /opt/rocm/lib /opt/rocm/lib64; do
    [[ -d "$_lib" ]] && LD_LIBRARY_PATH="${_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done
export LD_LIBRARY_PATH

# Detect gfx for legacy RDNA1/2 overrides and MIOpen/rocBLAS DB paths.
GPU_ARCH=""
if [[ -x venv/bin/python && -f detect_gpu_linux.py ]]; then
    GPU_ARCH="$(venv/bin/python detect_gpu_linux.py 2>/dev/null | tail -n1 | tr -d '[:space:]')" || true
fi
IS_LEGACY_GPU=0
case "${GPU_ARCH}" in
    gfx101*|gfx103*) IS_LEGACY_GPU=1 ;;
esac

if [[ "$IS_LEGACY_GPU" -eq 1 ]]; then
    echo "[AMD-ROCm] Legacy GPU ${GPU_ARCH} - RDNA1/2 SDP overrides (no aotriton experimental)"
    export TORCH_BACKENDS_CUDA_FLASH_SDP_ENABLED=0
    export TORCH_BACKENDS_CUDA_MEM_EFF_SDP_ENABLED=0
    export TORCH_BACKENDS_CUDA_MATH_SDP_ENABLED=1
    unset TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL
else
    if [[ -n "$GPU_ARCH" ]]; then
        echo "[AMD-ROCm] GPU arch: ${GPU_ARCH}"
    else
        echo "[AMD-ROCm] GPU arch: unknown"
    fi
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
fi

# MIOpen / rocBLAS tensile DBs under _rocm_sdk_devel; skip on gfx1100
# where those trees are often missing.
if [[ "${GPU_ARCH}" != "gfx1100" ]]; then
    for _devel in venv/lib/python*/site-packages/_rocm_sdk_devel/bin; do
        if [[ -d "$_devel" ]]; then
            export MIOPEN_SYSTEM_DB_PATH="$(cd "$_devel" && pwd)"
            if [[ -d "$_devel/rocblas" ]]; then
                export ROCBLAS_TENSILE_DB_PATH="$(cd "$_devel/rocblas" && pwd)"
            fi
            if [[ -d "$_devel/rocblas/library" ]]; then
                export ROCBLAS_TENSILE_LIBPATH="$(cd "$_devel/rocblas/library" && pwd)"
            fi
            break
        fi
    done
fi

unset ROCBLAS_USE_HIPBLASLT_BATCHED

if [[ -n "${BNB_ROCM_VERSION:-}" ]]; then
    echo "[AMD-ROCm] BNB_ROCM_VERSION=${BNB_ROCM_VERSION}  ROCM_PATH=${ROCM_PATH:-}"
else
    echo "[AMD-ROCm] BNB_ROCM_VERSION unset (bitsandbytes selects lib)  ROCM_PATH=${ROCM_PATH:-}"
fi

python lora_trainer_gui.py &
disown
