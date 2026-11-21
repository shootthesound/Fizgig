#!/usr/bin/env bash
# Fizgig launcher for AMD ROCm on Linux - mirrors Launch Fizgig AMD.bat env tuning.
# HIGHLY EXPERIMENTAL: Linux AMD training is best-effort only.
cd "$(dirname "$0")"

# shellcheck disable=SC1091
source venv/bin/activate

export MIOPEN_FIND_MODE=2
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export FIZGIG_GPU_BACKEND=rocm
# Cache fast-exit: src/fizgig/rocm/cache_exit.py (Linux ROCm only). Opt out: FIZGIG_ROCM_NO_FAST_EXIT=1

# BNB_ROCM_VERSION / ROCM_PATH / HIP_PATH - write_rocm_env.py
# (install / first launch only; re-run write_rocm_env.py after a pull if rocm_env.sh is stale).
if [[ -f user_data/rocm_env.sh ]]; then
    # shellcheck source=user_data/rocm_env.sh
    source user_data/rocm_env.sh
elif [[ -x venv/bin/python ]]; then
    venv/bin/python write_rocm_env.py >/dev/null 2>&1 || true
    if [[ -f user_data/rocm_env.sh ]]; then
        # shellcheck source=user_data/rocm_env.sh
        source user_data/rocm_env.sh
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
# Prefer install-time FIZGIG_AMD_GFX from rocm_env.sh.
GPU_ARCH="${FIZGIG_AMD_GFX:-}"
if [[ -z "$GPU_ARCH" && -x venv/bin/python && -f src/install/detect_gpu_linux.py ]]; then
    GPU_ARCH="$(venv/bin/python src/install/detect_gpu_linux.py 2>/dev/null | tail -n1 | tr -d '[:space:]')" || true
fi
IS_LEGACY_GPU=0
case "${GPU_ARCH}" in
    gfx101*|gfx103*) IS_LEGACY_GPU=1 ;;
esac
[[ -n "$GPU_ARCH" ]] && export FIZGIG_AMD_GFX="$GPU_ARCH"     # Fizgig decides the same (src/fizgig/utils/rocm_alloc.py)

# Allocator: expandable segments (under ROCm's own name too: it doesn't read PYTORCH_CUDA_ALLOC_CONF) on AMD's
# datacenter cards only. A 16 GB-capped MI350X's Krea 2 LoKR run at 0.5 MP ran out with 1.8 GB held in unusable pieces
# without them (#341). On consumer Radeon cards (RDNA2/3/4) they corrupt memory silently on current ROCm: zeroed
# tensors, NaN weights, garbage pictures (#363). An unknown card gets none. FIZGIG_NO_EXPANDABLE=1 opts out everywhere.
if [[ "${FIZGIG_NO_EXPANDABLE:-}" != "1" ]]; then
    case "${GPU_ARCH}" in
        gfx908|gfx90a|gfx94*|gfx95*)
            export PYTORCH_ALLOC_CONF=expandable_segments:True
            export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            echo "[AMD-ROCm] Expandable memory segments: on (datacenter card)" ;;
        *)
            echo "[AMD-ROCm] Expandable memory segments: off (consumer or unknown card)" ;;
    esac
fi

if [[ "$IS_LEGACY_GPU" -eq 1 ]]; then
    echo "[AMD-ROCm] Legacy GPU ${GPU_ARCH} - RDNA1/2 SDP overrides (no aotriton experimental)"
    export TORCH_BACKENDS_CUDA_FLASH_SDP_ENABLED=0
    export TORCH_BACKENDS_CUDA_MEM_EFF_SDP_ENABLED=0
    export TORCH_BACKENDS_CUDA_MATH_SDP_ENABLED=1
    unset TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL
    # gfx1031 / gfx1032 run as gfx1030, the RDNA2 chip ROCm builds for; never RDNA1 (gfx101x), which it would break
    case "${GPU_ARCH}" in
        gfx1031|gfx1032) export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-10.3.0}" ;;
    esac
else
    if [[ -n "$GPU_ARCH" ]]; then
        echo "[AMD-ROCm] GPU arch: ${GPU_ARCH}"
    else
        echo "[AMD-ROCm] GPU arch: unknown"
    fi
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
fi

# Locate the packaged MIOpen and rocBLAS databases. Skip gfx1100
# where those trees are often missing.
if [[ "${GPU_ARCH}" != "gfx1100" ]]; then
    for _db in venv/lib/python*/site-packages/_rocm_sdk_{libraries,devel}/share/miopen/db; do
        if [[ -d "$_db" ]]; then
            export MIOPEN_SYSTEM_DB_PATH="$(cd "$_db" && pwd)"
            break
        fi
    done
    for _devel in venv/lib/python*/site-packages/_rocm_sdk_devel/bin; do
        if [[ -d "$_devel" ]]; then
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

python fizgig.py &
disown
