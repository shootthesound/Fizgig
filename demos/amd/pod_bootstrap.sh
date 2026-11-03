#!/usr/bin/env bash
# One AMD GPU, everything at once: the start command of a pod on AMD's rocm/pytorch image (an MI300X on RunPod).
# Installs Fizgig with its ROCm installer, Fizgig's models, llama.cpp's ROCm build and the language model, then
# starts both servers and keeps the pod up. Progress goes to the pod's log, each line marked [amd].
# Needs FIZGIG_PASSWORD (Fizgig's login) and LLM_KEY (the language model server's key) in the pod's environment.
set -uo pipefail
BRANCH="${FIZGIG_BRANCH:-wip-tools}"
LLAMA_BUILD=b10666
LLM_FILE="${LLM_FILE:-Qwen3.8-27B-Q8_0.gguf}"
W=/workspace
log() { echo "[amd $(date +%H:%M:%S)] $*"; }
mkdir -p "$W" && cd "$W" || exit 1

command -v sudo >/dev/null || { printf '#!/bin/sh\nexec "$@"\n' > /usr/local/bin/sudo; chmod +x /usr/local/bin/sudo; }
log "packages"
apt-get update -qq >/dev/null && apt-get install -y -qq git curl python3-venv python3-dev libnuma-dev libgl1 libglib2.0-0 >/dev/null   # libGL: OpenCV (faces)
log "GPU:"; (rocm-smi --showproductname --showmeminfo vram 2>/dev/null || amd-smi static 2>/dev/null) | head -30
rocminfo 2>/dev/null | grep -m3 "gfx"

if [ ! -d Fizgig ]; then git clone -q -b "$BRANCH" https://github.com/shootthesound/Fizgig; fi
cd Fizgig && git pull -q
if ! venv/bin/python -c "import huggingface_hub, torch" 2>/dev/null; then
    log "Fizgig's ROCm installer"
    printf 'y\n' | FIZGIG_PYTHON=/usr/bin/python3 bash install_fizgig_rocm.sh 2>&1 | tail -60
fi
log "chip detected: $(python3 src/install/detect_gpu_linux.py 2>/dev/null)"
KTEST="import torch; x = torch.ones(1024, device='cuda'); print('[amd] torch', torch.__version__, torch.cuda.get_device_name(0), 'kernel test', (x * 2).sum().item())"
if ! venv/bin/python -c "$KTEST"; then
    log "the nightly wheels can't run here: Fizgig's installer again on its stable channel"
    printf 'y\n' | ROCM_CHANNEL=stable FIZGIG_PYTHON=/usr/bin/python3 bash install_fizgig_rocm.sh 2>&1 | tail -40
    venv/bin/python -c "$KTEST" || { log "FAILED: PyTorch can't run a kernel on this GPU"; sleep infinity; }
fi

venv/bin/python - <<'EOF'
import json, os                      # the Assistant on the user's computer drives this Fizgig over /mcp (logged in)
p = "user_data/prefs.json"
d = json.load(open(p)) if os.path.isfile(p) else {}
d["mcp_server"] = "1"
os.makedirs("user_data", exist_ok=True)
json.dump(d, open(p, "w"), indent=2)
EOF
log "Fizgig's models"
venv/bin/python -m fizgig.scripts.fetch_models --family tools --family qwen_image21 2>&1 | grep -v "it/s\]" | tail -25

log "llama.cpp $LLAMA_BUILD (ROCm) and $LLM_FILE"
mkdir -p "$W/llm" && cd "$W/llm" || exit 1
if [ ! -d llama.cpp ]; then
    curl -fsSL -o llama.tgz "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/llama-$LLAMA_BUILD-bin-ubuntu-rocm-7.14-x64.tar.gz"
    mkdir llama.cpp && tar -xzf llama.tgz -C llama.cpp && rm llama.tgz
fi
SERVER="$(find "$W/llm/llama.cpp" -name llama-server -type f | head -1)"
chmod +x "$SERVER"
# this image's ROCm is pip packages (no /opt/rocm): Fizgig's installer unpacked the full ROCm tree into its venv
RSDK="$(find "$W/Fizgig/venv/lib" -maxdepth 3 -type d -name _rocm_sdk_devel | head -1)"
RLIBS="$RSDK/lib:$(find "$W/Fizgig/venv/lib" -maxdepth 3 -type d -name _rocm_sdk_core | head -1)/lib"
devices() { LD_LIBRARY_PATH="$(dirname "$1"):$RLIBS:${LD_LIBRARY_PATH:-}" "$1" --list-devices 2>&1; }
DEVS="$(devices "$SERVER")"
echo "$DEVS" | sed 's/^/[amd llama] /' | head -12
if ! echo "$DEVS" | grep -qiE "ROCm[0-9]|HIP"; then        # the prebuilt build has no code for this chip: build it
    GFX="$(python3 "$W/Fizgig/src/install/detect_gpu_linux.py" 2>/dev/null)"
    CLANG="$(find "$RSDK" -path '*llvm/bin/clang++' | head -1)"
    log "llama.cpp's prebuilt ROCm build doesn't see the GPU: building $LLAMA_BUILD for $GFX against $RSDK"
    apt-get install -y -qq cmake build-essential >/dev/null
    [ -d llama-src ] || git clone -q --depth 1 -b "$LLAMA_BUILD" https://github.com/ggml-org/llama.cpp llama-src 2>/dev/null
    (cd llama-src && HIP_PATH="$RSDK" ROCM_PATH="$RSDK" cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS="$GFX" \
        -DCMAKE_HIP_COMPILER="$CLANG" -DCMAKE_HIP_COMPILER_ROCM_ROOT="$RSDK" -DCMAKE_PREFIX_PATH="$RSDK" \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF 2>&1 | grep -iE "error|hip|rocm" | head -20 \
        ; cmake --build build --config Release -j"$(nproc)" --target llama-server 2>&1 | grep -iE "error|built target" | tail -8)
    [ -x "$W/llm/llama-src/build/bin/llama-server" ] && SERVER="$W/llm/llama-src/build/bin/llama-server"
    devices "$SERVER" | sed 's/^/[amd llama] /' | head -12
fi
"$W/Fizgig/venv/bin/python" - <<EOF
from huggingface_hub import hf_hub_download
for f in ("$LLM_FILE", "mmproj-F16.gguf"):
    print("[amd]", hf_hub_download("unsloth/Qwen3.8-27B-GGUF", f, local_dir="$W/llm/models"))
EOF
cat > presets.ini <<EOF
[*]
parallel = 1
temp = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0

[demo-llm]
model = $W/llm/models/$LLM_FILE
mmproj = $W/llm/models/mmproj-F16.gguf
n-gpu-layers = all
flash-attn = on
ctx-size = 131072
cache-type-k = q8_0
cache-type-v = q8_0
EOF

log "starting the language model's server (port 38100) and Fizgig (port 8081)"
LD_LIBRARY_PATH="$(dirname "$SERVER"):$RLIBS:${LD_LIBRARY_PATH:-}" nohup "$SERVER" --models-preset presets.ini --models-max 1 \
    --host 0.0.0.0 --port 38100 --jinja --api-key "${LLM_KEY:?}" > "$W/llm.log" 2>&1 &
cd "$W/Fizgig" || exit 1
export MIOPEN_FIND_MODE=2 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE FIZGIG_NO_EXPANDABLE=1 FIZGIG_GPU_BACKEND=rocm
[ -f user_data/rocm_env.sh ] || venv/bin/python write_rocm_env.py >/dev/null 2>&1
# shellcheck disable=SC1091
[ -f user_data/rocm_env.sh ] && source user_data/rocm_env.sh
nohup venv/bin/python fizgig.py --web --host 0.0.0.0 --port 8081 > "$W/fizgig.log" 2>&1 &
ok=""
for i in $(seq 1 120); do curl -fs -o /dev/null http://127.0.0.1:8081/ && ok=1 && break; sleep 2; done
if [ -n "$ok" ]; then log "READY"; else log "FAILED: Fizgig didn't start"; tail -40 "$W/fizgig.log"; fi
tail -F "$W/fizgig.log" "$W/llm.log" 2>/dev/null | grep --line-buffered -iE "error|traceback|warn|offload|ROCm|using device|load_tensors" &
sleep infinity
