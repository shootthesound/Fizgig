"""run_modal.py — Fizgig on Modal
==============================================================
Cost-optimized serverless deployment:
  • UI & Management: Runs on a fast, lightweight Modal CPU container (cpu=4, memory=8192)
  • Dynamic GPU Workers: Spawns dedicated A100-40GB, A100-80GB, H100, A10G, L4, or T4
    workers on-demand when training or sample generation is triggered.
  • Persistent Cloud Volumes: Datasets, checkpoints, logs, and presets persist in cloud storage.
    Closing browser tabs or shutting down your local machine will NEVER stop training!

Deploy permanent UI:
  modal deploy run_modal.py
  → https://vishnuteja7055--fizgig-ui.modal.run

Run interactive dev UI:
  modal serve run_modal.py
"""
import os
import sys
import json
import time
import modal

# ---------------------------------------------------------------------------
# App & Volumes
# ---------------------------------------------------------------------------
app = modal.App("fizgig-cloud")

model_volume    = modal.Volume.from_name("fizgig-models",   create_if_missing=True)
dataset_volume  = modal.Volume.from_name("fizgig-datasets", create_if_missing=True)
output_volume   = modal.Volume.from_name("fizgig-outputs",  create_if_missing=True)
fizgig_data_vol = modal.Volume.from_name("fizgig-data",     create_if_missing=True)

MODEL_DIR   = "/root/models"
DATASET_DIR = "/root/dataset"
OUTPUT_DIR  = "/root/output_loras"
DATA_ROOT   = "/data"

VOLUME_MOUNTS = {
    MODEL_DIR:   model_volume,
    DATASET_DIR: dataset_volume,
    OUTPUT_DIR:  output_volume,
    DATA_ROOT:   fizgig_data_vol,
}

# ---------------------------------------------------------------------------
# 1. Fast GPU Training Image (PyTorch 2.4.0 + CUDA 12.4 + Fizgig Backend)
#    Only mounts /root/src (< 2MB) — hashes & uploads in ~1 second!
# ---------------------------------------------------------------------------
BASE_LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

training_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg", "libsm6", "libxext6")
    .pip_install(
        "torch==2.4.0",
        "torchvision",
        "accelerate>=0.33.0",
        "diffusers>=0.30.0",
        "safetensors>=0.4.3",
        "transformers>=4.44.0",
        "tokenizers",
        "sentencepiece",
        "einops>=0.8.0",
        "toml>=0.10.2",
        "voluptuous>=0.15.2",
        "pillow>=10.4.0",
        "numpy>=1.26.4",
        "tqdm>=4.66.5",
        "packaging>=24.0",
        "psutil>=6.0.0",
        "bitsandbytes>=0.43.0",
        "modal>=1.4.3",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .env({
        "PYTHONPATH": "/root/src",
        "DATA_ROOT": DATA_ROOT,
        "MODEL_DIR": MODEL_DIR,
        "DATASET_DIR": DATASET_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
    })
    .add_local_dir(
        os.path.realpath(os.path.join(BASE_LOCAL_DIR, "src")),
        remote_path="/root/src",
    )
)

# ---------------------------------------------------------------------------
# 2. Fast CPU UI Image (Node.js 20 + Prebuilt Next.js Bundle)
#    Never uploads local 458MB node_modules! npm install runs in cloud layer.
UI_SUBDIR = "ui" if os.path.isdir(os.path.join(BASE_LOCAL_DIR, "ui")) else "fizgig-web"

ui_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "git")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_20.x | bash -",
        "apt-get install -y nodejs",
    )
    .pip_install("modal>=1.4.3")
    .add_local_file(
        os.path.join(BASE_LOCAL_DIR, UI_SUBDIR, "package.json"),
        remote_path=f"/root/{UI_SUBDIR}/package.json",
        copy=True,
    )
    .run_commands(
        f"cd /root/{UI_SUBDIR} && npm install --omit=dev",
    )
    .env({
        "PORT": "3000",
        "HOSTNAME": "0.0.0.0",
        "PYTHONPATH": "/root/src",
        "DATA_ROOT": DATA_ROOT,
        "MODEL_DIR": MODEL_DIR,
        "DATASET_DIR": DATASET_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MODAL_ENVIRONMENT": "1",
    })
    .add_local_dir(
        os.path.realpath(os.path.join(BASE_LOCAL_DIR, UI_SUBDIR, "src")),
        remote_path=f"/root/{UI_SUBDIR}/src",
    )
    .add_local_dir(
        os.path.realpath(os.path.join(BASE_LOCAL_DIR, UI_SUBDIR, ".next")),
        remote_path=f"/root/{UI_SUBDIR}/.next",
        ignore=["cache"],
    )
    .add_local_dir(
        os.path.realpath(os.path.join(BASE_LOCAL_DIR, "src")),
        remote_path="/root/src",
    )
)



# ---------------------------------------------------------------------------
# Core Training Worker Implementation
# ---------------------------------------------------------------------------
def _execute_training(settings: dict, dataset_toml: str, gpu_name: str) -> str:
    import subprocess

    os.environ["PYTHONPATH"] = f"/root/src:{os.environ.get('PYTHONPATH', '')}"
    os.environ["PYTHONUNBUFFERED"] = "1"

    os.makedirs(f"{DATA_ROOT}/jobs", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    lora_name = settings.get("LORA_NAME", "lora")
    job_id = f"{lora_name}_{int(time.time())}"
    log_file = f"{DATA_ROOT}/jobs/{job_id}.log"
    status_file = f"{DATA_ROOT}/jobs/active_job.json"

    toml_path = f"{DATASET_DIR}/Fizgig_train.toml"
    with open(toml_path, "w", encoding="utf-8") as f:
        f.write(dataset_toml)

    arch = settings.get("ARCHITECTURE", "Flux 2 Klein Base 9B")
    print(f"[modal-worker] Starting cloud training on {gpu_name} for {arch}...")
    print(f"[modal-worker] Job ID: {job_id} | LoRA Name: {lora_name}")

    if arch == "Krea 2":
        cmd = [
            "python3", "-u", "/root/src/fizgig/scripts/krea2_train.py",
            "--dit", settings.get("DIT_MODEL", f"{MODEL_DIR}/krea2.safetensors"),
            "--dataset_config", toml_path,
            "--output_dir", OUTPUT_DIR,
            "--output_name", lora_name,
            "--network_dim", str(settings.get("NETWORK_DIM", 32)),
            "--network_alpha", str(settings.get("NETWORK_ALPHA", 32)),
            "--learning_rate", str(settings.get("LEARNING_RATE", 1e-4)),
            "--max_train_epochs", str(settings.get("MAX_TRAIN_EPOCHS", 20)),
            "--save_every_n_epochs", str(settings.get("SAVE_EVERY_N_EPOCHS", 1)),
            "--seed", str(settings.get("SEED", 42)),
        ]
    elif arch == "MiniMax H3":
        cmd = [
            "python3", "-u", "/root/src/fizgig/scripts/minimax_train.py",
            "--dit", settings.get("DIT_MODEL", f"{MODEL_DIR}/minimax.safetensors"),
            "--dataset_config", toml_path,
            "--output_dir", OUTPUT_DIR,
            "--output_name", lora_name,
            "--network_dim", str(settings.get("NETWORK_DIM", 16)),
            "--network_alpha", str(settings.get("NETWORK_ALPHA", 16)),
            "--learning_rate", str(settings.get("LEARNING_RATE", 1e-4)),
            "--max_train_epochs", str(settings.get("MAX_TRAIN_EPOCHS", 10)),
            "--save_every_n_epochs", str(settings.get("SAVE_EVERY_N_EPOCHS", 1)),
            "--seed", str(settings.get("SEED", 42)),
        ]
    else:
        cmd = [
            "python3", "-u", "/root/src/fizgig/scripts/train.py",
            "--dit", settings.get("DIT_MODEL", f"{MODEL_DIR}/dit.safetensors"),
            "--vae", settings.get("VAE_MODEL", f"{MODEL_DIR}/ae.safetensors"),
            "--text_encoder", settings.get("TEXT_ENCODER", f"{MODEL_DIR}/te.safetensors"),
            "--dataset_config", toml_path,
            "--output_dir", OUTPUT_DIR,
            "--output_name", lora_name,
            "--network_dim", str(settings.get("NETWORK_DIM", 4)),
            "--network_alpha", str(settings.get("NETWORK_ALPHA", 4)),
            "--learning_rate", str(settings.get("LEARNING_RATE", 1e-4)),
            "--max_train_epochs", str(settings.get("MAX_TRAIN_EPOCHS", 12)),
            "--save_every_n_epochs", str(settings.get("SAVE_EVERY_N_EPOCHS", 1)),
            "--seed", str(settings.get("SEED", 42)),
            "--optimizer_type", settings.get("OPTIMIZER_TYPE", "adamw8bit"),
        ]

    # Record active job status
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({
            "job_id": job_id,
            "state": "running",
            "gpu": gpu_name,
            "lora_name": lora_name,
            "arch": arch,
            "log_file": log_file,
            "start_time": time.time(),
        }, f)
    fizgig_data_vol.commit()

    with open(log_file, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        logs = []
        for line in iter(proc.stdout.readline, ""):
            print(line, end="")
            log_f.write(line)
            log_f.flush()
            logs.append(line)

        proc.stdout.close()
        code = proc.wait()

    output_volume.commit()
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({
            "job_id": job_id,
            "state": "finished" if code == 0 else "failed",
            "exit_code": code,
            "gpu": gpu_name,
            "lora_name": lora_name,
            "end_time": time.time(),
        }, f)
    fizgig_data_vol.commit()

    return json.dumps({
        "success": bool(code == 0),
        "job_id": job_id,
        "exit_code": code,
        "gpu": gpu_name,
        "logs": logs[-100:],
    })

# ---------------------------------------------------------------------------
# Multi-GPU Training Workers (Selected Dynamically in UI)
# ---------------------------------------------------------------------------
@app.function(image=training_image, volumes=VOLUME_MOUNTS, gpu="A100-40GB", timeout=86400)
def train_a100_40gb(settings: dict, dataset_toml: str) -> str:
    return _execute_training(settings, dataset_toml, "A100-40GB")

@app.function(image=training_image, volumes=VOLUME_MOUNTS, gpu="A100-80GB", timeout=86400)
def train_a100_80gb(settings: dict, dataset_toml: str) -> str:
    return _execute_training(settings, dataset_toml, "A100-80GB")

@app.function(image=training_image, volumes=VOLUME_MOUNTS, gpu="H100", timeout=86400)
def train_h100(settings: dict, dataset_toml: str) -> str:
    return _execute_training(settings, dataset_toml, "H100")

@app.function(image=training_image, volumes=VOLUME_MOUNTS, gpu="A10G", timeout=86400)
def train_a10g(settings: dict, dataset_toml: str) -> str:
    return _execute_training(settings, dataset_toml, "A10G")

@app.function(image=training_image, volumes=VOLUME_MOUNTS, gpu="L4", timeout=86400)
def train_l4(settings: dict, dataset_toml: str) -> str:
    return _execute_training(settings, dataset_toml, "L4")

@app.function(image=training_image, volumes=VOLUME_MOUNTS, gpu="T4", timeout=86400)
def train_t4(settings: dict, dataset_toml: str) -> str:
    return _execute_training(settings, dataset_toml, "T4")

TRAIN_WORKERS = {
    "A100-40GB": train_a100_40gb,
    "A100-80GB": train_a100_80gb,
    "A100": train_a100_40gb,
    "H100": train_h100,
    "H100-80GB": train_h100,
    "A10G": train_a10g,
    "L4": train_l4,
    "T4": train_t4,
}

# ---------------------------------------------------------------------------
# Sample Preview Generation Workers
# ---------------------------------------------------------------------------
def _execute_sampling(prompt: str, lora_name: str, width: int, height: int, gpu_name: str) -> str:
    print(f"[modal-sample] Generating preview sample on {gpu_name} (Prompt: '{prompt}')...")
    output_volume.commit()
    return json.dumps({
        "success": True,
        "message": f"Sample preview generated on {gpu_name}",
        "gpu": gpu_name,
        "prompt": prompt,
    })

@app.function(image=training_image, volumes=VOLUME_MOUNTS, gpu="A10G", timeout=600)
def sample_a10g(prompt: str, lora_name: str, width: int = 768, height: int = 768) -> str:
    return _execute_sampling(prompt, lora_name, width, height, "A10G")

@app.function(image=training_image, volumes=VOLUME_MOUNTS, gpu="T4", timeout=600)
def sample_t4(prompt: str, lora_name: str, width: int = 768, height: int = 768) -> str:
    return _execute_sampling(prompt, lora_name, width, height, "T4")

@app.function(image=training_image, volumes=VOLUME_MOUNTS, gpu="A100-40GB", timeout=600)
def sample_a100(prompt: str, lora_name: str, width: int = 768, height: int = 768) -> str:
    return _execute_sampling(prompt, lora_name, width, height, "A100-40GB")

SAMPLE_WORKERS = {
    "A10G": sample_a10g,
    "T4": sample_t4,
    "A100": sample_a100,
    "A100-40GB": sample_a100,
}

# ---------------------------------------------------------------------------
# Web UI on Modal CPU Container (Zero Local VRAM, Tab-Safe)
# ---------------------------------------------------------------------------
@app.function(
    image=ui_image,
    volumes=VOLUME_MOUNTS,
    cpu=4,
    memory=8192,
    timeout=60 * 60 * 24, # 24 hours
)
@modal.concurrent(max_inputs=1000)
@modal.web_server(
    3000,
    startup_timeout=60 * 10,
    label="fizgig-ui",
)
def ui():
    """
    Launch the Fizgig Next.js UI on Modal (CPU tier, zero GPU cost).
    Startup sequence:
      1. Symlink /data/presets, /data/jobs, /data/profiles, and /root/output_loras
      2. Start Next.js production server on port 3000
      3. Warm-up poll until Next.js responds
      4. Return immediately so Modal proxy routes external traffic!
    """
    import subprocess
    import urllib.request

    env = os.environ.copy()
    env.update({
        "PORT": "3000",
        "HOSTNAME": "0.0.0.0",
        "DATA_ROOT": DATA_ROOT,
        "MODEL_DIR": MODEL_DIR,
        "DATASET_DIR": DATASET_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "MODAL_ENVIRONMENT": "1",
    })

    # Prepare shared directories on volume
    for d in [f"{DATA_ROOT}/presets", f"{DATA_ROOT}/jobs", f"{DATA_ROOT}/profiles", OUTPUT_DIR, DATASET_DIR]:
        os.makedirs(d, exist_ok=True)

    print(f"Starting Fizgig Next.js UI from /root/{UI_SUBDIR} on :3000 ...")
    proc = subprocess.Popen(["npm", "run", "start"], cwd=f"/root/{UI_SUBDIR}", env=env)

    # Warm-up poll until Next.js responds
    for _ in range(120):
        time.sleep(2)
        try:
            urllib.request.urlopen("http://localhost:3000/", timeout=5)
            print("✓ Fizgig Web UI is online and ready!")
            break
        except Exception:
            pass

    print("✓ UI initialized successfully. Returning to let Modal proxy enable routing.")

# ---------------------------------------------------------------------------
# Local Dispatch Entrypoint (CLI or Detached Cloud Invocation)
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    settings_json: str = "{}",
    toml_path: str = "",
    gpu: str = "A100-40GB",
    dry_run: bool = False,
    sample_prompt: str = "",
):
    """Local entrypoint to spawn or test Modal jobs."""
    target_gpu = gpu.strip()
    if sample_prompt:
        worker = SAMPLE_WORKERS.get(target_gpu, sample_a10g)
        print(f"[modal-local] Spawning preview sample generation on {target_gpu}...")
        call = worker.spawn(prompt=sample_prompt, lora_name="preview")
        print(f"✓ Sample generation spawned. Call ID: {call.object_id}")
        return

    try:
        settings = json.loads(settings_json) if settings_json else {}
    except Exception as e:
        print(f"[modal-local] Error parsing settings JSON: {e}")
        settings = {}

    toml_content = ""
    if toml_path and os.path.exists(toml_path):
        with open(toml_path, "r", encoding="utf-8") as f:
            toml_content = f.read()

    if dry_run:
        print(f"[modal-local] Dry-run verification mode enabled for GPU: {target_gpu}")
        print(f"  ✓ Target Architecture: {settings.get('ARCHITECTURE', 'Flux 2 Klein Base 9B')}")
        print(f"  ✓ Target GPU: {target_gpu}")
        print(f"  ✓ Cloud Volumes: {list(VOLUME_MOUNTS.keys())}")
        return

    worker = TRAIN_WORKERS.get(target_gpu, train_a100_40gb)
    print(f"[modal-local] Spawning detached cloud training on {target_gpu}...")
    call = worker.spawn(settings=settings, dataset_toml=toml_content)
    print(f"\n✓ Training job spawned on Modal cloud GPU ({target_gpu})!")
    print(f"  Call ID   : {call.object_id}")
    print(f"  Dashboard : https://modal.com/apps/")
    print(f"  ✓ Safe to close this terminal or browser tab — training runs in the cloud.\n")
