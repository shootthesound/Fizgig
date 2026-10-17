# Install and requirements

[← Back to the README](../README.md)

## No GPU? Rent one

Fizgig ships as a ready-made cloud image: the whole app in a browser tab, not a cut-down web version. Drag datasets in and LoRAs out with the built-in file manager, download models in one click, and optionally have the pod shut itself down when training finishes. Models and datasets persist between sessions.

**[⚡ Deploy on RunPod →](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=bdrdew7hxw&ref=vkb387ep)**  ·  [Read the guide first](../docker/README.md)  ·  Need the old pod? [Fizgig Classic v7.x](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)

## Requirements

- **GPU** — NVIDIA RTX 30 / 40 / 50-series, or **AMD Radeon** with ROCm (RDNA1 through RDNA4, Strix Point / Halo, Instinct MI300+). **Klein 9B** trains on 10 GB (the 4-bit base below 16 GB), **Flux 3 Image Klein** and **Krea 2** train on 8 GB, **MiniMax H3** on 16 GB, **Qwen Image 2.1** on 10 GB — see [VRAM guidance](#vram-guidance). The fp8 Base's VRAM savings apply on NVIDIA Ada+; on AMD, NF4 and INT8 are the primary quant paths.
- **NVIDIA driver** — 555+ on Windows, 550+ on Linux (CUDA 12.8 wheels).
- **AMD ROCm** — **Windows:** `install_fizgig_rocm.bat` (supported path; launcher and installer improvements by [@scryptio](https://github.com/scryptio)). **Linux:** `./install_fizgig_rocm.sh`, **highly experimental**: newer gfx such as RDNA4, a desktop compositor sharing the training GPU, and driver resets are common. Use Windows ROCm or NVIDIA Linux for production training. Optional system `amdrocm-amdsmi` gives accurate status-bar VRAM via `amd-smi`.
- **OS** — Windows 10 / 11 or Linux. macOS handles captioning and image prep; training needs CUDA or ROCm.
- **Python** — 3.10 – 3.13.
- **System RAM** — 32 GB recommended; see [System RAM](#system-ram).
- **Disk** — ~10 GB for the venv, plus ~40 GB for model files.
- **Full fine-tuning** (experimental, every model; Flux 3 Image Klein, Krea 2 and MiniMax H3 in the guides) — see [Full fine-tuning](#full-fine-tuning).
- **Visual Studio Build Tools** (Windows only) — for InsightFace and the torch.compile speedup: **[aka.ms/vs/17/release/vs_BuildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe)**, tick **"Desktop development with C++"**. Without it everything works except the compile speedup.
- **Microsoft Edge WebView2 Runtime** (Windows 10) — Fizgig's app window runs on it. Windows 11 has it built in, but it isn't always installed on Windows 10. If the window opens white, install it from **[developer.microsoft.com/microsoft-edge/webview2](https://developer.microsoft.com/microsoft-edge/webview2/)** (the **Evergreen Bootstrapper**). Without it, Fizgig opens in your browser instead.

### System RAM

32 GB is recommended; 16 GB is workable for **Klein 9B**, **Flux 3 Image Klein** and **Krea 2**.

**MiniMax H3 needs more.** Its text encoder is a 15.7 GB file that streams from system RAM while captions are cached, and INT8 block streaming stages a similar amount again during training. 32 GB is comfortable for caching and training; 24 GB works only with other apps closed.

**Previews and LoRA Royale need more still.** Each preview parks the training base (~21 GB) in RAM and streams the text encoder through it. With sample generation on, or when rendering epochs in LoRA Royale, plan on **48 GB**, or leave the Windows paging file system-managed on a fast drive so commit can spill. On a card below 32 GB the trainer warns at launch when RAM is under 40 GB.

**Running out of RAM doesn't look like a memory error.** The app closes with a *"not enough memory resources"* dialog, or an RDP session drops. During caching, a RAM shortage shows up as **"CUDA error: out of memory"** even though the GPU is nearly empty: close what else is running and retry the caching step before suspecting VRAM.

To lower the load, use lighter samples: a smaller canvas such as 512×768, 22 frames rather than 56, or a still. As a last resort, train without sample generation and judge the checkpoints in LoRA Royale afterwards.

### Full fine-tuning

Full fine-tuning (experimental, Krea 2 & MiniMax H3) needs more than the above and tiers itself to your card. On the default 4-bit NF4 base, **24 GB** runs the full-depth component cycle at full speed and **16 GB** streams the frozen blocks from RAM at about 1.5× the step time on MiniMax H3 and 2.8× on Krea 2, for both families. Add the bf16 master in RAM (spilled to disk automatically on H3), and disk for saves: **~26 GB per Krea 2 checkpoint, ~21 GB per H3 one**. Each family fine-tunes its normal training base — Krea 2 the RAW bf16 model, H3 the pruned int8 checkpoint (H3's ~66 GB bf16 file works for LoRA training only, not fine-tuning). See [FINETUNE.md](FINETUNE.md).

## Install

Clone the repo:

```bash
git clone https://github.com/shootthesound/Fizgig.git
cd Fizgig
```

**Clone it rather than downloading the ZIP:** `update_fizgig.bat` updates by pulling with git, which a ZIP install can't do.

<details>
<summary><b>Already installed from a ZIP? Fix it without starting over</b></summary>

Open a terminal in your Fizgig folder and run:

```bash
git init
git remote add origin https://github.com/shootthesound/Fizgig.git
git fetch --depth 1 origin master
git reset --hard FETCH_HEAD
git branch -M master
git branch --set-upstream-to=origin/master master
```

Your model paths, output LoRAs, caches, presets and the venv are left alone. `update_fizgig.bat` works normally from then on.

</details>

### Windows (NVIDIA, one-click)

Double-click `install_fizgig.bat`. It creates a venv, installs CUDA 12.8 PyTorch and all dependencies, pre-downloads the InsightFace models, and verifies CUDA is visible to PyTorch. Launch with `Launch Fizgig.bat` (its own window) or `Launch Fizgig in browser.bat` (in your web browser); update with `update_fizgig.bat`.

### Windows (AMD ROCm)

Install a full **Python 3.12** first: the ROCm bitsandbytes wheel is cp312-only, and Gizmo needs Tkinter. Do not use the embeddable zip. Install from [Windows downloads](https://www.python.org/downloads/windows/):

- **Recommended** — [Python Install Manager](https://www.python.org/downloads/latest/pymanager) from the [Microsoft Store](https://apps.microsoft.com/detail/9NQ7512CXL7T), then `py install 3.12`.
- **Alternative** — [python-3.12.10-amd64.exe](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe); tick **Add python.exe to PATH** and **tcl/tk and IDLE**.

Then double-click `install_fizgig_rocm.bat` (NVIDIA users never run this). It picks 3.12 via `py -3.12` / `python3.12`, not whatever `python` defaults to (e.g. 3.14). It detects the GPU, then installs pinned multi-arch wheels from **AMD ROCm nightlies** (`https://rocm.nightlies.amd.com/whl-multi-arch/`, not built by Fizgig):

- `torch==2.12.0+rocm7.15.0a20260728`
- `torchvision==0.27.0+rocm7.15.0a20260728`
- `rocm-sdk-devel==7.15.0a20260728`

Override with `TORCH_PIN` / `TORCHVISION_PIN` / `ROCM_SDK_DEVEL_PIN` if needed. **bitsandbytes** is a pinned community Windows ROCm wheel from [0xDELUXA/bitsandbytes_win_rocm](https://github.com/0xDELUXA/bitsandbytes_win_rocm), built by neither AMD nor Fizgig. Shared deps come from `requirements.txt` with CUDA `torch`/`bitsandbytes` and NVIDIA-only `nvidia-ml-py` filtered out (`filter_requirements_rocm.py`). Launch with `Launch Fizgig AMD.bat`; update with `update_fizgig_rocm.bat` — **not** `update_fizgig.bat`, which installs CUDA torch and would wipe the ROCm stack.

**`--experimental` (unsupported):** `install_fizgig_rocm.bat --experimental` installs unpinned `torch[device-ARCH]` / `torchvision[device-ARCH]` / `rocm-sdk-devel` from AMD's **whl-next** nightlies (`https://nightly.repo.amd.com/rocm/whl-next/`, the index comfyui-rocm uses for floating installs; the pinned TORCH_PIN wheels are not on it) and leaves `BNB_ROCM_VERSION` unset so bitsandbytes auto-selects its highest matching DLL. It is **not** the same as Linux `ROCM_CHANNEL=nightly` (which stays on the constrained 7.14 / bitsandbytes 714 lane), and not the default Windows multi-arch pin index. Local experimentation only. **Do not open GitHub issues for crashes, install failures, or training problems when `--experimental` was used** — those reports are not supported. Use the pinned install (no flag) for anything you expect help with.

### Linux (AMD ROCm — highly experimental)

Expect crashes, GPU resets, and incomplete model support on many setups. Best-effort only; Windows ROCm or NVIDIA Linux are the supported training paths. Prerequisites: amdgpu driver loaded (`/dev/kfd`), user in the `render`/`video` groups. See [Install ROCm](https://rocm.docs.amd.com/en/latest/install/rocm.html) and [PyTorch for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). Then:

```bash
chmod +x install_fizgig_rocm.sh
./install_fizgig_rocm.sh
./launch_fizgig_amd.sh
```

The script detects your gfx target (`detect_gpu_linux.py`). **Nightly is the Linux default**: the [TheRock multi-arch RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) index plus a `[device-gfx*]` extra for your GPU (e.g. `gfx1201` → `device-gfx1201`). Unpinned nightly resolves the latest **torch 2.12** + **ROCm 7.14.0a\*** stack (matches `libbitsandbytes_rocm714.so`). Override with `TORCH_PIN=…`, `ROCM_META_PIN=…`, or `TORCH_NIGHTLY_MINOR=…`.

**Stable** (repo.amd.com, no nightly alphas) pins **`torch==2.12.0+rocm7.14.0`** + **`rocm-sdk==7.14.0`** (cp310–cp314):

```bash
ROCM_CHANNEL=stable ./install_fizgig_rocm.sh
```

**torch 2.14** is nightly-only and can increase sampling VRAM pressure compared with 2.12:

```bash
ROCM_CHANNEL=nightly TORCH_NIGHTLY_MINOR=2.14 ./install_fizgig_rocm.sh
# or an explicit pin, e.g.:
# TORCH_PIN=2.14.0a0+rocm7.14.0a20260625 ROCM_CHANNEL=nightly ./install_fizgig_rocm.sh
# (paired torchvision ~0.29.0a0+rocm7.14.0a… — installer resolves the match)
```

The installer then adds shared deps from `requirements.txt` (filtered) and `bitsandbytes>=0.50.0` for ROCm.

Linux ROCm cache scripts import `fizgig.rocm.cache_exit` only when `FIZGIG_GPU_BACKEND=rocm` (set by `launch_fizgig_amd.sh`); NVIDIA and other platforms call `main()` directly. Opt out with `FIZGIG_ROCM_NO_FAST_EXIT=1 ./launch_fizgig_amd.sh`.

### Linux / macOS (NVIDIA CUDA)

`install_fizgig.py` is CUDA-only (captioning / image prep on macOS; training needs a CUDA or ROCm GPU). On AMD-only Linux hosts it points you to the ROCm installer and exits:

```bash
python install_fizgig.py
./launch_fizgig.sh            # its own window (./launch_fizgig_browser.sh: in your web browser)
```

Run it from your system Python, not a conda environment (`conda deactivate` first): conda bundles its own Tk, which can't see the system's fonts, so the app comes up with missing text. The installer refuses when conda is active; `FIZGIG_ALLOW_CONDA=1` overrides it.

### The VRAM meter on AMD

NVIDIA uses `pynvml` / `nvidia-smi`; the AMD readers (`vram_monitor.read_amd_gpu_vram`) run only as a fallback. Windows ROCm uses `typeperf`; Linux ROCm uses the **`amd-smi`** CLI when available ([AMD SMI / ROCm Core SDK](https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html), e.g. `sudo apt install amdrocm-amdsmi`), falling back to `rocm-smi`. Fizgig picks the GPU with the largest VRAM total and skips empty iGPU entries. Do not `pip install amdsmi` — the PyPI package is outdated.

### Helper models

Three small models download on first use: InsightFace `buffalo_l` (~300 MB, during install), Florence-2 (~500 MB–1.5 GB, first AI caption), and Helsinki-NLP `opus-mt-en-zh` (~300 MB, first bilingual translation). If you use the furry / anime face backends, two more download on first use (or with `fetch_models --family tools`): the deepghs anime head model (~40 MB) and DINOv2-base (~350 MB). See [face backends](FACE_BACKENDS.md).

## Model downloads

Fizgig doesn't bundle weights; download only the family you use. **Preferences has a ⬇ Download models for me button** under each model card that downloads, verifies, and fills in the paths. Klein and Flux 3 Image Klein need a free HuggingFace token for BFL's licence; Krea 2 needs no account. Every row also has a manual **Download** link. From the command line:

```bash
python -m fizgig.scripts.fetch_models --family krea2   # ~32 GB, no account needed
python -m fizgig.scripts.fetch_models --family klein   # ~34 GB, needs a token
python -m fizgig.scripts.fetch_models --family flux3 --include-optional   # ~31 GB, needs a token
python -m fizgig.scripts.fetch_models --family minimax # ~47 GB, no account needed
python -m fizgig.scripts.fetch_models --family qwen_image21 --include-optional   # ~38 GB, no account needed
python -m fizgig.scripts.fetch_models --family tools   # Florence-2, face model, translator
```

### Klein 9B

| Model | File | Size | Source |
|---|---|---|---|
| **Base DiT (fp8) — recommended** | `flux-2-klein-base-9b-fp8.safetensors` | ~9.5 GB | [black-forest-labs/FLUX.2-klein-base-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8) |
| Base DiT (bf16) | `flux-2-klein-base-9b.safetensors` | ~17 GB | [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| Distilled DiT | `flux-2-klein-9b-fp8.safetensors` | ~9 GB | [black-forest-labs/FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) |
| VAE / AE | `ae.safetensors` | ~320 MB | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) (from root, **not** the `vae/` subfolder) |
| Text Encoder | `qwen_3_8b.safetensors` | ~15 GB | [Comfy-Org/vae-text-encorder-for-flux-klein-9b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors) |

Training runs on the **Base DiT**; the fp8 version is recommended on every GPU (same quality, half the VRAM). The **Distilled DiT** powers the 4-step previews and the workbench. More in [KLEIN.md](KLEIN.md).

### Flux 3 Image Klein

All files live in [black-forest-labs/flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy) (gated: accept the licence on the model page first).

| Model | File | Size |
|---|---|---|
| **Base DiT (bf16) — training** | `flux3_image_klein_bf16.safetensors` | ~11.1 GB |
| **Distilled DiT (bf16) — previews and workbench** | `flux3_image_klein_sd_bf16.safetensors` | ~11.1 GB |
| Autoencoder | `flux2-vae.safetensors` | ~340 MB |
| Text Encoder | `qwen3vl_4b_bf16.safetensors` | ~8.9 GB |

Files with `_sd_` in the name are the distilled model; train on the one without. More in [FLUX3.md](FLUX3.md).

### Krea 2

All files live in the [**Comfy-Org/Krea-2**](https://huggingface.co/Comfy-Org/Krea-2) repo.

| Model | File | Size |
|---|---|---|
| **RAW DiT (bf16) — training** | `krea2_raw_bf16.safetensors` | ~26 GB |
| **Turbo DiT (fp8) — workbench** | `krea2_turbo_fp8_scaled.safetensors` | ~13 GB |
| Turbo LoRA *(auto-downloads)* | `krea2_turbo_lora_rank_64_bf16.safetensors` | ~470 MB |
| Qwen-Image VAE | `qwen_image_vae.safetensors` | ~250 MB |
| **Text Encoder — recommended** | `qwen3vl_4b_fp8_scaled.safetensors` | ~5.2 GB |
| Text Encoder — full precision | `qwen3vl_4b_bf16.safetensors` | ~8.9 GB |

The text-encoder slot is open: any Qwen3-VL-4B in the ComfyUI layout loads — fp8_scaled (recommended; we couldn't tell its captions apart from bf16's), bf16, or a community fine-tune/abliterated build, which changes how your dataset gets captioned. More in [KREA2.md](KREA2.md).

### MiniMax H3

The file list is in [MINIMAX_H3.md](MINIMAX_H3.md).

### Qwen Image 2.1

Use the **Download models for me** button in the Qwen Image 2.1 section of the Preferences tab: about 34 GB, plus Krea 2's ~5 GB Qwen3-VL-4B captioner if you don't already have it. Details in [QWEN_IMAGE.md](QWEN_IMAGE.md).

## VRAM guidance

### Klein 9B

**Training:** the fp8 Base stays resident at ~9.6 GB, so a 9B LoRA fits **16 GB** (~14 GB observed). On smaller cards, the **4-bit (NF4) base** (Training tab → Base precision, which Auto picks on cards under 16 GB) drops the base to ~5.6 GB; a full LoRA trains in about 8.5 GB at 0.5 MP, fitting **10–12 GB cards with no swap**.

**Workbench** (Distilled 4-step):

| Block Swap | Min VRAM |
|---|---|
| 0 | 16 GB+ with the fp8 Distilled model, 24 GB+ with bf16 |
| 8 | 16 GB (bf16) |
| 12 | 14 GB |
| 16 | 12 GB |

On first launch Fizgig detects your VRAM and picks the default (0 on 16 GB cards and up); your own choice sticks.

### Flux 3 Image Klein

Trains on cards down to **8 GB**. Auto picks bf16, INT8 or 4-bit NF4 from your free VRAM; a rank-16 LoRA at 1 MP peaked at 12.7 GB on bf16, 9.1 GB on INT8 and 6.2 GB on NF4. On small cards the training model steps off the GPU while previews render. See [FLUX3.md](FLUX3.md#memory-8-gb-and-up).

### Krea 2

| Your card | What to do |
|---|---|
| **8 GB** | Everything on **Auto**, batch size 1, stock preset defaults |
| **10–12 GB** | Same — headroom to raise batch size or resolution |
| **16 GB+** | Same — Auto will usually pick the faster INT8 path |

Auto budgets from your *free* VRAM and the console explains its choice. If a preview can't fit, previews disable themselves and **training keeps running and saving**.

### MiniMax H3

See [MINIMAX_H3.md](MINIMAX_H3.md) for the Auto table and preview limits per card.

### Qwen Image 2.1

Trains on cards down to **10 GB**. At the presets' 0.5 MP, Auto picks bf16 on 24 GB+, INT8 with no block swap on 12–16 GB, and 4-bit NF4 on 10 GB. See [QWEN_IMAGE.md](QWEN_IMAGE.md).

### 12 GB cards + previews (Windows): leave the paging file system-managed

Each preview parks the training model and optimizer in system RAM while the decoder runs. A fixed small paging file (e.g. 4 GB) can't cover that commit spike, and the run dies with **Windows error 1455** ("paging file is too small"), which says nothing about previews. Set Settings → System → About → Advanced system settings → Performance → Advanced → Virtual memory → *Automatically manage*. Reported and confirmed on a 12 GB RTX 5070 by [@mabseyuk](https://github.com/mabseyuk).

### Desktop feels juddery while training? (Windows)

Turn off **Hardware-accelerated GPU scheduling** (Settings → System → Display → Graphics → *Default graphics settings*), then reboot. With it off, Fizgig runs training at low priority so your desktop stays smooth; training speed is unaffected.
