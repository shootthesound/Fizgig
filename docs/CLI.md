# Fizgig Headless CLI

Everything the GUI does for training runs through `src/fizgig/families/cache.py` and `train.py` (plus the tools in `src/fizgig/scripts/`) — the GUI is a front-end that builds these exact commands and runs them as subprocesses. That means the CLI is always feature-complete for every family — Klein 9B, Flux 3 Image Klein, MOSS-Image, Krea 2, MiniMax H3, Qwen Image 2.1, Z-Image Turbo and Base, SDXL, Anima and ACE-Step: adaptive LR, the per-image loss watch, auto-recaptioning, Context LoRA, video and voice training, pause/resume — all of it is available from a plain terminal, on **Windows and Linux alike** (including display-less boxes).

All commands below are run from the repo root. The scripts add `src/` to `sys.path` themselves, so the direct form always works:

```bash
# Windows (bundled venv, cmd or PowerShell)
venv\Scripts\python.exe src\fizgig\families\train.py --help

# Linux / macOS
python src/fizgig/families/train.py --help
```

Every script supports `--help` for the full argument list. This document covers the workflow, the dataset config format, and the flags that matter.

**Windows notes** — the examples below are written in bash style for compactness; on Windows the flags are identical, only the shell dressing changes:

- Use `venv\Scripts\python.exe` instead of `python`.
- The trailing `\` at line ends is bash line-continuation. In **PowerShell** use a backtick `` ` `` at line ends, in **cmd** use `^` — or simply put the whole command on one line.
- Forward slashes are fine in all path arguments (`S:/models/ae.safetensors` works everywhere, including inside the dataset TOML — no need to escape backslashes).
- Quote any path containing spaces: `--dit "C:\my models\klein.safetensors"`.
- Where the docs say `touch <file>` (the pause sentinel), the Windows equivalent is `type nul > <file>` (cmd) or `New-Item <file>` (PowerShell).

---

## Contents

- [Model files: where they come from, where they go](#model-files-where-they-come-from-where-they-go)
- [What's family-specific at a glance](#whats-family-specific-at-a-glance)
- [The three-step pipeline](#the-three-step-pipeline)
- [Dataset config (TOML)](#dataset-config-toml)
- [Preparing images and captions](#preparing-images-and-captions)
- [Klein 9B training](#klein-9b-training)
- [Flux 3 Image Klein training](#flux-3-image-klein-training)
- [Fibo 2.0 training](#fibo-20-training)
- [MOSS-Image training](#moss-image-training)
- [Krea 2 training](#krea-2-training)
- [MiniMax H3 training](#minimax-h3-training)
- [Qwen Image 2.1 training](#qwen-image-21-training)
- [Z-Image Turbo and Z-Image Base training](#z-image-turbo-and-z-image-base-training)
- [SDXL training](#sdxl-training)
- [Anima training](#anima-training)
- [ACE-Step training (songs)](#ace-step-training-songs)
- [Masked training](#masked-training)
- [Start at a smaller size](#start-at-a-smaller-size)
- [Batch size](#batch-size)
- [Caption dropout](#caption-dropout)
- [Detail crops (style LoRAs)](#detail-crops-style-loras)
- [Keeping a subject to its trigger (DOP)](#keeping-a-subject-to-its-trigger-dop)
- [Sample previews during training](#sample-previews-during-training)
- [Pause and resume](#pause-and-resume)
- [VRAM guidance (block swap)](#vram-guidance-block-swap)
- [LoRA extraction (rank reduction)](#lora-extraction)
- [LoRA profiling](#lora-profiling)
- [Analyzing a per-image loss log](#analyzing-a-per-image-loss-log)
- [Captioning, translating and extracting from a fine-tune](#captioning-translating-and-extracting-from-a-fine-tune)
- [Training maps](#training-maps)
- [Other training flags](#other-training-flags)

---

## Model files: where they come from, where they go

Headless, there is no Preferences tab: **model locations are passed as flags on every command** (`--dit`, `--vae`, `--text_encoder`, `--speed_lora` for a family's Turbo / speed LoRA, `--training_adapter` for a training adapter, and for MiniMax H3's audio VAE `--aux audio_vae=PATH` when caching and `--family_option audio_vae=PATH` when training). The CLI does not read the GUI's `prefs.json` — put the paths in a shell script or Makefile once and forget about them. The files themselves are the same ones the GUI's Preferences tab links to:

**Klein 9B:**

| File | Download | Used for |
|---|---|---|
| Base DiT (fp8, recommended) | [FLUX.2-klein-base-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/tree/main) | training (`--dit`) |
| Base DiT (bf16, big cards) | [FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/tree/main) | training (`--dit`) |
| Distilled DiT (fp8) | [FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/tree/main) | fast 4-step previews (`--preview_checkpoint`) and the workbench tabs |
| VAE `ae.safetensors` | [FLUX.2-dev → ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) | `--vae` |
| Text encoder `qwen_3_8b.safetensors` | [Comfy-Org Klein text encoder](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors) | `--text_encoder` |

> **VAE trap:** use `ae.safetensors` from the FLUX.2-dev repo **root** — not `vae/diffusion_pytorch_model.safetensors`, which is the Diffusers-format file and will not load.

**Flux 3 Image Klein** (all in [flux-3-image-klein-comfy](https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/tree/main), gated):

| File | Download | Used for |
|---|---|---|
| Base DiT `flux3-image-klein-base-bf16.safetensors` | diffusion_models/ | training (`--dit`); quantised to INT8 or NF4 at load when `--precision` asks |
| Distilled DiT `flux3-image-klein-bf16.safetensors` | diffusion_models/ | 16-step previews (`--preview_checkpoint`) and the workbench tabs |
| Autoencoder `flux2-vae.safetensors` | vae/ | `--vae` (cache step: `--model`) |
| Text encoder `qwen3vl_4b_bf16.safetensors` | text_encoders/ | `--text_encoder` (cache step: `--model`) |

**Fibo 2.0** (all in [Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0/tree/main), none gated):

| File | Download | Used for |
|---|---|---|
| Dev DiT `fibo2_dev_bf16.safetensors` | diffusion_models/ | training (`--dit`); quantised to INT8 or NF4 at load when `--precision` asks. The 8.4 GB `fibo2_dev_int8_convrot.safetensors` there loads too (decoded at load) |
| Autoencoder `flux2-vae.safetensors` | vae/ | `--vae` (cache step: `--model`) |
| Text encoder `qwen3vl_4b_bf16.safetensors` | text_encoders/ | `--text_encoder` (cache step: `--model`) |
| Turbo LoRA `fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors` | loras/ | 4-step previews (`--speed_lora`, with `--sample_steps 4 --sample_cfg_scale 1`) |

**MOSS-Image** (none gated):

| File | Download | Used for |
|---|---|---|
| Base DiT `moss_image_base_bf16.safetensors` | [Comfy-Org/MOSS-Image → diffusion_models](https://huggingface.co/Comfy-Org/MOSS-Image/tree/main/diffusion_models) | training (`--dit`); quantised to INT8 or NF4 at load when `--precision` asks |
| Autoencoder `flux2-vae_small.safetensors` | [Comfy-Org/MOSS-Image → vae](https://huggingface.co/Comfy-Org/MOSS-Image/tree/main/vae): FLUX.2's encoder with the small decoder MOSS-Image ships with (Flux 3's `flux2-vae.safetensors` works too) | `--vae` (cache step: `--model`) |
| SFT DiT `moss_image_sft_int8_convrot.safetensors` (optional) | [Comfy-Org/MOSS-Image → diffusion_models](https://huggingface.co/Comfy-Org/MOSS-Image/tree/main/diffusion_models) | previews on the SFT model (`--preview_checkpoint`) |
| Turbo LoRA `8-steps-dmd-lora.safetensors` (optional) | [lodestones/Kromoss](https://huggingface.co/lodestones/Kromoss/blob/main/8-steps-dmd-lora.safetensors) | fast previews (`--speed_lora`, with `--sample_steps 12 --sample_cfg_scale 1`) on the Base, never on SFT |
| Text encoder `qwen3.5_9b_bf16.safetensors` | [Comfy-Org/Qwen3.5 → text_encoders](https://huggingface.co/Comfy-Org/Qwen3.5/tree/main/text_encoders) | `--text_encoder` (cache step: `--model`) |

**Krea 2:**

| File | Download | Used for |
|---|---|---|
| RAW DiT | [krea2_raw_bf16.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors) | training (`--dit`) |
| Turbo LoRA | [krea2_turbo_lora_rank_64_bf16.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/loras/krea2_turbo_lora_rank_64_bf16.safetensors) | 8-step previews (`--speed_lora`) |
| VAE `qwen_image_vae.safetensors` | [Krea-2 → vae](https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors) | `--vae` (cache step: `--model`) |
| Text encoder `qwen3vl_4b_fp8_scaled.safetensors` | [Krea-2 → text_encoders](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) | `--text_encoder` (cache step: `--model`); the bf16 file works too |
| Turbo DiT (fp8) | [krea2_turbo_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors) | GUI only: the Repair Studio and Royale previews |

**MiniMax H3:**

| File | Download | Used for |
|---|---|---|
| DiT, pruned int8 (~21 GB) | [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | training (`--dit`) — the same file ComfyUI runs |
| DiT, reference build *(optional)* | [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | `--dit` for LoRAs that live in the reference-to-video workflow, and for reference distillation (`--family_option distill=1`) |
| Text encoder, nvfp4 (~15.7 GB) | [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | `--text_encoder` (caching, and pre-encoding preview prompts) |
| Video VAE | [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors) | `--vae` |
| Audio VAE *(optional)* | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `--aux audio_vae=` (caching) and `--family_option audio_vae=` (training) — sound in clips, voice recordings, previews with sound |
| Turbo LoRA *(optional)* | [minimax_h3_turbo_v4_step600.safetensors](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/blob/main/minimax_h3_turbo_v4_step600.safetensors) | 6-step previews (`--speed_lora`) |
| Training adapter *(recommended)* | [minimax_h3_image_training_adapter.safetensors](https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter/blob/main/minimax_h3_image_training_adapter.safetensors) (Circlestone, the GUI default — one file for both bases) | `--training_adapter` |
| Training adapter, Ostris *(best for videos)* | [minimax_h3_training_adapter_v1.safetensors](https://huggingface.co/ostris/minimax_h3_training_adapter/blob/main/minimax_h3_training_adapter_v1.safetensors) (fl2va) · [ref2va file](https://huggingface.co/ostris/minimax_h3_training_adapter/blob/main/minimax_h3_ref2va_training_adapter_v1.safetensors) | `--training_adapter` — match it to `--dit` |

**Qwen Image 2.1:**

| File | Download | Used for |
|---|---|---|
| DiT (bf16) | [qwen_image_2.1_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/diffusion_models/qwen_image_2.1_bf16.safetensors) | training (`--dit`); quantised to INT8 or NF4 at load when `--precision` asks |
| VAE | [Qwen-Image-2.1 → vae/diffusion_pytorch_model.safetensors](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/vae/diffusion_pytorch_model.safetensors) | `--vae` (cache step: `--model`) — Qwen 2.1's own VAE, not the Krea 2 / Qwen-Image one |
| Text encoder `qwen3vl_8b_bf16.safetensors` | [Qwen-Image-2.1 → text_encoders](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/text_encoders/qwen3vl_8b_bf16.safetensors) | `--text_encoder` (cache step: `--model`); loaded 8-bit automatically on cards under ~20 GB free |
| Fizgig training adapter *(recommended)* | [fizgig_qwen_image_2.1_training_adapter_v2.safetensors](https://huggingface.co/ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter) | `--training_adapter` |
| Turbo LoRA | [qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/loras/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors) | fast previews on the model being trained (`--speed_lora`) |
| Turbo DiT (int8) | [qwen_image_2.1_turbo_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors) | the workbench tabs' previews; training previews with `--preview_checkpoint` |

**Z-Image Turbo** (`--family zimage`):

| File | Download | Used for |
|---|---|---|
| DiT `z_image_turbo_bf16.safetensors` | [Comfy-Org/z_image_turbo → diffusion_models](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/diffusion_models) | training (`--dit`) |
| VAE `ae.safetensors` | [Comfy-Org/z_image_turbo → vae](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/vae) | `--vae` (cache step: `--model`) |
| Text encoder `qwen_3_4b.safetensors` | [Comfy-Org/z_image_turbo → text_encoders](https://huggingface.co/Comfy-Org/z_image_turbo/tree/main/split_files/text_encoders) | `--text_encoder` (cache step: `--model`); loads 8-bit with less than 11 GB free |
| Fizgig training adapter *(recommended)* | [fizgig_z_image_turbo_training_adapter.safetensors](https://huggingface.co/ShootTheSound/Fizgig-Z-Image-Turbo-Training-Adapter) | `--training_adapter` |

**Z-Image Base** (`--family zimage_base`): the DiT is `z_image_bf16.safetensors` from [Comfy-Org/z_image → diffusion_models](https://huggingface.co/Comfy-Org/z_image/tree/main/split_files/diffusion_models); the VAE and text encoder are the same files as Turbo's, and the Turbo DiT above is `--preview_checkpoint` for Turbo previews. No training adapter: Base isn't distilled.

**SDXL** (`--family sdxl`): any SDXL checkpoint as `--dit`, its text encoders read from inside it (the GUI's default is [Juggernaut-XL v9](https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/blob/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors)); for the text cache step pass the same checkpoint as `--model`. A separate VAE, [sdxl_vae.safetensors](https://huggingface.co/stabilityai/sdxl-vae/blob/main/sdxl_vae.safetensors), is optional (`--vae`, cache step: `--model`).

**Anima** (`--family anima`, all in [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima/tree/main/split_files)):

| File | Download | Used for |
|---|---|---|
| DiT `anima-base-v1.0.safetensors` | diffusion_models/ | training (`--dit`) |
| VAE `qwen_image_vae.safetensors` | vae/ | `--vae` (cache step: `--model`) |
| Text encoder `qwen_3_06b_base.safetensors` | text_encoders/ | `--text_encoder` (cache step: `--model`) |
| Turbo LoRA *(optional)* | [anima-turbo-lora-v0.2.safetensors](https://huggingface.co/circlestone-labs/Anima-Official-LoRAs/blob/main/anima-turbo-lora-v0.2.safetensors) | fast previews (`--speed_lora`) |

**ACE-Step 1.5 XL** (`--family acestep15xl`, all in [Comfy-Org/ace_step_1.5_ComfyUI_files](https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files/tree/main/split_files)):

| File | Download | Used for |
|---|---|---|
| XL Turbo DiT `acestep_v1.5_xl_turbo_bf16.safetensors` | diffusion_models/ | training (`--dit`), the usual choice |
| XL SFT DiT `acestep_v1.5_xl_sft_bf16.safetensors` *(optional)* | diffusion_models/ | `--dit` to train on the non-distilled model, with `--family_option base=sft` |
| VAE `ace_1.5_vae.safetensors` | vae/ | `--vae` (cache step: `--model`) |
| Text encoder `qwen_0.6b_ace15.safetensors` | text_encoders/ | `--text_encoder` (cache step: `--model`) |

The quickest way to get every file is the fetcher the Preferences download button runs — `python -m fizgig.scripts.fetch_models --family qwen_image21 --include-optional` (from the repo root with `src` on `PYTHONPATH`). It also caches the tokenizer files so training works offline.

---

## What's family-specific at a glance

Every family trains through the same two scripts — `src/fizgig/families/cache.py` and `src/fizgig/families/train.py`, selected with `--family` (`klein`, `flux3`, `fibo2`, `moss_image`, `krea2`, `minimax`, `qwen_image21`, `zimage`, `zimage_base`, `sdxl`, `anima`, `acestep15xl`) — and shares the dataset format, pause/resume, Context LoRA, LoKR, sliders, fine-tuning and the memory planner. What differs:

| Feature | Klein 9B | Flux 3 Image Klein | MiniMax H3 | Krea 2 | Qwen Image 2.1 |
|---|---|---|---|---|---|
| Base precision (`--precision`) | auto / bf16 / int8 / nf4 | auto / bf16 / int8 / nf4 | auto / int8 / nf4 / hqq | auto / bf16 / int8 / nf4 | auto / bf16 / int8 / nf4 |
| Adaptive LR (`--adaptive_lr`) | ✅ | ✅ | ❌ | ✅ | ✅ |
| Per-image loss watch, per-image LR, auto-recaption, look-outlier warm-up | ✅ | ✅ | ❌ | ✅ | ✅ |
| Model Area / block targeting (`--train_blocks`) | ✅ (identity, style, details areas) | ❌ | ✅ (Blocks to Train, Training mode) | ❌ | ❌ |
| Timestep range (`--min/max_timestep`, 0-1000) | ✅ | ✅ | ❌ | ✅ | ✅ |
| torch.compile (`--compile_blocks auto`) | ✅ | ✅ | ❌ | ✅ | ✅ |
| Edit LoRAs (original + edited pairs) | ✅ | ✅ | ❌ | ❌ | ✅ |
| Photos, video clips with sound, voice | photos | photos | all three | photos | photos |

The table covers the families with the longest history; Z-Image, SDXL, Anima and ACE-Step have their own sections below. Masked training (`--masked_training`), caption dropout (`--caption_dropout`) and Start at a smaller size (`--stage_dataset_config`) work on every picture family. Auto-recaption captions with Krea 2's Qwen3-VL-4B, the same model the GUI's Captions tab uses, whichever model you're training (`--captioner`). MiniMax H3's own settings (training structure, Training mode, TREAD, clip stills, distillation, retirement of a finished category) reach the trainer as `--family_option name=value` pairs; see [MiniMax H3 training](#minimax-h3-training).

---

## The three-step pipeline

Training is always three steps: cache the VAE latents, cache the text-encoder outputs, then train. Latents and text embeddings are computed once and reused across epochs (and across runs, if the dataset hasn't changed).

**Klein 9B:**

```bash
python src/fizgig/families/cache.py --family klein --stage latents --dataset_config my_dataset.toml --model /models/ae.safetensors
python src/fizgig/families/cache.py --family klein --stage text    --dataset_config my_dataset.toml --model /models/qwen_3_8b.safetensors
python src/fizgig/families/train.py --family klein --dataset_config my_dataset.toml ...   # full example below
```

**Krea 2:**

```bash
python src/fizgig/families/cache.py --family krea2 --stage latents --dataset_config my_dataset.toml --model /models/qwen_image_vae.safetensors
python src/fizgig/families/cache.py --family krea2 --stage text    --dataset_config my_dataset.toml --model /models/qwen3vl_4b_fp8_scaled.safetensors
python src/fizgig/families/train.py --family krea2 --dataset_config my_dataset.toml ...   # full example below
```

**MiniMax H3:**

```bash
python src/fizgig/families/cache.py --family minimax --stage latents --dataset_config my_dataset.toml --model /models/minimax_h3_video_vae_fp16.safetensors --aux audio_vae=/models/minimax_h3_audio_vae_fp32.safetensors --aux clip_still=1
python src/fizgig/families/cache.py --family minimax --stage text    --dataset_config my_dataset.toml --model /models/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
python src/fizgig/families/train.py --family minimax --dataset_config my_dataset.toml ...   # full example below
```

**Qwen Image 2.1** (one cache script and one trainer, both taking `--family qwen_image21`):

```bash
python src/fizgig/families/cache.py --family qwen_image21 --stage latents --dataset_config my_dataset.toml --model /models/qwen_image_2.1_vae_diffusers.safetensors
python src/fizgig/families/cache.py --family qwen_image21 --stage text    --dataset_config my_dataset.toml --model /models/qwen3vl_8b_bf16.safetensors
python src/fizgig/families/train.py --family qwen_image21 --dataset_config my_dataset.toml ...   # full example below
```

Every other family follows the same pattern with its own VAE and text encoder from the [download table](#model-files-where-they-come-from-where-they-go): `--family zimage`, `zimage_base`, `sdxl` (the checkpoint itself as the text step's `--model`), `anima`, `flux3`, `moss_image`, `acestep15xl`.

Re-running a cache step is cheap: pass `--skip_existing` to only encode new images. Stale cache files for images that were removed from the dataset are deleted automatically (pass `--keep_cache` to keep them, but see the warning in the next section).

---

## Dataset config (TOML)

The dataset config is a small TOML file. The GUI writes one for each run (in `user_data/dataset/`); headless you write it yourself:

```toml
[general]
resolution = [1024, 1024]     # target training resolution (see megapixel note below)
caption_extension = ".txt"    # caption sidecar extension
batch_size = 1
num_repeats = 1               # times each image is seen per epoch
enable_bucket = true          # multi-aspect-ratio bucketing (recommended)
bucket_no_upscale = true      # never upscale images smaller than the target

[[datasets]]
image_directory = "/data/my_subject/images"
cache_directory = "/data/my_subject/cache"
```

Field notes:

- **`resolution`** — `[width, height]`. With bucketing enabled this is the *area* target: each image is assigned to the nearest aspect-ratio bucket of roughly `width × height` pixels, so mixed portrait/landscape/square datasets train at their natural aspect ratios. `[1024, 1024]` ≈ 1 MP is the sweet spot for both Klein and Krea 2; `[768, 768]` trains faster on smaller cards at some quality cost.
- **`batch_size`** — 1 trains one picture a step, the default. Families that support it take more; see [Batch size](#batch-size). For a larger effective batch at the memory of batch 1, keep 1 and pass `--gradient_accumulation_steps N` to the trainer.
- **`num_repeats`** — multiplies how often each image appears per epoch. Leave at 1 and train more epochs instead, unless you're balancing multiple `[[datasets]]` blocks against each other.
- **`cache_directory`** — where latents and text embeddings are stored. **Give every dataset its own cache directory.** The training set is built from the cache, and a shared cache directory can mix a previous dataset's images into your run. (Fizgig cross-checks the cache against `image_directory` and skips orphans with a warning, but a dedicated directory per dataset is the clean way.)
- **Multiple `[[datasets]]` blocks** are supported — each with its own `image_directory`, `cache_directory`, and optional per-dataset overrides of any `[general]` key (e.g. a different `num_repeats` to weight one folder more heavily).
- **`caption_replace = [["zwxem, a woman", "zwxemwoman"]]`** — rewrites captions as they're read (whole words, any case, in order); your `.txt` files aren't changed. Fizgig LoRA Isolation uses it to join each trigger to its class.
- **`photos_only = true`** — a family that trains clips or voice (MiniMax H3) reads only the pictures in this folder. Fizgig LoRA Isolation sets it on H3, which trains people from their photos.
- **`is_reg = true`** — marks a fine-tune's regularisation set: those images train at `--reg_lr_multiplier` (default 0.2) times the learning rate, as a light anchor on the class. A LoRA run trains such a block as ordinary images and says so in the log.

If you change captions, re-run the text cache step. If you add/remove/edit images, re-run both cache steps (a picture edited after it was cached is encoded again even with `--skip_existing`).

---

## Preparing images and captions

### Images

A dataset is just a folder of images with caption sidecars — no manifest, no subfolder structure (each folder is read at its top level; the web UI's **Also train on its sub-folders** writes one `[[datasets]]` block per sub-folder, with `item_prefix = "<path inside the training folder>/"` so the loss watch keys each picture by its folder). Accepted formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` (plus `.jxl` if jxlpy is installed). You do **not** need to pre-resize or crop to a fixed size: with `enable_bucket = true` the loader assigns each image to its nearest aspect-ratio bucket at the target megapixel area, and `bucket_no_upscale = true` keeps undersized images at their native resolution instead of upscaling them. Aim for source images at or above ~1 MP; a mix of framings (close-up, half-body, full-body, varied backgrounds) trains better identity than 40 near-identical headshots.

The GUI's Image Prep tab (batch resize, PNG conversion, face-crop detection) is convenience, not requirement — anything that produces ordinary image files works.

**Likeness at 0.25 MP: face crops are what make it work.** Training defaults to 0.25 MP (a 512×512 area in whatever aspect the image has), and the VAE compresses a further 8× on each axis — so a face that fills ~80 px of a full-body shot reaches the model as roughly 10×10 latent pixels, carrying almost no identity signal. A face crop of the same photo gives that face the entire frame instead — about **40× the face area** for the model to learn from. So if likeness is coming out soft, the first fix isn't raising the megapixels: add tight face crops alongside the wider shots (Image Prep's face-crop mode makes them in one pass). The crops feed identity, the wide shots teach body and context, and you keep 0.25 MP's fast steps. Raising the MP target helps too but costs speed and VRAM on every image — crops put the resolution only where the identity lives.

### Caption modes

One `.txt` file per image, same basename (`portrait_012.png` → `portrait_012.txt`), plain text. The GUI's Captions tab offers three modes; all three produce plain sidecar files you can equally write yourself headless:

1. **Trigger word only** — every caption is just `ohwx`. Fastest to make, and workable for single-concept datasets, but the model has to absorb *everything* in each image into the trigger, backgrounds included.
2. **Trigger + description** (recommended) — trigger first, then describe the image: `ohwx man, close-up portrait, grey backdrop, soft window light`. What you describe, the model can separate out; what you leave silent gets baked into the trigger. Headless, any captioner works (the GUI uses Florence-2) — or write them by hand for small sets.
3. **Bilingual English + Chinese** — each caption becomes `<trigger>, <english> - <chinese>`. Empirically improves Klein visual quality at identical loss (both Klein's Qwen3 and Krea 2's Qwen3-VL have deep Chinese training). The GUI translates with Helsinki-NLP MT; headless, any MT model works — keep the trigger word verbatim, translate only the description.

Captioning rules that come from real runs:

- **Caption anything the model's prior would call a lie** — viewpoint especially. A profile shot captioned like a frontal portrait teaches the model to fight itself; caption it `..., profile view from the left`. On Krea 2 this matters doubly: caption-viewpoint mismatches are the single most common thing the per-image loss watch convicts.
- That rule includes the trigger itself: if the subject isn't actually recognizable in a shot (back of head, extreme distance), consider leaving the trigger out of that caption.
- **For a style dataset, caption the contents and never the style itself.** Describe what's in each image — the subject, the setting, the colours of the things themselves — and say nothing about medium, brushwork, texture, colour grade or lighting. Then the trigger word is the only thing every caption has in common, which is exactly what you want the look to bind to. Lighting is the one people get wrong: caption it and the style only fires under the lighting it saw. Describe generously — richer captions account for more of the image and leave the look as the cleaner remainder. The GUI's *Style* preset does this for you.
- Changed captions require re-running the text-cache step. (During a Krea 2 run, `--auto_recaption` handles stuck images' captions for you, including the re-encode.)

### Dataset sidecar files (Krea 2 intelligence)

Two JSON files can live alongside the images and travel with the dataset:

- `fizgig_look_scores.json` — written by the GUI's Look Consistency Filter scan (ArcFace similarity of every image to 3 baseline picks). `--warmup_look_outliers` reads it to give real-but-unusual images a gentle LR ramp. Keys are image basenames, so the file survives the folder being moved or copied. There's no headless generator for it yet — run the Look Filter once in the GUI, or skip the flag.
- `fizgig_excluded.json` — the per-image watch's record of images it excluded, per model family. A later run of the same family still trains them from the start (they teach the run until they get stuck), but their two AI recaptions are spent, so if one is confirmed stuck again it is excluded at once. Editing an image's caption clears its record. Delete the file to give everything a clean slate.

---

## Klein 9B training

You need the Klein model files from the [download table](#model-files-where-they-come-from-where-they-go) above (Base DiT, `ae.safetensors`, `qwen_3_8b.safetensors`, and the Distilled DiT for fast previews).

### Full example

A realistic identity LoRA run (rank 16, the Identity model area, adaptive LR, per-epoch checkpoints), as the GUI builds it:

```bash
python src/fizgig/families/train.py --family klein \
  --dataset_config my_dataset.toml \
  --dit /models/flux-2-klein-base-9b-fp8.safetensors \
  --vae /models/ae.safetensors \
  --text_encoder /models/qwen_3_8b.safetensors \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject \
  --network_dim 16 --network_alpha 16 \
  --learning_rate 1e-4 \
  --adaptive_lr --adaptive_lr_min 5e-5 --adaptive_lr_max 4e-4 \
  --max_train_epochs 55 --save_every_n_epochs 1 --seed 42 \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --precision auto --blocks_to_swap -1 --compile_blocks auto \
  --optimizer_type adamw8bit \
  --train_blocks single_1,single_2,single_3,single_4,single_5,single_6,single_7,single_8,single_9,single_10,single_11,single_12,single_13,single_14,single_15,single_16 \
  --sample_prompts sample_prompts.txt --sample_every_n_epochs 1 --sample_at_first \
  --preview_checkpoint /models/flux-2-klein-9b-fp8.safetensors
```

Checkpoints land as `output_loras/my_subject/my_subject-000001.safetensors` (epoch 1) etc., with the final LoRA as `my_subject.safetensors` — all directly loadable in ComfyUI, no conversion.

### The flags that matter

**Precision / VRAM** — `--precision auto --blocks_to_swap -1` reads the free VRAM at launch: INT8 when it compiles, else the Base file as it is (BFL's fp8 file stays fp8), NF4 on small cards, swapping blocks only when nothing else fits. Set either by hand to override. Gradient checkpointing turns itself off on big cards where the run fits (see `--gradient_checkpointing` under [Other training flags](#other-training-flags)).

**LoRA shape** — `--network_dim` / `--network_alpha` (the GUI presets use matched pairs: 4/4 style and details, 8/8, 16/16 identity); `--network_type lokr --lokr_factor 8` for LoKR.

**Learning rate** — `--learning_rate`, or `--adaptive_lr` (the bi-directional plateau tracker: probes LR up on steady descent, cuts it with weight rollback on plateau or instability). Bounds via `--adaptive_lr_min` (5e-5 for single-subject sets, 1e-4 for noisy multi-subject sets) and `--adaptive_lr_max` (4e-4 is the empirical ceiling). **`--learning_rate` is ignored while adaptive is on**: the run starts at the geometric midpoint of the window and the watcher owns the LR from there. If it has to cut the rate within three epochs of raising it, the raised rate counts as too high for that run: later raises stop one step below it (the log says "ceiling … learned"), and a resume keeps it. Without it, `--lr_scheduler` / `--lr_warmup_steps` apply.

**Training only part of the model** (the GUI's Model Area) — `--train_blocks` takes block ids (`double_0`-`double_7`, `single_0`-`single_23`) and `--min_timestep` / `--max_timestep` (0-1000) the noise range:

| Model Area | `--train_blocks` | Timesteps |
|---|---|---|
| Full Model | *(omit)* | *(omit)* |
| Identity | `single_1` … `single_16` | — |
| Style | `double_0` … `double_7`, `single_0`, `single_1` | `--min_timestep 0 --max_timestep 400` |
| Style+Composition | `double_0` … `double_7`, `single_0`, `single_1` | — |
| Details | `single_12` … `single_23` | — |

Style genuinely lives at late timesteps (0-400) on Klein — the style blocks with that range is the validated recipe for style LoRAs.

**Previews** — `--preview_checkpoint` renders on the Distilled DiT (4-step, matches ComfyUI) with the training model parked beside it; without it previews render on the Base at `--sample_steps 40 --sample_cfg_scale 4.5`. `--sample_reference FILE` makes every preview an edit of that photo.

**Context LoRA** — `--context_lora_path FILE --context_lora_strength 1.0` trains on top of an existing LoRA, frozen and active; deploy the pair together at the same strength. Accepts kohya, PEFT/Diffusers, OneTrainer, LoKR and LoHa formats.

**Checkpoints / state** — `--save_state` writes resumable state dirs (`<name>-NNNNNN-state/`, NNNNNN = epochs done); `--save_state_on_train_end` lets a finished run be trained further; `--keep_last_n_states N` keeps only the newest N; `--resume <state dir>` continues (see [Pause and resume](#pause-and-resume)).

---

## Flux 3 Image Klein training

Flux 3 trains through the shared cache script and trainer with `--family flux3`. You need the four files from the [download table](#model-files-where-they-come-from-where-they-go); the distilled DiT is optional but is what makes previews quick. The dataset TOML is the same as the other families'. 8 GB is the smallest card.

### Full example

The GUI's **Ultra Fast** preset, as it builds it — rank 16, 0.25 MP (`resolution = [512, 512]` in the TOML), Adaptive LR 1e-4 to 2e-4, 30 epochs:

```bash
python src/fizgig/families/cache.py --family flux3 --stage latents --dataset_config my_dataset.toml --model /models/flux2-vae.safetensors
python src/fizgig/families/cache.py --family flux3 --stage text    --dataset_config my_dataset.toml --model /models/qwen3vl_4b_bf16.safetensors

python src/fizgig/families/train.py \
  --family flux3 \
  --dataset_config my_dataset.toml \
  --dit /models/flux3-image-klein-base-bf16.safetensors \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject \
  --network_dim 16 --network_alpha 16 \
  --learning_rate 1e-4 \
  --adaptive_lr --adaptive_lr_min 1e-4 --adaptive_lr_max 2e-4 \
  --max_train_epochs 30 \
  --save_every_n_epochs 1 \
  --seed 42 \
  --optimizer_type adamw8bit \
  --precision auto --blocks_to_swap -1 --compile_blocks auto \
  --ema_decay 0.98 \
  --log_per_image_loss --per_image_lr \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --vae /models/flux2-vae.safetensors \
  --text_encoder /models/qwen3vl_4b_bf16.safetensors \
  --preview_checkpoint /models/flux3-image-klein-bf16.safetensors \
  --sample_prompts sample_prompts.txt \
  --sample_every_n_epochs 1 --sample_at_first \
  --sample_width 1024 --sample_height 1024 --sample_seed 1234
```

**Standard** is `--network_dim 32 --network_alpha 32 --max_train_epochs 64` with no `--adaptive_lr` flags (a flat 1e-4). **Style** is `--network_dim 16 --network_alpha 16 --adaptive_lr_min 5e-5 --adaptive_lr_max 2e-4 --max_train_epochs 64`.

### The flags that matter

- `--precision auto|bf16|int8|nf4` and `--blocks_to_swap -1` — as for Qwen Image 2.1 below: the most precise base that fits your free VRAM, quantised before it swaps.
- `--preview_checkpoint FILE` — previews on the distilled model (16 steps, no CFG). Without it they render on the Base at 50 steps with `--sample_cfg_scale 3.5` against an empty negative, as BFL's reference does.
- **Edit LoRAs**, **slider LoRAs**, **fine-tuning**, `--context_lora_path` and the per-image loss watch work as for [Qwen Image 2.1](#qwen-image-21-training); fine-tuning starts at 5e-5 in the GUI. `--sample_reference FILE` makes every preview an edit of that photo.
- `--train_blocks` and Model Area presets aren't offered yet: Flux 3's block map isn't charted.

The saved LoRA loads in ComfyUI's standard LoRA loader.

---

## Fibo 2.0 training

Fibo 2.0 trains through the shared cache script and trainer with `--family fibo2`. The four files are in the [download table](#model-files-where-they-come-from-where-they-go); none needs a token. See [the Fibo 2.0 guide](FIBO2.md) for the presets, memory and ComfyUI.

### Full example

The GUI's **Fibo 2.0 (rank 32, full model)** preset, as it builds it: rank 32, 0.25 MP (`resolution = [512, 512]` in the TOML), Automagic from 1e-6, 40 epochs, with Turbo previews:

```bash
python src/fizgig/families/cache.py --family fibo2 --stage latents --dataset_config my_dataset.toml --model /models/flux2-vae.safetensors
python src/fizgig/families/cache.py --family fibo2 --stage text    --dataset_config my_dataset.toml --model /models/qwen3vl_4b_bf16.safetensors

python src/fizgig/families/train.py \
  --family fibo2 \
  --dataset_config my_dataset.toml \
  --dit /models/fibo2_dev_bf16.safetensors \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject \
  --network_dim 32 --network_alpha 32 \
  --optimizer_type automagic3 --learning_rate 1e-6 \
  --max_train_epochs 40 --save_every_n_epochs 1 --seed 42 \
  --precision auto --blocks_to_swap -1 --compile_blocks auto \
  --ema_decay 0.98 --max_grad_norm 1.0 \
  --vae /models/flux2-vae.safetensors \
  --text_encoder /models/qwen3vl_4b_bf16.safetensors \
  --speed_lora /models/fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors --speed_lora_strength 1 \
  --sample_prompts sample_prompts.txt --sample_steps 4 --sample_cfg_scale 1 \
  --sample_every_n_epochs 5 --sample_at_first --sample_width 1024 --sample_height 1024 --sample_seed 1234
```

Without `--speed_lora`, previews render on the Dev model: `--sample_steps 30 --sample_cfg_scale 5`.

### The flags that matter

- `--precision auto|bf16|int8|nf4` and `--blocks_to_swap -1`: as for Qwen Image 2.1 below.
- **Edit LoRAs** (`control_directory` in the TOML; the caption is the edit instruction), **slider LoRAs** (`--slider_pairs`, `--slider_prompts`), **masked training** (`--masked_training`, `--mask_prior`) and **fine-tuning** (`--finetune` with `--ft_rotations`, `--ft_rotate_every`; the GUI's preset uses 1e-5, 3 rotations, 2 epochs a part) work as on [Qwen Image 2.1](#qwen-image-21-training).
- `--train_blocks` takes the block map's ids (`layers.0` ... `layers.43`, `noise_refiner.0/1`, `perceiver.layers.0/1`, `x_embedder`, `t_embedder`, `final_layer`).
- Every saved file (LoRA, Studio save, Extract, fine-tune checkpoint) is in ComfyUI's layout and loads there as it is.

---

## MOSS-Image training

MOSS-Image trains through the shared cache script and trainer with `--family moss_image`. You need the three files from the [download table](#model-files-where-they-come-from-where-they-go); the dataset TOML is the same as the other families'. Previews run res_multistep for 30 steps with CFG 4. The LoRA trains every Linear in all 42 blocks, block 22 included; with MOSS-Image SFT or a fine-tune of it as `--dit`, add `--family_option lora_sft=1` so it trains attention only (the GUI's Train attention only; it also lowers a fine-tune's VRAM on Base, at a possible cost in quality). `--family_option lownoise_pct=40` sets the clean-end share (a single value; leave it out for the model's own schedule, about 11%); `lownoise_pct=60:7:30:8:3` is the GUI's Slide (the default), 60% until epoch 7, then sliding in a straight line to 30% by epoch 8 (here one step), and after that 60% with every 3rd epoch at 30%, at half the learning rate (the every-N epochs take the lower share, whichever comes first; leave off `:3` for none); `60:7:30:15:2` slides over eight epochs instead; `lownoise_pct=75:5:40` drops at once, 75% for the first 5 epochs, then 40%. Train on the Base: a LoRA trained on Kromoss began to overbake after about 750 steps in our test. With SFT as `--preview_checkpoint`, `--speed_lora` (the Turbo LoRA) is left out of the previews; it joins previews on the Base.

### Full example

The **Fast** preset, as the GUI's planner builds it: rank 16, 0.25 MP (`resolution = [512, 512]` in the TOML), LR 1e-4, 70 epochs, clean-end share Slide (60% until epoch 7, 30% at epoch 8, then 60% with every 3rd epoch at 30%):

```bash
python src/fizgig/families/cache.py --family moss_image --stage latents --dataset_config my_dataset.toml --model /models/flux2-vae_small.safetensors
python src/fizgig/families/cache.py --family moss_image --stage text    --dataset_config my_dataset.toml --model /models/qwen3.5_9b_bf16.safetensors

python src/fizgig/families/train.py \
  --family moss_image \
  --dataset_config my_dataset.toml \
  --dit /models/moss_image_base_bf16.safetensors \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject \
  --network_dim 16 --network_alpha 16 \
  --learning_rate 1e-4 \
  --max_train_epochs 70 \
  --save_every_n_epochs 1 \
  --seed 42 \
  --optimizer_type adamw \
  --precision auto --compile_blocks auto \
  --ema_decay 0.98 \
  --family_option lownoise_pct=30:7:60:8:3 \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --vae /models/flux2-vae_small.safetensors \
  --text_encoder /models/qwen3.5_9b_bf16.safetensors \
  --sample_prompts sample_prompts.txt \
  --sample_every_n_epochs 1 \
  --sample_width 1024 --sample_height 1024 --sample_steps 30 --sample_cfg_scale 4.0 \
  --sample_negative "worst quality, bad anatomy, watermark, jpeg artifact"
```

See [MOSS-Image](MOSS_IMAGE.md) for the presets and preview settings.

---

## Krea 2 training

Krea 2 (12.9B) trains through Fizgig's driver system — the same cache script and trainer as Qwen Image 2.1 (`src/fizgig/families/`), selected with `--family krea2` — and is the home of the intelligent-trainer features: the per-image loss watch, auto-recaptioning, per-image adaptive LR and look-outlier warm-up started here. You need the files from the [download table](#model-files-where-they-come-from-where-they-go): the RAW DiT for training, the Qwen-Image VAE, the Qwen3-VL-4B text encoder (which doubles as the vision model for auto-recaptioning), and the Turbo LoRA for 8-step previews.

### Full example

The GUI's default **Ultra Fast** preset as it builds it — rank 8, Adaptive LR 2e-4 to 4e-4, EMA 0.98, the loss watch with per-image LR, the base precision and block swap planned from free VRAM:

```bash
python src/fizgig/families/train.py \
  --family krea2 \
  --dataset_config my_dataset.toml \
  --dit /models/krea2_raw_bf16.safetensors \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject \
  --network_dim 8 --network_alpha 8 \
  --learning_rate 1e-4 \
  --adaptive_lr --adaptive_lr_min 2e-4 --adaptive_lr_max 4e-4 \
  --max_train_epochs 30 \
  --save_every_n_epochs 1 \
  --seed 42 \
  --optimizer_type adamw8bit \
  --precision auto --blocks_to_swap -1 \
  --ema_decay 0.98 \
  --log_per_image_loss --per_image_lr \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --vae /models/qwen_image_vae.safetensors \
  --text_encoder /models/qwen3vl_4b_fp8_scaled.safetensors \
  --speed_lora /models/krea2_turbo_lora_rank_64_bf16.safetensors \
  --sample_prompts sample_prompts.txt \
  --sample_every_n_epochs 1 --sample_at_first \
  --sample_width 1024 --sample_height 1024 --sample_steps 8 --sample_seed 1234
```

Add `--auto_recaption --captioner /models/qwen3vl_4b_fp8_scaled.safetensors --trigger_word ohwx` to let stuck images' captions be rewritten, and `--warmup_look_outliers` once you've run the GUI's Look Filter on the dataset (see below).

**Classic-recipe variant** — if you'd rather drive the LR yourself than hand it to the adaptive watcher: a cosine decay with warmup, and an effective batch of 2 via accumulation. Swap these lines into the run above, dropping `--adaptive_lr*` (the watcher and a schedule are mutually exclusive — adaptive wins, and the schedule is ignored with a log line):

```bash
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1.0 \
```

Pause/resume continues the cosine curve where it left off rather than restarting it, with or without accumulation.

The non-obvious flags (`--help` lists them all):

**Core**

- `--precision auto|int8|nf4|bf16` — the frozen base. `auto` reads free VRAM at launch and your training resolution and picks INT8 where it fits, else NF4, and logs the choice as an `Auto plan` line; bf16 is a manual choice. INT8 is the fastest and ~7× more accurate than NF4 in forward error; NF4 (~5.6 GB base) fits 10-12 GB cards. The CLI default is `bf16`, so pass `auto` to get the GUI's behaviour.
- `--blocks_to_swap N` — blocks streamed between CPU and GPU; `-1` plans the fewest that fit. **Swapping is the slow path** (4.4× the time, 4× the CPU): Auto quantises first and only swaps when even NF4 will not fit. NF4 cannot swap.
- `--network_type lokr` + `--lokr_factor N` — train **LoKR** (LyCORIS Kronecker) instead of standard LoRA (the GUI's Network Type dropdown, headless here). One dial: the factor sets the Kronecker split, and dim/alpha are ignored. Lower factor ≈ more capacity and bigger files (factor 8 ≈ 400 MB, 16 ≈ 100 MB); **8 is the validated default, and going above it isn't worth it** — measured head-to-head, higher factors keep LoKR's ~20% step-time cost over standard LoRA while losing the quality edge that justifies it. Want smaller/faster? Use standard LoRA at low rank instead. Output is standard LyCORIS format — loads directly in ComfyUI and back into every Fizgig tool. In our validation runs LoKR at factor 8 hit the highest likeness we've ever measured, with noticeably more natural skin than standard LoRA on the same data.
- `--compile_blocks auto|on|outside|off` — torch.compile the transformer blocks (`outside` = the high-resolution boundary: checkpoint kept outside the compiled region, eager-level memory, chosen automatically by `auto`/`on` where inside-the-graph would not fit): roughly **2× faster steady-state steps** on the INT8 path after a one-off warm-up (~90 s). Each block is compiled for any sequence length from the start, so new picture sizes and caption lengths don't rebuild it; a run that alternates passes (DOP, Fizgig LoRA Isolation) settles within the first epoch. `auto` (default) weighs the warm-up against the run length and only compiles when it pays. Needs triton (installs with requirements; `triton-windows` on Windows) and, on Windows, the MSVC C++ Build Tools — direct installer: https://aka.ms/vs/17/release/vs_BuildTools.exe ("Desktop development with C++" workload). Missing either → a console note and the run continues uncompiled.
- `--ema_decay 0.98` — weight averaging: checkpoints and previews come from a smoothed average of the weights. `0` (the CLI default) turns it off.
- `--min_timestep` / `--max_timestep` — restrict training to a noise band, on a 0-1 scale.
- `--preview_lora FILE` — a LoRA on previews of the training model only, never in training (MOSS-Image's SFT look LoRA, `--preview_lora_strength 1.5`); on CPU between previews, and a file that isn't there yet is picked up at a later preview.
- `--speed_lora FILE` — the Turbo LoRA, attached for 8-step previews on the training model and absent from the saved LoRA; `--speed_lora_strength` defaults to 1; `0` renders previews without it until a live preview change asks for a strength, when it loads at the next preview.

**The per-image loss watch** (any of these enables the watcher)

- `--log_per_image_loss` — tracks every image's loss residual against its timestep bucket, classifies each image at every epoch boundary (`easy` / `suspect` / `stuck` / `exhausted` / `excluded`), and writes:
  - `<output_dir>/loss_log/per_image_loss.jsonl` — the raw per-step log
  - `<output_dir>/loss_log/problem_images.json` — current verdicts + trends (the per-image loss view in Runs & outputs reads this; it's plain JSON, perfectly greppable headless)
  - a **plateau banner** in the console when ≤~5% of images are still improving, with a best-checkpoint epoch estimate — your "you're done" signal.
- `--per_image_lr` — acts on the verdicts: stuck images get throttled (×0.5, then ×0.25), mined-out images ease off (×0.6), the healthy cohort gets a gentle boost (×1.1). It is a true per-image learning rate at any Batch size: in a batch, each picture's loss is weighed on its own.
- `--auto_recaption` — between epochs, confirmed-stuck images get their captions rewritten by Qwen3-VL from what's actually visible, the text cache is re-encoded, and the image gets a fresh start. Two failed attempts and a still-stuck image is excluded from the run entirely. Requires `--text_encoder`. Pass `--trigger_word` so rewritten captions keep your trigger (appended as `, <trigger>`).
- `--warmup_look_outliers` — curriculum entry (×0.4 LR ramping to ×1.0) for real-but-unusual images. Reads `<dataset>/fizgig_look_scores.json`, which is produced by the GUI's Look Consistency Filter scan — **GUI-only prerequisite**; without the file this flag logs a warning and disables itself.

Persistent artifacts: exclusions are recorded per model family in `<image_directory>/fizgig_excluded.json`; a later run of that family trains the image normally and excludes it at once if it gets stuck again (no new recaptions). Editing an excluded image's caption clears the record. Fresh runs rotate the old JSONL to `.bak`; `--resume` replays the log to restore full watch history.

**LR schedule and batching**

- `--lr_scheduler` — `constant` (default), `constant_with_warmup`, `cosine`, `cosine_with_restarts`, `linear`, `polynomial`; with `--lr_warmup_steps` (plus `--lr_scheduler_num_cycles` / `--lr_scheduler_power` for the two that use them). **Ignored when `--adaptive_lr` is on** — the plateau watcher owns the LR and says so in the log. Resume continues the curve rather than restarting it.
- `--gradient_accumulation_steps N` — accumulate over N micro-batches per optimizer step (effective batch = N). The loss is averaged over the group, and a partial group is flushed at the epoch boundary. Per-image LR still applies per image.
- `--max_grad_norm` — gradient clipping (default 1.0, matching the reference recipe; 0 disables).

**Optimizer**

- `--optimizer_type` — `muon` (every picture model's built-in presets: step size matched to AdamW's, so AdamW's rates apply; its arguments default to `adjust_lr_fn=match_rms_adamw weight_decay=0.01 momentum=0.95 nesterov=True ns_steps=5`), `adamw8bit`, `adamw` (fp32 state, CUDA-fused where available; the trainer's default when the flag is left out), `adafactor`, `pagedadamw8bit`, `ademamix8bit`, `pagedademamix8bit`, `lion8bit`, `prodigy` (estimates its own rate: pass `--learning_rate 1.0`), `automagic3` / `automagic4` (set their own rate from the LR you give as a start). Anything else is taken as a full `module.path.ClassName`, so an optimizer Fizgig has never heard of works without a release. The list the CLI actually offers on your machine is in `--help` — entries whose package is missing are filtered out. `adafactor` runs at a fixed rate (the Learning Rate box). A full fine-tune defaults to `adafactor` and also takes `adamw8bit`, `adamw`, `pagedadamw8bit`, `ademamix8bit`, `pagedademamix8bit` or `lion8bit`; given a LoRA optimizer (`muon`, `prodigy`, `automagic3/4`) it trains with `adafactor` and says so in the log.
- `--optimizer_args` — extra kwargs, e.g. `--optimizer_args "weight_decay=0.01 betas=0.9,0.99"`. Values are parsed as Python literals.

Two warnings worth internalising before you reach for the dropdown. **Learning rates do not transfer between families:** Lion applies the *sign* of the update and wants roughly a tenth of an AdamW LR. Fizgig logs a warning when the LR looks wrong for the family, but it will not override you. And **saving optimizer memory buys you little here** — a LoRA's state is tens of MB against a 13–19 GB base, so the reason to change optimizer is update *behaviour*, not VRAM. If a run fails to construct the optimizer it falls back to plain AdamW and says so in the log; the choice is recorded in the output LoRA as `ss_optimizer`.

**Fine-tuning the base model** — `--finetune` trains the model's own weights a component at a time instead of a LoRA; see [FINETUNE.md](FINETUNE.md). A Context LoRA doesn't combine with it (a fine-tune trains the base itself), so the trainer refuses `--context_lora_path` with `--finetune`. **Sliders** work as on [Qwen Image 2.1](#qwen-image-21-training) (`--slider_pairs`, `--slider_prompts`).

**Not offered (by design):** block targeting (no Krea 2 block map yet).

---

## MiniMax H3 training

MiniMax H3 (33B) trains on photos, video clips with their sound, and voice recordings, all from one dataset folder — the [dataset TOML](#dataset-config-toml) is the same, and clips and audio files sit beside the images with their own `.txt` captions. You need the files from the [download table](#model-files-where-they-come-from-where-they-go): the pruned int8 DiT, the nvfp4 text encoder, the video VAE, and, for sound and voice, the audio VAE. The Turbo LoRA and the training adapter are optional but both are in the default recipe.

The cache steps are the ones in [the three-step pipeline](#the-three-step-pipeline). `--aux audio_vae=PATH` on the latent cache is what makes a clip's sound a training target (and is required by voice recordings); `--aux clip_still=1` picks each clip's sharpest face frame and caches it as a still, for `--family_option clip_still_as_photo=1` at training time.

### Full example

The GUI's default recipe as it builds it (rank 8 LoRA, the Default training mode, the training adapter, clip previews with sound):

```bash
python src/fizgig/families/train.py --family minimax \
  --dataset_config my_dataset.toml \
  --dit /models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --vae /models/minimax_h3_video_vae_fp16.safetensors \
  --text_encoder /models/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --output_dir ./output_loras/my_subject --output_name my_subject_mmh3 \
  --network_dim 8 --network_alpha 8 --learning_rate 2e-4 \
  --max_train_epochs 50 --save_every_n_epochs 1 --seed 42 \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --precision auto --blocks_to_swap -1 --ema_decay 0.98 --optimizer_type adamw8bit \
  --training_adapter /models/minimax_h3_image_training_adapter.safetensors \
  --family_option lownoise_pct=60 --family_option highnoise_lr_pct=100 \
  --family_option photo_blocks=20-49 --family_option clip_blocks=20-49 --family_option audio_blocks=20-49 \
  --family_option tread=0.5@2-47 --family_option clip_still_as_photo=1 --family_option caption_dropout=0.05 \
  --family_option audio_vae=/models/minimax_h3_audio_vae_fp32.safetensors \
  --sample_prompts sample_prompts.txt --sample_every_n_epochs 1 --sample_at_first \
  --family_option preview_frames=56 --family_option preview_audio=1 \
  --speed_lora /models/minimax_h3_turbo_v4_step600.safetensors --sample_steps 6
```

### The flags that matter

H3's own settings are `--family_option name=value` pairs (the GUI's Training-tab rows, which build them):

- `photo_blocks` / `clip_blocks` / `audio_blocks` — the Training mode: Default trains photos, clips and voice on blocks 20-49 (More Blocks: 6-49); leave them out and use `--train_blocks SPEC` to hand-pick.
- `lownoise_pct` — the training structure: the share of steps on nearly-clean images (60 = Likeness and Style, 8 = the model's own movement-first schedule); `highnoise_lr_pct` scales the LR of the noisy steps and is best left at 100.
- `tread=0.5@2-47` — TREAD token routing on clip steps (half the video tokens skip blocks 2-46).
- `clip_still_as_photo=1` — each clip's cached sharpest face frame trains as a photo (needs `--aux clip_still=1` at caching).
- `caption_dropout` — a few percent of steps train with no caption.
- `audio_weight=W` — command line only: the weight on the sound part of the loss for clips that carry sound (default 1.0, equal to the picture). Sound is only ~4% of each clip's sequence, so raise it if a voice isn't being learned. Stills and muted clips ignore it.
- `stop_category=audio|visual stop_epoch=N stop_mode=anchor|stop` — mixed datasets: finish the smaller category early (anchor = 10% LR with its epoch report live; stop = skip its steps).
- `distill=1 distill_weight=W distill_phase1=N` with `--dit` on the ref2va DiT — reference distillation (the text cache needs `--aux distill=1 --aux distill_refs=K`).
- `train_token_refiner=1` — adds the text token refiner to the LoRA targets (recommended off).
- `adapter_ramp=R` — adapter-relative LR: the Learning Rate becomes a ceiling the run climbs toward.
- `preview_frames=N preview_audio=1` — previews as clips on the 17n+5 frame grid (1 = a still), with their sound.

`--training_adapter` is frozen at 1.0 for every training step, off for previews and absent from the saved LoRA; with Ostris's adapters, use the fl2va or ref2va file to match `--dit`. `--ema_decay 0.98` saves a smoothed average of the weights. `--precision auto --blocks_to_swap -1` picks the base precision (int8 ~21 GB resident, NF4 ~11 GB, HQQ by hand) and the swap together from free VRAM. Full fine-tuning (`--finetune`) trains the base in component windows on an NF4-resident trunk and saves exact int8 checkpoints; the GUI's Fine-tune choice builds it — see `docs/FINETUNE_HOWDOI.md` before running one headless.

---

## Qwen Image 2.1 training

Qwen Image 2.1 is the first model on Fizgig's driver system: one cache script and one trainer (`src/fizgig/families/`) serve it, selected with `--family qwen_image21`. You need the files from the [download table](#model-files-where-they-come-from-where-they-go): the bf16 DiT, the VAE and the Qwen3-VL-8B text encoder, plus the training adapter and the Turbo LoRA, which are optional but both in the default recipe. The dataset TOML is the same as the other families'. 10 GB is the smallest card Qwen trains on.

### Full example

The GUI's default **Fast** preset, exactly as it builds it — rank 8, 0.5 MP (`resolution = [704, 704]` in the TOML), Adaptive LR 2e-4 to 4e-4, 30 epochs:

```bash
python src/fizgig/families/train.py \
  --family qwen_image21 \
  --dataset_config my_dataset.toml \
  --dit /models/qwen_image_2.1_bf16.safetensors \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject \
  --network_dim 8 --network_alpha 8 \
  --learning_rate 1e-4 \
  --adaptive_lr --adaptive_lr_min 2e-4 --adaptive_lr_max 4e-4 \
  --max_train_epochs 30 \
  --save_every_n_epochs 1 \
  --seed 42 \
  --optimizer_type adamw8bit \
  --precision auto --blocks_to_swap -1 \
  --ema_decay 0.98 \
  --training_adapter /models/fizgig_qwen_image_2.1_training_adapter_v2.safetensors \
  --log_per_image_loss \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --vae /models/qwen_image_2.1_vae_diffusers.safetensors \
  --text_encoder /models/qwen3vl_8b_bf16.safetensors \
  --speed_lora /models/qwen_image_2.1_turbo_lora_avg_rank_178_bf16.safetensors --sample_steps 8 \
  --sample_prompts sample_prompts.txt \
  --sample_every_n_epochs 1 --sample_at_first \
  --sample_width 1024 --sample_height 1024 --sample_seed 1234
```

The other two presets change only a few flags. **Standard** is `--network_dim 16 --network_alpha 16 --adaptive_lr_min 1e-4 --adaptive_lr_max 2e-4`. **Style** is `--network_dim 16 --network_alpha 16 --learning_rate 1.5e-4` with no `--adaptive_lr` flags (a flat rate).

### The flags that matter

**Memory and precision**

- `--precision auto|bf16|int8|nf4` — the base precision. `auto` reads free VRAM at launch and your training resolution, picks the most precise base that fits without block swap (bf16, then INT8, then NF4), and logs the choice as an `Auto plan` line. The CLI default is `bf16`, so pass `auto` to get the GUI's behaviour.
- `--blocks_to_swap N` — blocks streamed between CPU and GPU; `-1` plans the fewest that fit. Auto quantises before it swaps, because swapping costs far more speed. NF4 cannot swap.

**The recipe**

- `--training_adapter FILE` — the Fizgig training adapter, frozen for every training step, off for previews and absent from the saved LoRA. Without it Qwen 2.1 LoRAs tend to collapse into texture or wobble part-way through.
- `--ema_decay 0.98` — weight averaging: checkpoints and previews come from a smoothed average of the weights. `0` (the CLI default) turns it off.
- `--optimizer_type adamw8bit|adamw|automagic3|automagic4|prodigy|muon` — `automagic3` and `automagic4` set their own learning rate from `--learning_rate` as a start, and turn off `--adaptive_lr`, `--per_image_lr` and `--warmup_look_outliers`. `prodigy` finds its own rate too: give `--learning_rate 1.0` (a multiplier); Adaptive LR and the look warm-up stand down, per-image LR stays. `muon` takes AdamW's learning rates.
- `--network_type lokr --lokr_factor 8` — LoKR instead of LoRA.
- `--min_timestep` / `--max_timestep` — restrict training to a noise band, on a 0-1 scale.

**Per-image loss watch** — `--log_per_image_loss`, `--per_image_lr`, `--auto_recaption` and `--warmup_look_outliers` work as they do on [Krea 2](#krea-2-training). Auto-recaption captions with Krea 2's Qwen3-VL-4B, as the Captions tab does. Pass it as `--captioner` (the `qwen3vl_4b_fp8_scaled.safetensors` file the Qwen download fetches), plus `--text_encoder` to re-encode the new caption. `--trigger_word` and `--trigger_position start|end` control where the trigger goes in rewritten captions.

**Edit LoRAs** — add `control_directory = "/data/before"` under `[[datasets]]` in the TOML, beside `image_directory` (the after-images). Before-images match after-images by file name (`photo.png` pairs with `photo.png`). Extra pictures for an edit (Flux 3 Image Klein and Qwen Image 2.1 up to 9 per photo, Klein 9B up to 3) go in their own folder, `extra_reference_directory = "/data/extras"` beside `control_directory`, named `photo_1.png`, `photo_2.png` … for `photo.png`, at any shape. The cache script then stores each before-image's latent and encodes the caption with the before-images, so run both cache stages again after changing the pairs or the resolution. `--sample_reference FILE` makes the previews edits of that photo. Everything else is as above.

**Slider LoRAs** — from photo pairs: put the -1 photos in `control_directory` (same file names as the +1 photos in `image_directory`), run both cache stages with `--slider` (it caches the -1 latents and encodes the captions as plain text), then train with `--slider_pairs`. `--slider_diff_weight` (default 1) concentrates the loss where the two photos of a pair differ; 0 is the plain loss. From prompts: no dataset and no caching, just `--slider_prompts "BASE" "BASE +1 WORDS" "BASE -1 WORDS"` with `--text_encoder` and `--vae`; `--slider_guidance` (default 3; the GUI starts at 3 on Krea 2 and 2 on Qwen Image 2.1, values 2 to 9 tested on both) is the push strength, `--slider_bank` (16) and `--slider_bank_res` (768) set the practice pictures. `--slider_anchors "PROMPT" ...` are prompts the dial must leave as they are (one a step, in turn; the frozen model's prediction for each is a target at +1 and -1), weighted by `--slider_anchor_weight` (default 1). Sliders train a plain LoRA with Adaptive LR, EMA and the per-image watch off; previews render at -1 / 0 / +1. `--train_blocks` (comma-separated block ids, e.g. `block_0,block_1,txt_lw_0`) limits the LoRA to those blocks; the GUI's Krea 2 **Ultra mode** sends blocks 0-7 and the four text-fusion blocks, so the slider holds up at much higher strengths. It works best for prompt sliders, and for photo pairs whose change is compositional (pose, framing, layout).

**Fine-tuning the base model** — `--finetune` (with `--ft_rotations`, `--ft_save_every_rotations`, `--ft_rotate_every`) trains Qwen's own weights a component at a time instead of a LoRA, as on Krea 2; the GUI starts it at a learning rate of 1e-5. See [FINETUNE.md](FINETUNE.md).

**Other** — `--context_lora_path FILE --context_lora_strength S` trains on top of an existing Qwen LoRA, frozen and active in training and previews (not with `--finetune`). `--metadata_title/author/description/license/tags/trigger_phrase` are recorded in the saved LoRA. The saved file loads in ComfyUI's standard LoRA loader.

---

## Z-Image Turbo and Z-Image Base training

Z-Image trains as two families: `--family zimage` on the Turbo model, with the Fizgig training adapter (the Turbo model is distilled, and the adapter keeps a LoRA from undoing that), and `--family zimage_base` on the undistilled Base, with no adapter. Most people generate with Turbo, so a Base run previews on the Turbo DiT by default (`--preview_checkpoint`, 8 steps, no CFG); leave it out to preview on Base at `--sample_steps 25 --sample_cfg_scale 4` with a negative prompt. Both do LoRAs, LoKR, sliders, masked training and fine-tunes. Guide: [Z-Image](ZIMAGE.md).

### Full example

The **Fast** preset on Turbo, as the GUI's planner builds it: rank 8, 0.25 MP (`resolution = [496, 496]` in the TOML), Adaptive LR 2e-4 to 4e-4, 30 epochs:

```bash
python src/fizgig/families/cache.py --family zimage --stage latents --dataset_config my_dataset.toml --model /models/ae.safetensors
python src/fizgig/families/cache.py --family zimage --stage text    --dataset_config my_dataset.toml --model /models/qwen_3_4b.safetensors

python src/fizgig/families/train.py \
  --family zimage \
  --dataset_config my_dataset.toml \
  --dit /models/z_image_turbo_bf16.safetensors \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject \
  --network_dim 8 --network_alpha 8 \
  --learning_rate 1e-4 \
  --adaptive_lr --adaptive_lr_min 2e-4 --adaptive_lr_max 4e-4 \
  --max_train_epochs 30 --save_every_n_epochs 1 --seed 42 \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --training_adapter /models/fizgig_z_image_turbo_training_adapter.safetensors \
  --precision auto --blocks_to_swap -1 --compile_blocks auto \
  --ema_decay 0.98 --optimizer_type adamw \
  --log_per_image_loss \
  --vae /models/ae.safetensors \
  --text_encoder /models/qwen_3_4b.safetensors \
  --sample_prompts sample_prompts.txt --sample_every_n_epochs 1 --sample_at_first \
  --sample_width 1024 --sample_height 1024 --sample_cfg_scale 1.0 --sample_seed 1234
```

For **Base**, use `--family zimage_base --dit /models/z_image_bf16.safetensors`, drop `--training_adapter`, and add `--preview_checkpoint /models/z_image_turbo_bf16.safetensors` for Turbo previews.

---

## SDXL training

`--family sdxl` trains on any SDXL checkpoint (`--dit`), reading the text encoders from inside it; caching the text uses the same checkpoint as `--model`. The GUI's presets train at 1 MP: **Strong** is rank 32 / alpha 16 at 5e-5 for 50 epochs, **Standard** rank 16 / alpha 8 at 5e-5. SDXL does LoRAs, sliders, masked training and fine-tunes. Guide: [SDXL](SDXL.md).

```bash
python src/fizgig/families/cache.py --family sdxl --stage latents --dataset_config my_dataset.toml --model /models/sdxl_vae.safetensors
python src/fizgig/families/cache.py --family sdxl --stage text    --dataset_config my_dataset.toml --model /models/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors

python src/fizgig/families/train.py \
  --family sdxl \
  --dataset_config my_dataset.toml \
  --dit /models/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject \
  --network_dim 32 --network_alpha 16 \
  --learning_rate 5e-5 \
  --max_train_epochs 50 --save_every_n_epochs 1 --seed 42 \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --precision auto --compile_blocks auto \
  --sample_prompts sample_prompts.txt --sample_every_n_epochs 1 --sample_at_first \
  --sample_width 1024 --sample_height 1024 --sample_seed 1234
```

Without a separate VAE file, pass the checkpoint to the latent cache as well.

---

## Anima training

`--family anima` trains Anima's Base DiT with its Qwen-Image VAE and Qwen3 0.6B text encoder. With the optional Turbo LoRA as `--speed_lora`, previews render fast on the training model. Presets: **Character** (rank 16, 1e-4, 50 epochs at 1 MP, below), **Style** (rank 16, 5e-5), **Official** (rank 32, 2e-5), a slider and fine-tunes. Guide: [Anima](ANIMA.md).

```bash
python src/fizgig/families/cache.py --family anima --stage latents --dataset_config my_dataset.toml --model /models/qwen_image_vae.safetensors
python src/fizgig/families/cache.py --family anima --stage text    --dataset_config my_dataset.toml --model /models/qwen_3_06b_base.safetensors

python src/fizgig/families/train.py \
  --family anima \
  --dataset_config my_dataset.toml \
  --dit /models/anima-base-v1.0.safetensors \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject \
  --network_dim 16 --network_alpha 16 \
  --learning_rate 1e-4 \
  --max_train_epochs 50 --save_every_n_epochs 1 --seed 42 \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --precision auto --blocks_to_swap -1 --compile_blocks auto \
  --ema_decay 0.98 --optimizer_type adamw \
  --log_per_image_loss \
  --vae /models/qwen_image_vae.safetensors \
  --text_encoder /models/qwen_3_06b_base.safetensors \
  --speed_lora /models/anima-turbo-lora-v0.2.safetensors \
  --sample_prompts sample_prompts.txt --sample_every_n_epochs 1 --sample_at_first \
  --sample_width 1024 --sample_height 1024 --sample_cfg_scale 4.5 --sample_seed 1234
```

---

## ACE-Step training (songs)

`--family acestep15xl` trains ACE-Step 1.5 XL on songs. A training song is an audio file with a caption `.txt` beside it; its lyrics go in `<name>.lyrics.txt` (none = an instrumental), and bpm, time signature, key and language in an optional `<name>.json`. Songs over 240 s are cut to the first 240 s. The dataset TOML is the same as for pictures. A sample prompt is a song: `caption | lyrics`, lyric lines separated by ` / `. Guide: [ACE-Step](ACESTEP.md).

The **Official LoRA** preset, as the GUI builds it: rank 64 / alpha 128, 1e-4, gradient accumulation 4, 800 epochs (the official recipe for 10-20 songs), a checkpoint and a preview every 40 epochs, each preview also played at strength 0.5:

```bash
python src/fizgig/families/cache.py --family acestep15xl --stage latents --dataset_config my_songs.toml --model /models/ace_1.5_vae.safetensors
python src/fizgig/families/cache.py --family acestep15xl --stage text    --dataset_config my_songs.toml --model /models/qwen_0.6b_ace15.safetensors

python src/fizgig/families/train.py \
  --family acestep15xl \
  --dataset_config my_songs.toml \
  --dit /models/acestep_v1.5_xl_turbo_bf16.safetensors \
  --output_dir ./output_loras/my_sound \
  --output_name my_sound \
  --network_dim 64 --network_alpha 128 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 4 \
  --max_train_epochs 800 --save_every_n_epochs 40 --seed 42 \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --blocks_to_swap -1 --ema_decay 0.98 --optimizer_type adamw \
  --vae /models/ace_1.5_vae.safetensors \
  --text_encoder /models/qwen_0.6b_ace15.safetensors \
  --sample_prompts song_prompts.txt --sample_every_n_epochs 40 \
  --sample_strengths 0.5 --family_option preview_seconds=60
```

To train on the SFT model, pass its DiT as `--dit` with `--family_option base=sft`; its previews take 30 guided steps. Masked training, caption dropout and Start at a smaller size don't apply to songs.

---

## Masked training

Every picture family. A mask is a black-and-white picture beside its photo, `photo-masklabel.png` for `photo.jpg`: white trains fully, grey partly, black only as much as `--mask_floor` says. The photo still trains whole; the mask weights its loss. Pictures without a mask train as normal, and a mask that isn't its picture's shape (aspect ratio more than 2% off) is named in the log and left out. A mask at a different size is scaled to its picture.

| Flag | Meaning |
|---|---|
| `--masked_training` | turn it on |
| `--mask_floor 0.1` | how much outside the mask counts: 0 = nothing, 1 = as much as inside (default 0.1) |
| `--mask_prior W` | prior preservation outside the mask: keep the LoRA to the model's own prediction there, at weight W (0 = off; the GUI offers 0.5, 1 and 2). One extra forward a step |

The saved LoRA records `ss_masked_training`, `ss_mask_floor` and how many pictures had a mask. The GUI's Image Prep tab makes masks (a person finder, or SAM 3.1 from words) and edits them by hand. Guide: [Masked training](MASKED_TRAINING.md).

---

## Start at a smaller size

Standard LoRAs on every picture family (not edits, sliders, fine-tunes or MiniMax H3). The first part of the run trains at a smaller size from its own dataset TOML, then blends into the full size.

1. Write a second dataset TOML at the smaller `resolution` (for 0.25 MP, `[496, 496]`) with **its own `cache_directory`**, and run both cache steps on it too.
2. Train with the full-size TOML as `--dataset_config`, plus:

| Flag | Meaning |
|---|---|
| `--stage_dataset_config small.toml` | the smaller size's TOML |
| `--stage_until 0.3` | the share of the run's epochs at the smaller size (the GUI offers 0.2, 0.3 and 0.5) |
| `--stage_blend 0.1` | the share of the run the change blends over (default 0.1) |

The log says when each part starts. In a Z-Image Turbo test, 0.25 MP for the first 30% of a 1 MP run took about a fifth less time, with likeness arriving a little later.

---

## Batch size

`batch_size = 2` (or 3, 4, 6, 8) in the dataset TOML's `[general]` trains that many pictures a step, on Klein 9B, Krea 2, Flux 3 Image Klein, Fibo 2.0, MOSS-Image, Qwen Image 2.1, Z-Image Turbo and Base, SDXL, Anima and MiniMax H3. On MiniMax H3 the batch is of pictures: a clip or a voice still trains on a step of its own. A step's loss is the average of each picture's own loss, as gradient accumulation averages it. A batch holds pictures of one bucket size; captions of different lengths are padded where the model allows it, or the batch is split by caption length.

It pays off most with block swap: each step's swap is spread over every picture in the batch. Batch 3 against batch 1 with Gradient Accumulation 3, at 0.25 MP on an A40:

| | 1 picture a step | Batch 3 |
|---|---|---|
| Qwen Image 2.1, 8 blocks swapped | 5.0 s a picture | 1.9 s (2.7× faster) |
| Qwen Image 2.1, 16 blocks swapped | 9.5 s | 3.5 s (2.7×) |
| MOSS-Image int8, 8 blocks swapped | 2.5 s | 1.4 s (1.8×) |

With no swap it helps the smaller models (Anima 2.6×, SDXL 2.3× on a 4090) and changes little for the larger ones. These speed-ups were measured at 0.25 MP; at 1 MP the gain is smaller unless blocks are swapping. Each extra picture costs roughly 0.2–0.75 GB at 0.25 MP, more at larger sizes with gradient checkpointing on.

**Batch size Auto** (the GUI's Auto choice, `--batch_size_auto` to the trainer with `batch_size = 1` in the TOML) picks the batch measured fastest per picture for the model at the run's size, on a 40 GB+ card: SDXL 8 at 0.25 MP and 4 at 1 MP, MiniMax H3 4, Anima 4, Flux 3 4 and Fibo 2.0 2 at 0.25 MP, otherwise 1. It stays at 1 on smaller cards, when a setting reads one picture a step, or when the batch would need block swap. Measured on an RTX PRO 6000 with checkpointing off, against batch 1 with it off: SDXL 7.3× / 3.3× faster a picture, H3 2.7× / 1.8×, Anima 2.3×, Fibo 1.7×, Flux 3 1.45×. Klein, Z-Image, MOSS-Image, Qwen Image 2.1 and Krea 2 aren't offered Auto: with no swap one picture already fills the GPU, and batching measured no faster. With checkpointing off each extra picture costs far more (SDXL 2.3 / 5.5 GB at 0.25 / 1 MP, H3 2.4 / 5.9, Anima 5.1, Flux 3 5.7, Fibo 22), so `--gradient_checkpointing auto` turns it off for a batch only when one picture's figure plus those fits.

Batch size above 1 isn't used with sliders, edit pairs, DOP, Fizgig LoRA Isolation or MiniMax H3's reference distillation, and turns off masked-training mask priors, Start at a smaller size and helper shares. The GUI greys those out; the trainer refuses a batch with them. The per-image loss watch (problem images, per-image LR, the look warm-up, auto-recaption) works at any Batch size: each picture in a batch is weighed, excluded and recorded on its own, as its own step would be. Full fine-tunes batch too: each picture past the first is planned off the free memory before the windows are sized, and a regularisation picture keeps its own LR multiplier.

---

## Caption dropout

`--caption_dropout 0.05` trains that share of steps with no caption (the empty caption's encoding, made once), so the LoRA leans less on the trigger word and the caption's exact words. The GUI offers 0.05 and 0.10. Every picture family; MiniMax H3 has its own, `--family_option caption_dropout=0.05`.

---

## Detail crops (style LoRAs)

Detail crops give each picture one square crop at its own scale, so a style's brushwork and texture train pixel for pixel. Headless, it's two steps:

1. **Make and caption the crops** for each size the run trains at (its `resolution`, plus the smaller size if it starts smaller). Switch the folder on first by writing `{"on": true}` to `<training folder>/fizgig_detail_crops.json`:

```bash
python src/fizgig/scripts/detail_crops.py --folder /data/my_style \
  --mp 0.5 --mp 0.25 \
  --captioner /models/qwen3vl_4b_fp8_scaled.safetensors --instruction_file style_instruction.txt \
  --trigger zwxstyle --ext .txt
```

The crops go in `<folder>/fizgig_detail_crops/<size>mp/`. A crop with no caption gets the same picture's caption from another size; a picture with none at any size is captioned once, on its largest crop, with `--captioner` and the instruction in `--instruction_file`, and that caption is copied to every size. Without a captioner, crops that need a caption stop it with exit code 1. A picture smaller than the crop gets none. Pictures kept in `<folder>/detail_masters/` (the GUI's Train on the crops only) are cropped as well, and never trained.

2. **Train them as their own dataset block**, one per size, beside the pictures' block:

```toml
[[datasets]]
image_directory = "/data/my_style/fizgig_detail_crops/0.5mp"
item_prefix = "fizgig_detail_crops/"
cache_directory = "/data/cache/my_style_crops_0.5mp"
```

Use `item_prefix = "fizgig_detail_crops/"` for every size, so each crop keeps one per-image loss history through Start at a smaller size; the smaller size's TOML names its own size folder.

---

## Keeping a subject to its trigger (DOP)

Every family, Standard LoRA and fine-tune runs. `--dop` adds a second pass to each step: the same picture, with the caption's trigger swapped for `--dop_class` (or dropped when that is empty), trained to match what the base model itself predicts there. Without the trigger the LoRA learns to change nothing, so the subject stops leaking into plain prompts ("a woman ..."). It costs about 1.5-2x the time per step. Differential Output Preservation comes from Ostris's AI-Toolkit.

| Flag | Meaning |
|---|---|
| `--dop` | turn it on (needs `--text_encoder`: the swapped captions are encoded once at the start) |
| `--dop_trigger zwxem` | the word in the captions that means the subject; several subjects comma-separated (`zwxem, qrtpl`). Default `--trigger_word` |
| `--dop_class woman` | what replaces each trigger: one per trigger in order (`woman, man`) or one for all; empty = the triggers are just left out |
| `--dop_weight 1.0` | the preservation loss's weight |
| `--dop_every 1` | a preservation pass every N steps; raise `--dop_weight` by the same factor to keep the same preservation on average (the web UI's "Every second step" sends `--dop_every 2` with the weight doubled) |
| `--dop_ft_samples 8` | fine-tunes only: how many of the original model's predictions are made per image before the first step (a fine-tune has no adapter to switch off, so these stand in for the base model; saved beside the run so a continued fine-tune reuses them) |

A caption with several triggers (a photo of both people) has each one swapped. Images whose caption has no trigger train as usual. Each epoch logs the preservation loss: how far the LoRA moves the trigger-free prompt from the base model.

The web app's **Subjects** card always uses the tested recipe, `--dop --dop_every 2 --dop_weight 2`. **Fizgig LoRA Isolation** adds three things, which the launch builds for you:
- each subject's 16 side-by-side pictures, built first by `src/fizgig/scripts/multi_person.py --subject FOLDER FUSED_WORD OUT_DIR` (one `--subject` per person) and trained as one more dataset block per subject;
- `--lr_boost fizgig_mp_=2`, so those pictures train at 2x the learning rate;
- a `caption_replace` list in each subject's dataset block, which reads `zwxem, a woman` as `zwxemwoman`, with `--dop_trigger zwxemwoman --dop_class "a woman"`.

How to caption for it: [People in group pictures](MULTI_PERSON.md).

## Sample previews during training

`--sample_prompts` takes a text file, one plain prompt per line, `#` lines are comments; geometry and seed come from `--sample_width` / `--sample_height` / `--sample_seed`.

**Klein** renders previews on the Distilled model (4-step, fast, matches ComfyUI) with `--preview_checkpoint <distilled>`; without it they render on the training Base, where `--sample_cfg_scale` (4.5 by default; 1.0 disables CFG) and `--sample_negative` steer them. `--sample_reference FILE` makes every preview an edit of that photo.

**Krea 2** and **Qwen Image 2.1** prompt files are plain prompts only; geometry and seed come from `--sample_width` / `--sample_height` / `--sample_seed`, and previews render on the training model, with the speed LoRA when `--speed_lora` is given (`--sample_steps`: 8 for Krea's Turbo LoRA, 6 for Qwen's). `--sample_cfg_scale` above 1 enables CFG on the previews — pair it with `--sample_negative` for a real uncond. `--sample_at_first` renders an epoch-0 preview (base + zero-init LoRA) before training. `--sample_until_epoch Y --sample_every_n_epochs_after Z` previews every `--sample_every_n_epochs` up to and including epoch Y, then on whole multiples of Z (1 / 10 / 5: epochs 1-10, then 15, 20, 25). `--save_until_epoch Y --save_every_n_epochs_after Z` does the same for checkpoints (not for a fine-tune, which saves at whole rotations). A reference picture: Krea 2's `--sample_image FILE` feeds it through the Qwen3-VL vision path (generate driven by a reference picture; works even with an empty prompt file), and Qwen's `--sample_reference FILE` makes every preview an edit of that photo. `--metadata_title/author/description/license/tags` are recorded in the saved LoRA as `modelspec.*` keys.

**MiniMax H3** prompt files are plain prompts too. Previews are clips: `--family_option preview_frames=N` sets the length on the model's 17n+5 frame grid (1 = a still, 56 ≈ 2.3 s, 124 = the trained minimum of ~5 s; off-grid values snap down), `--family_option preview_audio=1` with `--family_option audio_vae=PATH` decodes the clip's generated sound to a `.wav` beside the `.mp4`, and `--speed_lora` with `--sample_steps 6` renders on the Turbo LoRA at `--speed_lora_strength 0.75` (20 steps without it, matching ComfyUI's shipped template). `--sample_width/height` default to H3's native 768. The training adapter is switched off for previews; a Context LoRA stays on. Rendering a 56-frame clip every epoch costs more than the epoch's training on a small dataset — set `--sample_every_n_epochs` higher, or preview stills, when speed is the point.

**Qwen Image 2.1** prompt files are plain prompts; `--sample_width/height/seed` set the rest. With `--speed_lora` the previews render with the Turbo LoRA at strength 1.0 for 8 steps at CFG 1, on the Turbo checkpoint's own schedule (`--speed_lora_strength` and `--sample_steps` override; other step counts use the model's standard schedule). `--preview_checkpoint <Turbo DiT>` renders them on the Turbo DiT instead (8 steps, CFG 1), with the training model parked beside it. Without either they render on the plain model at `--sample_steps 25 --sample_cfg_scale 3` with `--sample_negative`; `--sample_cfg_scale 1` turns CFG off, at roughly half the time per preview. The training adapter is off for previews and a Context LoRA stays on. `--sample_at_first` renders an epoch-0 preview.

Samples are written to `<output_dir>/sample/` with the epoch number in the filename. Prefer ~1024×1024 on Klein and Krea 2 — sub-1024 previews degrade anatomy and undersell the checkpoint.

---

## Pause and resume

Pause is a file, which makes it fully scriptable:

```bash
# request a graceful pause (every family)
touch ./output_loras/my_subject/.pause_requested        # Linux/macOS
New-Item ./output_loras/my_subject/.pause_requested     # Windows PowerShell
```

At the next epoch boundary the trainer force-saves a full state dir, logs `[pause] requested ... Exiting cleanly`, and exits 0 — GPU freed, no quality loss. Every family watches `<output_dir>/.pause_requested`.

Resume by pointing at the state directory:

```bash
python src/fizgig/families/train.py --family klein ... --resume ./output_loras/my_subject/my_subject-000012-state
```

The epoch number is parsed from the dir name; optimizer, scheduler, RNG, dataloader state, adaptive-LR scalars, and (where the family has it) the full per-image watch history are all restored. Pass the same flags as the original run plus `--resume`.

**Training a finished LoRA further.** Resume its end-of-run state and raise `--max_train_epochs` in the same command — the state records how many epochs are already done, so a run resumed at its final epoch has nothing left to do and simply rewrites the final LoRA (the trainer says so plainly in the log rather than pretending it trained). Raising the ceiling is what gives it epochs to run:

```bash
# LoRA finished at 30 epochs; take it to 45
python src/fizgig/families/train.py --family krea2 ... --max_train_epochs 45 \
    --resume ./output_loras/my_subject/my_subject-000030-state
```

---

## VRAM guidance (block swap)

`--blocks_to_swap` parks transformer blocks in CPU RAM and streams them over PCIe — slower per step, but fits big models on small cards. Every family plans for itself: `--precision auto --blocks_to_swap -1` reads the free VRAM at launch and your training resolution, quantises before it swaps (INT8, then NF4), and logs the plan as an `Auto plan` line. Set either by hand to override.

- **Klein 9B**: INT8 when it compiles, else the Base file as it is (BFL's fp8 file stays fp8, ~9.6 GB resident, so 16 GB+ cards skip swap), NF4 on small cards.
- **Flux 3 Image Klein**: bf16, INT8 or NF4 by free VRAM (a rank-16 LoRA at 1 MP peaked at 12.7 / 9.1 / 6.2 GB); on small cards the NF4 model steps off the GPU for previews, so 8 GB cards train and preview.
- **MiniMax H3**: int8 at ~21 GB resident on 32 GB cards, 4-bit at ~11 GB below that, the swap planned with it.
- **Qwen Image 2.1**: bf16 on 24 GB+ cards, INT8 with no swap on 12 and 16 GB cards and NF4 on 10 GB cards, at the presets' 0.5 MP. The text encoder only encodes and loads 8-bit below about 20 GB free, which sets the 10 GB floor.

---

## LoRA extraction

`extract_lora.py` reduces a LoRA to a lower rank — exact weight SVD straight from the file, no model loaded, in the family's own key format. It handles LyCORIS (LoKR) sources too:

```bash
python src/fizgig/scripts/extract_lora.py --family klein \
  --source big_r32.safetensors --output small_r8.safetensors --rank 8
```

`--identity` keeps 4x the rank in the model's identity blocks (MOSS-Image blocks 24-35, Qwen Image 2.1 blocks 10-14), so a character LoRA keeps its likeness at a small rank; the Extract tab's Identity mode tick does the same. `--preset` keeps one of the family's block groups — Klein: `Identity` (single 1-16), `Style+Composition` (double 0-7 + single 0-1, single 2 at half strength), `Details` (single 12-23) — and `--blocks single_1,single_2,...` picks blocks by id. The Extract tab does the same with a Custom block picker.

---

## LoRA profiling

`profile_lora.py` writes a LoRA's weights report — how much of its change each rank keeps (what Extract keeps at that rank) and how big each block's real update is — plus the sidecar Repair Studio reads, with no model loaded, in seconds:

```bash
python src/fizgig/scripts/profile_lora.py --family klein --lora my_subject.safetensors
```

Writes `<name>_<suffix>_profile.html` next to the LoRA (or pass `--output report.html`). The rendered profile — each group of blocks on its own and left out, with likeness and bleed scored against photos of your subject — runs on the Profiler tab.

---

## Analyzing a per-image loss log

For a quick headless read of a Krea 2 run's per-image data (no GUI needed):

```bash
python src/fizgig/scripts/analyze_loss_log.py ./output_loras/my_subject --top 20
```

Points at the output dir (or the `per_image_loss.jsonl` directly) and prints the N hardest images by mean residual — the same signal that drives the watch's verdicts. Cross-reference against your captions: the top offenders are usually caption-viewpoint mismatches, and fixing the caption beats letting the throttle handle it.

---

## Captioning, translating and extracting from a fine-tune

**AI captions** — `batch_caption.py` is what the Captions tab runs. Qwen3-VL (the Krea 2 text encoder file, or a bigger Qwen3-VL as the GUI's secondary captioning model) captions every file listed in a text file, one path a line, with your instruction, putting the trigger word first:

```bash
python -m fizgig.scripts.batch_caption --backend qwen --text_encoder /models/qwen3vl_4b_fp8_scaled.safetensors \
  --list_file images.txt --instruction_file instruction.txt --trigger zwxem --max_new_tokens 160
```

The GUI's caption tasks (person or character, object or product, style, concept) are instructions; write the one you want to a file. `--backend florence` uses Florence-2 instead.

**Bilingual captions** — `python -m fizgig.scripts.translate_captions <folder> [--skip_existing] [--ext .txt]` adds a Chinese translation to every caption (`<trigger>, <english> - <chinese>`), keeping the trigger word as it is.

**A fine-tune as LoRAs** — `extract_finetune.py` takes the difference between a fine-tuned checkpoint and the base it started from, one LoRA per rank from a single pass:

```bash
python -m fizgig.scripts.extract_finetune --base /models/base.safetensors --tuned ./output/my_finetune.safetensors \
  --out-dir ./output_loras --name my_finetune --ranks 32,64
```

`--identity` gives the model's identity blocks 4x each rank (MOSS-Image blocks 24-35, Qwen Image 2.1 blocks 10-14; `_id` in the file name), which keeps a character's likeness in a smaller LoRA.

**Model files** — `python -m fizgig.scripts.fetch_models --family <key> --include-optional` downloads a family's files (`--all` for every family, `--dry-run` to list them first, `--token` for gated repos, `--models-dir` to save somewhere other than `user_data/models`, `--choose PREF=alt` for a file offered in a second build).

---

## Training maps

- `--training_maps` — before training and after every epoch, three of the training pictures (spread through the dataset; at least one per subject, and one of each subject's Fizgig LoRA Isolation side-by-side pictures) are run at three fixed noise levels with fixed noise, and the run's output folder gets `training_maps/<LoRA name>/probes.json` and `maps.jsonl`: each picture's error, the LoRA's effect (on minus off) and, on every still-picture model except SDXL, the trigger word's share of the attention. The trigger comes from `--dop_trigger`, `--trigger_word` or `--metadata_trigger_phrase` (a comma list is several), found in each picture's cached caption. A few seconds an epoch; off for sliders and fine-tunes. The Training maps button in the app reads the files; see [TRAINING_MAPS.md](TRAINING_MAPS.md).
- `--training_maps_images N` — how many pictures the maps follow (default 3; more when there are more subjects).

## Other training flags

- `--gradient_checkpointing auto|on` — `auto` (default) turns it off for a compiled LoRA run with no block swap when the measured peak fits free VRAM, on 40 GB+ cards only (13–27% faster per step; SDXL, Anima, Flux 3 and Klein measured 1.3–2.3× faster); at a Batch size above 1, when that peak plus each extra picture's measured cost fits. Where it doesn't all fit, the last blocks go without it (batch 1). `on` keeps it everywhere.
- `--training_adapter_strength S` — the training adapter's strength (default 1.0), for the families that use one.
- `--sample_strengths 0.5` — also render each preview at these LoRA strengths, comma-separated.
- `--preview_int8` — INT8 matmuls on the preview checkpoint, for smaller cards.
- **Auto-recaption's tasks** — `--recaption_instruction TEXT` and `--recaption_instruction_detailed TEXT` are the first and second tries' instructions (the GUI fills them from its **First try uses** / **Second try uses** picks); `--recaption_task` and `--recaption_task_detailed` name those tasks for the log. `--recaption_triggers '{"outfits/red/": "zwxred"}'` gives other dataset blocks their own trigger word, keyed by `item_prefix`.
- **Fine-tune windows** — `--ft_fused_backward` steps each weight as its gradient lands and frees it (the GUI's "Free each gradient as soon as it's used", on by default: less VRAM, no clipping or accumulation); `--ft_max_parts N` caps the parts per window (0 = as many as fit; 1 = one part per window, the most headroom).
- `--preview_small_card_cap PX` — on a card under 20 GB, the previews' long side is capped at this many pixels (the Samples tab's Faster, smaller previews).
- **Cache steps** — `--batch_size`, `--num_workers` and `--device` tune the cache scripts; the defaults suit most machines.

The trainer also has options Fizgig itself uses between computers (`--share_worker`, the `--ft_*` window and continue options, `--preview_checkpoint_cache`); you don't need them by hand.
