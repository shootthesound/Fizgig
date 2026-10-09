# Fizgig Headless CLI

Everything the GUI does for training runs through `src/fizgig/families/cache.py` and `train.py` (plus the tools in `src/fizgig/scripts/`) — the GUI is a front-end that builds these exact commands and runs them as subprocesses. That means the CLI is always feature-complete for every family — Klein 9B, Krea 2, MiniMax H3 and Qwen Image 2.1: adaptive LR, the per-image loss watch, auto-recaptioning, Context LoRA, video and voice training, pause/resume — all of it is available from a plain terminal, on **Windows and Linux alike** (including display-less boxes).

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
- [Krea 2 training](#krea-2-training)
- [MiniMax H3 training](#minimax-h3-training)
- [Qwen Image 2.1 training](#qwen-image-21-training)
- [Sample previews during training](#sample-previews-during-training)
- [Pause and resume](#pause-and-resume)
- [VRAM guidance (block swap)](#vram-guidance-block-swap)
- [LoRA extraction (rank reduction)](#lora-extraction)
- [LoRA profiling](#lora-profiling)
- [Analyzing a per-image loss log](#analyzing-a-per-image-loss-log)

---

## Model files: where they come from, where they go

Headless, there is no Preferences tab: **model locations are passed as flags on every command** (`--dit`, `--vae`, `--text_encoder`, for MiniMax H3 `--audio_vae`, `--turbo_lora_path` and `--training_adapter_path`, for Krea 2 and Qwen Image 2.1 `--speed_lora`, and for Qwen `--training_adapter`). The CLI does not read the GUI's `prefs.json` — put the paths in a shell script or Makefile once and forget about them. The files themselves are the same ones the GUI's Preferences tab links to:

**Klein 9B:**

| File | Download | Used for |
|---|---|---|
| Base DiT (fp8, recommended) | [FLUX.2-klein-base-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/tree/main) | training (`--dit`) |
| Base DiT (bf16, big cards) | [FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/tree/main) | training (`--dit`) |
| Distilled DiT (fp8) | [FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/tree/main) | fast 4-step previews (`--preview_checkpoint`) and the workbench tabs |
| VAE `ae.safetensors` | [FLUX.2-dev → ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) | `--vae` |
| Text encoder `qwen_3_8b.safetensors` | [Comfy-Org Klein text encoder](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors) | `--text_encoder` |

> **VAE trap:** use `ae.safetensors` from the FLUX.2-dev repo **root** — not `vae/diffusion_pytorch_model.safetensors`, which is the Diffusers-format file and will not load.

**Krea 2:**

| File | Download | Used for |
|---|---|---|
| RAW DiT | [krea2_raw_bf16.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors) | training (`--dit`) |
| Turbo LoRA | [krea2_turbo_lora_rank_64_bf16.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/loras/krea2_turbo_lora_rank_64_bf16.safetensors) | 8-step previews (`--speed_lora`) |
| VAE `qwen_image_vae.safetensors` | [Krea-2 → vae](https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors) | `--vae` (cache step: `--model`) |
| Text encoder `qwen3vl_4b_fp8_scaled.safetensors` | [Krea-2 → text_encoders](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors) | `--text_encoder` (cache step: `--model`); the bf16 file works too |
| Turbo DiT (fp8) | [krea2_turbo_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors) | GUI only: the Repair Studio, Explorer and Royale previews |

**MiniMax H3:**

| File | Download | Used for |
|---|---|---|
| DiT, pruned int8 (~21 GB) | [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | training (`--dit`) — the same file ComfyUI runs |
| DiT, reference build *(optional)* | [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | `--dit` for LoRAs that live in the reference-to-video workflow, and for `--distill` |
| Text encoder, nvfp4 (~15.7 GB) | [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | `--text_encoder` (caching, and pre-encoding preview prompts) |
| Video VAE | [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors) | `--vae` |
| Audio VAE *(optional)* | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `--audio_vae` — sound in clips, voice recordings, previews with sound |
| Turbo LoRA *(optional)* | [minimax_h3_turbo_v4_step600.safetensors](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/blob/main/minimax_h3_turbo_v4_step600.safetensors) | 6-step previews (`--turbo_lora_path`) |
| Training adapter *(recommended)* | [minimax_h3_image_training_adapter.safetensors](https://huggingface.co/circlestone-labs/MiniMax-H3-Image-Training-Adapter/blob/main/minimax_h3_image_training_adapter.safetensors) (Circlestone, the GUI default — one file for both bases) | `--training_adapter_path` |
| Training adapter, Ostris *(best for videos)* | [minimax_h3_training_adapter_v1.safetensors](https://huggingface.co/ostris/minimax_h3_training_adapter/blob/main/minimax_h3_training_adapter_v1.safetensors) (fl2va) · [ref2va file](https://huggingface.co/ostris/minimax_h3_training_adapter/blob/main/minimax_h3_ref2va_training_adapter_v1.safetensors) | `--training_adapter_path` — match it to `--dit` |

**Qwen Image 2.1:**

| File | Download | Used for |
|---|---|---|
| DiT (bf16) | [qwen_image_2.1_bf16.safetensors](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/diffusion_models/qwen_image_2.1_bf16.safetensors) | training (`--dit`); quantised to INT8 or NF4 at load when `--precision` asks |
| VAE | [Qwen-Image-2.1 → vae/diffusion_pytorch_model.safetensors](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/vae/diffusion_pytorch_model.safetensors) | `--vae` (cache step: `--model`) — Qwen 2.1's own VAE, not the Krea 2 / Qwen-Image one |
| Text encoder `qwen3vl_8b_bf16.safetensors` | [Qwen-Image-2.1 → text_encoders](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/blob/main/text_encoders/qwen3vl_8b_bf16.safetensors) | `--text_encoder` (cache step: `--model`); loaded 8-bit automatically on cards under ~20 GB free |
| Fizgig training adapter *(recommended)* | [fizgig_qwen_image_2.1_training_adapter.safetensors](https://huggingface.co/ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter) | `--training_adapter` |
| Viggle turbo LoRA *(optional)* | [Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo) | fast previews (`--speed_lora`) |

The quickest way to get every file is the fetcher the Preferences download button runs — `python -m fizgig.scripts.fetch_models --family qwen_image21 --include-optional` (from the repo root with `src` on `PYTHONPATH`). It also caches the tokenizer files so training works offline.

---

## What's family-specific at a glance

Every family trains through the same two scripts — `src/fizgig/families/cache.py` and `src/fizgig/families/train.py`, selected with `--family` (`klein`, `minimax`, `krea2`, `qwen_image21`) — and shares the dataset format, pause/resume, Context LoRA, LoKR, sliders, fine-tuning and the memory planner. What differs:

| Feature | Klein 9B | MiniMax H3 | Krea 2 | Qwen Image 2.1 |
|---|---|---|---|---|
| Base precision (`--precision`) | auto / bf16 / int8 / nf4 | auto / int8 / nf4 / hqq | auto / bf16 / int8 / nf4 | auto / bf16 / int8 / nf4 |
| Adaptive LR (`--adaptive_lr`) | ✅ | ❌ | ✅ | ✅ |
| Per-image loss watch, per-image LR, auto-recaption, look-outlier warm-up | ✅ | ❌ | ✅ | ✅ |
| Model Area / block targeting (`--train_blocks`) | ✅ (identity, style, details areas) | ✅ (Blocks to Train, Training mode) | ❌ | ❌ |
| Timestep range (`--min/max_timestep`, 0-1000) | ✅ | ❌ | ✅ | ✅ |
| torch.compile (`--compile_blocks auto`) | ✅ | ❌ | ✅ | ✅ |
| Edit LoRAs (original + edited pairs) | ✅ | ❌ | ❌ | ✅ |
| Photos, video clips with sound, voice | photos | all three | photos | photos |

Auto-recaption captions with Krea 2's Qwen3-VL-4B, the same model the GUI's Captions tab uses, whichever model you're training (`--captioner`). MiniMax H3's own settings (training structure, Training mode, TREAD, clip stills, distillation, retirement of a finished category) reach the trainer as `--family_option name=value` pairs; see [MiniMax H3 training](#minimax-h3-training).

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

Re-running a cache step is cheap: pass `--skip_existing` to only encode new images. Stale cache files for images that were removed from the dataset are deleted automatically (pass `--keep_cache` to keep them, but see the warning in the next section).

---

## Dataset config (TOML)

The dataset config is a small TOML file. The GUI writes one automatically (`dataset/Fizgig_train.toml`); headless you write it yourself:

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
- **`batch_size`** — keep it at 1: Fizgig trains one image a step. For a larger effective batch, pass `--gradient_accumulation_steps N` to the trainer: the same averaged gradient, at the memory of batch 1.
- **`num_repeats`** — multiplies how often each image appears per epoch. Leave at 1 and train more epochs instead, unless you're balancing multiple `[[datasets]]` blocks against each other.
- **`cache_directory`** — where latents and text embeddings are stored. **Give every dataset its own cache directory.** The training set is built from the cache, and a shared cache directory can mix a previous dataset's images into your run. (Fizgig cross-checks the cache against `image_directory` and skips orphans with a warning, but a dedicated directory per dataset is the clean way.)
- **Multiple `[[datasets]]` blocks** are supported — each with its own `image_directory`, `cache_directory`, and optional per-dataset overrides of any `[general]` key (e.g. a different `num_repeats` to weight one folder more heavily).

If you change captions, re-run the text cache step. If you add/remove/edit images, re-run both cache steps.

---

## Preparing images and captions

### Images

A dataset is just a folder of images with caption sidecars — no manifest, no subfolder structure. Accepted formats: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp` (plus `.jxl` if jxlpy is installed). You do **not** need to pre-resize or crop to a fixed size: with `enable_bucket = true` the loader assigns each image to its nearest aspect-ratio bucket at the target megapixel area, and `bucket_no_upscale = true` keeps undersized images at their native resolution instead of upscaling them. Aim for source images at or above ~1 MP; a mix of framings (close-up, half-body, full-body, varied backgrounds) trains better identity than 40 near-identical headshots.

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

**Precision / VRAM** — `--precision auto --blocks_to_swap -1` reads the free VRAM at launch: INT8 when it compiles, else the Base file as it is (BFL's fp8 file stays fp8), NF4 on small cards, swapping blocks only when nothing else fits. Set either by hand to override. Gradient checkpointing is always on.

**LoRA shape** — `--network_dim` / `--network_alpha` (the GUI presets use matched pairs: 4/4 style and details, 8/8, 16/16 identity); `--network_type lokr --lokr_factor 8` for LoKR.

**Learning rate** — `--learning_rate`, or `--adaptive_lr` (the bi-directional plateau tracker: probes LR up on steady descent, cuts it with weight rollback on plateau or instability). Bounds via `--adaptive_lr_min` (5e-5 for single-subject sets, 1e-4 for noisy multi-subject sets) and `--adaptive_lr_max` (4e-4 is the empirical ceiling). **`--learning_rate` is ignored while adaptive is on**: the run starts at the geometric midpoint of the window and the watcher owns the LR from there. Without it, `--lr_scheduler` / `--lr_warmup_steps` apply.

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
- `--compile_blocks auto|on|outside|off` — torch.compile the transformer blocks (`outside` = the high-resolution boundary: checkpoint kept outside the compiled region, eager-level memory, chosen automatically by `auto`/`on` where inside-the-graph would not fit): roughly **2× faster steady-state steps** on the INT8 path after a one-off warm-up (~90 s + a pause on each new latent shape). `auto` (default) weighs the warm-up against the run length and only compiles when it pays. Needs triton (installs with requirements; `triton-windows` on Windows) and, on Windows, the MSVC C++ Build Tools — direct installer: https://aka.ms/vs/17/release/vs_BuildTools.exe ("Desktop development with C++" workload). Missing either → a console note and the run continues uncompiled.
- `--ema_decay 0.98` — weight averaging: checkpoints and previews come from a smoothed average of the weights. `0` (the CLI default) turns it off.
- `--min_timestep` / `--max_timestep` — restrict training to a noise band, on a 0-1 scale.
- `--speed_lora FILE` — the Turbo LoRA, attached for 8-step previews on the training model and absent from the saved LoRA; `--speed_lora_strength` defaults to 1.

**The per-image loss watch** (any of these enables the watcher)

- `--log_per_image_loss` — tracks every image's loss residual against its timestep bucket, classifies each image at every epoch boundary (`easy` / `suspect` / `stuck` / `exhausted` / `excluded`), and writes:
  - `<output_dir>/loss_log/per_image_loss.jsonl` — the raw per-step log
  - `<output_dir>/loss_log/problem_images.json` — current verdicts + trends (the GUI's Problem Images window reads this; it's plain JSON, perfectly greppable headless)
  - a **plateau banner** in the console when ≤~5% of images are still improving, with a best-checkpoint epoch estimate — your "you're done" signal.
- `--per_image_lr` — acts on the verdicts: stuck images get throttled (×0.5 → ×0.25 → ×0.125), mined-out images ease off (×0.6), the healthy cohort gets a gentle boost (×1.1). Batch size 1 makes this a true per-image learning rate.
- `--auto_recaption` — between epochs, confirmed-stuck images get their captions rewritten by Qwen3-VL from what's actually visible, the text cache is re-encoded, and the image gets a fresh start. Two failed attempts and a still-stuck image is excluded from the run entirely. Requires `--text_encoder`. Pass `--trigger_word` so rewritten captions keep your trigger (appended as `, <trigger>`).
- `--warmup_look_outliers` — curriculum entry (×0.4 LR ramping to ×1.0) for real-but-unusual images. Reads `<dataset>/fizgig_look_scores.json`, which is produced by the GUI's Look Consistency Filter scan — **GUI-only prerequisite**; without the file this flag logs a warning and disables itself.

Persistent artifacts: exclusions are recorded per model family in `<image_directory>/fizgig_excluded.json`; a later run of that family trains the image normally and excludes it at once if it gets stuck again (no new recaptions). Editing an excluded image's caption clears the record. Fresh runs rotate the old JSONL to `.bak`; `--resume` replays the log to restore full watch history.

**LR schedule and batching**

- `--lr_scheduler` — `constant` (default), `constant_with_warmup`, `cosine`, `cosine_with_restarts`, `linear`, `polynomial`; with `--lr_warmup_steps` (plus `--lr_scheduler_num_cycles` / `--lr_scheduler_power` for the two that use them). **Ignored when `--adaptive_lr` is on** — the plateau watcher owns the LR and says so in the log. Resume continues the curve rather than restarting it.
- `--gradient_accumulation_steps N` — accumulate over N micro-batches per optimizer step (effective batch = N). The loss is averaged over the group, and a partial group is flushed at the epoch boundary. Per-image LR still applies per image.
- `--max_grad_norm` — gradient clipping (default 1.0, matching the reference recipe; 0 disables).

**Optimizer**

- `--optimizer_type` — `adamw8bit` (default, the validated recipe), `adamw` (fp32 state, CUDA-fused where available), `pagedadamw8bit`, `ademamix8bit`, `pagedademamix8bit`, `lion8bit`. Anything else is taken as a full `module.path.ClassName`, so an optimizer Fizgig has never heard of works without a release. The list the CLI actually offers on your machine is in `--help` — entries whose package is missing are filtered out. (`adafactor`, `prodigy` and `came` were removed: they manage their own learning rate, which conflicts with Adaptive LR and produced real failures; reach them via the module-path form if you know what you're doing.)
- `--optimizer_args` — extra kwargs, e.g. `--optimizer_args "weight_decay=0.01 betas=0.9,0.99"`. Values are parsed as Python literals.

Two warnings worth internalising before you reach for the dropdown. **Learning rates do not transfer between families:** Lion applies the *sign* of the update and wants roughly a tenth of an AdamW LR. Fizgig logs a warning when the LR looks wrong for the family, but it will not override you. And **saving optimizer memory buys you little here** — a LoRA's state is tens of MB against a 13–19 GB base, so the reason to change optimizer is update *behaviour*, not VRAM. If a run fails to construct the optimizer it falls back to plain AdamW and says so in the log; the choice is recorded in the output LoRA as `ss_optimizer`.

**Fine-tuning the base model** — `--finetune` trains the model's own weights a component at a time instead of a LoRA; see [FINETUNE.md](FINETUNE.md). A Context LoRA doesn't combine with it (a fine-tune trains the base itself), so the trainer refuses `--context_lora_path` with `--finetune`. **Sliders** work as on [Qwen Image 2.1](#qwen-image-21-training) (`--slider_pairs`, `--slider_prompts`).

**Not offered (by design):** block targeting (no Krea 2 block map yet) and turning off gradient checkpointing (always on).

---

## MiniMax H3 training

MiniMax H3 (33B) trains on photos, video clips with their sound, and voice recordings, all from one dataset folder — the [dataset TOML](#dataset-config-toml) is the same, and clips and audio files sit beside the images with their own `.txt` captions. You need the files from the [download table](#model-files-where-they-come-from-where-they-go): the pruned int8 DiT, the nvfp4 text encoder, the video VAE, and, for sound and voice, the audio VAE. The Turbo LoRA and the training adapter are optional but both are in the default recipe.

The cache steps are the ones in [the three-step pipeline](#the-three-step-pipeline). `--audio_vae` on the latent cache is what makes a clip's sound a training target (and is required by voice recordings); `--clip_still` picks each clip's sharpest face frame and caches it as a still, for `--clip_still_as_photo` at training time.

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

Qwen Image 2.1 is the first model on Fizgig's driver system: one cache script and one trainer (`src/fizgig/families/`) serve it, selected with `--family qwen_image21`. You need the files from the [download table](#model-files-where-they-come-from-where-they-go): the bf16 DiT, the VAE and the Qwen3-VL-8B text encoder, plus the training adapter and the Viggle turbo LoRA, which are optional but both in the default recipe. The dataset TOML is the same as the other families'. 10 GB is the smallest card Qwen trains on.

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
  --training_adapter /models/fizgig_qwen_image_2.1_training_adapter.safetensors \
  --log_per_image_loss \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --vae /models/qwen_image_2.1_vae_diffusers.safetensors \
  --text_encoder /models/qwen3vl_8b_bf16.safetensors \
  --speed_lora /models/Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors \
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
- `--optimizer_type adamw8bit|adamw|automagic3` — `automagic3` sets its own learning rate from `--learning_rate` as a start, and turns off `--adaptive_lr`, `--per_image_lr` and `--warmup_look_outliers`.
- `--network_type lokr --lokr_factor 8` — LoKR instead of LoRA.
- `--min_timestep` / `--max_timestep` — restrict training to a noise band, on a 0-1 scale.

**Per-image loss watch** — `--log_per_image_loss`, `--per_image_lr`, `--auto_recaption` and `--warmup_look_outliers` work as they do on [Krea 2](#krea-2-training). Auto-recaption captions with Krea 2's Qwen3-VL-4B, as the Captions tab does. Pass it as `--captioner` (the `qwen3vl_4b_fp8_scaled.safetensors` file the Qwen download fetches), plus `--text_encoder` to re-encode the new caption. `--trigger_word` and `--trigger_position start|end` control where the trigger goes in rewritten captions.

**Edit LoRAs** — add `control_directory = "/data/before"` under `[[datasets]]` in the TOML, beside `image_directory` (the after-images). Before-images match after-images by file name, one each (`photo.png` pairs with `photo.png`). The cache script then stores each before-image's latent and encodes the caption with the before-images, so run both cache stages again after changing the pairs or the resolution. `--sample_reference FILE` makes the previews edits of that photo. Everything else is as above.

**Slider LoRAs** — from photo pairs: put the -1 photos in `control_directory` (same file names as the +1 photos in `image_directory`), run both cache stages with `--slider` (it caches the -1 latents and encodes the captions as plain text), then train with `--slider_pairs`. `--slider_diff_weight` (default 1) concentrates the loss where the two photos of a pair differ; 0 is the plain loss. From prompts: no dataset and no caching, just `--slider_prompts "BASE" "BASE +1 WORDS" "BASE -1 WORDS"` with `--text_encoder` and `--vae`; `--slider_guidance` (default 3; the GUI starts at 3 on Krea 2 and 2 on Qwen Image 2.1, values 2 to 9 tested on both) is the push strength, `--slider_bank` (16) and `--slider_bank_res` (768) set the practice pictures. Sliders train a plain LoRA with Adaptive LR, EMA and the per-image watch off; previews render at -1 / 0 / +1. `--train_blocks` (comma-separated block ids, e.g. `block_0,block_1,txt_lw_0`) limits the LoRA to those blocks; the GUI's Krea 2 **Ultra mode** sends blocks 0-7 and the four text-fusion blocks, so the slider holds up at much higher strengths. It works best for prompt sliders, and for photo pairs whose change is compositional (pose, framing, layout).

**Fine-tuning the base model** — `--finetune` (with `--ft_rotations`, `--ft_save_every_rotations`, `--ft_rotate_every`) trains Qwen's own weights a component at a time instead of a LoRA, as on Krea 2; the GUI starts it at a learning rate of 1e-5. See [FINETUNE.md](FINETUNE.md).

**Other** — `--context_lora_path FILE --context_lora_strength S` trains on top of an existing Qwen LoRA, frozen and active in training and previews (not with `--finetune`). `--metadata_title/author/description/license/tags/trigger_phrase` are recorded in the saved LoRA. The saved file loads in ComfyUI's standard LoRA loader.

---

## Sample previews during training

`--sample_prompts` takes a text file, one plain prompt per line, `#` lines are comments; geometry and seed come from `--sample_width` / `--sample_height` / `--sample_seed`.

**Klein** renders previews on the Distilled model (4-step, fast, matches ComfyUI) with `--preview_checkpoint <distilled>`; without it they render on the training Base, where `--sample_cfg_scale` (4.5 by default; 1.0 disables CFG) and `--sample_negative` steer them. `--sample_reference FILE` makes every preview an edit of that photo.

**Krea 2** and **Qwen Image 2.1** prompt files are plain prompts only; geometry and seed come from `--sample_width` / `--sample_height` / `--sample_seed`, and previews render on the training model, with the speed LoRA when `--speed_lora` is given (`--sample_steps`: 8 for Krea's Turbo LoRA, 6 for Qwen's). `--sample_cfg_scale` above 1 enables CFG on the previews — pair it with `--sample_negative` for a real uncond. `--sample_at_first` renders an epoch-0 preview (base + zero-init LoRA) before training. A reference picture: Krea 2's `--sample_image FILE` feeds it through the Qwen3-VL vision path (generate driven by a reference picture; works even with an empty prompt file), and Qwen's `--sample_reference FILE` makes every preview an edit of that photo. `--metadata_title/author/description/license/tags` are recorded in the saved LoRA as `modelspec.*` keys.

**MiniMax H3** prompt files are plain prompts too. Previews are clips: `--sample_frames` sets the length on the model's 17n+5 frame grid (1 = a still, 56 ≈ 2.3 s, 124 = the trained minimum of ~5 s; off-grid values snap down), `--sample_audio` with `--audio_vae` decodes the clip's generated sound to a `.wav` beside the `.mp4`, and `--turbo_lora_path` with `--sample_steps 6` renders on the Turbo LoRA at `--turbo_lora_strength 0.75` (20 steps without it, matching ComfyUI's shipped template). `--sample_width/height` default to H3's native 768. The training adapter is switched off for previews; a Context LoRA stays on. Rendering a 56-frame clip every epoch costs more than the epoch's training on a small dataset — set `--sample_every_n_epochs` higher, or preview stills, when speed is the point.

**Qwen Image 2.1** prompt files are plain prompts; `--sample_width/height/seed` set the rest. With `--speed_lora` the previews render with the Viggle turbo LoRA at strength 1.0 for 6 steps, on its own schedule (`--speed_lora_strength` and `--sample_steps` override; other step counts use the model's standard schedule). Without it they render at 25 steps. `--sample_cfg_scale` above 1 applies with or without the turbo LoRA, with `--sample_negative`; no turbo, `--sample_steps 20 --sample_cfg_scale 3` works very well, at roughly twice the time per preview. The training adapter is off for previews and a Context LoRA stays on. `--sample_at_first` renders an epoch-0 preview.

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
- **MiniMax H3**: int8 at ~21 GB resident on 32 GB cards, 4-bit at ~11 GB below that, the swap planned with it.
- **Qwen Image 2.1**: bf16 on 24 GB+ cards, INT8 with no swap on 12 and 16 GB cards and NF4 on 10 GB cards, at the presets' 0.5 MP. The text encoder only encodes and loads 8-bit below about 20 GB free, which sets the 10 GB floor.

---

## LoRA extraction

`extract_lora.py` reduces a LoRA to a lower rank — exact weight SVD straight from the file, no model loaded, in the family's own key format. It handles LyCORIS (LoKR) sources too:

```bash
python src/fizgig/scripts/extract_lora.py --family klein \
  --source big_r32.safetensors --output small_r8.safetensors --rank 8
```

`--preset` keeps one of the family's block groups — Klein: `Identity` (single 1-16), `Style+Composition` (double 0-7 + single 0-1, single 2 at half strength), `Details` (single 12-23) — and `--blocks single_1,single_2,...` picks blocks by id. The Extract tab does the same with a Custom block picker.

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
