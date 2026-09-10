# Fizgig Headless CLI

Everything the GUI does for training runs through the scripts in `src/fizgig/scripts/` — the GUI is a front-end that builds these exact commands and runs them as subprocesses. That means the CLI is always feature-complete for all three families — Klein 9B, Krea 2 and MiniMax H3: adaptive LR, the per-image loss watch, auto-recaptioning, Context LoRA, video and voice training, pause/resume — all of it is available from a plain terminal, on **Windows and Linux alike** (including display-less boxes).

All commands below are run from the repo root. The scripts add `src/` to `sys.path` themselves, so the direct form always works:

```bash
# Windows (bundled venv, cmd or PowerShell)
venv\Scripts\python.exe src\fizgig\scripts\train.py --help

# Linux / macOS
python src/fizgig/scripts/train.py --help
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
- [Sample previews during training](#sample-previews-during-training)
- [Pause and resume](#pause-and-resume)
- [VRAM guidance (block swap)](#vram-guidance-block-swap)
- [LoRA extraction (rank reduction / specialization)](#lora-extraction)
- [LoRA profiling](#lora-profiling)
- [Analyzing a per-image loss log](#analyzing-a-per-image-loss-log)

---

## Model files: where they come from, where they go

Headless, there is no Preferences tab: **model locations are passed as flags on every command** (`--dit`, `--vae`, `--text_encoder`, for Krea 2 previews `--turbo_dit`, for MiniMax H3 `--audio_vae`, `--turbo_lora_path` and `--training_adapter_path`). The CLI does not read the GUI's `prefs.json` — put the paths in a shell script or Makefile once and forget about them. The files themselves are the same ones the GUI's Preferences tab links to:

**Klein 9B:**

| File | Download | Used for |
|---|---|---|
| Base DiT (fp8, recommended) | [FLUX.2-klein-base-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/tree/main) | training (`--dit`) |
| Base DiT (bf16, big cards) | [FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/tree/main) | training (`--dit`) |
| Distilled DiT (fp8) | [FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/tree/main) | fast 4-step previews (`--sample_dit`), extraction/profiling |
| VAE `ae.safetensors` | [FLUX.2-dev → ae.safetensors](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) | `--vae` |
| Text encoder `qwen_3_8b.safetensors` | [Comfy-Org Klein text encoder](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors) | `--text_encoder` |

> **VAE trap:** use `ae.safetensors` from the FLUX.2-dev repo **root** — not `vae/diffusion_pytorch_model.safetensors`, which is the Diffusers-format file and will not load.

**Krea 2:**

| File | Download | Used for |
|---|---|---|
| RAW DiT | [krea2_raw_bf16.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors) | training (`--dit`) |
| fp8 Turbo DiT | [krea2_turbo_fp8_scaled.safetensors](https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors) | 8-step previews (`--turbo_dit`) |
| VAE `qwen_image_vae.safetensors` | [Krea-2 → vae](https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors) | `--vae` |
| Text encoder `qwen3vl_4b_bf16.safetensors` | [Krea-2 → text_encoders](https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_bf16.safetensors) | `--text_encoder` |

**MiniMax H3:**

| File | Download | Used for |
|---|---|---|
| DiT, pruned int8 (~21 GB) | [minimax_h3_fl2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | training (`--dit`) — the same file ComfyUI runs |
| DiT, reference build *(optional)* | [minimax_h3_ref2va_pruned_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) | `--dit` for LoRAs that live in the reference-to-video workflow, and for `--distill` |
| Text encoder, nvfp4 (~15.7 GB) | [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | `--text_encoder` (caching, and pre-encoding preview prompts) |
| Video VAE | [minimax_h3_video_vae_fp16.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors) | `--vae` |
| Audio VAE *(optional)* | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `--audio_vae` — sound in clips, voice recordings, previews with sound |
| Turbo LoRA *(optional)* | [minimax_h3_turbo_v4_step600.safetensors](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/blob/main/minimax_h3_turbo_v4_step600.safetensors) | 6-step previews (`--turbo_lora_path`) |
| Training adapter *(recommended)* | [minimax_h3_training_adapter_v1.safetensors](https://huggingface.co/ostris/minimax_h3_training_adapter/blob/main/minimax_h3_training_adapter_v1.safetensors) (fl2va) · [ref2va file](https://huggingface.co/ostris/minimax_h3_training_adapter/blob/main/minimax_h3_ref2va_training_adapter_v1.safetensors) | `--training_adapter_path` — match it to `--dit` |

---

## What's family-specific at a glance

The two model families share the dataset format and most of the workflow, but not every feature exists on both sides:

| Feature | Klein 9B | Krea 2 |
|---|---|---|
| Adaptive LR (`--adaptive_lr`) | ✅ | ✅ |
| Context LoRA | ✅ | ✅ |
| Pause / resume | ✅ | ✅ |
| NF4 4-bit base (`--quant_4bit` / `--quantize_4bit`) | ✅ | ✅ |
| Per-image loss watch (`--log_per_image_loss`) | ❌ | ✅ Krea 2 only |
| Per-image LR (`--per_image_lr`) | ❌ | ✅ Krea 2 only |
| Auto-recaption (`--auto_recaption`) | ❌ | ✅ Krea 2 only |
| Look-outlier warm-up (`--warmup_look_outliers`) | ❌ | ✅ Krea 2 only |
| LR scheduler (cosine, linear, warmup, ...) | ✅ | ✅ |
| Gradient accumulation | ✅ | ✅ |
| Block targeting (`include_patterns`) / Model Area | ✅ Klein only | ❌ (no Krea 2 block map yet) |
| Timestep range (`--min/max_timestep`) | ✅ Klein only | ❌ (fixed `krea2_shift` recipe) |
| Optimizer choice (`--optimizer_type`) | ✅ | ✅ |
| INT8 W8A8 base (`--quant_int8`) | ❌ | ✅ Krea 2 only |
| torch.compile speedup | ✅ (`--compile` + flags) | ✅ (`--compile_blocks auto`, on by default when it pays) |
| Weight-only extraction (rank reduction, `--samples 0`) | ✅ | ✅ |
| Profiling | ✅ full activation profile | ✅ weight-only (`--krea2`) |
| Activation-weighted (specialized) extraction | ✅ Klein only | ❌ (needs the Klein pipeline) |

The four intelligence toggles are Krea 2-only because auto-recaption needs a text encoder that can *see* — Krea 2's Qwen3-VL is a full vision-language model; Klein's stripped Qwen3-8B can't generate text or look at images.

**MiniMax H3** is the third family and has its own three scripts and flag set: photos, video clips with their sound, and voice recordings in one dataset; Optimised Likeness Learning on by default; an int8 or 4-bit base sized to your card; weight averaging; Turbo previews as clips with sound; Context LoRA, the training adapter, TREAD, pause/resume, and full fine-tuning. See [MiniMax H3 training](#minimax-h3-training).

---

## The three-step pipeline

Training is always three steps: cache the VAE latents, cache the text-encoder outputs, then train. Latents and text embeddings are computed once and reused across epochs (and across runs, if the dataset hasn't changed).

**Klein 9B:**

```bash
python src/fizgig/scripts/cache_latents.py --dataset_config my_dataset.toml --vae /models/ae.safetensors
python src/fizgig/scripts/cache_text.py    --dataset_config my_dataset.toml --text_encoder /models/qwen_3_8b.safetensors
python src/fizgig/scripts/train.py         --dataset_config my_dataset.toml ...   # full example below
```

**Krea 2:**

```bash
python src/fizgig/scripts/krea2_cache_latents.py --dataset_config my_dataset.toml --vae /models/qwen_image_vae.safetensors
python src/fizgig/scripts/krea2_cache_text.py    --dataset_config my_dataset.toml --text_encoder /models/qwen3vl_4b_bf16.safetensors
python src/fizgig/scripts/krea2_train.py         --dataset_config my_dataset.toml ...   # full example below
```

**MiniMax H3:**

```bash
python src/fizgig/scripts/minimax_cache_latents.py --dataset_config my_dataset.toml --vae /models/minimax_h3_video_vae_fp16.safetensors --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors --clip_still --skip_existing
python src/fizgig/scripts/minimax_cache_text.py    --dataset_config my_dataset.toml --text_encoder /models/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
python src/fizgig/scripts/minimax_train.py         --dataset_config my_dataset.toml ...   # full example below
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
- `fizgig_excluded.json` — the per-image watch's persistent exclusion list, written during training. It follows the images across runs; editing an excluded image's caption auto-pardons it. Delete the file to give everything a clean slate.

---

## Klein 9B training

You need the four Klein model files from the [download table](#model-files-where-they-come-from-where-they-go) above (Base DiT, optional Distilled for previews, `ae.safetensors`, `qwen_3_8b.safetensors`).

### Full example

A realistic identity LoRA run (rank 16, adaptive LR, per-epoch checkpoints, Distilled previews every epoch), mirroring the GUI's ✨Identity r16 preset:

```bash
python src/fizgig/scripts/train.py \
  --dataset_config my_dataset.toml \
  --dit /models/flux-2-klein-base-9b.safetensors \
  --vae /models/ae.safetensors \
  --text_encoder /models/qwen_3_8b.safetensors \
  --sdpa \
  --mixed_precision bf16 \
  --fp8_base --fp8_scaled \
  --gradient_checkpointing \
  --blocks_to_swap 0 \
  --optimizer_type adamw8bit \
  --learning_rate 1e-4 \
  --network_module fizgig.networks.lora_klein \
  --network_dim 16 --network_alpha 16 \
  --timestep_sampling flux2_shift \
  --max_train_epochs 55 \
  --save_every_n_epochs 1 \
  --save_state \
  --seed 42 \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject \
  --pause_flag_path ./output_loras/my_subject/.pause_requested \
  --adaptive_lr --adaptive_lr_min 5e-5 --adaptive_lr_max 4e-4 \
  --sample_dit /models/flux-2-klein-9b.safetensors \
  --sample_prompts sample_prompts.txt \
  --sample_every_n_epochs 1
```

Checkpoints land as `output_loras/my_subject/my_subject-000001.safetensors` (epoch 1) etc., with the final LoRA as `my_subject.safetensors` — all directly loadable in ComfyUI, no conversion.

### The flags that matter

**Required plumbing**

- An attention flag is mandatory: `--sdpa` (PyTorch native, always available — use this) or `--flash_attn` / `--flash3` / `--xformers` / `--sage_attn` if you have them installed.
- `--mixed_precision bf16` and `--gradient_checkpointing` should be considered defaults. Turning checkpointing off is ~20-30% faster steps but much higher VRAM — only sensible on big cards with no block swap.
- `--network_module fizgig.networks.lora_klein` is the LoRA implementation; always pass it.

**Precision / VRAM**

- Training on a **bf16** Base: pass `--fp8_base --fp8_scaled` (quantizes to fp8 at load, ~halves DiT memory, this is the validated recipe).
- Training on a **pre-quantized fp8** Base file: pass **neither** — the file is already fp8, and re-quantizing degrades it.
- `--quant_4bit` — QLoRA-style NF4 4-bit frozen base for very low-VRAM cards. Supersedes the fp8 flags and forces block swap off.
- `--blocks_to_swap N` — see [VRAM guidance](#vram-guidance-block-swap).

**LoRA shape**

- `--network_dim` / `--network_alpha` — rank and alpha. The GUI presets use matched pairs: 4/4 (style, details), 8/8, 16/16 (identity). Higher rank captures more but overfits sooner.
- `--network_args loraplus_lr_ratio=4` — optional LoRA+ (higher LR on the up-projection).

**Learning rate**

- `--learning_rate` — the trainer's raw default is a placeholder; always pass one. GUI presets use 1e-4 (rank 16 identity) to 4e-4 (rank 4 style/details).
- `--adaptive_lr` — the bi-directional plateau tracker: probes LR up on steady descent, cuts it (with weight rollback) on plateau or instability. When active it takes over from `--lr_scheduler`. Bounds via `--adaptive_lr_min` (floor — 5e-5 for single-subject sets, 1e-4 for noisy multi-subject sets; the rank-4/8 Identity presets ship 2e-4) and `--adaptive_lr_max` (4e-4 is the empirical ceiling). **`--learning_rate` is ignored while adaptive is on**: the run starts at the geometric midpoint of the Min/Max window (1e-4 & 4e-4 → 2e-4) and the watcher owns the LR from there. Resumed runs keep their mid-flight LR.
- Without adaptive LR, the usual `--lr_scheduler` options exist (`constant`, `cosine`, `constant_with_warmup`, ...) with `--lr_warmup_steps` / `--lr_decay_steps`.

**Batching**

- `--gradient_accumulation_steps N` — accumulate over N micro-batches per optimizer step, so the effective batch is `batch_size × N`. The usual reason to reach for it is a bigger effective batch than VRAM allows at once.
- `--max_grad_norm` — gradient clipping (default 1.0; 0 disables).

**Training only part of the model** (the GUI's "Model Area to Train")

`--network_args include_patterns=[...]` restricts which blocks get LoRA modules, and `--min_timestep` / `--max_timestep` (0-1000) restrict the noise range. The GUI presets translate to:

| Preset | `include_patterns` | Timesteps |
|---|---|---|
| Full Model | *(omit)* | *(omit)* |
| Identity | `[".*single_blocks\\.(1[0-6]|[1-9])\\..*"]` | — |
| Style / Style+Composition | `[".*double_blocks\\..*",".*single_blocks\\.[01]\\..*"]` | Style adds `--min_timestep 0 --max_timestep 400` |
| Details | `[".*single_blocks\\.(1[2-9]|2[0-3])\\..*"]` | — |

Shell-quoting example (note the single quotes — the value contains regex backslashes and brackets):

```bash
--network_args 'include_patterns=[".*double_blocks\..*",".*single_blocks\.[01]\..*"]' \
--min_timestep 0 --max_timestep 400
```

Style genuinely lives at late timesteps (0-400) on Klein — combining the style blocks with that range is the validated recipe for style LoRAs.

**Context LoRA** (train on top of an existing LoRA)

```bash
--context_lora_path /loras/style_anchor.safetensors --context_lora_strength 1.0
```

Loads the existing LoRA frozen-but-active on the base, so your new LoRA learns to coexist with it — identity-on-style, outfit-on-character, compatibility patches. At inference, deploy the pair together at the same strength. Accepts kohya, PEFT/Diffusers, OneTrainer, LoKR and LoHa formats.

**Checkpoints / state**

- `--save_every_n_epochs 1` + `--save_state` — per-epoch LoRA checkpoints plus resumable state dirs (`<name>-NNNNNN-state/`). State is what makes pause/resume and epoch-scrubbing possible; keep it on.
- `--save_state_on_train_end` — write a state dir when the run finishes, so a completed LoRA can be trained further later by raising `--max_train_epochs` and resuming. Without it, a finished run leaves only the LoRA and can't be continued.
- `--keep_last_n_states 2` — keep only the N newest state dirs. Each is the LoRA plus optimizer moments (~470 MB at rank 32), so an unpruned 55-epoch run leaves tens of GB behind. Only dirs matching the current `--output_name` are touched, and the newest is always kept (values below 1 are clamped).
- `--resume <path-to-state-dir>` — continue a run (see [Pause and resume](#pause-and-resume)).

Both trainers write the same `<name>-NNNNNN-state/` layout, where `NNNNNN` is the number of completed epochs. Pause always writes state regardless of the flags above.

---

## Krea 2 training

Krea 2 (12.9B) is the second model family and the home of the intelligent-trainer features: the per-image loss watch, auto-recaptioning, per-image adaptive LR, and look-outlier warm-up are **Krea 2-only** (see the [family table](#whats-family-specific-at-a-glance)). You need the four Krea 2 files from the [download table](#model-files-where-they-come-from-where-they-go): the RAW DiT for training (fp8-quantized at load, ~14 GB resident), the fp8 Turbo for previews, the Qwen-Image VAE, and the Qwen3-VL-4B text encoder (which doubles as the vision model for auto-recaptioning).

### Full example

Everything on — the self-adapting run: rank 16:16, adaptive LR, all four intelligence toggles, and NF4 4-bit training so it fits low-VRAM cards:

```bash
python src/fizgig/scripts/krea2_train.py \
  --dataset_config my_dataset.toml \
  --dit /models/Krea-2-raw.safetensors \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject \
  --network_dim 16 --network_alpha 16 \
  --learning_rate 1e-4 \
  --max_train_epochs 40 \
  --save_every_n_epochs 1 \
  --quantize_4bit \
  --seed 42 \
  --adaptive_lr --adaptive_lr_min 5e-5 --adaptive_lr_max 4e-4 \
  --log_per_image_loss \
  --per_image_lr \
  --auto_recaption \
  --warmup_look_outliers \
  --trigger_word ohwx \
  --turbo_dit /models/krea2-turbo-fp8.safetensors \
  --vae /models/qwen_image_vae.safetensors \
  --text_encoder /models/qwen3vl_4b_bf16.safetensors \
  --sample_prompts sample_prompts.txt \
  --sample_every_n_epochs 1 \
  --sample_width 1024 --sample_height 1024
```

(`--quantize_4bit` forces block swap off, so no `--blocks_to_swap` here. On a 24 GB+ card you'd drop `--quantize_4bit` and use the default dynamic fp8 with `--blocks_to_swap` from the [VRAM table](#vram-guidance-block-swap). Drop `--warmup_look_outliers` unless you've run the GUI's Look Filter on the dataset — see below.)

**Classic-recipe variant** — if you'd rather drive the LR yourself than hand it to the adaptive watcher: a cosine decay with warmup, and an effective batch of 2 via accumulation. Swap these lines into the run above, dropping `--adaptive_lr*` (the watcher and a schedule are mutually exclusive — adaptive wins, and the schedule is ignored with a log line):

```bash
  --lr_scheduler cosine \
  --lr_warmup_steps 100 \
  --gradient_accumulation_steps 2 \
  --max_grad_norm 1.0 \
```

Pause/resume continues the cosine curve where it left off rather than restarting it, with or without accumulation.

The Krea 2 parser is small enough to know in full: run `krea2_train.py --help`. The non-obvious flags:

**Core**

- `--network_type lokr` + `--lokr_factor N` — train **LoKR** (LyCORIS Kronecker) instead of standard LoRA (the GUI's Network Type dropdown, headless here). One dial: the factor sets the Kronecker split, and dim/alpha are ignored. Lower factor ≈ more capacity and bigger files (factor 8 ≈ 400 MB, 16 ≈ 100 MB); **8 is the validated default, and going above it isn't worth it** — measured head-to-head, higher factors keep LoKR's ~20% step-time cost over standard LoRA while losing the quality edge that justifies it. Want smaller/faster? Use standard LoRA at low rank instead. Output is standard LyCORIS format (`diffusion_model.*.lokr_*`) — loads directly in ComfyUI and back into every Fizgig tool, where Repair Studio and Explorer save it natively (lossless; SVD only on donor-blended blocks). In our validation runs LoKR at factor 8 hit the highest likeness we've ever measured, with noticeably more natural skin than standard LoRA on the same data.
- `--no_fp8` — train the base in bf16 instead of dynamic fp8 (needs a lot more VRAM; fp8 is the validated default).
- `--quantize_4bit` — NF4 4-bit frozen base, ~5.6 GB DiT residency, fits 10-12 GB cards (block swap forced off).
- `--quant_int8 bf16` — INT8 W8A8 frozen base: ~18.6 GB, so it needs a 24 GB card, and in exchange it is both the fastest option measured (0.637 s/it vs NF4's 0.709 on an RTX 5090) and ~7× more accurate than NF4 in forward error, since 8 bits beat 4. `bf16` keeps gradients exact; `int8` quantises the backward too — faster again, lossier. The GUI picks this automatically when there is free VRAM for it; on the CLI it is opt-in. Mutually exclusive with `--quantize_4bit`. Block swap is force-zeroed under INT8 (the staged quantise makes the model fully resident anyway).
- `--compile_blocks auto|on|outside|off` — torch.compile the transformer blocks (`outside` = the high-resolution boundary: checkpoint kept outside the compiled region, eager-level memory, chosen automatically by `auto`/`on` where inside-the-graph would not fit): roughly **2× faster steady-state steps** on the INT8 path after a one-off warm-up (~90 s + a pause on each new latent shape). `auto` (default) weighs the warm-up against the run length and only compiles when it pays. Needs triton (installs with requirements; `triton-windows` on Windows) and, on Windows, the MSVC C++ Build Tools — direct installer: https://aka.ms/vs/17/release/vs_BuildTools.exe ("Desktop development with C++" workload). Missing either → a console note and the run continues uncompiled.
- `--blocks_to_swap` — see [VRAM guidance](#vram-guidance-block-swap). `--preview_blocks_to_swap` is the separate, forward-only swap for the preview Turbo. **Swapping is the slow path** (4.4× the time, 4× the CPU): quantise first, and only swap when even NF4 will not fit.

**The per-image loss watch** (any of these enables the watcher)

- `--log_per_image_loss` — tracks every image's loss residual against its timestep bucket, classifies each image at every epoch boundary (`easy` / `suspect` / `stuck` / `exhausted` / `excluded`), and writes:
  - `<output_dir>/loss_log/per_image_loss.jsonl` — the raw per-step log
  - `<output_dir>/loss_log/problem_images.json` — current verdicts + trends (the GUI's Problem Images window reads this; it's plain JSON, perfectly greppable headless)
  - a **plateau banner** in the console when ≤~5% of images are still improving, with a best-checkpoint epoch estimate — your "you're done" signal.
- `--per_image_lr` — acts on the verdicts: stuck images get throttled (×0.5 → ×0.25 → ×0.125), mined-out images ease off (×0.6), the healthy cohort gets a gentle boost (×1.1). Batch size 1 makes this a true per-image learning rate.
- `--auto_recaption` — between epochs, confirmed-stuck images get their captions rewritten by Qwen3-VL from what's actually visible, the text cache is re-encoded, and the image gets a fresh start. Two failed attempts and a still-stuck image is excluded from the run entirely. Requires `--text_encoder`. Pass `--trigger_word` so rewritten captions keep your trigger (appended as `, <trigger>`).
- `--warmup_look_outliers` — curriculum entry (×0.4 LR ramping to ×1.0) for real-but-unusual images. Reads `<dataset>/fizgig_look_scores.json`, which is produced by the GUI's Look Consistency Filter scan — **GUI-only prerequisite**; without the file this flag logs a warning and disables itself.

Persistent artifacts: exclusions are stored in `<image_directory>/fizgig_excluded.json` so they travel with the dataset across runs; editing an excluded image's caption auto-pardons it. Fresh runs rotate the old JSONL to `.bak`; `--resume` replays the log to restore full watch history.

**LR schedule and batching**

- `--lr_scheduler` — `constant` (default), `constant_with_warmup`, `cosine`, `cosine_with_restarts`, `linear`, `polynomial`; with `--lr_warmup_steps` (plus `--lr_scheduler_num_cycles` / `--lr_scheduler_power` for the two that use them). **Ignored when `--adaptive_lr` is on** — the plateau watcher owns the LR and says so in the log. Resume continues the curve rather than restarting it.
- `--gradient_accumulation_steps N` — accumulate over N micro-batches per optimizer step (effective batch = N). The loss is averaged over the group, and a partial group is flushed at the epoch boundary. Per-image LR still applies per image.
- `--max_grad_norm` — gradient clipping (default 1.0, matching the reference recipe; 0 disables).

**Optimizer**

- `--optimizer_type` — `adamw8bit` (default, the validated recipe), `adamw` (fp32 state, CUDA-fused where available), `pagedadamw8bit`, `ademamix8bit`, `pagedademamix8bit`, `lion8bit`. Anything else is taken as a full `module.path.ClassName`, so an optimizer Fizgig has never heard of works without a release. The list the CLI actually offers on your machine is in `--help` — entries whose package is missing are filtered out. (`adafactor`, `prodigy` and `came` were removed: they manage their own learning rate, which conflicts with Adaptive LR and produced real failures; reach them via the module-path form if you know what you're doing.)
- `--optimizer_args` — extra kwargs, e.g. `--optimizer_args "weight_decay=0.01 betas=0.9,0.99"`. Values are parsed as Python literals.

Two warnings worth internalising before you reach for the dropdown. **Learning rates do not transfer between families:** Lion applies the *sign* of the update and wants roughly a tenth of an AdamW LR. Fizgig logs a warning when the LR looks wrong for the family, but it will not override you. And **saving optimizer memory buys you little here** — a LoRA's state is tens of MB against a 13–19 GB base, so the reason to change optimizer is update *behaviour*, not VRAM. If a run fails to construct the optimizer it falls back to plain AdamW and says so in the log; the choice is recorded in the output LoRA as `ss_optimizer`.

**Not in the Krea 2 parser (by design):** timestep sampling (fixed `krea2_shift` recipe), block targeting (no Krea 2 block map yet), gradient checkpointing (always on). What's absent is deliberate, not missing.

---

## MiniMax H3 training

MiniMax H3 (33B) trains on photos, video clips with their sound, and voice recordings, all from one dataset folder — the [dataset TOML](#dataset-config-toml) is the same, and clips and audio files sit beside the images with their own `.txt` captions. You need the files from the [download table](#model-files-where-they-come-from-where-they-go): the pruned int8 DiT, the nvfp4 text encoder, the video VAE, and, for sound and voice, the audio VAE. The Turbo LoRA and the training adapter are optional but both are in the default recipe.

The cache steps are the ones in [the three-step pipeline](#the-three-step-pipeline). `--audio_vae` on the latent cache is what makes a clip's sound a training target (and is required by voice recordings); `--clip_still` picks each clip's sharpest face frame and caches it as a still, for `--clip_still_as_photo` at training time.

### Full example

The GUI's **Fast** preset, exactly as it builds it — rank 8, flat 2e-4, 50 epochs, everything the app has on by default:

```bash
python src/fizgig/scripts/minimax_train.py \
  --dataset_config my_dataset.toml \
  --dit /models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --output_dir ./output_loras/my_subject \
  --output_name my_subject_mmh3 \
  --network_dim 8 --network_alpha 8 \
  --learning_rate 2e-4 \
  --max_train_epochs 50 \
  --save_every_n_epochs 1 \
  --seed 42 \
  --optimizer_type adamw \
  --base_quant auto --blocks_to_swap auto --gradient_checkpointing auto \
  --shift 0.666667 \
  --ema_decay 0.98 \
  --caption_dropout 0.05 \
  --photo_blocks 20-49 --clip_blocks 20-49 --audio_blocks 34-49 \
  --no_train_adaln \
  --training_adapter_path /models/minimax_h3_training_adapter_v1.safetensors \
  --tread_ratio 0.5 --tread_start 2 --tread_end 47 \
  --clip_still_as_photo \
  --save_state --save_state_on_train_end --keep_last_n_states 2 \
  --text_encoder /models/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
  --vae /models/minimax_h3_video_vae_fp16.safetensors \
  --sample_prompts sample_prompts.txt \
  --sample_every_n_epochs 1 --sample_at_first \
  --sample_width 768 --sample_height 768 --sample_seed 1234 \
  --sample_frames 56 --sample_audio --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --turbo_lora_path /models/minimax_h3_turbo_v4_step600.safetensors --turbo_lora_strength 0.75 --sample_steps 6
```

Two things are on with no flag at all: under `--photo_blocks` / `--clip_blocks` / `--audio_blocks` the backward pass stops at the trained window, and the LoRA does not train the model's text token refiner. Both are what the app ships; `--likeness_full_backward` and `--train_token_refiner` are the opt-outs, for A/Bs of your own.

### The flags that matter

**The recipe**

- `--photo_blocks 20-49 --clip_blocks 20-49 --audio_blocks 34-49` — Optimised Likeness Learning: photos and clips train the identity blocks, voice the audio zone. Leave them out to train the whole model (style and scene LoRAs; the GUI's Style preset does).
- `--training_adapter_path` — the training adapter, frozen at 1.0 for every training step, off for previews and absent from the saved LoRA. Use the fl2va or ref2va file to match `--dit`.
- `--ema_decay 0.98` — weight averaging: checkpoints and previews come from a smoothed average of the weights. `0` turns it off.
- `--shift 0.666667` — the GUI's **Likeness and Style** training structure, most of the run on nearly-clean images. Unset = the model's own movement-first schedule. `--highnoise_lr_scale` scales the LR of the noisy-half steps and is best left at 1.
- `--no_train_adaln` — the app always passes it: the deployed pruned builds cannot load AdaLN LoRA keys.
- `--tread_ratio 0.5 --tread_start 2 --tread_end 47` — TREAD token routing on clip steps; `--tread_ratio 0` turns it off.
- `--clip_still_as_photo` — each clip's cached sharpest face frame trains as a photo of its own (needs `--clip_still` at caching).
- `--caption_dropout 0.05` — a few percent of steps train with no caption (needs the empty-prompt embed the text cache writes).
- `--network_type lokr --lokr_factor 8` — LoKR instead of LoRA. Standard LoRA stays the recommendation on H3.

**Memory and precision**

- `--base_quant auto|int8|nf4|hqq` — `auto` reads free VRAM at launch and picks the base precision and block swap together: int8 is the checkpoint's own weights and the most accurate (~21 GB resident), nf4 fits smaller cards (~11 GB), hqq sits between (never picked by auto). `--blocks_to_swap auto` plans the swap from free VRAM and the largest bucket; `--gradient_checkpointing auto` turns it on only when needed.
- `--gradient_accumulation_steps N` — sum N batches per optimizer step.

**Video and voice**

- `--audio_weight` — the weight on the audio term for clips with sound; raise it if the per-epoch `[audio]` line shows sound taking a negligible share of the loss.
- `--visual_stop_epoch N --visual_stop_mode anchor|stop`, `--audio_stop_epoch N --audio_stop_mode` — mixed datasets: finish the smaller category early. `anchor` keeps it at 10% LR with its epoch report live; `stop` skips its steps.

**Previews** — see [Sample previews during training](#sample-previews-during-training) for the H3 flags (`--sample_frames`, `--sample_audio`, the Turbo LoRA).

**Learning-rate helpers**

- `--adapter_ramp R` — adapter-relative LR: the Learning Rate becomes a ceiling the run climbs toward (0.005 is the GUI's On value).
- `--lr_warmup_epochs N` — linear warm-up over the first N epochs.

**Experiments and opt-outs**

- `--train_token_refiner` — adds the text token refiner to the LoRA targets. Recommended off; does not affect the ability to use a trigger word.
- `--likeness_full_backward` — the old full backward under the likeness masks, with the refiner training on every step. A/B only.
- `--train_blocks SPEC` — train only these of the 50 blocks (`'20-49'`, `'3-12, 22, 31-33'`). `--slow_blocks SPEC --slow_block_lr_scale X` trains a range at a reduced LR. `--block_limit N` caps any block's single-step movement at N× the median block's.
- `--distill --distill_weight W --distill_phase1_epochs N` — reference distillation, trained on the ref2va DiT; needs `--reference_count K` on the text cache.
- `--context_lora_path FILE --context_lora_strength S` — train with an existing H3 LoRA frozen and active underneath, in training and previews.
- `--metadata_title/author/description/license/tags/trigger_phrase` — recorded in the saved LoRA.

**Full fine-tuning** — `--finetune_rotation N` trains the base model itself in component windows on an NF4-resident base and saves exact int8 checkpoints; `--finetune_rotate_every`, `--finetune_scope all|photo`, `--finetune_blocks`, `--finetune_master auto|ram|disk`, `--finetune_scratch_dir` and `--reg_lr_multiplier` go with it. The GUI's Fine-tune checkbox builds these; see `docs/FINETUNE_HOWDOI.md` before running one headless. Context LoRA, the training adapter and TREAD are LoRA-only and refused under rotation.

---

## Sample previews during training

`--sample_prompts` takes a text file, one prompt per line, `#` lines are comments.

**Klein** supports kohya-style per-prompt overrides appended after the prompt:

```
# sample_prompts.txt
ohwx man, portrait, studio light --w 1024 --h 1024 --d 42
ohwx man, full body, city street --w 832 --h 1216 --d 42 --l 4.0 --n blurry, low quality
```

Recognized: `--w` width, `--h` height, `--d` seed, `--s` steps, `--g` guidance (Distilled), `--l` CFG scale (Base — Base has no guidance embed, so `--l` is what actually steers it; 1.0 disables CFG), `--n` negative prompt, `--ci` control image.

Pass `--sample_dit <distilled>` to render previews on the Distilled model (4-step, fast) instead of the training Base; `--sample_blocks_to_swap` gives the sample model its own swap setting.

**Krea 2** prompt files are plain prompts only; geometry and seed come from `--sample_width` / `--sample_height` / `--sample_seed`, and previews render on the fp8 Turbo (`--turbo_dit`; `--sample_steps`, default 8). `--sample_cfg_scale` above 1 enables CFG on the previews — pair it with `--sample_negative` for a real uncond (with CFG at 1.0 a negative is ignored, with a log note). `--sample_at_first` renders an epoch-0 preview (base + zero-init LoRA) before training, and works even with `--sample_every_n_epochs 0`. `--sample_ref_image` enables the Qwen3-VL vision path (generate driven by a reference picture; works even with an empty prompt file). `--metadata_title/author/description/license/tags` are recorded in the saved LoRA as `modelspec.*` keys.

**MiniMax H3** prompt files are plain prompts too. Previews are clips: `--sample_frames` sets the length on the model's 17n+5 frame grid (1 = a still, 56 ≈ 2.3 s, 124 = the trained minimum of ~5 s; off-grid values snap down), `--sample_audio` with `--audio_vae` decodes the clip's generated sound to a `.wav` beside the `.mp4`, and `--turbo_lora_path` with `--sample_steps 6` renders on the Turbo LoRA at `--turbo_lora_strength 0.75` (20 steps without it, matching ComfyUI's shipped template). `--sample_width/height` default to H3's native 768. The training adapter is switched off for previews; a Context LoRA stays on. Rendering a 56-frame clip every epoch costs more than the epoch's training on a small dataset — set `--sample_every_n_epochs` higher, or preview stills, when speed is the point.

Samples are written to `<output_dir>/sample/` with the epoch number in the filename. Prefer ~1024×1024 on Klein and Krea 2 — sub-1024 previews degrade anatomy and undersell the checkpoint.

---

## Pause and resume

Pause is a file, which makes it fully scriptable:

```bash
# request a graceful pause (Klein, Krea 2 and MiniMax H3)
touch ./output_loras/my_subject/.pause_requested        # Linux/macOS
New-Item ./output_loras/my_subject/.pause_requested     # Windows PowerShell
```

At the next epoch boundary the trainer force-saves a full state dir, logs `[pause] requested ... Exiting cleanly`, and exits 0 — GPU freed, no quality loss. (Klein: pass `--pause_flag_path` as in the example so the trainer knows where to look. Krea 2 and MiniMax H3 watch `<output_dir>/.pause_requested` automatically.)

Resume by pointing at the state directory:

```bash
python src/fizgig/scripts/train.py       ... --resume ./output_loras/my_subject/my_subject-000012-state
python src/fizgig/scripts/krea2_train.py ... --resume ./output_loras/my_subject/my_subject-000012-state
python src/fizgig/scripts/minimax_train.py ... --resume ./output_loras/my_subject/my_subject_mmh3-000012-state
```

The epoch number is parsed from the dir name; optimizer, scheduler, RNG, dataloader state, adaptive-LR scalars, and (Krea 2) the full per-image watch history are all restored. Pass the same flags as the original run plus `--resume`.

**Training a finished LoRA further.** Resume its end-of-run state and raise `--max_train_epochs` in the same command — the state records how many epochs are already done, so a run resumed at its final epoch has nothing left to do and simply rewrites the final LoRA (the trainer says so plainly in the log rather than pretending it trained). Raising the ceiling is what gives it epochs to run:

```bash
# LoRA finished at 30 epochs; take it to 45
python src/fizgig/scripts/krea2_train.py ... --max_train_epochs 45 \
    --resume ./output_loras/my_subject/my_subject-000030-state
```

---

## VRAM guidance (block swap)

`--blocks_to_swap` parks transformer blocks in CPU RAM and streams them over PCIe — slower per step, but fits big models on small cards. What the GUI auto-detects:

| GPU VRAM | Klein `--blocks_to_swap` | Krea 2 `--blocks_to_swap` |
|---|---|---|
| 32 GB | 0 | 0 |
| 24 GB | 0 | 12 |
| 16 GB | 0 | 20 |
| 10-14 GB | 12 | 26 |
| < 10 GB | 16 | *(use `--quantize_4bit` instead)* |

Klein's fp8 Base is only ~9.6 GB resident, so 16 GB+ cards skip swap entirely (faster — no PCIe transfers). Krea 2's fp8 RAW is ~14 GB resident, hence the more aggressive ladder. `--quantize_4bit` (both trainers) is the below-10 GB escape hatch and forces swap off.

MiniMax H3 plans for itself: `--base_quant auto --blocks_to_swap auto` reads the free VRAM at launch and picks the base precision (int8 at ~21 GB resident on 32 GB cards, 4-bit at ~11 GB below that) and the swap together, and logs the plan as a `[vram]` line. Set either by hand to override.

---

## LoRA extraction

`extract_lora.py` distills an existing LoRA to a lower rank, optionally specialized to block categories and timestep ranges.

**Rank reduction (the common case)** — pure weight SVD with `--samples 0`: no model paths, no GPU models loaded, runs straight from the safetensors. Model-agnostic, so it handles **both Klein and Krea 2** LoRAs:

```bash
python src/fizgig/scripts/extract_lora.py \
  --source big_r32.safetensors --output small_r8.safetensors \
  --rank 8 --blocks all --samples 0
```

For Krea 2 sources always use `--blocks all` — the named block categories are Klein's semantic map. Rank-reducing a full-model LoRA takes **~5 minutes on Klein** and **~25 minutes on Krea 2** (264 modules with 6144-wide dense deltas to SVD) — the long quiet stretch is normal, not a hang.

**Specialized extraction** (Klein only) — activation-weighted SVD (`--samples 16`, loads the full Klein pipeline, needs the model paths). Example: pull just the style out of a character LoRA:

```bash
python src/fizgig/scripts/extract_lora.py \
  --source character.safetensors --output character_style_only.safetensors \
  --dit ... --vae ... --text_encoder ... \
  --rank 4 --blocks style_composition --timesteps late \
  --samples 16 --prompt "a photo in the style of ohwx"
```

- `--blocks`: `all`, `style_composition` (double 0-7 + single 0-2), `identity` (single 1-16), `details` (single 12-23), or `custom` with `--custom_blocks "double_blocks.5,single_blocks.12"`.
- `--timesteps`: `all`, `late` (0.0-0.4 — where style lives), `mid`, `early`, `midlate`, `earlymid`.
- LyCORIS (LoKR/LoHa) sources work — the dense delta is materialized and SVD'd to a standard LoRA.

---

## LoRA profiling

`profile_lora.py` measures which blocks a LoRA actually uses and writes a report plus a JSON sidecar the GUI's Repair Studio picks up.

**Klein** — full activation profile (weight norms + activation probes across timestep bins, needs the model paths):

```bash
python src/fizgig/scripts/profile_lora.py \
  --lora my_subject.safetensors \
  --dit /models/flux-2-klein-9b.safetensors \
  --vae /models/ae.safetensors --text_encoder /models/qwen_3_8b.safetensors \
  --prompt "ohwx man, portrait" \
  --output my_subject_profile.png \
  --blocks_to_swap 12
```

Use the Distilled DiT for speed. PEFT and LyCORIS LoRAs auto-convert on load.

**Krea 2** — weight-only per-block profile, no models loaded, runs in seconds:

```bash
python src/fizgig/scripts/profile_lora.py --lora my_krea2_subject.safetensors --krea2
```

Writes `<name>_krea2_profile.html` next to the LoRA (or pass `--output report.html`) with per-block bars ranked by depth, plus the Repair Studio sidecar. Krea 2's block *roles* aren't mapped yet — this report is the instrument for discovering them, so if you spot patterns, share them on GitHub.

---

## Analyzing a per-image loss log

For a quick headless read of a Krea 2 run's per-image data (no GUI needed):

```bash
python src/fizgig/scripts/analyze_loss_log.py ./output_loras/my_subject --top 20
```

Points at the output dir (or the `per_image_loss.jsonl` directly) and prints the N hardest images by mean residual — the same signal that drives the watch's verdicts. Cross-reference against your captions: the top offenders are usually caption-viewpoint mismatches, and fixing the caption beats letting the throttle handle it.
