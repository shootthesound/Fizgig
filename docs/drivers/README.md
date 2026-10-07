# Adding a model to Fizgig: the driver system

Every model Fizgig trains — Klein 9B, MiniMax H3, Krea 2, Qwen Image 2.1, SDXL and Anima — runs through one shared layer in `src/fizgig/families/`. A model is added as a **family**, and a family is three pieces:

| Piece | Where | What it holds |
|---|---|---|
| **Description** | `src/fizgig/families/<family>.py` | The facts: model files, latent rules, block layout, LoRA key format, sampling settings, presets, and which optional abilities the family has. Plain data, importable without torch. |
| **Driver** | `src/fizgig/<family>/driver.py` | The model code behind one interface (`FamilyDriver`): load, encode, one training step, generate, decode. |
| **Model package** | `src/fizgig/<family>/` | The DiT (or UNet), VAE and text-encoder code the driver calls. |

Register the description in `src/fizgig/families/registry.py`. That is the only shared file you edit. From those three pieces Fizgig builds:

- the caching and training entry points (`families/cache.py`, `families/train.py`), on the command line and from the GUI
- the GUI: the Base Model picker, its Preferences section and download buttons, the Training and Samples tabs, presets, validation and the training queue
- previews during training, LoRA saving in the family's own key format, resume and pause
- the workbench tabs: Repair Studio, LoRA the Explorer, Profiler, Extract and LoRA Royale

You don't edit the GUI, the trainer or the tools. If your model needs something the layer doesn't have yet, see [When the layer is missing something](#when-the-layer-is-missing-something).

## Before you start: open a Discussion

Adding a model (or fine-tuning for a model Fizgig already has) starts with a post in the repo's [New models & fine-tuning](https://github.com/shootthesound/Fizgig/discussions/new?category=new-models-fine-tuning) discussions, before any code: which model, where its weights and reference implementation live, and what you plan to support. Discussions are for that only; bugs, questions and feature requests go in [Issues](https://github.com/shootthesound/Fizgig/issues). It's a prerequisite for a model pull request. It lets us agree the scope, avoids two people building the same model, and gets you answers about the driver system early.

## The guides

1. **[Entry points](#entry-points-what-calls-your-driver)** (below): what calls your driver, and when.
2. **[A stills model](STILLS.md)**: the minimum description and driver for an image model, step by step.
3. **[A video model](VIDEO.md)**: what a model that trains on clips (and optionally sound) adds.
4. **[Abilities](ABILITIES.md)**: the optional features a description switches on, and what each one asks of the driver.
5. **[Fine-tuning](FINETUNE.md)**: optional. Full fine-tuning of the base model, which you, the maintainer or anyone else can add after the family ships.
6. **[Before you ship](CHECKLIST.md)**: the checks a new family must pass, and what a finished family includes (research, parity, fast presets, compile, INT8 attention, default negative and more).
7. **[Driver excerpts](EXCERPTS.md)**: real code from Qwen Image 2.1 (stills) and MiniMax H3 (clips and sound) at the places new drivers most often go wrong, kept identical to the source by `src/fizgig/families/doc_excerpts.py --check`.

The existing families are the best reference. Read them side by side:

| Family | Description | Driver | Worth reading for |
|---|---|---|---|
| Qwen Image 2.1 | `families/qwen_image.py` | `qwen_image21/driver.py` | The leanest complete stills driver: edit training, sliders, prompt travel, fine-tune |
| Klein 9B | `families/klein.py` | `klein/driver.py` | A preview checkpoint (Distilled), fp8 files, Model Area presets, Repair Studio categories |
| Krea 2 | `families/krea2.py` | `krea2/driver.py` | Named block areas (text fusion, I/O), a vision path for reference pictures, compile |
| MiniMax H3 | `families/minimax.py` | `minimax/driver.py` | Video and sound, family options, its own cache layout and workbench engine |
| SDXL | `families/sdxl.py` | `sdxl/driver.py` | A UNet through diffusers, any single-file checkpoint, conv LoRAs, kohya LDM names (`alias_flat`), a default negative |
| Anima | `families/anima.py` | `anima/driver.py` | A vendored DiT, parity measured against ComfyUI, fine-tuning with measured memory figures and a `file_prefix` |

## Entry points: what calls your driver

The driver is created by `description.load_driver()`. Nothing else in Fizgig knows your model's classes. These are the five places that call it.

### 1. Caching — `families/cache.py`

Run once per stage, before training. `--stage latents` and `--stage text` are separate processes, so only one model is ever in memory.

| Stage | Driver calls, in order |
|---|---|
| latents | `cache_stage(...)` first: return `True` to cache in your own layout and skip the rest. Otherwise `load_vae`, then `encode_images(vae, [image])` per item (and once more per control image, for edit and slider pairs) |
| text | `load_text_encoder` → `encode_text(te, captions)` per batch → `unload_text_encoder`. Edit pairs use `load_reference_text_encoder` → `encode_text_with_references` instead |

Each conditioning dict you return is stored as is and handed back, batched, to `training_loss` and `generate`. The keys are yours. Family options reach caching only as `--aux KEY=VALUE` pairs, in the `aux` argument of `cache_stage`.

### 2. Training — `families/train.py`

In order, for one run:

1. `set_options(...)` with the run's `--family_option` pairs (before the dataset is built), then `expand_train_blocks(...)`.
2. **Memory plan.** With Auto precision or swap, `plan_run(...)` (return `None` to use the shared plan, which reads the description's `train_memory`, `precisions` and `max_blocks_to_swap()`).
3. **Preview prompts** are encoded once, up front: `load_text_encoder` → `encode_text` → `unload_text_encoder`.
4. **Base model.** `load_planned(...)` (return `None` for the shared load), else `load_dit`, quantisation of the block map's Linears to INT8 / NF4, then `enable_block_swap` if blocks stream. Then `enable_gradient_checkpointing(dit, True)`.
5. **LoRA.** The trainable LoRA wraps `lora_target_names(dit)` (by default every module in `block_map(dit)`), written in the description's `LoRAFormat`. Frozen files (training adapter, context LoRA, preview speed LoRA) go through `convert_lora_state_dict` and `alias_flat`, then `frozen_file_added(...)`. `prepare_training(dit, group, net)` runs last.
6. **Each step:** `step_policy(batch, epoch)` (skip or scale this step), `batch_cond(batch, device)` (the cached dict, batched), then `training_loss(dit, latents, cond, generator, min_t=, max_t=)` → `(loss, info)`. The trainer reads `info["t"]`, and `info["lr_mult"]` if you set it. `after_optimizer_step()` follows each optimizer step.
7. **Previews** at the chosen epochs: `block_swap_mode(dit, inference=True)` if blocks stream, `generate(...)` per prompt, `park_for(...)` to make room for the decode, `decode(...)`, `save_preview(...)`, then everything back.
8. **Save.** The LoRA is written in the family's key format with Fizgig's metadata plus your `run_metadata()`.

Pause, resume, EMA, Adaptive LR, gradient clipping, the per-image loss watch, LoKR and the context LoRA need nothing from the driver.

### 3. The workbench — `families/workbench.py`

Repair Studio, the Explorer and Royale share one engine (`WorkbenchEngine`):

- **Load:** the same base-model load as training (forward only), `load_vae`, the LoRA over `lora_target_names`, and the preview speed LoRA if the family has one.
- **Sliders:** one per block in `block_map(dit)`; `block_of(module)` maps each LoRA module to its slider.
- **Prompts:** `load_text_encoder` → `encode_text` (or `encode_text_with_image` / `encode_text_with_references`, by the family's reference kind) → `unload_text_encoder`. Encodings are cached per prompt.
- **Render:** `generate(dit, cond, w, h, steps=, seed=, cfg=, sigmas=, options=, noise=, on_step=)` → `decode(...)`. `on_step(done, total)` must be called before every step; it raises to cancel and drives Turbo Preview.
- **Seed travel** blends two `initial_noise(seed, w, h)` starts. **Prompt travel** needs `pad_conditioning(conds)`; without it the family simply has no prompt travel.
- **Save Repaired** bakes the live sliders into a new LoRA in your key format.

A video family can replace this engine with its own (see [VIDEO.md](VIDEO.md)).

### 4. The GUI launch — `families/launch.py`

The GUI never loads your driver to start a run. `launch.plan()` reads only the description and turns the Training tab into the cache and train commands above: model paths from `model_files`, dataset TOML from `media` and the training kind, flags from the abilities you declared, and your `FamilyOption` controls as `--family_option`, trainer flags or `--aux` tokens. That's why the same run can be started from the command line with no GUI at all.

### 5. Profiler and Extract — `families/block_profile.py`, `families/extract.py`

These read LoRA files directly. They need your `LoRAFormat`, `alias_flat` and **`block_map()` called with no model loaded**. The default `block_map` works without one; if you override it, make sure yours does too. The Profiler's rendered measurements go through the workbench engine.

## When the layer is missing something

Every model brings quirks: reference images, frames, sound, an unusual mask or quantisation. Ask first whether it's something other models will share.

- **Shared:** add it to the layer as an optional driver method or description field whose default means "not supported", and use it in the generic code only when a family provides it. Families without it are unaffected.
- **One-off:** keep it inside your driver.

Either way, no `if family == ...` in shared code. Every optional feature in [ABILITIES.md](ABILITIES.md) started this way.

Please open a discussion before starting a family, so two people don't build the same one.
