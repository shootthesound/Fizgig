# Abilities

Everything beyond the basics is switched on per family in the description. Each ability's default means "not offered", so a family that leaves a field alone simply doesn't show that control. When an ability needs model code, the driver method is listed; the rest is handled by the layer.

**Offer every ability your model can support.** Users expect the same tools on every model. Switch each one on unless the model can't do it or a measurement shows it hurts, and say why in the description when you leave one out.

The full field list, with comments, is `FamilyDescription` in `src/fizgig/families/description.py`.

## Fast previews

| Field | What users get | Driver |
|---|---|---|
| `speed_loras`, `preview_speed_lora`, `preview_speed_steps`, `preview_speed_strength` | Training previews and the workbench render with a Turbo / Lightning / DMD LoRA when its file is set in Preferences. The Samples tab's Turbo strength controls it. | Nothing, unless the speed LoRA carries weights the LoRA layer can't wrap: then `frozen_file_added(dit, path, strength, "speed")` applies them (H3's AdaLN rows). |
| A model file with `role="preview_dit"` + `preview_checkpoint_sampling` | The workbench previews on a separate fast checkpoint (Klein Distilled). | — |
| `train_preview_checkpoint` | Training previews also render on that checkpoint. | `park_for_preview`, `load_preview_checkpoint`, `unpark_after_preview` (free room beside the training model, render, put it back). |
| `workbench_follows_samples` | The workbench takes the Samples tab's steps, CFG, negative and turbo strength instead of a fixed recipe. | — |

## Memory and speed

| Field | What users get | Driver |
|---|---|---|
| `precisions`, `auto_precisions`, `train_memory` | The Base precision choice (bf16 / INT8 / NF4) and an Auto plan that picks precision and block swap from free VRAM at launch. See [STILLS.md](STILLS.md#4-memory-let-auto-choose). | Optional: `plan_run` and `load_planned` for a family-specific plan or loader. |
| — | Block swap: blocks stream between CPU and GPU on small cards, in training, previews and fine-tunes. | `max_blocks_to_swap`, `enable_block_swap`, `block_swap_mode`. |
| `compiles`, `compile_payback_steps`, `compile_boundary`, `compile_memory` | The Compile Blocks control (torch.compile); Auto compiles only when the run is long enough to repay the warm-up. | `compile_targets(dit)`: the ModuleList of blocks. Blocks spread over several lists or called with keyword arguments: override `compile_blocks` and call `families/compile.ready_to_compile()` first (SDXL, Anima). |
| `int8_attention` | Workbench renders use INT8 attention where the kernel exists. | The model's attention calls `fizgig.modules.int8_attention.attend()` first. A diffusers model can install an attention processor that routes through it. Measure it: a launch-bound model gains nothing (SDXL: 1.01x), so leave it off there. |

## Kinds of training

| Field | What users get | Driver |
|---|---|---|
| `network_types=("lora", "lokr")` | LoKR as well as LoRA. | — . A kohya family saves LoKR under `diffusion_model.<module path>`; if your loaded model's names aren't ComfyUI's (SDXL loads in diffusers names), set `lokr_kohya_stems=True` in its `LoRAFormat` so LoKR uses the LoRA's `lora_unet_` names. Check the file loads in ComfyUI with no unmapped keys. |
| `edit_training` | Edit LoRAs from before/after photo pairs, with edit previews. | `supports_references = True`, `load_reference_text_encoder`, `encode_text_with_references`, and `refs=` in `training_loss` / `generate`. |
| `slider_training`, `slider_guidance`, `slider_ultra_blocks` | Slider LoRAs (a strength dial between two looks), from image pairs or prompts. | `training_loss(diff_ref=, diff_weight=)` for pairs; `noise_latents` and `predict` for prompt sliders. |
| `finetune`, `ft_learning_rate` | Full fine-tuning of the base model on an NF4 trunk: the whole model at once when the card holds it, otherwise the fewest windows that fit. Optional, and can be added after the family ships. | `ft_spec(dit)` returning an `FTSpec`, plus measured memory figures. [FINETUNE.md](FINETUNE.md) covers it step by step. |
| `lr_hint` | The Training tab's large-dataset hint: a standard LoRA at or above the rate (adaptive: its Min LR), on more steps an epoch than the threshold, gets a tip with a one-click cooler setting (Min 1e-4 / Max 2e-4, or 1e-4). Defaults to `(2e-4, 125)`, inherited by every family. | Nothing. Set your own `(rate, steps)` once measured, or `None` to turn it off. |

## Training aids

| Field | What users get |
|---|---|
| `training_adapter`, `training_adapter_note` | A frozen adapter (e.g. a de-distillation LoRA) on for every training step, off for previews, never saved. |
| `ema_default` | The weight-averaging (EMA) control with this default. |
| `adaptive_lr` (default on) | Adaptive LR: the learning rate moves up or down with the loss. Turn it off where the per-epoch loss is too noisy for the plateau detector (SDXL). |
| `loss_watch` (default on) | The per-image loss watch: how each image trains, per-image learning rates, auto-recaption. |
| `training_maps` (default on) | Training maps (and, on a still model's workbench, Studio's Maps, through `encode_images` and the same hooks): heatmaps over a few training pictures each epoch (still wrong, learned since the start, where the LoRA acts). Uses the driver's `noise_latents` / `predict` and `map_target` (flow matching's `(x_t - x0) / t` by default; override it when `predict` returns another target, e.g. SDXL's epsilon or v, or a token layout). Stills only: off for clip and song models, with a `not_offered` reason. |
| `trigger_attention` | The Trigger word map, in training maps and in Studio's Maps: each picture area's attention to the trigger word. The model calls `training/attn_probe.py`'s `layout(img, txt, total, valid, kv_total)` once its sequence is built and `see(q, k, mask)` in its attention, after QK-norm and RoPE (both no-ops outside a capture; the capture runs eagerly). The driver gives `trigger_text(tokenizer, caption)` (the prompt as its encoder builds it, and the prefix it drops), or overrides `trigger_tokenizer()` / `trigger_key_positions()` when its text keys aren't its caption tokenizer's pieces one for one (Anima's T5 tokens, SDXL's 77-token chunks), and `trigger_attention_hooks(dit)` when the attention needs hooks to reach (SDXL's processors). Check it on a real run: the trigger found in the captions, and a map with structure. |
| `train_areas` | Model Area to Train: named block sets with optional timestep windows, plus a Custom block pick. |
| `identity_blocks` | Fast Identity Mode: a standard LoRA trains only the identity blocks, faster. |
| `multi_concept`, `multi_concept_defaults`, `multi_concept_hint` | The Subjects card's "Who is this LoRA of?" (2 to 4 subjects): extra subject folders, each its own dataset block. The card's protection choices (DOP, the side-by-side pictures) need nothing more from a driver than training does. |
| `batch_training` | Batch size in Other Options: several pictures of one size bucket in one forward. The driver gives `training_loss_batch` (each picture draws its noise and timestep in the order its own `training_loss` would, the loss is their mean) and, when its captions vary in length, `stack_conds` (pad with a mask the model honours per picture, or keep each caption apart; the default stacks only same-shape captions and the trainer groups the rest). Opt in only after a parity check: from the same generator, the batched loss equals the mean of the pictures' own losses. Sliders, edits, fine-tunes, DOP, LoRA Isolation, the per-image loss watch, Start at a smaller size and the mask prior stay at batch 1 by themselves. |
| `caption_dropout` (default on) | Caption dropout in Other Options: that share of steps trains on the empty caption's encoding, made once with the family's own `encode_text`. Turn it off for a family with its own (H3) or no pictures (ACE-Step). |
| `options` (`FamilyOption`) | Your own Training / Samples tab controls; each value becomes trainer flags, `--family_option` pairs for `set_options`, or cache `--aux` values. |

The Context LoRA (train on top of another LoRA, frozen) needs nothing; every family has it.

## Start checks

Before a run starts, Fizgig lists what looks wrong, as warnings with **Start anyway** (never a block): pictures with no caption, captions without the trigger word, and a learning rate outside the model's usual range (`launch.start_checks`). The first two need nothing from a family. The third is declared:

| Field | What users get |
|---|---|
| `start_lr_bands` | `((kind, low, high), ...)` for `"lora"`, `"finetune"` and `"slider"`: a rate outside is a warning at Start (only with Adaptive LR off, and never for an optimizer that sets its own rate). |

Set each band from your shipped presets: a third of the lowest preset's learning rate to three times the highest, for each kind the family offers. Leave a kind out when nothing measured says what's usual (MiniMax H3's LoRAs train on Automagic, so it declares only its slider band). Update the band when you change a preset's rate.

## Workbench

| Field | What users get | Driver |
|---|---|---|
| `workbench` | Which tabs offer the family: any of `"repair"`, `"profiler"`, `"extract"`, `"royale"`. | — |
| `activation_cache` | Repair Studio's Reuse unchanged blocks tick: a slider change replays the unchanged blocks on step 1, the same picture as a full render. | `generate` calls `on_step(i, steps)` before every step, and your blocks sit in a `ModuleList` the block map points into. |
| `repair_presets`, `block_categories`, `category_masters` | Repair Studio presets, category colours, and master sliders per category. | — |
| `extract_presets` | Extract's block presets (`(name, ((block id, multiplier), ...))`), beside Custom. | — |
| `preview_image` | A reference picture conditions previews through the text encoder's vision path. | `encode_text_with_image`. |
| `reference_strength` | An edit family's reference has a strength, and LoRA Royale's travel can chain frames. | — |
| — | Prompt travel in LoRA Royale. | `pad_conditioning(conds)`. Seed travel needs `initial_noise`. |

## The GUI's wording

These change text only, so the family reads right without per-family GUI code:

- **Preferences:** `prefs_title`, `prefs_intro`, `prefs_note`, `fetch_note`, `fetch_optional_label`, and per model file `hint`, `download_label`, `download_note`, `alt_repo` / `alt_path` (a second build of the same file), `gated`.
- **New files:** `ModelFile.announce` + `announce_intro`, a one-time popup for users who set the family up before the file existed.
- **Training tab:** `model_note` (under the Base Model picker; usage only, never the licence, which goes in the family's guide), `preset_notes`, `network_hint`, `ema_hint`, `precision_labels`, `precision_hint`, `compile_hint`.
- **Samples tab:** `samples_text` (per-place wording, including `"sampler"`: the sampler and schedule named beside the Steps box), `samples_cfg_free` (hide CFG and the negative for a CFG-free model), `preview_cfg_note`, `preview_negative` (the default negative prompt; each family keeps the user's own edit, and `None` greys the box).
- **Repair Studio:** `repair_size` (its starting preview size; 1024 for a 1 MP model).
