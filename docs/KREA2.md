# Krea 2

[← Back to the README](../README.md)

Krea 2 is a from-scratch native port: 12.9B single-stream MMDiT, Qwen-Image VAE, Qwen3-VL-4B text encoder. Train on the **RAW model**; previews render on the training model itself with the official Turbo LoRA (auto-downloads) applied for the render only. Pick Krea 2 from the **Base Model selector** on the Training tab and the **✨ Krea 2 Ultra Fast (rank 8, adaptive LR)** preset applies on your first visit; the other built-in presets, including **✨ Krea 2 Standard (rank 32, full model)** and **✨ Krea 2 Style (rank 16, 5e-5)** (CAME at a fixed rate), are in Load Preset.

## What works

Everything works on Krea 2:

- All five workbench tools ([TRAINING.md](TRAINING.md#the-workbench)), **Pause/Resume**, **Context LoRA**, **Adaptive LR**, reference images and the live sample override.
- **Weight averaging (EMA)**, on by default at 0.98 — checkpoints and previews come from a running average of the adapter's recent steps. Measured on Krea 2 and MiniMax H3, it lifts late-epoch likeness and steadies the epochs.
- **LoKR training** — pick it from Network Type, factor 8 or below, and drop the learning rate to 5e-5 whichever preset you started from (or, with Adaptive LR, set Min 5e-5 and Max 1e-4). Standard LoRA is ~20% faster and is the default.
- **Automagic v3** (`automagic3`) and **v4** (`automagic4`) in the Optimizer Type row. Each sets its own learning rate, so the scheduler and Adaptive LR stand down while it owns the rate; the Learning Rate box is only its start. See the optimizer notes in [MINIMAX_H3.md](MINIMAX_H3.md).
- **Full fine-tuning** of the base model itself — see [FINETUNE.md](FINETUNE.md).

Output is ComfyUI-ready.

## VRAM: 8 GB is enough

Users train full Krea 2 LoRAs on 8 GB with everything on **Auto** and batch size 1. Auto reads your *free* VRAM and picks INT8 or NF4 plus the right block swap — the console explains its choice. On longer runs the transformer blocks **torch.compile** automatically for roughly 2× faster steps.

LoRA runs with block swap can go faster with **Preferences → Speed-ups for your graphics card → Faster block swap (send-only)**: the base model never changes while a LoRA trains, so each swapped block is only copied to the card and never copied back. Same VRAM, same results; 1.5x on an RTX 5090 with 16 blocks swapped, about the same on a 16 GB laptop RTX 4090, so compare the s/step in the log. Off by default; it stays off for fine-tunes, NF4 and runs with fewer than 4 blocks swapped, and the log says which swap ran.

| Your card | What to do |
|---|---|
| **8 GB** | Everything on **Auto**, batch size 1, stock preset defaults |
| **10–12 GB** | Same — headroom to raise batch size or resolution |
| **16 GB+** | Same — Auto will usually pick the faster INT8 path |

If a preview can't fit, previews auto-disable and **training keeps running and saving**.

## SageAttention image generation on AMD

With a compatible SageAttention build installed in Fizgig's Python environment, an AMD launch offers
**Preferences → Speed-ups for your graphics card → Force SageAttention for Krea 2 image generation**.
It takes effect on the next image in training previews, Studio, Royale and Profiler. Training keeps its own backend.

Generation follows ComfyUI's fixed-length attention calls: expanded K/V heads, explicit HND layout and
`smooth_k=False` (the RDNA4 package can override smoothing for accuracy). Equal-length samples share one call;
ragged samples are split so padding never enters attention. Sage generation also skips the extra 256-token
shape padding, including Studio's cached forwards. A failing forced kernel stops generation.
Reference: [ComfyUI attention](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/modules/attention.py)
and [Krea 2 model](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/krea2/model.py).

For a ComfyUI comparison, match the checkpoint, precision, resolution, prompt, steps and CFG.
Compare warmed-up renders; weight offloading and VAE decoding can still differ.

## High-resolution generation

On ROCm, **Blocks Swap: Auto** starts Krea 2 with most block weights on CPU, then budgets GPU
residency against the requested resolution and available VRAM. Sampling OOM retries use the same seed
with additional blocks offloaded. Manual block counts remain fixed; NF4 does not support block swapping.
Auto preserves stored FP8 weights and uses INT8 for dense checkpoints. NVIDIA keeps the generic
workbench loading defaults and Studio/Profiler block-swap preferences.

ROCm separates text encoding, sampling and VAE decoding into low-memory stages. The DiT and VAE
park on CPU after rendering. The allocator cap respects existing allocations and stricter limits,
restores the previous limit afterward, and adds no fixed desktop reserve. The activation budget remains.
Keep the AMD launcher's `garbage_collection_threshold:0.8`; disabling collection caused desktop
framebuffer allocation failures during earlier hardware tests.

On Linux ROCm, stored FP8 DiT Linear weights use private file-backed CPU views. Pinned staging avoids
registering those mappings directly with ROCm. Block swaps upload immutable CPU weights without a
return copy; parking rebinds the CPU views. Adapters and mutable biases retain their normal handling.
The text encoder is deleted without first copying it to CPU. Keep the base checkpoint unchanged while loaded.
Training and additional quantization retain their existing loading policies.

VAE OOM recovery retries the existing tiled decoder. Tiling can change pixels slightly and does not
ensure every resolution fits. Single-frame VAE cache removal is proposed separately. Turbo previews retain
**8 steps, CFG 1, Euler / simple**, preferred for predictable results in this workflow. The beta scheduler
experiment has been reverted; adopting it remains open for separate discussion.

CPU checks cover ROCm selection, NVIDIA default preservation, immutable weight storage, offloading,
OOM recovery and attention correctness. Earlier RX 9070 XT testing reduced peak process RSS from
22.66 to 6.02 GiB across four renders, but included the separately proposed schedule and VAE changes.
Those measurements do not validate performance or stability of the split branch; new GPU validation is pending.

The activation reserve follows [ComfyUI's Krea 2 estimate](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_base.py),
using Fizgig's existing block offloader.

Profiler logs include the total render count, Auto's memory budget, sampling backend, VAE decode and image
completion. Two seeds run sequentially for every ablation condition, not as a two-image batch. PyTorch's
allocator can print OOM warnings and recover by releasing cached allocations; a warning alone is not a failed
render. Check whether image completion and progress continue, or whether the job reports an exception.
An unexpected Studio worker exit now includes its process exit code in the error and log. Native crashes
also print a Python fault traceback to the worker log where the runtime can provide one.

## The trainer curates your dataset while it trains (experimental)

Four Training-tab toggles no other trainer has. They work on Krea 2 and [Qwen Image 2.1](QWEN_IMAGE.md).

Images don't start out as problems: early on, the hard ones are often the most valuable in the set. The watch follows every image's loss across epochs, and only an image that has stopped teaching the model (stuck without improving, or mined out after a good run) gets treated as a problem.

- **Track how each image trains** — per-image loss is tracked across epochs (noise-normalised). Most images do their work early and are simply learned by the end; the few that stay hard without improving get flagged in the **Per-image loss** view (on the run card while it trains, and in Runs & outputs), with thumbnails and trends. In real runs the top flags were all caption/image mismatches.
- **Per-image adaptive LR** — flagged images are throttled so one bad caption can't yank the weights all run; healthy images get a gentle boost. Matched-epoch A/Bs: faster likeness *and* a higher ceiling.
- **Auto-recaption stuck images** — between epochs, Krea 2's Qwen3-VL-4B *looks at* each stuck image and rewrites its caption from what's visible. Still stuck after two attempts and the image is excluded for the run (remembered per-dataset; fix the caption and it's re-admitted).
- **Warm up look outliers** — real-but-unusual shots (tight angles, profiles) ease in at reduced LR while the identity forms, then release to full. It reads the scores saved by the [Look Consistency Filter](TRAINING.md#dataset-prep).

Edit any caption yourself mid-run from that view — no restart. When nothing is improving any more, a plateau banner names the best-checkpoint window to scrub in LoRA Royale. Pause, resume, restart: a resumed run replays its own loss log and loses nothing.

## Captioning

Fizgig's AI captioning — the Captions tab's Qwen3-VL option and auto-recaption — uses Krea 2's Qwen3-VL-4B for every model. Florence-2 is the zero-setup alternative on the Captions tab.

## Help map Krea 2's blocks

Krea 2's per-block roles aren't charted yet, which is why Repair Studio's colour-coded sliders and Model Area targeting are Klein-only. The Profiler's weight-only report is the instrument — [open an issue](https://github.com/shootthesound/Fizgig/issues) with what you find and it drives the presets and Repair Studio colour-coding to come.

The four **text fusion** blocks have more to give. Load a Krea 2 LoRA in Repair Studio and pick **✨Text fusion ×2** or **×3** (experimental) — measured across several LoRAs, ×3 lifted the detail meter and likeness on every one with the composition unchanged; an overtrained LoRA is the one case it works against. Save Repaired LoRA bakes it in, so the boosted file works anywhere at strength 1.0.

## Model files

All files live in the one [**Comfy-Org/Krea-2**](https://huggingface.co/Comfy-Org/Krea-2) repo. **Preferences → ⬇ Download models for me** under the Krea 2 card downloads, verifies and fills in the paths, with no account needed; every row also has a manual **Download** link. From the command line (~32 GB):

```bash
python -m fizgig.scripts.fetch_models --family krea2
```

| Model | File | Size |
|---|---|---|
| **RAW DiT (bf16) — training** | `krea2_raw_bf16.safetensors` | ~26 GB |
| **Turbo DiT (fp8) — workbench** | `krea2_turbo_fp8_scaled.safetensors` | ~13 GB |
| Turbo LoRA *(auto-downloads)* | `krea2_turbo_lora_rank_64_bf16.safetensors` | ~470 MB |
| Qwen-Image VAE | `qwen_image_vae.safetensors` | ~250 MB |
| **Text Encoder — recommended** | `qwen3vl_4b_fp8_scaled.safetensors` | ~5.2 GB |
| Text Encoder — full precision | `qwen3vl_4b_bf16.safetensors` | ~8.9 GB |

The text-encoder slot is **open**: any Qwen3-VL-4B in the ComfyUI layout loads — fp8_scaled (recommended, captions we couldn't tell apart), bf16, or a community fine-tune/abliterated build, which changes how your dataset gets captioned.

The two DiT slots are open too: point them at your own Krea 2 checkpoints from ComfyUI's `diffusion_models` folder, fine-tunes included. fp8 builds (fp8_scaled, fp8mixed) run as they are. int8 builds (int8 ConvRot, int8mixed) and older scaled_fp8 files are decoded to bf16 as they load, and then Fizgig's own precision applies, so they take a couple of minutes longer to load. A Turbo build goes in the Turbo slot, since previews render it in 8 steps without CFG.

General install steps are in [INSTALL.md](INSTALL.md).
