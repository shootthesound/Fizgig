# Krea 2

[← Back to the README](../README.md)

Krea 2 is a from-scratch native port: 12.9B single-stream MMDiT, Qwen-Image VAE, Qwen3-VL-4B text encoder. Train on the **RAW model**; previews render on the training model itself with the official Turbo LoRA (auto-downloads) applied for the render only. Pick Krea 2 from the **Base Model selector** on the Training tab and the **✨ Krea 2 Ultra Fast (rank 8, adaptive LR)** preset applies on your first visit; the other built-in presets, including **✨ Krea 2 Standard (rank 32, full model)** and **✨ Krea 2 Style (rank 16, gentle LR)**, are in Load Preset.

## What works

Everything works on Krea 2:

- All five workbench tools ([TRAINING.md](TRAINING.md#the-workbench)), **Pause/Resume**, **Context LoRA**, **Adaptive LR**, reference images and the live sample override.
- **Weight averaging (EMA)**, on by default at 0.98 — checkpoints and previews come from a running average of the adapter's recent steps. Measured on Krea 2 and MiniMax H3, it lifts late-epoch likeness and steadies the epochs.
- **LoKR training** — pick it from Network Type, factor 8 or below, and drop the learning rate to 5e-5 whichever preset you started from (or, with Adaptive LR, set Min 5e-5 and Max 1e-4). Standard LoRA is ~20% faster and is the default.
- **Automagic v3** (`automagic3`) in the Optimizer Type row. It sets its own learning rate, so the scheduler and Adaptive LR stand down while it owns the rate; the Learning Rate box is only its start. See the optimizer notes in [MINIMAX_H3.md](MINIMAX_H3.md).
- **Full fine-tuning** of the base model itself — see [FINETUNE.md](FINETUNE.md).

Output is ComfyUI-ready.

## VRAM: 8 GB is enough

Users train full Krea 2 LoRAs on 8 GB with everything on **Auto** and batch size 1. Auto reads your *free* VRAM and picks INT8 or NF4 plus the right block swap — the console explains its choice. On longer runs the transformer blocks **torch.compile** automatically for roughly 2× faster steps.

| Your card | What to do |
|---|---|
| **8 GB** | Everything on **Auto**, batch size 1, stock preset defaults |
| **10–12 GB** | Same — headroom to raise batch size or resolution |
| **16 GB+** | Same — Auto will usually pick the faster INT8 path |

If a preview can't fit, previews auto-disable and **training keeps running and saving**.

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
