# SDXL

[← Back to the README](../README.md)

Fizgig trains any SDXL checkpoint, from the single `.safetensors` file ComfyUI loads: Juggernaut, Illustrious or your own merge. Its VAE and text encoders come from the same file, so one Preferences row is all SDXL needs. LoRAs, LoKRs and slider LoRAs train from an 8 GB card, and so do full fine-tunes. Every workbench tool works on it: Studio, LoRA Royale, the Profiler and Extract, and they load community SDXL LoRAs too. Saved LoRAs load in ComfyUI's standard LoRA loader.

SDXL support is marked experimental.

## Getting set up

1. On the **Preferences** tab, the SDXL section's **SDXL checkpoint** row downloads Juggernaut XL v9 (a strong photo model), with Illustrious XL v0.1 (anime and illustration) offered beside it. Or point the row at any SDXL checkpoint you already have.
2. On the **Training** tab, pick **SDXL (any checkpoint)** as the Base Model and load a preset.

The **VAE** row is optional: leave it empty to use the checkpoint's own VAE, or point it at [stabilityai/sdxl-vae](https://huggingface.co/stabilityai/sdxl-vae) to use that instead. The text encoders (CLIP-L and CLIP-G) are always read from the checkpoint.

From the command line:

```bash
python -m fizgig.scripts.fetch_models --family sdxl
```

| Model | File | Size |
|---|---|---|
| Checkpoint (Juggernaut XL v9) | `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` | 7.1 GB |
| Or: Illustrious XL v0.1 | `Illustrious-XL-v0.1.safetensors` | 6.9 GB |
| VAE (optional) | `sdxl_vae.safetensors` | 0.3 GB |

A LoRA works best on the checkpoint it was trained on: train on the one you generate with.

## Presets

Every preset trains at 1 MP with Muon, EMA 0.98, a flat learning rate and the per-image loss watch, saving every epoch. Strong and Standard also have per-image learning rates and auto-recaption on:

| Preset | Rank / alpha | Learning rate | Epochs |
|---|---|---|---|
| **SDXL Strong** (the default) | 32 / 16 | 5e-5 | 50 |
| **SDXL Standard** | 16 / 8 | 5e-5 | 50 |
| **SDXL Slider** | 32 / 16 | 5e-5 | 30 |
| **SDXL Fine-tune** | — | 1e-5 | 20 |
| **SDXL Fine-tune Official** | — | 3e-6 | 20 |

Auto-recaption rewrites the captions of images that keep failing to train, with Krea 2's Qwen3-VL-4B captioner (the Captions tab's), so that file needs to be downloaded too. Scrub the epochs in LoRA Royale to pick the best one.

## What it trains

SDXL trains as it was made: epsilon prediction on its scaled-linear noise schedule, uniform timesteps, plain MSE, and size conditioning from each bucket's size. The LoRA covers the UNet's attention and feed-forward layers: input blocks 4, 5, 7 and 8, the middle block and output blocks 0-5 (the outer blocks have no attention). The VAE runs in fp32, since SDXL's VAE overflows in half precision.

Two family options on the Training tab:

- **Min-SNR weighting** (off by default): evens out how much the noisiest and cleanest steps count. Common in other SDXL trainers; in Fizgig's A/B it gave no gain.
- **Noise offset** (0 by default): 0.0357 is the offset SDXL base was trained with; 0 trained as well in Fizgig's A/B.

**Long captions and prompts aren't cut off.** Anything past CLIP's 77 tokens is encoded in 77-token chunks side by side, as ComfyUI does.

## Memory: 8 GB and up

**Base precision: Auto** picks bf16, INT8 or 4-bit NF4 from your free VRAM. The plan works from these LoRA training peaks at 1 MP: bf16 10.3 GB, INT8 8.2 GB, NF4 7.4 GB.

**Compile Blocks** on Auto compiles SDXL's 70 transformer blocks for runs over about 200 steps: about 1.4× faster on a 5090 with no extra memory. The first steps pause to compile (about 25 s, then a few seconds for each new image shape). Auto doesn't compile the INT8 and NF4 bases (On still does).

A **full fine-tune** trains the UNet's attention and feed-forward layers with the text encoders frozen. It runs from an 8 GB card, and trains the whole model at once from 10 GB. Each checkpoint is a complete SDXL file, text encoders and VAE included, that loads like any other. See [Full fine-tuning](FINETUNE.md).

## Previews

Previews render at 30 steps with DPM++ 2M SDE on the Karras schedule (in ComfyUI: `dpmpp_2m_sde` / `karras`) at CFG 3, with a default negative prompt. SDXL needs CFG: 3 to 5 is the usual range, and Juggernaut's skin turns waxy above 6 or 7. Plain Euler gives a softer look with less pore detail. Studio has its own negative prompt box, saved separately from the Samples tab's, and starts at 1024.

## Community LoRAs in the workbench

SDXL LoRAs from other trainers load in every workbench tab: kohya files, LoCon, LyCORIS, diffusers naming, and speed LoRAs such as Lightning and LCM. Extract has two block presets beside All Blocks and Custom: **Input blocks** and **Middle + output blocks**.

## Slider LoRAs

A slider is a dial between two looks: strength +1 moves toward one, -1 toward the other. Pick **Kind of LoRA: Slider** or load **SDXL Slider**, then train it from photo pairs (two folders of the same shots, one per end) or from three prompts. The steps are the same as for Qwen Image 2.1: see [Slider LoRAs](QWEN_IMAGE.md#slider-loras).

## Several subjects in one LoRA

**Multi Concept** and **Fizgig LoRA Isolation** both work on SDXL: train two people into one LoRA and keep strangers looking like themselves. See [Subjects](TRAINING.md#subjects-more-than-one-person-and-keeping-them-to-their-trigger) and [Fizgig LoRA Isolation](LORA_ISOLATION.md).

## Using your LoRAs in ComfyUI

Load the LoRA with the standard LoRA loader on your SDXL checkpoint. Fizgig saves it with kohya names, which ComfyUI and most other SDXL tools read. A fine-tune checkpoint loads in place of the base, like any SDXL checkpoint.

## Licence

Each checkpoint has its own licence: SDXL base is CreativeML Open RAIL++-M, Juggernaut XL v9 CreativeML Open RAIL-M, and Illustrious XL v0.1 the Fair AI Public License 1.0-SD. Read your checkpoint's before publishing or selling anything made with it.

## Command line

SDXL trains headless through `src/fizgig/families/cache.py` and `src/fizgig/families/train.py` with `--family sdxl`. See [the three-step pipeline](CLI.md#the-three-step-pipeline).
