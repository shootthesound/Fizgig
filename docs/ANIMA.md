# Anima

[← Back to the README](../README.md)

Anima is CircleStone Labs' 2B model for anime and illustration. Fizgig trains it on **Anima Base v1.0**: LoRAs, LoKRs and slider LoRAs from an 8 GB card, and full fine-tunes from 8 GB too. Every workbench tool works on it: Studio, LoRA Royale, the Profiler and Extract. Saved LoRAs load in ComfyUI's standard LoRA loader.

Anima support is marked experimental. Note its non-commercial licence (below) before you start.

## Getting set up

1. On the **Preferences** tab, press **Download models for me** in the Anima section. It fetches the Base model, the Qwen-Image VAE and the Qwen3 0.6B text encoder (about 5.6 GB). The Turbo LoRA for fast previews is optional: the Download link on its row fetches it.
2. On the **Training** tab, pick **Anima** as the Base Model and load a preset.

Train on Base, not Turbo or Aesthetic.

From the command line:

```bash
python -m fizgig.scripts.fetch_models --family anima --include-optional
```

| Model | File | Size |
|---|---|---|
| Base DiT | `anima-base-v1.0.safetensors` | 4.2 GB |
| VAE (Qwen-Image) | `qwen_image_vae.safetensors` | 0.25 GB |
| Text encoder | `qwen_3_06b_base.safetensors` | 1.2 GB |
| Turbo LoRA (previews, optional) | `anima-turbo-lora-v0.2.safetensors` | 0.15 GB |

The first three come from [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima), the Turbo LoRA from [circlestone-labs/Anima-Official-LoRAs](https://huggingface.co/circlestone-labs/Anima-Official-LoRAs).

## Presets

Every preset trains at 1 MP with Muon, EMA 0.98, a flat learning rate and the per-image loss watch, saving every epoch:

| Preset | Rank | Learning rate | Epochs |
|---|---|---|---|
| **Anima Character** | 16 | 1e-4 | 50 |
| **Anima Style** | 16 | 5e-5 | 30 |
| **Anima Official** (the model card's recipe) | 32 | 2e-5 | 30 |
| **Anima Slider** | 8 | 2e-4 | 30 |
| **Anima Fine-tune** | — | 1e-5 | 30 |
| **Anima Fine-tune Official** | — | 1e-6 | 30 |

Scrub the epochs in LoRA Royale to pick the best one.

## Captions and what it trains

Anima reads danbooru tags or prose. Its text goes through the Qwen3 0.6B encoder and an LLM adapter that turns it into the DiT's context; captions over 512 tokens are cut. The model card's quality prefix is `masterpiece, best quality, score_7, safe,`.

Training is rectified flow with logit-normal timesteps and plain MSE, as diffusion-pipe and sd-scripts train it. The LoRA covers the 28 DiT blocks' self-attention, cross-attention and MLP layers. The LLM adapter is never trained: the model card warns it degrades easily. Noise offset washes Anima's colours out, and fp8 bases train badly, so neither is offered.

## Memory: 8 GB and up

**Base precision: Auto** picks bf16, INT8 or 4-bit NF4 from your free VRAM. The plan works from these LoRA training peaks at 1 MP: bf16 8.9 GB, INT8 7.3 GB, NF4 6.6 GB.

**Compile Blocks** on Auto compiles Anima's 28 blocks for runs over about 300 steps: about 2.3× faster on a 5090, for about 0.4 GB more memory. The first steps pause to compile (about 40 s, then a few seconds for each new image shape). Auto doesn't compile the INT8 and NF4 bases (On still does).

A **full fine-tune** trains every block's attention and feed-forward layers, with the LLM adapter left untouched. It runs from an 8 GB card, and trains the whole model at once from 10 GB. Each checkpoint loads in ComfyUI in place of the base. See [Full fine-tuning](FINETUNE.md).

## Previews

Previews render on Base at 20 steps, CFG 4.5, Euler with shift 3 (ComfyUI's default for Anima), with the official negative prompt. CFG around 4.5 works best: below 2 washes out, above about 7 oversaturates. The model card suggests 30-50 steps; 20 is enough for previews.

**Turbo LoRA previews:** with the Turbo LoRA downloaded, set **Turbo LoRA strength** to 1 on the Samples tab: previews render at 10 steps with CFG 1, much quicker. Studio and LoRA Royale follow the Samples tab's settings, and Studio has its own negative prompt box.

## Slider LoRAs

A slider is a dial between two looks: strength +1 moves toward one, -1 toward the other. Pick **Kind of LoRA: Slider** or load **Anima Slider**, then train it from picture pairs (two folders of the same shots, one per end) or from three prompts. The steps are the same as for Qwen Image 2.1: see [Slider LoRAs](QWEN_IMAGE.md#slider-loras).

## Several subjects in one LoRA

**Multi Concept** and **Fizgig LoRA Isolation** both work on Anima: train two characters into one LoRA and keep the others looking like themselves. See [Subjects](TRAINING.md#subjects-more-than-one-person-and-keeping-them-to-their-trigger) and [Fizgig LoRA Isolation](LORA_ISOLATION.md).

## Using your LoRAs in ComfyUI

Load the LoRA with the standard LoRA loader on ComfyUI's Anima workflow, on Base or the Turbo LoRA's setup. No conversion is needed.

## Not charted yet

Anima's block map (which blocks carry identity, style and detail) isn't measured yet, so **Model Area to Train**, Extract's block presets and Studio's colour-coded categories aren't offered. Every block's slider works in Studio, and Extract's All Blocks and Custom work.

## Licence

Anima is released under the [CircleStone Labs Non-Commercial License](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md), with NVIDIA's Open Model License for the model it builds on. LoRAs are derivatives of the model and fall under it; images you generate may be used commercially. Read the licence before publishing anything made with it. The Qwen3 0.6B text encoder is Apache 2.0.

## Command line

Anima trains headless through `src/fizgig/families/cache.py` and `src/fizgig/families/train.py` with `--family anima`. See [the three-step pipeline](CLI.md#the-three-step-pipeline).
