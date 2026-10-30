# Z-Image Turbo

[← Back to the README](../README.md)

Z-Image Turbo is Tongyi-MAI's 6B photo model: 8 steps, no CFG, and one of the most-used image models in ComfyUI. Fizgig trains on Turbo itself, the model you generate with, with Fizgig's own training adapter keeping its 8-step look intact while your LoRA learns: LoRAs, LoKRs and slider LoRAs from a 12 GB card, and full fine-tunes from 16 GB. Every workbench tool works on it: Studio with Turbo Preview, LoRA Royale, the Profiler and Extract. Saved LoRAs load in ComfyUI's standard LoRA loader.

Z-Image Turbo support is marked experimental.

## Getting set up

1. On the **Preferences** tab, press **Download models for me** in the Z-Image Turbo section. It fetches the Turbo model, the Qwen3-4B text encoder, the FLUX.1 VAE and the Fizgig training adapter (about 21 GB).
2. On the **Training** tab, pick **Z-Image Turbo** as the Base Model and load a preset. The training adapter is ticked by default: leave it on.

If you already use Z-Image in ComfyUI, point the Preferences rows at your own files instead. The VAE is the FLUX.1 one (often saved as `ae.safetensors` or `z_image_vae.safetensors`), not the FLUX.2 VAE Klein uses. Once its files are downloaded, Z-Image trains and previews with no internet connection.

From the command line:

```bash
python -m fizgig.scripts.fetch_models --family zimage
```

| Model | File | Size |
|---|---|---|
| Turbo DiT | `z_image_turbo_bf16.safetensors` | 12.3 GB |
| Text encoder | `qwen_3_4b.safetensors` | 8.0 GB |
| VAE (FLUX.1) | `ae.safetensors` | 0.3 GB |
| Fizgig training adapter | `fizgig_z_image_turbo_training_adapter.safetensors` | 70 MB |

The first three come from [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo), the adapter from [ShootTheSound/Fizgig-Z-Image-Turbo-Training-Adapter](https://huggingface.co/ShootTheSound/Fizgig-Z-Image-Turbo-Training-Adapter).

## The training adapter

A LoRA trained on Turbo the usual way undoes the distillation that makes it fast, and its pictures go soft and hazy at 8 steps. Fizgig's training adapter sits under your LoRA, frozen at full strength, for every training step. It's off in previews and never in your saved file. Against the adapter most people train Z-Image with, it gives better likeness, image quality and detail: clean, natural skin instead of gloss and speckle, and prompts that don't mention your subject still look like Turbo. It's free on Hugging Face for any trainer.

## Presets

Every preset trains 30 epochs with AdamW, EMA 0.98 and the per-image loss watch, saving every epoch:

| Preset | Rank | Learning rate | Size | For |
|---|---|---|---|---|
| **Z-Image Turbo Fast** | 8 | Adaptive LR 2e-4 to 4e-4 | 0.25 MP | Characters and most LoRAs, quickest to a likeness |
| **Z-Image Turbo Standard** | 16 | Adaptive LR 1e-4 to 2e-4 | 0.5 MP | Bigger or more varied datasets |
| **Z-Image Turbo Style** | 16 | Flat 1.5e-4 | 0.5 MP | Styles |
| **Z-Image Turbo Slider** | 8 | Flat 2e-4 | 0.5 MP | Slider LoRAs |
| **Z-Image Turbo Fine-tune** | — | 5e-5 | 0.5 MP | A full fine-tune (10 rotations) |
| **Z-Image Turbo Fine-tune Strong** | — | 1e-4 | 0.5 MP | A fine-tune that should move the model further, such as a strong style |

Z-Image fine-tunes at 5e-5 rather than the 1e-5 the other models use: its weights are larger, so 1e-5 hardly moves it. Ticking Fine-tune on the Training tab sets 5e-5 for you. Scrub the epochs in LoRA Royale to pick the best one.

## Memory: 12 GB and up

**Base precision: Auto** picks bf16, INT8 or 4-bit NF4 from your free VRAM and sizes Block Swap to match. With the Auto plan, previews included:

| Your card | LoRA training | Full fine-tune |
|---|---|---|
| 12 GB | INT8 with block swap | — |
| 16 GB | INT8, no block swap | three parts of the model at a time |
| 24 GB | bf16, no block swap | two parts at a time |
| 32 GB | bf16 | the whole model at once |

The plan works from these LoRA training peaks at 1 MP: bf16 17.2 GB, INT8 12.3 GB, NF4 10.1 GB. The workbench tabs run on a 12 GB card.

The **text encoder** only runs while captions cache, before training starts. With less than 11 GB free it loads as 8-bit weights: about 4.5 GB instead of 7.7, and the captions encode the same to within rounding (cosine 1.00000 to bf16). With room, it stays bf16.

**Compile Blocks** on Auto turns on for runs long enough to repay the warm-up: about 1.49× faster per step on the INT8 base and 1.18× on bf16, with almost no extra memory. It waits for runs over about 300 steps on INT8 and 600 on bf16, isn't used by Auto on NF4, and never with Block Swap.

## Previews

Previews render on Turbo itself: 8 steps, no CFG, shift 3, as the reference does. The Samples tab hides CFG and the negative prompt for Z-Image, since they do nothing without CFG. Studio and LoRA Royale follow the Samples tab's settings; Studio starts at 768, its default size, and **Turbo Preview** makes late-block tweaks near-instant.

## Slider LoRAs

A slider is a dial between two looks: strength +1 moves toward one, -1 toward the other. Pick **Kind of LoRA: Slider** or load **Z-Image Turbo Slider**, then train it from photo pairs (two folders of the same shots, one per end) or from three prompts. The steps are the same as for Qwen Image 2.1: see [Slider LoRAs](QWEN_IMAGE.md#slider-loras).

## Several subjects in one LoRA

**Multi Concept** and **Fizgig LoRA Isolation** both work on Z-Image: train two people into one LoRA and keep strangers looking like themselves. See [Subjects](TRAINING.md#subjects-more-than-one-person-and-keeping-them-to-their-trigger) and [Fizgig LoRA Isolation](LORA_ISOLATION.md).

## Using your LoRAs in ComfyUI

Load the LoRA with the standard LoRA loader on Z-Image Turbo and generate at 8 steps, CFG 1. Fizgig saves it under the names AI-Toolkit writes for Z-Image, which ComfyUI reads directly; the training adapter is never in the file, so nothing else is needed.

## Z-Image Base

**Z-Image Base** is Tongyi's undistilled Z-Image, the model Turbo was made from. Pick it as the Base Model on the Training tab to train on Base instead of Turbo. It trains without the training adapter, since Base isn't distilled, and offers the same kinds of training as Turbo: LoRAs, LoKR, sliders from prompts or photo pairs, masked training, and full fine-tunes. Memory use, block swap and the big-card settings are the same as Turbo's.

**Download models for me** in its Preferences section fetches Base, the FLUX.1 VAE, the Qwen3-4B text encoder and Z-Image Turbo for the previews (about 33 GB), from [Comfy-Org/z_image](https://huggingface.co/Comfy-Org/z_image) and [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo).

**Previews on Turbo.** Most people generate with Z-Image Turbo, so training samples, Studio and LoRA Royale render your Base LoRA on Turbo by default (8 steps, no CFG). Untick **Use Z-Image Turbo for samples** on the Samples tab to preview on Base itself, at ComfyUI's Base settings: 25 steps, CFG 4, with a negative prompt. Sliders and fine-tunes always preview on Base, since there is no LoRA to carry over to Turbo.

In ComfyUI, load a Base LoRA with the standard LoRA loader, on Turbo at 8 steps and CFG 1, or on Base at 25 steps and CFG 4.

## Not charted yet

Z-Image's block map (which of its 30 layers carry identity, style and detail) isn't measured yet, so **Model Area to Train**, Extract's block presets and Studio's colour-coded categories aren't offered. Every layer's slider works in Studio, and Extract's All Blocks and Custom work.

## Licence

Z-Image Turbo is released under the Apache 2.0 licence, as is Fizgig's training adapter. The Qwen3-4B text encoder is Apache 2.0 too.

## Command line

Z-Image trains headless through `src/fizgig/families/cache.py` and `src/fizgig/families/train.py` with `--family zimage`, with the training adapter passed as `--training_adapter`. Z-Image Base is `--family zimage_base`, with no training adapter; `--preview_checkpoint` with the Turbo DiT previews on Turbo. See [the three-step pipeline](CLI.md#the-three-step-pipeline).
