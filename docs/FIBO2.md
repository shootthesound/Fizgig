# Fibo 2.0

[← Back to the README](../README.md)

Fibo 2.0 Dev is BRIA's 8.2B model for text-to-image and editing, trained on long structured captions. Fizgig trains LoRAs, edit LoRAs, slider LoRAs and full fine-tunes on it, from a 12 GB card. Training previews render in 4 steps with Comfy-Org's Turbo LoRA. Every workbench tool works on it: Studio with Reuse unchanged blocks, LoRA Royale (epochs, seed and prompt travel), the Profiler and Extract. Saved LoRAs and fine-tunes load in ComfyUI as they are.

Thanks to @spaghettified26, who brought Fibo 2.0 to Fizgig.

## Getting set up

1. On the **Preferences** tab, open the Fibo 2.0 section and press **Download models for me**. No Hugging Face token is needed. It fetches the Dev DiT, the FLUX.2 autoencoder, the Qwen3-VL-4B text encoder and the Turbo LoRA (about 28 GB), plus Krea 2's Qwen3-VL-4B captioner for the Captions tab (about 5 GB) if you don't have it.
2. On the **Training** tab, pick **Fibo 2.0** as the Base Model and load a preset.

From the command line:

```bash
python -m fizgig.scripts.fetch_models --family fibo2
```

| Model | File | Size |
|---|---|---|
| Dev DiT (training) | `fibo2_dev_bf16.safetensors` | 16.5 GB |
| Autoencoder | `flux2-vae.safetensors` | 0.3 GB |
| Text encoder | `qwen3vl_4b_bf16.safetensors` (saved as `fibo2_text_encoder.safetensors`) | 8.9 GB |
| Turbo LoRA (previews) | `fibo_2.0_turbo_lora_avg_rank_270_bf16.safetensors` | 2.0 GB |

All of them come from [Comfy-Org/Fibo-2.0](https://huggingface.co/Comfy-Org/Fibo-2.0). The DiT row also offers the 8.4 GB int8 build, `fibo2_dev_int8_convrot.safetensors`, the one ComfyUI's Fibo workflows use; Fizgig decodes it when it loads, so it trains the same way as the bf16 file.

## Presets

| Preset | Rank | Learning rate | Epochs |
|---|---|---|---|
| **Fibo 2.0 (rank 32, full model)** | 32 | Muon 1e-4 | 40 |
| **Fibo 2.0 Edit** | 16 | Muon 1e-4 | 6 |
| **Fibo 2.0 Fine-tune** | — | 1e-5, 3 rotations | 2 a part |
| **Fibo 2.0 Slider** | 8 | Muon 2e-4, at 0.5 MP | 20 |

The LoRA presets train at 0.25 MP with Muon and EMA 0.98, saving every epoch. Muon's steps match AdamW's in size, so its rates are AdamW's. Automagic v3 is still in the Optimizer Type row: it sets its own rate from a 1e-6 start (rising to about 1e-4 within ten epochs on a 16-photo likeness set); starting it at 1e-4 overcooks the LoRA. Scrub the epochs in LoRA Royale to pick the best one.

The LoRA trains the whole model, every Linear, as Bria's own trainer does.

## Memory: 12 GB and up

**Base precision: Auto** picks INT8 or 4-bit NF4 from your free VRAM and sizes Block Swap to match; bf16 is there to pick by hand. Measured on an RTX 5090, previews included:

| Base precision | 0.25 MP | 1 MP |
|---|---|---|
| bf16 | 26.6 GB | 27.8 GB |
| INT8 | 18.7 GB | 20.5 GB |
| NF4 | 15.5 GB | 17.5 GB |

The 1024 preview sets these peaks; training alone takes about 8 GB less. On cards under 20 GB the previews render at 768 with the model parked for the decode, so Auto plans those cards from training's own peak: a 16 GB card trains INT8 with no block swap at 0.25 MP (12.4 GB measured on an RTX 4090 Laptop GPU, 2.1 s a step) and NF4 at 1 MP (about 13 GB), and a 12 GB card trains NF4 (about 10 GB at 0.25 MP). **Compile Blocks** speeds training up on longer runs (INT8 at 0.25 MP: 0.82 → 0.47 s a step). On cards of 40 GB and up, Auto trains without gradient checkpointing when it fits.

A **full fine-tune** trains the 44 main layers. It fits a 16 GB card, streaming the frozen layers from system RAM; a 24 GB card trains it in three windows, and a 32 GB card in two. See [Full fine-tuning](FINETUNE.md).

## Captions

Fibo 2.0 was trained on long structured JSON captions, and plain-text captions work too. Its default negative prompt is the empty JSON caption the reference pipeline uses; the Samples tab shows it, and you can change it there.

## Previews

Training previews render with the Turbo LoRA: 4 steps, CFG 1, a few seconds at 1024. Set the Samples tab's Turbo strength to 0 to render on the Dev model instead: 30 steps at CFG 5 against the negative prompt, as BRIA's pipeline does, which is much slower. Studio and LoRA Royale follow the Samples tab's settings.

## Edit LoRAs

Fibo 2.0 edits photos as well as generating them, so a LoRA can learn an edit: a grade, a look, a relight, a retouch. Train it on pairs of the same photo, the original and your edited version. The text encoder sees the original, and the model renders the edit on top of it, as BRIA's pipeline does.

1. Pick **Fibo 2.0 Edit**; it ticks **Edit LoRA** under Training Parameters.
2. **Originals folder:** the unedited photos.
3. **Edited folder:** the same photos after your edit, with the same file names. This is the Start tab's folder.
4. **Captions:** the edit instruction, e.g. "Apply my concert grade." A JSON caption of the result with an `edit_instruction` key works too.
5. **Test photo for previews (optional):** an original in neither folder; the previews show the edit applied to it.

## Slider LoRAs

A slider is a dial between two looks: strength +1 moves toward one, -1 toward the other. Pick **Kind of LoRA: Slider** or load **Fibo 2.0 Slider**, then train it from photo pairs (two folders of the same shots, one per end) or from three prompts. The steps are the same as for Qwen Image 2.1: see [Slider LoRAs](QWEN_IMAGE.md#slider-loras).

## Masked training

Masks work as on the other models, prior preservation outside the mask included. See [Masked training](MASKED_TRAINING.md).

## Using your LoRAs in ComfyUI

Load the LoRA with the standard LoRA loader on ComfyUI's Fibo 2.0 workflow. ComfyUI's Fibo 2.0 model fuses attention's q, k and v into one tensor, and the FFN's w1 and w3; Fizgig trains them separately and saves every LoRA in ComfyUI's layout, so it loads with every module and no conversion. That includes LoRAs saved from Studio and Extract. A fine-tune loads with ComfyUI's Load Diffusion Model node in place of the Dev file.

LoKR isn't offered on Fibo 2.0: a LoKR change can't be written exactly onto ComfyUI's fused tensors.

## Licence

BRIA releases the Fibo 2.0 weights under [Creative Commons Attribution-NonCommercial 4.0](https://creativecommons.org/licenses/by-nc/4.0/): non-commercial use only. Commercial use needs a separate agreement with BRIA ([pricing](https://platform.bria.ai/licensing/pricing?model=fibo-2)). A LoRA or fine-tune trained on it is a derivative and carries the same restriction. The Qwen3-VL-4B text encoder is Apache-2.0.

## Command line

Fibo 2.0 trains headless through `src/fizgig/families/cache.py` and `src/fizgig/families/train.py` with `--family fibo2`. See [the CLI guide](CLI.md#fibo-20-training).
