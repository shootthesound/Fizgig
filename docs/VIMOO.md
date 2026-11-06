# Vimoo 1.0 Lite

[← Back to the README](../README.md)

Vimoo 1.0 Lite is Vimoo-AI's 9.5B video model: one checkpoint for text-to-video and image-to-video, 720p at 24 fps, released under Apache 2.0. Fizgig trains LoRAs on it from photos and clips, renders training previews as clips in 12 steps with Vimoo's distilled LoRA, and saves LoRAs that load in ComfyUI with Vimoo's own nodes.

## Getting set up

1. On the **Preferences** tab, open the Vimoo 1.0 Lite section and press **Download models for me** (about 40 GB, no Hugging Face token needed).
2. On the **Training** tab, pick **Vimoo 1.0 Lite** as the Base Model and load the preset.

| Model | File | Size |
|---|---|---|
| Vimoo 1.0 Lite | `vimoo_v1_lite_bf16.safetensors` | 19.3 GB |
| VAE (Wan 2.2 TI2V) | `vimoo_v1_wan22_ti2v_vae_fp32.safetensors` | 2.8 GB |
| Text encoder (Qwen2.5-VL-7B) | `vimoo_v1_qwen2_5_vl_7b_bf16.safetensors` | 14.1 GB |
| Glyph-ByT5 (text in quotes) | `vimoo_v1_glyph_byt5_fp32.safetensors` | 0.9 GB |
| SigLIP (image-to-video) | `vimoo_v1_siglip_so400m_fp16.safetensors` | 0.9 GB |
| Distilled LoRA (previews) | `vimoo_v1_ti2v_distilled_lora_fp32.safetensors` | 2.3 GB |

All of them come from [Vimoo-AI/Vimoo-1.0-Lite-ComfyUI](https://huggingface.co/Vimoo-AI/Vimoo-1.0-Lite-ComfyUI), the files ComfyUI's Vimoo workflows use. If you already have them in ComfyUI's models folder, point Preferences at them instead of downloading.

## Training data

- **Photos** train as one-frame clips.
- **Clips** must be 24 fps, with a frame count of 4n + 1 (17, 25, 33, 49 ... 121, up to 241), and width and height in multiples of 32. Fizgig refuses a clip off that grid rather than resample it. Gizmo prepares clips to the spec: pick Vimoo 1.0 Lite when it opens and its lengths, sizes and memory notes follow Vimoo (no sound options - Vimoo trains picture only).
- **Captions** in plain English. Text in "double quotes" is read by the glyph encoder as lettering in the frame (a sign, a title).

## Memory

**Base precision: Auto** picks INT8 or 4-bit NF4 from your free VRAM and sizes Block Swap to match. Measured on an RTX 5090 with the default 2-second previews:

| Base precision | Training, 1 MP photos | Training, a 49-frame clip at 0.4 MP | With a preview |
|---|---|---|---|
| bf16 | 19.3 GB | — | 28.0 GB |
| INT8 | 11.0 GB | 13.4 GB | 19.5 GB |
| NF4 | 7.5 GB | 9.7 GB | 15.9 GB |

The previews set these peaks. On cards under 20 GB the model steps off the card for each preview's decode, so a 16 GB card trains NF4 (9.6 GB) or INT8 with no block swap (13.1 GB): measured on an RTX 4090 Laptop GPU with photos and a 33-frame clip, 3.0 and 3.8 s a step. On a card under 20 GB, Auto picks NF4, in training and in Studio, for the room it leaves; pick INT8 yourself for the closer base. Speed on the 5090, INT8: 0.87 s a step for 1 MP photos, 3.2 s for a 49-frame clip at 0.4 MP. A clip costs memory by its token load (frames x area), so long clips at full 720p need a big card.

Vimoo's INT8 is weight-only: the weights are stored in 8 bits and turned back to bf16 for each multiply, and the activations stay bf16. The faster INT8 the other models use also rounds the activations, and on Vimoo that showed as mottled, flickering backgrounds. INT8 attention is off for Vimoo for the same reason. The INT8 speeds above were measured before this change.

## Previews

Training previews render with the distilled LoRA: 12 steps at CFG 1, a 2-second 832x480 clip in about 20 seconds on a 5090, decode included (Samples tab: **Sample length** sets the clip length, or a still). Set the distilled LoRA's strength to 0 to render on the base model instead: 50 steps at CFG 4 against the negative prompt, as Vimoo's pipeline does, about six times slower.

**Image-to-video previews:** set a reference picture on the Samples tab; the previews start on it and move from there.

## Using your LoRAs in ComfyUI

Install [ComfyUI-Vimoo](https://github.com/Vimoo-AI/ComfyUI-Vimoo) (ComfyUI 0.34 or later, PyTorch 2.9 or later) and load the LoRA with ComfyUI's standard **Load LoRA (model only)** node after **Vimoo Model Loader**. Fizgig saves every LoRA on the Lite file's own module names, so all of its modules map.

## Licence

Vimoo releases Vimoo 1.0 Lite, its distilled LoRA and its ComfyUI nodes under Apache 2.0. The text encoder (Qwen2.5-VL-7B), SigLIP and the Wan 2.2 VAE are Apache 2.0 as well; the Glyph-ByT5 weights come from Glyph-SDXL-v2, which states no licence on Hugging Face.

## Command line

Vimoo trains headless through `src/fizgig/families/cache.py` and `src/fizgig/families/train.py` with `--family vimoo`. See [the CLI guide](CLI.md).
