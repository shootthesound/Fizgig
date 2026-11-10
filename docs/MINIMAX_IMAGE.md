# MiniMax H3 Image

[← Back to the README](../README.md)

MiniMax H3 Image is MiniMax's picture model from the H3 family, released 3 November 2026: text-to-image and edits with up to 16 reference pictures in one checkpoint. It is MiniMax H3's 33B diffusion transformer without its audio stream, distilled to render in **6 steps with no CFG**, at about 4 to 20 megapixels (best around 16 MP). Its text encoder is MiniMax H3's Qwen3-VL-32B, unchanged.

Fizgig trains LoRAs, LoKRs, edit LoRAs and slider LoRAs on it, with masked training, Multi Concept, DOP and Fizgig LoRA Isolation. Every workbench tool works with it: Studio with Reuse unchanged blocks, LoRA Royale with seed and prompt travel, the Profiler and Extract. Saved LoRAs load in ComfyUI's standard LoRA loader.

> **Status:** the family is being built (branch `h3-image`); this guide fills in as each part is measured.

## Getting set up

Pick **MiniMax H3 Image** on the Training tab, then press **Download models for me** in Preferences. None of the files are gated.

```bash
python -m fizgig.scripts.fetch_models --family minimax_image
```

| Model | File | Size |
|---|---|---|
| DiT (training) | `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors` from [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) | 21.0 GB |
| VAE | `vae/minimax_h3_image_vae_fp16.safetensors` from [Comfy-Org/MiniMax-H3-Image](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) | 1.2 GB |
| Text encoder | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | 15.7 GB |
| Fizgig training adapter | `fizgig_minimax_h3_image_training_adapter.safetensors` from [ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter](https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter) | 0.15 GB |

- **Already have MiniMax H3?** The text encoder is the same file, so it isn't downloaded again.
- **The VAE is its own:** MiniMax H3's video VAE encoder with a decoder made for pictures. The H3 video VAE won't decode H3 Image's pictures properly.
- **Not the same weights as MiniMax H3:** the layers are H3's, but the weights moved a long way in H3 Image's training (a quarter to a half of each weight's size on the attention and MLP layers). LoRAs trained on MiniMax H3 don't carry over, and neither do H3's training adapters.

## How it trains

- **The checkpoint's own noise levels:** H3 Image is distilled to 6 fixed noise levels (1.0, 0.968, 0.888, 0.750, 0.533, 0.232) with one output head per level. Fizgig trains a LoRA at those levels, each through its own head: the exact inputs the model sees when it renders.
- **Size:** each DiT token covers 32×32 pixels, so 1 MP is 1,024 tokens. The presets are Krea 2's (Ultra Fast, Standard, Style, Edit, Slider) and train at 0.25 MP, the Slider at 0.5 MP.
- **Fizgig's training adapter:** on by default, frozen at 1.0 for every training step, off for previews and never in your saved file. Without it, people trained on this model turned into caricatures; with it, the same dataset gave photographic skin and a clear likeness, and pictures without the trigger kept the model's own quality. It was trained on 7,600 real photos and 1,000 of the model's own renders, and it's free on [Hugging Face](https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter) for any trainer.

## Memory

Base precision **Auto** picks int8 (the checkpoint's own storage) when it fits, else 4-bit, and swaps blocks as needed. The 2048×2048 preview sets the peak, so training size barely changes it. Measured on an RTX 5090 (rank 8, the training adapter on):

| Base | Training, 0.25 MP / 1 MP | Peak with the preview | Each swapped block saves |
|---|---|---|---|
| int8 | 21.0 / 21.7 GB | 22.7 GB | 0.29 GB |
| 4-bit (NF4) | 11.3 / 11.9 GB | 12.8 GB | 0.15 GB |
| 4-bit HQQ (pick it yourself) | 17.0 GB at 1 MP | 17.3 GB | — |

A 16 GB laptop with 32 GB of RAM trains Ultra Fast on Auto and runs Studio (14.7 GB peak). A 12 GB card trains too (4-bit, 11.2 GB peak), but the text encoder (Qwen3-VL-32B, shared with MiniMax H3) then streams through about 19 GB of free system RAM, and Studio on a 12 GB card needs about 48 GB of RAM: the model waits in RAM while each new prompt is encoded.

## Previews

- 6 steps, no CFG, Euler on the model's own schedule, 2048×2048 by default (ComfyUI's default size for this model). Negative prompts don't apply.
- Studio renders with INT8 attention, as the checkpoint asks and as ComfyUI does: on the same LoRA, prompt and seed at 2048×2048, Studio and ComfyUI differ by a mean of 5 in 255 (27 dB PSNR). In-training previews use standard attention, which differs from ComfyUI by a mean of 10 in 255 (22 dB); judge the final look in Studio or ComfyUI.

## Other trainers' LoRAs

LoRAs for H3 Image made elsewhere load in Studio, LoRA Royale, the Profiler, Extract and as a Context LoRA, in kohya names (`lora_unet_blocks_…`), ComfyUI names (`diffusion_model.blocks.…`) or diffusers names (`transformer_blocks.…`, whose separate q/k/v Fizgig fuses as ComfyUI does). Tested on public files from three trainers: every module landed.

## In ComfyUI

| Node | Setting |
|---|---|
| Load Diffusion Model | `minimax_h3_image_pruned_int8_convrot.safetensors` |
| MiniMax H3 Image Conditioning | your prompt; width and height (2048×2048 or larger) |
| KSampler | sampler `euler`, scheduler `simple`, 6 steps, CFG 1 |
| Your LoRA | Load LoRA |

## Licence

MiniMax H3 Image is released under the [MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE), which differs from MiniMax H3's licence. Its terms include:

- Written permission from MiniMax is needed if your (and your affiliates') annual revenue exceeds 1 million USD, or if you provide a hosted service to third parties (requested through [platform.minimax.io/license](https://platform.minimax.io/license)).
- A commercial product or service that uses the model must show "MiniMax-H3-Image" in its user interface.
- The model and its outputs may not be used to improve any other AI model.

Read the licence itself for the full terms.
