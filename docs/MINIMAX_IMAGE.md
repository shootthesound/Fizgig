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

- **Already have MiniMax H3?** The text encoder is the same file, so it isn't downloaded again.
- **The VAE is its own:** MiniMax H3's video VAE encoder with a decoder made for pictures. The H3 video VAE won't decode H3 Image's pictures properly.
- **Not the same weights as MiniMax H3:** the layers are H3's, but the weights moved a long way in H3 Image's training (a quarter to a half of each weight's size on the attention and MLP layers). LoRAs trained on MiniMax H3 don't carry over, and neither do H3's training adapters.

## How it trains

- **The checkpoint's own noise levels:** H3 Image is distilled to 6 fixed noise levels (1.0, 0.968, 0.888, 0.750, 0.533, 0.232) with one output head per level. Fizgig trains a LoRA at those levels, each through its own head: the exact inputs the model sees when it renders.
- **Size:** each DiT token covers 32×32 pixels, so 1 MP is 1,024 tokens. The presets train at 1 MP.

## Previews

- 6 steps, no CFG, Euler on the model's own schedule, 2048×2048 by default (ComfyUI's default size for this model). Negative prompts don't apply.
- Measured against ComfyUI on the same LoRA, prompt and seed: a Fizgig preview and ComfyUI's render are 40 dB PSNR apart (a mean difference of 1.2 in 255). The preview is what ComfyUI will give you.

## In ComfyUI

| Node | Setting |
|---|---|
| Load Diffusion Model | `minimax_h3_image_pruned_int8_convrot.safetensors` |
| MiniMax H3 Image Conditioning | your prompt; width and height (2048×2048 or larger) |
| KSampler | sampler `euler`, scheduler `simple`, 6 steps, CFG 1 |
| Your LoRA | Load LoRA |

## Licence

MiniMax H3 Image is released under the [MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE), which differs from MiniMax H3's licence. Its terms include:

- **Territory:** the licence applies only in its "Applicable Territory": worldwide, excluding the European Union, the United Kingdom, the Republic of Korea and the United States. It does not authorise using, reproducing, modifying, distributing or displaying the model, its derivatives or their outputs outside that territory.
- Written permission from MiniMax is needed if your (and your affiliates') annual revenue exceeds 1 million USD, or if you provide a hosted service to third parties (requested through [platform.minimax.io/license](https://platform.minimax.io/license)).
- A commercial product or service that uses the model must show "MiniMax-H3-Image" in its user interface.
- The model and its outputs may not be used to improve any AI model other than MiniMax H3 Image and its derivatives.
- Distributing the model or a derivative of it (a LoRA, for example) requires passing on a copy of the licence, a notice that the files were modified, and the attribution notice the licence specifies, and binding recipients to its use restrictions.

Read the licence itself for the full terms.
