# MiniMax H3 Image

[← Back to the README](../README.md)

MiniMax H3 Image is MiniMax's picture model from the H3 family, released 3 November 2026: text-to-image and edits with up to 16 reference pictures in one checkpoint. It is MiniMax H3's 33B diffusion transformer without its audio stream, distilled to render in **6 steps with no CFG**, at about 4 to 20 megapixels (best around 16 MP). Its text encoder is MiniMax H3's Qwen3-VL-32B, unchanged.

Fizgig trains LoRAs, LoKRs, edit LoRAs and slider LoRAs from photo pairs on it, with masked training, Multi Concept, DOP and Fizgig LoRA Isolation. Every workbench tool works with it: Studio with Reuse unchanged blocks, LoRA Royale with seed and prompt travel, the Profiler and Extract. Saved LoRAs load in ComfyUI's standard LoRA loader.

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
- **Clean-end share** (Training Parameters): how much of training goes to the last of the model's 6 steps, the nearly finished picture (noise level 0.232), where skin, eyes and fine detail are made. The rest is shared evenly by the five earlier steps, which set out pose, layout and the shape of things. The presets use 40. H3 Image's own default is about 17, an even share for every step: leave the box empty for that. Higher gives more detail and likeness, lower more layout and composition.
- **Size:** each DiT token covers 32×32 pixels, so 1 MP is 1,024 tokens. The presets (Standard, Style, Edit, Slider) train on Automagic v3, which finds its own learning rate, at 0.5 MP.
- **Fizgig's training adapter:** on by default, frozen at 1.0 for every training step, off for previews and never in your saved file. Without it, people trained on this model turned into caricatures; with it, the same dataset gave photographic skin and a clear likeness, and pictures without the trigger kept the model's own quality. It was trained on 7,600 real photos and 1,000 of the model's own renders, and it's free on [Hugging Face](https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter) for any trainer.

## Memory

Base precision **Auto** picks int8 (the checkpoint's own storage) when it fits, else 4-bit, and swaps blocks as needed. The 2048×2048 preview sets the peak, so training size barely changes it. Measured on an RTX 5090 (rank 8, the training adapter on):

| Base | Training, 0.25 MP / 1 MP | Peak with the preview | Each swapped block saves |
|---|---|---|---|
| int8 | 21.0 / 21.7 GB | 22.7 GB | 0.29 GB |
| 4-bit (NF4) | 11.3 / 11.9 GB | 12.8 GB | 0.15 GB |
| 4-bit HQQ (pick it yourself) | 17.0 GB at 1 MP | 17.3 GB | 0.22 GB |

On a card of 40 GB or more, a compiled int8 run trains without gradient checkpointing, about 1.5× faster (23.4 GB at 0.25 MP, 30.6 GB at 1 MP). Compile Blocks on its own adds about 10% at 1 MP and little at 0.25 MP, so Auto compiles only long runs.

A 16 GB laptop with 32 GB of RAM trains a rank 8 LoRA on Auto and runs Studio (14.7 GB peak). A 12 GB card trains too (4-bit, 11.2 GB peak), but the text encoder (Qwen3-VL-32B, shared with MiniMax H3) then streams through about 19 GB of free system RAM, and Studio on a 12 GB card needs about 48 GB of RAM: the model waits in RAM while each new prompt is encoded.

## Previews

- 6 steps, no CFG, Euler on the model's own schedule, 2048×2048 by default (ComfyUI's default size for this model). Negative prompts don't apply.
- Studio renders with INT8 attention, as the checkpoint asks and as ComfyUI does: on the same LoRA, prompt and seed at 2048×2048, Studio and ComfyUI differ by a mean of 5 in 255 (27 dB PSNR). In-training previews use standard attention, which differs from ComfyUI by a mean of 10 in 255 (22 dB); judge the final look in Studio or ComfyUI.

## Sliders

Slider LoRAs train from photo pairs. Sliders from prompts are off on this model for now: a smiling / frowning prompt slider moved clothes, styling and light rather than the expression, because on H3 Image a change of prompt reshuffles the whole picture and the slider learns all of it. If you can fix that, a pull request is very welcome.

## Other trainers' LoRAs

LoRAs for H3 Image made elsewhere load in Studio, LoRA Royale, the Profiler, Extract and as a Context LoRA, in kohya names (`lora_unet_blocks_…`), ComfyUI names (`diffusion_model.blocks.…`) or diffusers names (`transformer_blocks.…`, whose separate q/k/v Fizgig fuses as ComfyUI does). Tested on public files from three trainers: every module landed.

## In ComfyUI

| Node | Setting |
|---|---|
| Load Diffusion Model | `minimax_h3_image_pruned_int8_convrot.safetensors` |
| MiniMax H3 Image Conditioning | your prompt; width and height (2048×2048 or larger) |
| KSampler | sampler `euler`, scheduler `simple`, 6 steps, CFG 1 |
| Your LoRA | Load LoRA |

**Your LoRA can look smoother in ComfyUI than in Fizgig.** ComfyUI's default loading (dynamic VRAM) currently applies a LoRA a little weaker on this model: it adds the LoRA to each weight in bf16, where the smallest changes round away. On our test LoRA about 15% was lost, and the same seed came out cleaner and slightly less like the subject. Fizgig's previews and Studio show the LoRA at its full strength, and so does ComfyUI started with `--disable-dynamic-vram`. To see ComfyUI's default look in Studio, set the LoRA's strength to about 0.8. Reported to ComfyUI as [#17690](https://github.com/Comfy-Org/ComfyUI/issues/17690).

## Licence

MiniMax H3 Image is released under the [MiniMax-H3-Image Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE), which differs from MiniMax H3's licence. Its terms include:

- Written permission from MiniMax is needed if your (and your affiliates') annual revenue exceeds 1 million USD, or if you provide a hosted service to third parties (requested through [platform.minimax.io/license](https://platform.minimax.io/license)).
- A commercial product or service that uses the model must show "MiniMax-H3-Image" in its user interface.
- The model and its outputs may not be used to improve any other AI model.

Read the licence itself for the full terms.
