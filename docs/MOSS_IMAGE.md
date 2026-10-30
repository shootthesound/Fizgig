# MOSS-Image

[← Back to the README](../README.md)

MOSS-Image is OpenMOSS's 5B single-stream model for text-to-image, released 16 October 2026 under Apache-2.0. It renders text well (English and Chinese), and takes prompts of up to 4,096 tokens as well as JSON layout prompts that place text and objects.

Fizgig trains it on the **Base** model, the checkpoint OpenMOSS recommend for training. It offers LoRAs, LoKRs, slider LoRAs, full fine-tunes, Multi Concept, DOP and Fizgig LoRA Isolation. Every workbench tool works with it: Repair Studio with Turbo Preview, LoRA Royale with seed and prompt travel, the Profiler and Extract. Saved LoRAs load in ComfyUI's standard LoRA loader. MOSS-Image came to Fizgig through @Murf-Code's pull request.

**No edit LoRAs:** the released model is text-to-image only. Its technical report says it "does not directly condition on reference images or support image editing". The "layout editing" it shows is regenerating from an edited layout prompt.

## Getting set up

Pick **MOSS-Image** on the Training tab, then press **Download models for me** in Preferences. None of the files are gated. From the command line:

```bash
python -m fizgig.scripts.fetch_models --family moss_image
```

| Model | File | Size |
|---|---|---|
| Base DiT (training) | `moss_image_base_bf16.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) | 10.2 GB |
| SFT DiT (previews) | `moss_image_sft_int8_convrot.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image) | 5.4 GB |
| Text encoder | [`qwen3.5_9b_bf16.safetensors`](https://huggingface.co/Comfy-Org/Qwen3.5/blob/main/text_encoders/qwen3.5_9b_bf16.safetensors) from Comfy-Org/Qwen3.5 (not in the Comfy-Org/MOSS-Image repo, which only has quantised ones) | 19.3 GB |
| Autoencoder | `flux2-vae_small.safetensors` from [Comfy-Org/MOSS-Image](https://huggingface.co/Comfy-Org/MOSS-Image): FLUX.2's encoder with the small decoder MOSS-Image ships with | 0.25 GB |

- **Kromoss:** lodestones' fine-tune of Base is not recommended, neither to train on (in our test a LoRA trained on Kromoss overbaked sooner) nor as the preview model (too different from Base for reliable previews of a Base-trained LoRA).
- **Train on Base, use the LoRA on SFT.** MOSS-Image SFT is OpenMOSS's fine-tuned model for everyday use, and Fizgig previews on it. In our tests so far Base trains more reliably than SFT, so train on Base and use the LoRA on SFT in ComfyUI. If you do train on SFT or a model fine-tuned from it, tick **Train attention only** on the Training tab, because training SFT's feed-forward layers breaks anatomy. It also uses less VRAM, so a fine-tune fits a smaller card; on Base it may cost quality or slow convergence.
- **Autoencoder:** Flux 3's or Klein 9B's full FLUX.2 autoencoder also works, with the same latents, but the small decoder avoids spots in the pictures.
- **Text encoder alternatives:** Comfy-Org's `qwen3.5_9b_int8_convrot.safetensors` (9.7 GB) also works. So do the official diffusers folders (OpenMOSS-Team/MOSS-Image-Base `transformer/` and `text_encoder/`): give the path of any shard.
- **Optional Turbo LoRA:** the Turbo LoRA row in Preferences downloads lodestones' 8-step LoRA, distilled from MOSS-Image Base. It works on previews of your Base, not on SFT: untick **Use the SFT model for samples** on the Samples tab to use it. Then set **Turbo LoRA strength** to 1: Steps move to 12 and CFG to 1 (12 looked much better than the card's 8).

## Presets

| Preset | Rank | Learning rate | Epochs | Size |
|---|---|---|---|---|
| **MOSS-Image Fast** (the default) | 16 | 1e-4 | 70 | 0.25 MP |
| **MOSS-Image Standard** | 32 | 1e-4 | 70 | 1 MP |
| **MOSS-Image Style** | 16 | 5e-5 | 50 | 1 MP |
| **MOSS-Image Slider** | 8 | 2e-4 | 20 | 1 MP |
| **MOSS-Image Fine-tune** | — | 1e-5 | rotations | 1 MP |

- **Size:** Standard trains at 1 MP, the size MOSS-Image finished its training at. **Fast** is Standard at 0.25 MP and rank 16: much quicker steps and less memory, and it performs very well on face-cropped datasets (Image Prep's face crop).
- **Learning rate:** every preset uses a fixed one, with AdamW and EMA 0.98 (Style has EMA off), saving every epoch. Track how each image trains and per-image LR are off; both are still on the Training tab. On MOSS-Image's flat loss, Adaptive LR walks itself down to its floor. It's still on the Training tab, off by default.
- **How long to train:** with Clean-end share at 30% (an earlier default; Slide is the default now), likeness arrives early and keeps improving. On the Standard preset a 62-image dataset reached 66% ArcFace likeness by epoch 12 and 72% by epoch 32, above the 71% that the dataset's own photos score against the same three reference photos, and it was still rising. By eye those epochs were superb. The presets run 70 epochs so smaller datasets get there too.
- **Picking an epoch:** scrub the epochs in LoRA Royale to choose the best one, and look at the late ones before settling on an early peak.
- **Clean-end share:** the share of training steps on nearly-clean images (noise below half), where faces and fine detail are refined. By default it's **Slide**, `60:7:30:8:3`: 60% until epoch 7 and 30% at epoch 8, so likeness comes in early and composition after. Then, with **then every 3rd epoch** ticked (the default), training runs at the higher share, 60%, with every 3rd epoch (11, 14, 17 and on) at the lower one, 30%, at half the learning rate: those epochs still settle composition but cost less of the likeness the 60% epochs build. On a 53-photo dataset this held likeness about 5 points higher from epoch 9 to 20 than the earlier default, which slid from 60 to 30 by epoch 15 with every 2nd epoch alternating; the slow slide's 30–40% epochs cost the most. Whichever share comes first, the every-N epochs are at the lower one. On a small dataset, likeness faded at a permanent 30%; the refresh epochs keep it up. That works well with datasets of up to about 55 images; with many more, drop from 60 earlier. **Single** sets one value for the whole run. The model's own schedule puts about 11% there and the rest on layout and composition, which made the picture jump from epoch to epoch and slowed likeness. As a single value, 40 did better than 30 in later testing, and 30 had composition and shape coming in sooner than 60; it's still an experimental figure, so share what you find. Empty the box for the model's own schedule. On a dataset that struggled, a sudden drop from 75 to 40 cost 11 points of likeness two epochs later; easing down cost 4. Of eight Slide settings tried there, 60 sliding to 30 by epoch 15 with every 2nd epoch back at 60 led from epoch 12 on and still held 71% likeness at epoch 60. By eye the Clean-end pictures were also far sharper and more accurate, with no body horror, and the composition holds steady between epochs instead of hopping.
- **Masked training:** it seems especially effective on MOSS-Image with the default Clean-end share, Slide. Make masks on the Image Prep tab and tick **Train on the parts your masks show** on the Training tab. [Masked training](MASKED_TRAINING.md)
- **What the LoRA trains:** every Linear in all 42 blocks, block 22 included. Leaving block 22 out, as AI Toolkit does, cost likeness in side-by-side runs.

## Memory

- **Training:** Auto trains on an INT8 base, or NF4 on smaller cards, and compiles the blocks for about 25% faster steps on longer runs. bf16 is a manual choice.
- **Text encoder:** it always runs at full precision. On cards without room for it (about 15 GB), it streams from system memory one layer at a time, still exact, so captions cache on any card. On a computer with less than 30 GB of RAM it streams from the file instead, a layer at a time, so it needs about 6 GB of RAM rather than about 20 GB.
- **12 GB cards:** these train with SFT previews and stay under about 10.5 GB.

## Fine-tuning

A full fine-tune trains the model itself rather than a LoRA (Training tab, **Fine-tune**; the **MOSS-Image Fine-tune** preset is LR 1e-5). The general guide is [FINETUNE.md](FINETUNE.md).

- **Fine-tune Base.** SFT has already been taken a long way in one direction.
- **Clean-end share for a fine-tune:** picking **Fine-tune** (or the preset) sets its own Slide, `11:4:30:8:2`: the model's own schedule (about 11%) until epoch 4, so body shape and the model's knowledge hold, then sliding to 30% by epoch 8, and after that 30% with every 2nd epoch at 11%, at half the learning rate. A fine-tune has no rank to limit it, and at the LoRA's 60% a plain "a woman" prompt had turned into the trained person by epoch 10, even with DOP. In a fine-tune the Slide's epochs are rotations. Picking a LoRA again puts the LoRA's Slide back.
- **What fits, at 1 MP.** Measured on real fine-tunes: training the whole model at once peaked at **12.6 GB** (all parts) and **8.3 GB** (Train attention only); a preview peaked at 9.9 GB, under training. Fizgig plans from these figures:

| Card | Base (all parts) | Train attention only |
|:--|:--|:--|
| 16 GB and up | The whole model at once, nothing rotates | The whole model at once |
| 12 GB | 2 parts per rotation | The whole model at once |
| 10 GB | 9 parts per rotation | Many small parts, with blocks streamed from system RAM (slow) |

  The 16 GB figures are measured; the 12 GB and 10 GB rows are the planner's, from the same measurements, not yet run on those cards. The fine-tune's bf16 master copy takes about 9.5 GB of system RAM.
- **Train attention only** is for SFT or a fine-tune of it. On Base it learns much more slowly: after 10 rotations on the same dataset, a full Base fine-tune scored 49% ArcFace likeness and an attention-only one 19% (both rendered on the fine-tuned Base). It's about 25% faster per step.
- **Previews** render on the model being fine-tuned, not on SFT (a fine-tune changes the model itself), so they look like Base renders: flatter and less polished than the SFT previews a LoRA run shows.
- **EMA** is off for a fine-tune.

## Captions

MOSS-Image was trained on long, detailed captions, with short, medium and long variants of each. Its text encoder reads up to 4,096 tokens, so describe the picture fully, including any text in it exactly as written.

Fizgig never trains the text encoder or the model's text adapter, as OpenMOSS advise.

## Previews

- **Samples render on MOSS-Image SFT** so you see your LoRA where you'll use it. Untick the box on the Samples tab to preview on the Base model being trained. Repair Studio, LoRA Royale and the other workbench tabs render on SFT too.
- **SFT look on Base previews:** with the SFT tick off, tick **Give Base previews the SFT look** to put Fizgig's SFT look LoRA on the previews. It gives previews of your Base much of SFT's look. It's not as good as SFT itself, but it's a really good indicator, and previews are much quicker with no model swap. It's used for previews only, never in training, and it's downloaded for you the first time you tick it (about 2.1 GB; offline, use the Download link on its Preferences row once you're back online). The SFT look LoRA needs about 2.1 GB more VRAM while a preview renders, and the same in RAM between previews. **We still recommend not using the Turbo LoRA for previews.**
- **With any other Base DiT** (a fine-tune of Base, or SFT itself), samples render on that model instead, the one you'll use the LoRA on: the SFT samples tick turns off by itself once when you set such a Base, and you can tick it back on.
- **Sampler:** previews render as ComfyUI does with the settings below: res_multistep on the simple schedule (shift 5) for 30 steps at CFG 4, with block 22 left out of the negative branch for the first half of the steps (ComfyUI's SLG switch on). That beat OpenMOSS's own recipe, Euler at 50 steps, CFG 3, in side-by-side tests.
- **Negative:** the default is `worst quality, bad anatomy, watermark, jpeg artifact`.

## In ComfyUI

| Node | Setting |
|---|---|
| KSampler | sampler `res_multistep`, scheduler `simple`, 30 steps, CFG 4 |
| SkipLayerGuidanceDiTSimple | single layer 22, start 0, end 0.5 (the MOSS template's SLG switch on) |
| Negative prompt | `worst quality, bad anatomy, watermark, jpeg artifact` |
| VAE | `flux2-vae_small.safetensors` |
| Text encoder | [`qwen3.5_9b_bf16.safetensors`](https://huggingface.co/Comfy-Org/Qwen3.5/blob/main/text_encoders/qwen3.5_9b_bf16.safetensors), in `models/text_encoders` |
| Your LoRA | Load LoRA |

- **Text encoder:** use the bf16 one. MOSS-Image's developers say the text encoder is more fragile than the model, so avoid heavy quants of it.
- **Same as Fizgig:** a LoRA trained on Base and run on SFT, with the settings above and the same seed and resolution, gives matching images in ComfyUI and the same likeness scores as Fizgig's previews.
