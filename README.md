<h1 align="center">Fizgig — LoRA & Fine-tune Studio for Klein 9B, Krea 2, MiniMax H3, Qwen Image 2.1, SDXL & Anima</h1>

<p align="center">
  <strong>Fine-tune base models on consumer GPUs — down to 8 GB. Fix broken LoRAs without retraining. Remix any LoRA into new variations in seconds.</strong><br>
  A train · fine-tune · repair · explore workbench built end-to-end for <strong>Flux 2 Klein 9B</strong>, <strong>Krea 2</strong>, <strong>MiniMax H3</strong>, <strong>Qwen Image 2.1</strong>, <strong>SDXL</strong> and <strong>Anima</strong> — training on photos, video, sound and voices, from quick LoRAs to the full base model.
</p>

<p align="center">
  <a href="#install"><img src="https://img.shields.io/badge/⬇%20Install%20Fizgig-2EA043?style=for-the-badge&logoColor=white" alt="Jump to the install instructions"></a>
  <a href="https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep"><img src="https://img.shields.io/badge/⚡%20Deploy%20on%20RunPod-673AB7?style=for-the-badge&logoColor=white" alt="Deploy Fizgig on RunPod"></a>
  <a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
  <a href="https://github.com/sponsors/shootthesound"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor on GitHub"></a>
</p>
<p align="center">
  <sub>No GPU, or want a bigger one? Fizgig runs on rented hardware — one click, nothing to install.<br>
  Deploying through that link supports Fizgig's development at no extra cost to you.</sub>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=yrz0l6URGGk"><img src="logo.jpg" alt="Fizgig LoRA & Fine-tune Studio — watch the full video tutorial" width="600"></a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=yrz0l6URGGk"><img src="https://img.shields.io/badge/▶%20Watch%20the%20full%20video%20tutorial-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch the full video tutorial on YouTube"></a><br>
  <sub>Start-to-finish walkthrough — install, prep, caption, train, and the workbench tools</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/models-Klein%209B%20%2B%20Krea%202%20%2B%20MiniMax%20H3%20%2B%20Qwen%20Image%202.1%20%2B%20SDXL%20%2B%20Anima-blue?style=for-the-badge" alt="Klein 9B + Krea 2 + MiniMax H3 + Qwen Image 2.1 + SDXL + Anima">
</p>

> ### 📰 Latest news
> - **Fizgig 7.0 — Anima and SDXL arrive, fine-tuning for every model, and the driver system is complete - meaning the community can now add models to Fizgig (Docs included).** Anima and SDXL join with LoRA and LoKR training, sliders, fine-tuning and the full workbench. Every model fine-tunes on a 16 GB card (Anima and SDXL from 8 GB), and bigger cards train the whole model at once. [Release notes](docs/RELEASE_NOTES_v7.0.0.md) · [Add your own model](docs/drivers/README.md)
> - **Fizgig 6.8 — Krea 2 sliders, Ultra mode and Qwen fine-tuning.** Slider LoRAs on Krea 2, with an Ultra mode that holds up at much higher strengths; full fine-tuning for Qwen Image 2.1 (a single character in as little as 30 minutes on a 5090); Repair Studio with nudge buttons, no strength limit and donor saves that look exactly as previewed; and Krea 2 on Fizgig's new driver system, with Klein and MiniMax H3 next. [Release notes](docs/RELEASE_NOTES_v6.8.1.md)
> - **Fizgig 6.7 — Qwen slider LoRAs.** A LoRA whose strength is a dial between two looks (sad to happy, cool to warm), trained from a few photo pairs or from a few words. [Release notes](docs/RELEASE_NOTES_v6.7.0.md)
> - **Fizgig 6.6 — Qwen edit LoRAs.** Teach Qwen Image 2.1 your own edit (a grade, a look, a relight) from pairs of original and edited photos, then apply it to any photo. [Release notes](docs/RELEASE_NOTES_v6.6.0.md)
> - **Fizgig 6.5.1 — Klein on 10 GB cards.** Base precision on the Training tab for Klein, with Auto picking 4-bit on cards under 16 GB; plus Qwen preview fixes. [Release notes](docs/RELEASE_NOTES_v6.5.1.md)
>
> [All releases →](https://github.com/shootthesound/Fizgig/releases)

---

## What Fizgig is

A local trainer and workbench for image and video models. Every trainer makes LoRAs; Fizgig is also built around what you do with them afterwards:

- **Fix** a LoRA block by block without retraining: drag a slider per block, save a new `.safetensors`.
- **Explore** variations: the app mutates a LoRA, you pick favourites, it evolves.
- **Find** the best epoch by eye: LoRA Royale renders every epoch on one seed with a crossfade, and exports shareable loops.
- **Profile** which blocks carry identity, style and detail before you touch anything.
- **Train beyond LoRAs:** LoKR, or the full base model itself: Anima on an 8 GB card, the larger models on 16 GB. A LoRA extracted from a fine-tune came out better than one trained directly at the same rank.

Memory plans itself: precision, block swap and previews size to your free VRAM, and if a preview can't fit, training keeps going. Fizgig loads kohya, PEFT, OneTrainer, AI-Toolkit and LyCORIS LoRAs, and saves `.safetensors` that drop straight into ComfyUI. Free and open source.

## Supported models

| Model | Trains on | LoRA | LoKR | Slider / Edit LoRA | Full fine-tune | Smallest card | Guide |
|---|---|---|---|---|---|---|---|
| **Flux 2 Klein 9B** | photos | ✅ | ✅ | ✅ slider + edit | ✅ experimental | 10 GB | [Klein 9B](docs/KLEIN.md) |
| **Krea 2** (12.9B) | photos | ✅ | ✅ | ✅ slider | ✅ experimental | 8 GB | [Krea 2](docs/KREA2.md) |
| **MiniMax H3** (33B) | photos, video clips, sound, voice | ✅ | ✅ | ✅ slider | ✅ experimental | 16 GB | [MiniMax H3](docs/MINIMAX_H3.md) |
| **Qwen Image 2.1** | photos | ✅ | ✅ | ✅ slider + edit | ✅ experimental | 10 GB | [Qwen Image 2.1](docs/QWEN_IMAGE.md) |
| **SDXL** (any checkpoint), experimental | photos | ✅ | ✅ | ✅ slider | ✅ experimental | 8 GB | [7.0 notes](docs/RELEASE_NOTES_v7.0.0.md) |
| **Anima** (2B), experimental | photos | ✅ | ✅ | ✅ slider | ✅ experimental | 8 GB | [7.0 notes](docs/RELEASE_NOTES_v7.0.0.md) |

Every model gets all five workbench tools. MiniMax H3 also makes **RefMods**, tuned against H3 itself rather than a plain encode ([how do I…?](docs/REFMOD_HOWDOI.md)). Fizgig's Qwen training adapter is [free on Hugging Face](https://huggingface.co/ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter) for any trainer. Full fine-tuning has [its own guide](docs/FINETUNE.md).

## The workbench

Each tool works on your own runs **or any LoRA you've downloaded**, and they hand off to each other. [More detail](docs/TRAINING.md).

- **Repair Studio** — a live slider per block with a side-by-side preview, donor-LoRA blending, and an exact baked save. On H3 the previews are clips with sound, with a background block library and first/last-frame pinning.
- **LoRA the Explorer** — evolutionary discovery: four mutated variants, pick one, repeat.
- **LoRA Royale** — every epoch on one seed, an optional likeness score, and MP4/GIF exports: epoch morphs, seed/prompt/strength travels, comparison sheets.
- **Profiler** — a colour-coded per-block report that Repair Studio reads inline.
- **Extract** — shrink any LoRA to a lower rank.

## Training features

- **Presets per model**: pick a ✨ preset on the Training tab and go.
- **Slider LoRAs** (Klein 9B, Krea 2, MiniMax H3, Qwen Image 2.1, SDXL, Anima): a LoRA whose strength is a dial between two looks, from photo pairs or three prompts. Krea 2's **Ultra mode** trains the composition blocks only, so the dial holds up at much higher strengths. It works best for sliders trained from prompts, and with photo pairs when the change is compositional.
- **Edit LoRAs** (Klein 9B, Qwen Image 2.1): teach an edit from pairs of original and edited photos, then apply it to any photo.
- **Adaptive LR**: a plateau tracker that raises or lowers the rate within your Min/Max, with rollback on instability.
- **Weight averaging (EMA)**, on by default where it's measured to help.
- **Context LoRA**: train on top of a frozen, active LoRA so the two coexist (a face on a style, an outfit on a character). No other trainer does this.
- **Pause and resume** with full state, plus a **training queue** for back-to-back runs.
- **The trainer curates your dataset while it trains** (every model except MiniMax H3). Every image starts as useful training data; the watch follows each one's loss, and only when an image stops teaching the model does it step in: throttle it, recaption it, and as a last resort set it aside. It also tells you when the whole run has plateaued. No other trainer does this.
- **A sample gallery that scores likeness** live, and a run visualiser to scrub epochs.
- **Dataset prep**: AI captions with Qwen3-VL or Florence-2, bilingual captions, face crops, and a Look Consistency Filter.
- **Gizmo** (MiniMax H3): cut clips with scene detection, crop to the subject, and record or segment a voice dataset with Whisper transcription.

Full list: [docs/TRAINING.md](docs/TRAINING.md). Everything also runs headless: [docs/CLI.md](docs/CLI.md).

## Install

Needs an NVIDIA RTX 30/40/50-series or an AMD Radeon with ROCm, Windows 10/11 or Linux, Python 3.10–3.13, and 32 GB of system RAM recommended. Full requirements, AMD ROCm and model downloads: [docs/INSTALL.md](docs/INSTALL.md).

Clone the repo rather than downloading the ZIP (the updater pulls with git):

```bash
git clone https://github.com/shootthesound/Fizgig.git
cd Fizgig
```

- **Windows (NVIDIA):** double-click `install_fizgig.bat`, launch with `run_fizgig.bat`, update with `update_fizgig.bat`.
- **Linux (NVIDIA):** `python install_fizgig.py`, then `./run_fizgig.sh`.
- **AMD ROCm:** see [docs/INSTALL.md](docs/INSTALL.md).

Model files download from the **Preferences** tab: one **Download models for me** button per model.

### No GPU? Rent one

The whole app runs in a browser tab on RunPod, with a file manager, one-click model downloads and an optional auto-stop when training finishes. **[⚡ Deploy on RunPod →](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)** · [Guide](docker/README.md)

## Getting started

Work left to right through the numbered tabs:

1. **Start** — set your training image folder.
2. **Image Prep** (optional) — resize, face-crop, and filter out images that don't match the look.
3. **Captions** — trigger-word or AI captions.
4. **Samples** — the preview prompts that render during training.
5. **Training** — pick a model and a preset, click **Start Training**.

**Tip: give one sample prompt something your dataset doesn't have,** such as a specific pose, angle or setting that none of your photos show (for example "looking over her shoulder from a low angle" when your photos are all straight-on portraits). Early epochs follow it. When a LoRA starts to overtrain, that is the first thing to fail: the previews drift back to your dataset's framing even though the prompt asks for something else. The last epoch that still follows it is a good one to keep.

The unnumbered tabs are the workbench (Profiler, Repair Studio, LoRA the Explorer, LoRA Royale, Extract) and Preferences. The [video tutorial](https://www.youtube.com/watch?v=yrz0l6URGGk) walks through all of it.

**Community translation:** [Korean (한국어)](https://github.com/ssain3d-lgtm/Fizgig-Korean-Translated-Ver) by @ssain3d-lgtm, an unofficial add-on that translates the UI at runtime without touching Fizgig's files and uninstalls with one script. If you hit a bug with it installed, uninstall and reproduce before reporting here; translation issues go to its repo.

---

## Support the project

Fizgig is free. If it saves you time or makes your LoRAs better, a coffee or a GitHub sponsorship keeps it going:

<a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a> <a href="https://github.com/sponsors/shootthesound"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor on GitHub"></a>

Renting through the [RunPod link](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep) supports development too, at no extra cost to you.

---

## Documentation

| Guide | What's in it |
|---|---|
| [Install and requirements](docs/INSTALL.md) | Hardware, system RAM, AMD ROCm, model downloads, VRAM tables |
| [Klein 9B](docs/KLEIN.md) · [Krea 2](docs/KREA2.md) · [MiniMax H3](docs/MINIMAX_H3.md) · [Qwen Image 2.1](docs/QWEN_IMAGE.md) | Each model: presets, features, memory |
| [Training and the workbench](docs/TRAINING.md) | The shared features and tools in detail |
| [Full fine-tuning](docs/FINETUNE.md) · [How do I…?](docs/FINETUNE_HOWDOI.md) | Training the base model itself |
| [RefMods](docs/REFMOD_HOWDOI.md) | RefMods for MiniMax H3 |
| [Command line](docs/CLI.md) | Every trainer, headless |
| [RunPod](docker/README.md) | The cloud image |

## License

Fizgig is open source under the **[Apache License 2.0](LICENSE)** — free to use, modify, and redistribute, including commercially, with attribution and no warranty. Third-party components under compatible permissive licenses (and other terms where noted) are listed in **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)**.

The **Automagic v3 optimizer**, the Krea 2 **MMDiT backbone and flow-matching sampler**, and Ostris's MiniMax H3 **training adapter** all come from **[@ostris](https://github.com/ostris)** — the first two from [AI-Toolkit](https://github.com/ostris/ai-toolkit) under the MIT licence, the adapter downloaded as a model rather than bundled. The default H3 training adapter comes from **[circlestone-labs](https://huggingface.co/circlestone-labs)**, also downloaded as a model. Anima's model code is adapted from kohya-ss's [sd-scripts](https://github.com/kohya-ss/sd-scripts) under the Apache licence.

Copyright © 2026 Peter Neill.

Model weights are **not** covered by this license — each model carries its own terms from its publisher (see the Download links in Preferences).

---

## Working with me

I'm available for consulting on local AI training pipelines, custom workflow tooling, and
private model work — the same engineering that's in Fizgig, applied to your studio's
hardware and IP. Get in touch: **peter@shootthesound.com**.
