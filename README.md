<h1 align="center">Fizgig — LoRA & Fine-tune Studio for Klein 9B, Flux 3 Image Klein, Krea 2, MiniMax H3, Qwen Image 2.1, MOSS-Image, Z-Image Turbo, SDXL & Anima</h1>

<p align="center">
  <strong>Fine-tune base models on consumer GPUs — down to 8 GB. Fix broken LoRAs without retraining. Find the best epoch by eye.</strong><br>
  A train · fine-tune · repair · explore workbench built end-to-end for <strong>Flux 2 Klein 9B</strong>, <strong>Flux 3 Image Klein</strong>, <strong>Krea 2</strong>, <strong>MiniMax H3</strong>, <strong>Qwen Image 2.1</strong>, <strong>MOSS-Image</strong>, <strong>Z-Image Turbo</strong>, <strong>SDXL</strong> and <strong>Anima</strong> — training on photos, video, sound and voices, from quick LoRAs to the full base model.
</p>

<p align="center">
  <a href="#install"><img src="https://img.shields.io/badge/⬇%20Install%20Fizgig-2EA043?style=for-the-badge&logoColor=white" alt="Jump to the install instructions"></a>
  <a href="https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=bdrdew7hxw&ref=vkb387ep"><img src="https://img.shields.io/badge/⚡%20Deploy%20on%20RunPod-673AB7?style=for-the-badge&logoColor=white" alt="Deploy Fizgig on RunPod"></a>
  <a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
  <a href="https://github.com/sponsors/shootthesound"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor on GitHub"></a>
</p>
<p align="center">
  <sub>No GPU, or want a bigger one? Fizgig runs on rented hardware — one click, nothing to install.<br>
  On RunPod's page, <b>add a 300 GB Volume disk before Deploy Pod</b> (<a href="docker/README.md#give-it-a-disk-on-runpods-deploy-page">how</a>).<br>
  Deploying through that link supports Fizgig's development at no extra cost to you.</sub>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=yrz0l6URGGk"><img src="docs/assets/fizgig_v8.jpg" alt="Fizgig LoRA & Fine-tune Studio v8.0 — watch the full video tutorial" width="700"></a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=yrz0l6URGGk"><img src="https://img.shields.io/badge/▶%20Watch%20the%20full%20video%20tutorial-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch the full video tutorial on YouTube"></a><br>
  <sub>Start-to-finish walkthrough — install, prep, caption, train, and the workbench tools</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/models-Klein%209B%20%2B%20Flux%203%20Image%20Klein%20%2B%20Krea%202%20%2B%20MiniMax%20H3%20%2B%20Qwen%20Image%202.1%20%2B%20MOSS--Image%20%2B%20Z--Image%20Turbo%20%2B%20SDXL%20%2B%20Anima%20%2B%20ACE--Step%201.5%20XL-blue?style=for-the-badge" alt="Klein 9B + Flux 3 Image Klein + Krea 2 + MiniMax H3 + Qwen Image 2.1 + MOSS-Image + Z-Image Turbo + SDXL + Anima + ACE-Step 1.5 XL">
</p>

> ### 📰 Latest news
> - **Fizgig 8.9.6 — masked training.** Train on the part of each picture that matters: the person, not the room around them. A black-and-white mask beside a picture says which part counts, and Image Prep makes them for a whole folder in one click (the person, just the face and hair, the body without the face, and more). In our tests of character LoRAs, masks significantly raised both the best likeness a run reached, by eye and in likeness scores, and how quickly it got there, on every recent model. It seems especially effective on MOSS-Image with its default Clean-end share, Slide. [Release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.6) · [Masked training guide](docs/MASKED_TRAINING.md)
> - **Fizgig 8.9.5 — faster training on 40 GB+ cards.** On most pods (A100, A40, A6000, L40S, H100, RTX PRO 6000) Auto now turns gradient checkpointing off when the run fits with room to spare: 13–27 % faster per step on Krea 2, Qwen Image 2.1, MOSS-Image, Z-Image and MiniMax H3. Where it doesn't all fit, the last blocks go without it: 2–21 % faster. Previews keep at least 4 GB free, and cards of 32 GB and under train as before. [Release notes](https://github.com/shootthesound/Fizgig/releases/tag/v8.9.5)
> - **Fizgig 8.9.0 — your other GPUs, at home or in the cloud, speeding up your run.** A helper is another computer running Fizgig: a second PC or laptop, a second GPU in this computer, or a RunPod or Vast.ai pod. It renders your previews so training never stops for them, takes over a run while you need your computer, or trains a share of each epoch alongside your GPU. In a share, each GPU trains its own copy of the LoRA on its part of the images and the copies' changes are added into one LoRA at every combine: in testing its likeness matched a one-GPU run. On a fast network, combine every epoch for the closest result. It pays most on long runs. Training on a pod? Your home GPU can help it, and its checkpoints come home as they're saved. [Release notes](docs/RELEASE_NOTES_v8.9.0.md) · [▶ Watch the video](https://www.youtube.com/watch?v=7Ao0gE2b2qw) · [Helpers guide](docs/HELPERS.md)
> - **Fizgig 8.8.1 — MOSS-Image likeness sooner, and previews with the SFT look.** A new Clean-end share default, **Slide**, brings likeness in early and keeps it: on our most difficult dataset, the best of eight settings we tried, around 70% likeness from epoch 20 to epoch 60. Base previews can take on much of MOSS-Image SFT's look with no model swap, every enlarged preview says how it was made, and Krea 2's Auto precision no longer picks a slow setup on 16 GB cards. 8.8.2 stops the web UI freezing when you open it during a long run. [Release notes](docs/RELEASE_NOTES_v8.8.1.md) · [8.8.2](docs/RELEASE_NOTES_v8.8.2.md) · [▶ Watch the video](https://www.youtube.com/watch?v=UCoCtM8yrBk) · [MOSS-Image guide](docs/MOSS_IMAGE.md)
> - **Fizgig 8.5 — MOSS-Image.** OpenMOSS's new 5B open image model trains in Fizgig: LoRAs, LoKRs, sliders, full fine-tuning, Multi Concept and Fizgig LoRA Isolation, with previews on MOSS-Image SFT, the model most people generate with, and every workbench tool. Brought to Fizgig by @Murf-Code. 8.5.1 fixes the GPU readout on AMD cards (thanks @scryptio); 8.5.3 fixes Linux launching; 8.5.4 makes MOSS-Image previews render as ComfyUI does; 8.5.5 trains MOSS-Image at 1 MP; 8.5.8 fixes MOSS-Image training with block swap; 8.5.9 makes MOSS-Image Studio renders faster; 8.6.0 improves MOSS-Image likeness and adds an option for training on SFT; 8.6.1 fixes noisy Qwen Image 2.1 previews; 8.7 brings MOSS-Image likeness much sooner with a new Clean-end share option, 30% by default since 8.7.1, fine-tunes included in 8.7.2, since 8.7.3 a fine-tune fits a 16 GB card, and 8.7.4 adds an 8-step Turbo LoRA for fast previews on Base, 8.7.5 lets MOSS-Image train on computers with 16 GB of RAM, 8.7.6 fixes the cache on a resume and for a refilled folder, 8.7.7 drops Kromoss, 8.7.8 adds preview and save schedules, 8.7.9 is a maintenance release, and 8.8.0 fixes resuming a run on Python 3.10. [Release notes](docs/RELEASE_NOTES_v8.8.0.md) · [MOSS-Image guide](docs/MOSS_IMAGE.md)
> - **Fizgig 8.4 — Fizgig LoRA Isolation.** Train two people into one LoRA and every stranger keeps their own face: with a normal LoRA 48 of 50 strangers came out looking like the trained person, with LoRA Isolation 0 of 50, at the same likeness. DOP from @ostris's AI-Toolkit at its base, with Fizgig's own techniques on top, set up on the Training tab's new Subjects card, on every model except ACE-Step. 8.4.2 adds a trigger word checker on the same card. [Demo](https://youtu.be/CNh4sN5vN_8) · [How to set it up](https://youtu.be/8uW5NdL5b20) · [Release notes](docs/RELEASE_NOTES_v8.4.0.md)
> - **Fizgig 8.1 — Flux 3 Image Klein, from day one.** Black Forest Labs' new 5.6B text-to-image and edit model trains in Fizgig the day it came out: LoRAs, LoKRs, edit LoRAs from before/after pairs, slider LoRAs and full fine-tuning, from an 8 GB card, with every workbench tool, prompt travel and edit chains in LoRA Royale included. [Release notes](docs/RELEASE_NOTES_v8.1.0.md) · [Flux 3 guide](docs/FLUX3.md)
> - **Fizgig 8 — the new app (yes, the web UI).** Fizgig opens in its own window or your browser, and your phone can watch a run. One **Pods** tab opens your other computers' GPUs, found on your network by themselves, and RunPod pods, logged in for you. What's training and what's up next sit in the sidebar on every tab, and **Runs & outputs** keeps every run. Updating keeps your settings, presets, folders and queued runs; a welcome tour on the first start shows where everything went. [Release notes](docs/RELEASE_NOTES_v8.0.0.md) · [The app](docs/WEB_UI.md)
> - **Fizgig 7.3 — music with ACE-Step 1.5 XL.** Train LoRAs on your own songs, music sliders and full fine-tuning, previews you play in the gallery, and Repair Studio as a DAW for AI music: covers, repaint and extend, a model EQ. Gizmo gets a Music tab for cutting songs into sections. [Release notes](docs/RELEASE_NOTES_v7.3.0.md) · [ACE-Step guide](docs/ACESTEP.md)
> - **Fizgig 7.2 — Qwen Image 2.1 Turbo.** Repair Studio, LoRA the Explorer and LoRA Royale preview on Qwen's new Turbo model, and training previews use Comfy-Org's new Turbo LoRA: 8 steps, CFG 1, about four times faster, and LoRAs trained on the base keep their likeness on it. Plus MiniMax H3 long captions on V100-class cards by @Hell-Bent-Fox. [Release notes](docs/RELEASE_NOTES_v7.2.0.md)
> - **Fizgig 7.1 — Z-Image Turbo, with its own training adapter.** LoRA and LoKR training, sliders, full fine-tuning and every workbench tab. Fizgig's training adapter keeps Turbo's 8-step look while your LoRA learns: better likeness, image quality and detail than the adapter most people use. LoRAs from 12 GB cards, fine-tunes from 16 GB. [Release notes](docs/RELEASE_NOTES_v7.1.0.md)
> - **Fizgig 7.0 — Anima and SDXL arrive, fine-tuning for every model, and the driver system is complete - meaning the community can now add models to Fizgig (Docs included).** Anima and SDXL join with LoRA and LoKR training, sliders, fine-tuning and the full workbench. Every model fine-tunes on a 16 GB card (Anima and SDXL from 8 GB), and bigger cards train the whole model at once. [Release notes](docs/RELEASE_NOTES_v7.0.0.md) · [Add your own model](docs/drivers/README.md)
> - **Fizgig 6.8 — Krea 2 sliders, Ultra mode and Qwen fine-tuning.** Slider LoRAs on Krea 2, with an Ultra mode that holds up at much higher strengths; full fine-tuning for Qwen Image 2.1 (a single character in as little as 30 minutes on a 5090); Repair Studio with nudge buttons, no strength limit and donor saves that look exactly as previewed; and Krea 2 on Fizgig's new driver system, with Klein and MiniMax H3 next. [Release notes](docs/RELEASE_NOTES_v6.8.1.md)
> - **Fizgig 6.7 — Qwen slider LoRAs.** A LoRA whose strength is a dial between two looks (sad to happy, cool to warm), trained from a few photo pairs or from a few words. [Release notes](docs/RELEASE_NOTES_v6.7.0.md)
> - **Fizgig 6.6 — Qwen edit LoRAs.** Teach Qwen Image 2.1 your own edit (a grade, a look, a relight) from pairs of original and edited photos, then apply it to any photo. [Release notes](docs/RELEASE_NOTES_v6.6.0.md)
>
> [All releases →](https://github.com/shootthesound/Fizgig/releases)

---

## What Fizgig is

A trainer and workbench for image, video and music models, in its own window on your computer, in your browser, or on a rented GPU — and you can watch a run from your phone. Every trainer makes LoRAs; Fizgig is also built around what you do with them afterwards:

- **Fix** a LoRA block by block without retraining: drag a slider per block, save a new `.safetensors`.
- **Explore** variations: 🎲 I'm feeling lucky in Studio rolls new block strengths within your limits; keep the ones you like.
- **Find** the best epoch by eye: LoRA Royale renders every epoch on one seed with a crossfade, and exports shareable loops.
- **Profile** which blocks carry identity, style and detail before you touch anything.
- **Train beyond LoRAs:** LoKR, or the full base model itself: Anima on an 8 GB card, the larger models on 16 GB. A LoRA extracted from a fine-tune came out better than one trained directly at the same rank.

Memory plans itself: precision, block swap and previews size to your free VRAM, and if a preview can't fit, training keeps going. Fizgig loads kohya, PEFT, OneTrainer, AI-Toolkit and LyCORIS LoRAs, and saves `.safetensors` that drop straight into ComfyUI. Free and open source.


https://github.com/user-attachments/assets/2db82d0d-e7ed-4f60-a79c-d2ac1f4984ac

<p align="center"><sub>The Fizgig 8 tour · <a href="https://youtu.be/F8F-vuiZd88">watch it on YouTube</a></sub></p>

## Supported models

| Model | Trains on | LoRA | LoKR | Slider / Edit LoRA | Full fine-tune | Smallest card | Guide |
|---|---|---|---|---|---|---|---|
| **Flux 2 Klein 9B** | photos | ✅ | ✅ | ✅ slider + edit | ✅ experimental | 10 GB | [Klein 9B](docs/KLEIN.md) |
| **Flux 3 Image Klein** (5.6B), experimental | photos | ✅ | ✅ | ✅ slider + edit | ✅ experimental | 8 GB | [Flux 3 Image Klein](docs/FLUX3.md) |
| **Krea 2** (12.9B) | photos | ✅ | ✅ | ✅ slider | ✅ experimental | 8 GB | [Krea 2](docs/KREA2.md) |
| **MiniMax H3** (33B) | photos, video clips, sound, voice | ✅ | ✅ | ✅ slider | ✅ experimental | 16 GB | [MiniMax H3](docs/MINIMAX_H3.md) |
| **Qwen Image 2.1** | photos | ✅ | ✅ | ✅ slider + edit | ✅ experimental | 10 GB | [Qwen Image 2.1](docs/QWEN_IMAGE.md) |
| **MOSS-Image** (5B), experimental | photos | ✅ | ✅ | ✅ slider | ✅ experimental | 12 GB | [MOSS-Image](docs/MOSS_IMAGE.md) |
| **Z-Image Turbo** (6B), experimental | photos | ✅ | ✅ | ✅ slider | ✅ experimental | 12 GB | [Z-Image Turbo](docs/ZIMAGE.md) |
| **SDXL** (any checkpoint), experimental | photos | ✅ | ✅ | ✅ slider | ✅ experimental | 8 GB | [SDXL](docs/SDXL.md) |
| **Anima** (2B), experimental | photos | ✅ | ✅ | ✅ slider | ✅ experimental | 8 GB | [Anima](docs/ANIMA.md) |
| **ACE-Step 1.5 XL** (4B), experimental | songs | ✅ | — | ✅ slider | ✅ experimental | 16 GB | [ACE-Step](docs/ACESTEP.md) |

Every image and video model gets all four workbench tools; ACE-Step has Studio and Extract. Qwen Image 2.1 trains on the base model and previews on Qwen's own Turbo (8 steps, CFG 1); your LoRAs work on both. MOSS-Image trains on Base and previews on MOSS-Image SFT, the model most people generate with. MiniMax H3 also makes **RefMods**, tuned against H3 itself rather than a plain encode ([how do I…?](docs/REFMOD_HOWDOI.md)). Fizgig's training adapters for [Qwen Image 2.1](https://huggingface.co/ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter) and [Z-Image Turbo](https://huggingface.co/ShootTheSound/Fizgig-Z-Image-Turbo-Training-Adapter) are free on Hugging Face for any trainer. Full fine-tuning has [its own guide](docs/FINETUNE.md).

## The workbench

Each tool works on your own runs **or any LoRA you've downloaded**, and they hand off to each other. [More detail](docs/TRAINING.md).

- **Studio** (was Repair Studio) — a live slider per block with a side-by-side preview, donor-LoRA blending, **🎲 I'm feeling lucky** for random variations within your limits, and an exact baked save. On H3 the previews are clips with sound, with a background block library and first/last-frame pinning.
- **LoRA Royale** — every epoch on one seed, an optional likeness score, and MP4/GIF exports: epoch morphs, seed/prompt/strength travels, comparison sheets.
- **Profiler** — a colour-coded per-block report that Studio reads inline.
- **Extract** — shrink any LoRA to a lower rank.

<p align="center"><img src="docs/assets/screenshots/studio.png" alt="Studio: a LoRA's blocks beside the baseline and tweaked previews" width="800"></p>

## Training features

- **Presets per model**: pick a ✨ preset on the Training tab and go.
- **Slider LoRAs** (Klein 9B, Flux 3 Image Klein, Krea 2, MiniMax H3, Qwen Image 2.1, MOSS-Image, Z-Image Turbo, SDXL, Anima): a LoRA whose strength is a dial between two looks, from photo pairs or three prompts, and a slider from prompts can be steered while it trains (change its ends, it trains on toward them). Krea 2's **Ultra mode** trains the composition blocks only, so the dial holds up at much higher strengths. It works best for sliders trained from prompts, and with photo pairs when the change is compositional.
- **Edit LoRAs** (Klein 9B, Flux 3 Image Klein, Qwen Image 2.1): teach an edit from pairs of original and edited photos, then apply it to any photo.
- **Fizgig LoRA Isolation, on the Subjects card** (every model except ACE-Step): up to four people in one LoRA, each from their own folder. Fizgig's own techniques combined with DOP keep your subjects to their trigger words: leave the trigger out and you get the model's own people, and in a group picture your subjects, strangers and people the model already knows each keep their own face. [See it in action](https://youtu.be/CNh4sN5vN_8) · [How it works](docs/LORA_ISOLATION.md) · [How to set it up (video)](https://youtu.be/8uW5NdL5b20) · [How to caption for it](docs/MULTI_PERSON.md).
- **Masked training** (every image model except MiniMax H3): train on the part of each picture that matters, the person and not the room around them. Image Prep makes the masks for a whole folder in one click, or you paint your own. In our tests of character LoRAs it significantly raised both the best likeness and how quickly runs got there. [How it works](docs/MASKED_TRAINING.md)
- **Adaptive LR**: a plateau tracker that raises or lowers the rate within your Min/Max, with rollback on instability.
- **Weight averaging (EMA)**, on by default where it's measured to help.
- **Context LoRA**: train on top of a frozen, active LoRA so the two coexist (a face on a style, an outfit on a character). No other trainer does this.
- **Pause and resume** with full state, plus a **training queue** for back-to-back runs.
- **The trainer curates your dataset while it trains** (every model except MiniMax H3). Every image starts as useful training data; the watch follows each one's loss, and only when an image stops teaching the model does it step in: throttle it, recaption it, and as a last resort set it aside. It also tells you when the whole run has plateaued. No other trainer does this.
- **Fast previews** in a few steps on a turbo or distilled model where one exists: Klein's and Flux 3 Image Klein's distilled models, the Turbo LoRAs for Krea 2, MiniMax H3, Qwen Image 2.1 and Anima, and Qwen Image 2.1's Turbo model.
- **A sample gallery that scores likeness** live, and a run visualiser to scrub epochs.
- **Dataset prep**: AI captions with Qwen3-VL or Florence-2, bilingual captions, face crops, and a Look Consistency Filter.
- **Face backends for every kind of character**: face crops, the Look Filter and likeness scores work on anime, furry and fursuit characters as well as people (see the section below).
- **Gizmo** (MiniMax H3): cut clips with scene detection, crop to the subject, and record or segment a voice dataset with Whisper transcription.

Full list: [docs/TRAINING.md](docs/TRAINING.md). Everything also runs headless: [docs/CLI.md](docs/CLI.md).

## One app, wherever you are

- **In its own window** (`Launch Fizgig.bat`), or **in your browser** (`Launch Fizgig in browser.bat`) — the same app either way.
- **The queue in sight**: what's training and what's queued sit in the sidebar on every tab; a held queue says why, with Start queue right there. **Continue** on any run in Runs & outputs picks it up again from its last save, so you can pause one project, train another and come back.
- **On your phone or another computer**: set a password in Preferences and open it over your network, or scan the Start tab's **Open on your phone** QR code and you're straight in; add it to your phone's home screen and it opens like an app.
- **On your other computer's GPU**: train on your desktop's card from your laptop. Fizgig finds your other computers running it, and the **Pods** tab opens one as it does a pod ([how](docs/WEB_UI.md#your-other-computers-use-ones-gpu-from-another)).
- **On a rented GPU**: the **Pods** tab finds the card that's cheapest per run among those RunPod has free right now (one card or two), deploys it, opens the pod in the same window (logged in for you), and sends a training folder from your computer straight to it. Prefer Vast.ai? Fizgig has a template there too, with https addresses and a stop when the run finishes ([Vast.ai](docs/VAST.md)).
- **Your other GPUs helping**: a second computer, a second GPU in this one, or a pod renders your previews so training never stops for them, trains a share of each epoch alongside yours, or takes over a run while you game or someone else needs the computer, then hands it back. A pod's checkpoints can come home as they're saved, and the pod can be deleted once they're all here ([Helpers](docs/HELPERS.md)).
- **Runs & outputs** keeps every run with its loss graph, notes, previews and checkpoints, and starts any of them again with the same settings.

<p align="center">
  <img src="docs/assets/screenshots/pods.png" alt="The Pods tab: a laptop found on the network, and a RunPod pod" width="640">
  <img src="docs/assets/screenshots/phone.png" alt="A run on a phone: progress, loss, time left and what's up next" width="150">
</p>

More: [docs/WEB_UI.md](docs/WEB_UI.md).

## Face backends

Face crops, the Look Consistency Filter and every likeness score can work on anime, furry and drawn characters and fursuits too, not just people.

**The defaults are InsightFace (face finder) and ArcFace (look match), the same as before.** People's sets need nothing changed. Everything else is opt-in: open **Image Prep → Anime/furry/drawn characters** and pick:

- **Face finder**: **Anime/furry/drawn (YOLO model)** for characters, or **Both** for a set with people and characters in it.
- **Look match**: **Anime/furry/drawn** for characters, or **Auto (Decide for each set)**. Each look match has its own thresholds, and saved scores record which one made them.

Setup (which face model to use and how to get it), numbers and limitations are in [docs/FACE_BACKENDS.md](docs/FACE_BACKENDS.md).

## Install

Needs an NVIDIA RTX 30/40/50-series or an AMD Radeon with ROCm, Windows 10/11 or Linux, Python 3.10–3.13, and 32 GB of system RAM recommended. Full requirements, AMD ROCm and model downloads: [docs/INSTALL.md](docs/INSTALL.md).

Clone the repo rather than downloading the ZIP (the updater pulls with git):

```bash
git clone https://github.com/shootthesound/Fizgig.git
cd Fizgig
```

- **Windows (NVIDIA):** double-click `install_fizgig.bat`, then launch with `Launch Fizgig.bat` (its own window) or `Launch Fizgig in browser.bat` (in your web browser); update with `update_fizgig.bat`.
- **Linux (NVIDIA):** `python install_fizgig.py`, then `./launch_fizgig.sh` or `./launch_fizgig_browser.sh`; update with `./update_fizgig.sh`.
- **AMD ROCm:** see [docs/INSTALL.md](docs/INSTALL.md).

Model files download from the **Preferences** tab: one **Download models for me** button per model.

### No GPU? Rent one

**Nothing to install on your computer:** the whole app runs on the pod, in your browser.

1. **[⚡ Deploy on RunPod](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=bdrdew7hxw&ref=vkb387ep)**, and add a **300 GB Volume disk** before **Deploy Pod**: RunPod doesn't add one for you ([how](docker/README.md#give-it-a-disk-on-runpods-deploy-page)).
2. Wait for **Ready** in the pod's **Logs** tab; the password is printed under it.
3. **Connect → HTTP port 8081**, and log in with that password. (Opened from Fizgig on your own computer, with your RunPod API key on its Pods tab, the pod logs you in by itself.)
4. **Preferences → Download models for me**, upload your training folder on the **Start** tab, and press **Start** on the **Training** tab.

Prefer Vast.ai? The same image runs there: **[Fizgig on Vast.ai](https://cloud.vast.ai/?ref_id=757773&creator_id=757773&name=Fizgig%20LoRa%20Training%20and%20Fine-tuning)** ([guide](docs/VAST.md)).

Step by step: [Quick start](docker/README.md#quick-start-nothing-installed-on-your-computer). The pod also has a drag-and-drop file manager for big datasets and fine-tunes (the **File manager** button in the sidebar), one-click model downloads and an optional auto-stop when training finishes. Opened from your own Fizgig, a pod runs the same version as your computer ([which Fizgig a pod runs](docker/README.md#which-fizgig-a-pod-runs)). [The full pod guide](docker/README.md).

Mid-job on the old pod with the remote desktop? It stays available as **[Fizgig Classic v7.x](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)**, and pods you already have keep running as they are.

## Getting started

Work down through the numbered tabs:

1. **Start** — set your training image folder.
2. **Image Prep** (optional) — resize, face-crop, and filter out images that don't match the look.
3. **Captions** — trigger-word or AI captions.
4. **Samples** — the preview prompts that render during training.
5. **Training** — pick a model and a preset, click **Start Training**.

<p align="center"><img src="docs/assets/screenshots/training_run.png" alt="The Training tab: a run going, with its progress, Pause, Stop and Add to queue" width="800"></p>

**Tip: give one sample prompt something your dataset doesn't have,** such as a specific pose, angle or setting that none of your photos show (for example "looking over her shoulder from a low angle" when your photos are all straight-on portraits). Early epochs follow it. When a LoRA starts to overtrain, that is the first thing to fail: the previews drift back to your dataset's framing even though the prompt asks for something else. The last epoch that still follows it is a good one to keep.

The unnumbered tabs are Runs & outputs, the workbench (Profiler, Studio, LoRA Royale, RefMod Toolkit, Extract, Metadata), Preferences, and Pods. The [video tutorial](https://www.youtube.com/watch?v=yrz0l6URGGk) walks through all of it.

### Updating from Fizgig 7

Run `update_fizgig.bat` as always. Fizgig 7's desktop app is replaced by the new one: start it with **Launch Fizgig.bat** from now on (`run_fizgig.bat` is gone, so point a shortcut at the new file). Everything of yours moves into one `user_data` folder: your settings, presets, models, cache and queued runs come across, Fizgig opens on the model you last trained, and the welcome on the first start shows where everything went.

<p align="center"><img src="docs/assets/screenshots/welcome.png" alt="The welcome on the first start: a short tour beside what's the same and what's new" width="720"></p>

**Community translation:** [Korean (한국어)](https://github.com/ssain3d-lgtm/Fizgig-Korean-Translated-Ver) by @ssain3d-lgtm, an unofficial add-on that translates the UI at runtime without touching Fizgig's files and uninstalls with one script. If you hit a bug with it installed, uninstall and reproduce before reporting here; translation issues go to its repo.

---

## Support the project

Fizgig is free. If it saves you time or makes your LoRAs better, a coffee or a GitHub sponsorship keeps it going:

<a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a> <a href="https://github.com/sponsors/shootthesound"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor on GitHub"></a>

Renting through the [RunPod link](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=bdrdew7hxw&ref=vkb387ep) supports development too, at no extra cost to you. **Add a 300 GB Volume disk on RunPod's page before Deploy Pod** ([how](docker/README.md#give-it-a-disk-on-runpods-deploy-page)).

---

## Documentation

| Guide | What's in it |
|---|---|
| [Install and requirements](docs/INSTALL.md) | Hardware, system RAM, AMD ROCm, model downloads, VRAM tables |
| [Klein 9B](docs/KLEIN.md) · [Flux 3 Image Klein](docs/FLUX3.md) · [Krea 2](docs/KREA2.md) · [MiniMax H3](docs/MINIMAX_H3.md) · [Qwen Image 2.1](docs/QWEN_IMAGE.md) · [MOSS-Image](docs/MOSS_IMAGE.md) · [Z-Image Turbo](docs/ZIMAGE.md) · [SDXL](docs/SDXL.md) · [Anima](docs/ANIMA.md) · [ACE-Step](docs/ACESTEP.md) | Each model: presets, features, memory |
| [Training and the workbench](docs/TRAINING.md) | The shared features and tools in detail |
| [Face backends](docs/FACE_BACKENDS.md) | Face finder (InsightFace / YOLO) and look match (ArcFace / DINOv2) for people, anime and furry |
| [Full fine-tuning](docs/FINETUNE.md) · [How do I…?](docs/FINETUNE_HOWDOI.md) | Training the base model itself |
| [RefMods](docs/REFMOD_HOWDOI.md) | RefMods for MiniMax H3 |
| [Command line](docs/CLI.md) | Every trainer, headless |
| [Fizgig LoRA Isolation](docs/LORA_ISOLATION.md) | How it works, the recipe and the results |
| [Masked training](docs/MASKED_TRAINING.md) | Train on the part of each picture that matters; making masks, or painting your own |
| [The app](docs/WEB_UI.md) | Starting Fizgig, its window and browser, your phone, your other computers, the port and password, pods |
| [Helpers](docs/HELPERS.md) · [Previews on a helper](docs/HELPER_PREVIEWS.md) · [Training share](docs/TRAINING_SHARE.md) · [Move a run](docs/MOVE_RUN.md) | Another computer or a pod renders your previews, trains a share of each epoch, or takes over a run while you use your computer |
| [RunPod](docker/README.md) · [Vast.ai](docs/VAST.md) | The cloud image, on RunPod and on Vast.ai |

## License

Fizgig is open source under the **[Apache License 2.0](LICENSE)** — free to use, modify, and redistribute, including commercially, with attribution and no warranty. Third-party components under compatible permissive licenses (and other terms where noted) are listed in **[THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)**.

The **Automagic v3 optimizer**, the Krea 2 **MMDiT backbone and flow-matching sampler**, and Ostris's MiniMax H3 **training adapter** all come from **[@ostris](https://github.com/ostris)** — the first two from [AI-Toolkit](https://github.com/ostris/ai-toolkit) under the MIT licence, the adapter downloaded as a model rather than bundled. The default H3 training adapter comes from **[circlestone-labs](https://huggingface.co/circlestone-labs)**, also downloaded as a model. Anima's model code is adapted from kohya-ss's [sd-scripts](https://github.com/kohya-ss/sd-scripts) under the Apache licence, Z-Image's from Tongyi-MAI's [reference implementation](https://github.com/Tongyi-MAI/Z-Image), and Flux 3 Image Klein's from Black Forest Labs' [reference implementation](https://github.com/black-forest-labs/flux-3-image-klein), both Apache too.

Copyright © 2026 Peter Neill.

Model weights are **not** covered by this license — each model carries its own terms from its publisher (see the Download links in Preferences).

---

## Working with me

I'm available for consulting on local AI training pipelines, custom workflow tooling, and
private model work — the same engineering that's in Fizgig, applied to your studio's
hardware and IP. Get in touch: **peter@shootthesound.com**.
