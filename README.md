<h1 align="center">Fizgig — Klein 9B, Krea 2 & MiniMax H3 LoRA Studio</h1>

<p align="center">
  <strong>Fine-tune base models on consumer GPUs — down to 16 GB. Fix broken LoRAs without retraining. Remix any LoRA into new variations in seconds.</strong><br>
  A train · fine-tune · repair · explore workbench built end-to-end for <strong>Flux 2 Klein 9B</strong>, <strong>Krea 2</strong> and <strong>MiniMax H3</strong> — training on photos, video, sound and voices, from quick LoRAs to the full base model.
</p>

<p align="center">
  <a href="#install"><img src="https://img.shields.io/badge/⬇%20Install%20Fizgig-2EA043?style=for-the-badge&logoColor=white" alt="Jump to the install instructions"></a>
  <a href="https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep"><img src="https://img.shields.io/badge/⚡%20Deploy%20on%20RunPod-673AB7?style=for-the-badge&logoColor=white" alt="Deploy Fizgig on RunPod"></a>
  <a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
</p>
<p align="center">
  <sub>No GPU, or want a bigger one? Fizgig runs on rented hardware — one click, nothing to install.<br>
  Deploying through that link supports Fizgig's development at no extra cost to you.</sub>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=yrz0l6URGGk"><img src="assets/hero.png" alt="Fizgig LoRA Studio — watch the full video tutorial" width="600"></a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=yrz0l6URGGk"><img src="https://img.shields.io/badge/▶%20Watch%20the%20full%20video%20tutorial-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="Watch the full video tutorial on YouTube"></a><br>
  <sub>Start-to-finish walkthrough — install, prep, caption, train, and the workbench tools</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/models-Klein%209B%20%2B%20Krea%202%20%2B%20MiniMax%20H3-blue?style=for-the-badge" alt="Klein 9B + Krea 2 + MiniMax H3">
</p>

> ### 📰 Latest news
> - **Fizgig 5.2 — likeness blocks + the training adapter: the combination wins on MiniMax H3.** Fizgig's Optimised Likeness Learning and **[@ostris](https://github.com/ostris)**'s training adapter each fall short alone; together they reach higher likeness sooner and finish above either — so the adapter is now on by default in every H3 preset (fetched by the updater and the download button). Also: Context LoRA for H3, and video clips follow likeness mode in LoRA runs. [Release notes](docs/RELEASE_NOTES_v5.2.0.md)
> - **Fizgig 5.1.1 — a launch splash.** Fizgig shows a status window within a second of launching and keeps you posted through the load — modules, then each tab as it's built — and the main window now appears fully drawn instead of blank and filling in. Suggested by **[@fm3at](https://github.com/fm3at)**. [Release notes](docs/RELEASE_NOTES_v5.1.1.md)
> - **Fizgig 5.1 — a third base precision for MiniMax H3: 4-bit HQQ.** Between int8 and 4-bit: ~6.3% base error (a third closer to int8 than NF4) for ~15 GB, streamed through the same H2D ring when the card needs swap — at about half NF4's step speed, so it's an explicit pick and Auto never chooses it. Built by **[@rintic-13](https://github.com/rintic-13)**, ring reviewed by **[@mabseyuk](https://github.com/mabseyuk)**. Also: Checkpoint to LoRA gets a button on the fine-tune cards and a Linux launcher. [Release notes](docs/RELEASE_NOTES_v5.1.0.md)
> - **🧪 Fizgig 5.0 — Full fine-tuning graduates: train the MiniMax H3 and Krea 2 base models themselves, on consumer GPUs down to 16 GB.** No adapter, no rank bottleneck — full-rank updates that change how the model *represents* a concept instead of filtering its output. One checkbox applies the whole recipe and the planner sizes the run to your card; **photos, voice and video clips all fine-tune** (2.3 s clips confirmed by measured runs on every tier, longer with video on the likeness blocks), and the built-in **Checkpoint to LoRA** utility turns the result into an ordinary shareable file — rank 64 was perceptually indistinguishable from the full checkpoint. Experimental; NVIDIA only for now. [Details ↓](#full-fine-tuning-krea-2--minimax-h3--experimental) · [Release notes](docs/RELEASE_NOTES_v5.0.0.md)
> - **Fizgig 4.3.1 — 12 GB cards confirmed training MiniMax H3** — a community field report on an RTX 5070 proved H3 LoRA training runs stable at 12 GB, and the two crashes in its way are fixed: checkpoint saves no longer die on low memory, and previews no longer fragment VRAM into a next-step OOM. Also in this maintenance release: **captioning no longer slows down your next training run** (it runs in its own process now — built by **[@scryptio](https://github.com/scryptio)**). [Release notes](docs/RELEASE_NOTES_v4.3.1.md)
> - **Fizgig 4.3 — AMD Radeon support arrives** — Fizgig now trains on AMD with ROCm (RDNA1 through RDNA4, Strix Point / Halo, Instinct). **Windows** is the supported path with its own one-click installer; **Linux** is experimental. Built by **[@scryptio](https://github.com/scryptio)** and tested in the open by the community. Also in the release: **identity distillation now fits 16 GB cards** — the 32B text encoder streams layer by layer, contributed by **[@rintic-13](https://github.com/rintic-13)** — the Repair Studio gains a **side-by-side compare view with likeness and quality metrics**, and Fizgig **speaks Korean** via a community add-on by **[@ssain3d-lgtm](https://github.com/ssain3d-lgtm/Fizgig-Korean-Translated-Ver)**. [Details ↓](#install) · [Release notes](docs/RELEASE_NOTES_v4.3.0.md)
> - **Fizgig 4.2 — the workbench opens to MiniMax H3, and what it found ships as features** — all five post-training tools now work on H3 LoRAs, with previews rendered as 22-frame clips judged by their middle frame. Using those tools on real LoRAs produced the first **H3 block map** — and its biggest finding is now **Optimised Likeness Learning**, a default-on checkbox that trains photos on the identity blocks only: sharper, more prompt-responsive, better sound, fewer epochs. Plus a **✨ MiniMax H3 Style** preset, an **Append Transcription** button that Whispers a clip's speech into its caption, and fully offline transcription. [Details ↓](#minimax-h3--third-model-family) · [Release notes](docs/RELEASE_NOTES_v4.2.0.md)
> - **Fizgig 4.0 — video, sound and voices** — MiniMax H3 now trains on **video clips**, on **their sound**, and on **voice recordings alone**: photos, clips and voice files in one folder train one LoRA in one run. **Gizmo**, a new bundled prep tool, cuts to-spec clips from any footage, auto-chops long videos at scene cuts, and records a voice dataset from nothing but a mic and ten minutes of reading. Training previews render in **6 steps** with the Turbo LoRA and can carry their **generated sound**, opening in the gallery as playable clips. And **16 GB / 24 GB cards now train on the accurate int8 base** — block swap streams one-way, ~6× faster, contributed by **[@rintic-13](https://github.com/rintic-13)**. [Details ↓](#minimax-h3--third-model-family) · [Release notes](docs/RELEASE_NOTES_v4.0.0.md)
> - **One-click cloud training on RunPod** — no GPU, or want a 5090 for the afternoon? The official Fizgig template deploys the full app to a rented GPU in your browser: nothing to install, your files persist until you terminate the pod, and the in-app RunPod panel can even **auto-stop the pod when your run finishes** so an idle GPU never bills overnight. [**⚡ Deploy →**](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep) · [Guide](docker/README.md)

---

## What Fizgig is

Every trainer makes LoRAs. Fizgig is built around what you do with them **afterwards** — and that's the part nobody else has.

- **Fix** a baked LoRA block-by-block, no retraining — overbaked identity, crushed style, drag a slider, save a new `.safetensors`.
- **Explore** new variations like a game — the app proposes mutations, you pick favourites, the LoRA evolves through selection.
- **Find** the best LoRA by eye — **LoRA Royale** renders every epoch of a run (or any folder of LoRAs) on one seed; crossfade to the one that *feels* right.
- **Share** what you made — LoRA Royale exports the epoch morph, or travels a single LoRA through seeds, prompts, or strength, as a looping MP4/GIF made to share.
- **Profile** exactly which blocks carry identity, style, and detail — so you know what to touch before you touch it.
- **Train beyond LoRAs.** Standard LoRA, **LoKR**, or the **full base model itself** — pick per run. A fine-tune comes back as a normal checkpoint, and **Checkpoint to LoRA** distils it into a shareable LoRA at any rank.

Under the workbench sits a fast, light trainer tuned to **fit your GPU**: a full **Klein 9B** LoRA trains on **16 GB**, the 12.9B **Krea 2** on **8 GB**, and the 33B **MiniMax H3** on **16 GB** — block swap, quantisation and previews all size themselves to your VRAM automatically, and if a preview can't fit, training keeps running and saving. It loads kohya / PEFT / OneTrainer / AI-Toolkit / LyCORIS LoRAs, auto-converted, and saves kohya `.safetensors` that drop straight into ComfyUI.

**Free and open source.** A good first run: pick a ✨ built-in preset on the Training tab and go.

---

## The workbench

Each tool works on a trained run's output **or any LoRA you've downloaded** — and they hand off to each other (profile → repair → explore → compare, one closed loop). All three families: Klein, Krea 2 and MiniMax H3 (H3 previews render a short clip, judged by its middle frame — the model's native regime).

### Repair Studio
A live slider per transformer block (32 on Klein, up to 50 + the token refiners on MiniMax H3) with a side-by-side preview that updates as you drag. **Turbo Preview** caches per-block activations so late-block edits redraw up to 97% faster; the baked save is always exact. Blend blocks from a second **donor** LoRA, balance the pair per block, condition previews on a reference photo, and save a `.safetensors` that works in ComfyUI at strength 1.0. On MiniMax H3 the previews are **clips with sound**, played side by side in the app, with a per-block library and first/last-frame conditioning — see [Repair Studio on H3](#repair-studio-on-h3-watch-the-clip-not-a-frame).

### LoRA the Explorer
Evolutionary discovery: the app mutates blocks and shows four variants — pick a favourite and it becomes the new baseline. Freeze what you like, set how far composition drifts, cycle seeds — and send any baseline to Repair Studio (and back) with one click.

### LoRA Royale
Point it at a training run and it renders **every epoch on one fixed seed**, with a crossfade slider — drag until it looks best and stop. An optional **likeness score** (ArcFace, CPU) rates each epoch against a training photo and jumps you to the best. Then make it shareable: epoch-morph clips, seed / prompt / strength **travels**, a **comparison sheet** (with/without-LoRA grid, same seed per row), all exportable as looping MP4/GIF with an optional deflicker pass. Works on any folder of LoRAs, or a single file.

### Profiler
A per-block activation profile as a colour-coded HTML report — which blocks carry style, identity, and detail, and where they overlap. Repair Studio reads its sidecar automatically and shows the findings inline when you load the same LoRA.

### Extract
Distil any Klein, Krea 2 or MiniMax H3 LoRA to a lower rank — Fast presets run weight-only SVD with no models loaded; Klein's activation-weighted presets add block and timestep targeting. PEFT and LyCORIS sources supported.

---

## Krea 2 — second model family

A from-scratch native port: 12.9B single-stream MMDiT, Qwen-Image VAE, Qwen3-VL-4B text encoder. Train on the **RAW model**; previews render on the training model itself with the official Turbo LoRA (auto-downloads) applied for the render only. Pick Krea 2 from the **Base Model selector** on the Training tab and the **✨ Krea 2 Defaults** preset applies itself.

Everything works on Krea 2: all five workbench tools, **Pause/Resume**, **Context LoRA**, **Adaptive LR**, reference images, the live sample override — and **LoKR training** (pick it from Network Type; factor 8 or below for the quality edge, standard LoRA is ~20% faster). Output is ComfyUI-ready.

> **8 GB is enough.** Users train full Krea 2 LoRAs on 8 GB with everything on **Auto** and batch size 1. Auto reads your *free* VRAM and picks INT8, NF4 or fp8 plus the right block swap — the console explains its choice. On longer runs the transformer blocks **torch.compile** automatically for roughly 2× faster steps.

### The trainer curates your dataset while it trains (Krea 2, experimental)

Four Training-tab toggles no other trainer has:

- **Detect problem images** — per-image loss is tracked across epochs (noise-normalised); images that stay hard without improving get flagged in a live **Problem Images window** with thumbnails and trends. In real runs the top flags were all caption/image mismatches.
- **Per-image adaptive LR** — flagged images are throttled so one bad caption can't yank the weights all run; healthy images get a gentle boost. Matched-epoch A/Bs: faster likeness *and* a higher ceiling.
- **Auto-recaption stuck images** — the text encoder *looks at* each stuck image between epochs and rewrites its caption from what's visible. Still stuck after two attempts and the image is excluded for the run (remembered per-dataset; fix the caption and it's re-admitted).
- **Warm up look outliers** — real-but-unusual shots (tight angles, profiles) ease in at reduced LR while the identity forms, then release to full.

Edit any caption yourself mid-run from the Problem Images window — no restart. When nothing is improving any more, a plateau banner names the best-checkpoint window to scrub in LoRA Royale. Pause, resume, restart: a resumed run replays its own loss log and loses nothing.

> **📣 Help map Krea 2's blocks — [open an issue](https://github.com/shootthesound/Fizgig/issues).** Krea 2's per-block roles aren't charted yet, which is why the colour-coded sliders and layer targeting are Klein-only for now. The Profiler's weight-only report is the instrument — share what you find and it drives the presets and Repair Studio colour-coding to come.

---

## MiniMax H3 — third model family

Fizgig trains LoRAs for **MiniMax H3**, MiniMax's open-weight ~33B video model, from ordinary still-image datasets — and from **short video clips, their sound, and voice recordings** ([details ↓](#training-on-video-clips--and-on-their-sound)) — on a single consumer GPU. Output loads straight into ComfyUI's H3 workflows, including the pruned inference builds.

**The full studio, as of 4.2.** H3 trains, previews and pauses/resumes like the other families — and all five workbench tools now work on H3 LoRAs too, with previews rendered as short clips (Repair Studio plays them, [below](#repair-studio-on-h3-watch-the-clip-not-a-frame); the other tools judge the middle frame). It was those tools, on real LoRAs, that produced the block map behind Optimised Likeness Learning below. Two things arrived in 5.2 for H3 LoRA runs. **Training adapter** — one tickbox on the Training tab loads Ostris's [training adapter](https://huggingface.co/ostris/minimax_h3_training_adapter) frozen under your LoRA: H3 is guidance-distilled, and the adapter pulls the base back toward plain flow so the gradient is all concept from step one. It's on for every training step, off for previews, and never in your saved file; the updater and the Preferences download button fetch both variants (fl2va and ref2va — the tickbox picks the one matching your Training Base). In our A/B with likeness mode on it reached 50% likeness seven epochs sooner and peaked higher — and its best window arrives earlier, so watch the gallery. **Context LoRA** — pick any existing H3 LoRA (AI-Toolkit files load as-is) and it rides frozen under the one you're training, in training and in previews, so the new LoRA learns to coexist with it and previews show the pair as you'll deploy it. Both record what they were trained against in the output metadata; both are LoRA runs only, not fine-tuning.

**How it works:** pick **MiniMax H3** from the Base Model selector and the usual flow applies — Start-tab folder, Captions, Samples, Training. Leave **Blocks Swap** and **Base Precision** on Auto: at launch the trainer reads your **free** VRAM (close ComfyUI first) and picks the base precision and block-swap count together:

| Free VRAM | What Auto does |
|---|---|
| ~30 GB | **int8**, no block swap, up to 1 MP |
| ~22 GB | **int8**, ~14 blocks streamed |
| ~15 GB | **int8**, ~36 blocks streamed |
| ≤12 GB | **4-bit**, as before |

int8 is the checkpoint's own storage and the most accurate base (~0.17% error). A third option, **4-bit HQQ**, sits between the two (~4.8% base error, ~15 GB on the pruned checkpoint). On a plan that streams blocks — the 12–24 GB tiers it exists for — it runs at NF4's speed, the dequant hiding behind the transfers; only on a big card with nothing streamed does it show as roughly half NF4's step speed. An explicit pick under Base precision that Auto never makes (contributed by [@rintic-13](https://github.com/rintic-13), [#102](https://github.com/shootthesound/Fizgig/issues/102)). Block swap **streams one way only** — ~6.4× faster than round-trip swap, which is what lets 16 and 24 GB cards keep the accurate base (design contributed by [@rintic-13](https://github.com/rintic-13), [#73](https://github.com/shootthesound/Fizgig/issues/73)). Hit an OOM anyway? Set Blocks Swap to a number to override the planner.

Three built-in presets ship; **Fast** applies the moment you pick the family:

| Preset | Settings |
|---|---|
| **✨ MiniMax H3 Fast** | LoRA dim/alpha **8, 50 epochs, flat 2e-4**, **0.25 MP**, Training Structure **Likeness and Style**, `adamw`. Reaches likeness in a few hundred steps, and the lower rank tends to come out more flexible |
| **✨ MiniMax H3 (Lower LR - slower)** | The same at **rank 16, 60 epochs, flat 1e-4** — more suitable for larger datasets with longer trains |
| **✨ MiniMax H3 Style** | The Fast recipe on the measured style blocks, `0-3, 6-47` — style lives almost everywhere in H3 except the few blocks that only do identity and voice |

<p align="center"><img src="assets/optimised_likeness.png" alt="Optimised Likeness Learning — the default-on Training-tab checkbox" width="713"></p>

**Optimised Likeness Learning** ships ticked (Fast and Lower LR; Style unticks it): photo steps
train only the identity blocks (**20-49**) while video and audio clips train the full model.
Measured against full-model photo training: sharper, much better prompt following, better sound,
fewer epochs — and the occasional deformed preview of full-model photo runs is gone. Untick it
for style or scene training; while it's on, Blocks to Train is disabled with a note.

**0.25 MP is the default, and it holds up** — four times cheaper per step than 1 MP, and the extra resolution has not paid for itself in testing. Raise it if a specific dataset asks for it.

**Previews default to 768×768, 56-frame clips with sound** — a short watchable clip with the model's generated audio, opened in the gallery as a playable video (never autoplay). Without the audio VAE set, clips render silent; stills and other lengths stay in the dropdown. Set the **Turbo LoRA** in Preferences and previews render in **6 steps instead of 20** — previews only, never the saved LoRA. On a plan that streams blocks (a 24 GB card on the int8 base), clip previews clamp to **22 frames up front** — the plan leaves previews ~4 GB and a 56-frame clip measurably doesn't fit there, so the trainer says so once and renders the 22-frame clip instead of failing its way down to it. A preview that still outgrows VRAM steps itself down a ladder rather than dying — a shorter clip first, then resolution to a 512×512 floor — and the size that fit is saved as the new default.

### Repair Studio on H3: watch the clip, not a frame

Pick **MiniMax H3** on the Repair Studio tab and every preview is a **22-frame clip with its sound** (or a still, or 56 frames — your pick). Click either preview and both clips play **side by side in the app**, looping in lockstep: space to pause, arrow keys to step a frame, a scrub bar, slow motion, and **S** to swap sides so the one you're judging carries the sound. The panel and the metrics strip still judge the middle frame.

Three render controls. **Steps** and **Turbo** (the strength of the bundled Turbo LoRA) are boxes you type into; **Render size** renders every clip at a fraction of your chosen size. 4 steps at Turbo 1.0 and ⅔ size is the fast loop (~4 s a move on a 5090); 6 steps at Turbo 0.75 and full size is the render you judge before saving, the same settings as training previews. Turbo 0 switches the Turbo LoRA off entirely, so a Turbo LoRA loaded as the primary can be edited on its own at the steps you choose. **Show early** puts a rough picture up after the second pass while the rest finishes. Everything you see is an exact render — nothing is approximated. A change to a late block skips the untouched blocks on the first pass (bit-for-bit the same result, up to a fifth faster).

**The block library.** After your first render, Fizgig renders every block switched off in the background at your current settings (about 3½ minutes for a 52-block LoRA at 4 steps and ⅔ size, ~8 at 6 steps full size; it pauses whenever you move a slider). From then on the **●** beside each slider is live: hover for a thumbnail, click to see the clip with that block removed, instantly — "what does this block do" for all 52 blocks without waiting. Every render you make is kept too, so a state you've already seen comes back in a couple of seconds, and the **History** strip under the previews holds all of them: click one to view it, right-click to **pin it as the baseline** (compare two tweaks head to head) or save it as an MP4. The library and history live in your cache folder and survive restarts; **Clear cache…** removes them.

**First / Last Frame.** Pin the clip's first and/or last frame to a photo: pick it (or drag it from Explorer straight onto the slot), drag an aspect-locked box over the part you want, done. The clip you judge then starts (or ends) on that picture on every render — sliders, library and Confirm alike — so what you're comparing is the LoRA's effect, not the shot the seed happened to pick.

**Reference mode.** A **Model** picker on the Setup card runs either H3 checkpoint: **First/last frame (fl2va)**, the standard one, or **Reference (ref2va)**, the fine-tune the r2v workflow uses. Under Reference the frame card becomes **Reference Images**: up to two photos the clip takes its subject from, cropped to the clip's shape and sized to its canvas, and you refer to them in the prompt as `<Picture 1>` and `<Picture 2>`. The same picture goes to the text encoder's vision blocks and to the condition rows, which is what makes the identity carry. The prompt-plus-pictures encode is paid once per combination, then cached. Switching model reloads on the next Start.

**The text encoder stays parked in RAM.** The first prompt of a session streams the 32B encoder in (a couple of minutes, and the status line says so). On a machine with the RAM to hold it (about 40 GB free — a 64 GB box), it then stays parked in system RAM for the session, so every prompt or reference after that costs seconds instead of a fresh load, and editing a long prompt word by word is finally painless. It's released again if your RAM runs short, or when the studio unloads. With less RAM the encoder loads per prompt and the base steps aside for it: unloaded and reloaded from disk on a 32 GB machine (a RAM copy would page), back bit-identical about 25 s later.

**Short clips.** The Length menu goes down to 5 frames, and to 9 and 13 (marked off-grid: the model was never trained on a part-filled temporal group, but it renders them fine). Fewer frames means fewer tokens, so a 5-frame Dial move comes back in a fraction of the 22-frame time — use it to read a block's effect on the picture, then go back to 22 to judge the motion. Every Clip-row setting, and the folder each Browse dialog last picked from, is remembered across restarts.

**No-LoRA clip.** Tick it (on the Clip row or in the player) and the player grows a third pane: the same seed and prompt rendered by the base model with no LoRA at all, so you can see what the LoRA adds, not just what your sliders changed. It renders once per setup and is cached, so it costs one extra render, not one per slider move. The player's speed menu goes down to 0.1×.

**Load strength.** Each LoRA has an "at strength" dial next to its Browse button: the strength it was designed to be used at. The block sliders stay relative to it (a block at 1.0 is that block at the load strength), the baseline is the LoRA at that strength, and the saved file keeps its original scale, so you use it at the same strength and it looks exactly like the preview.

**24 GB cards.** Repair Studio on H3 plans its base from the VRAM free when you press Start. A 32 GB card keeps the int8 base resident. A 24 GB card runs the **same int8 base** with its last ~24 blocks streamed in from system RAM each pass, keeping the 0.2% base error rather than the NF4 base's 9.5% — when you're judging what one block does, the base's own error must not be in the picture. Below about 18 GB free the NF4 base takes over. Measured on a simulated 24 GB card: load 26 s, a Dial move at ⅔ size 4.5 s (3.4 s on 32 GB), the No-LoRA clip 3.7 s, a 56-frame 768×640 clip at 6 steps about a minute. The pass-1 resume sits out on the streamed plan; everything else is the same.

### Video and sound: how do I…

**…train on video clips?** Cut them with **Gizmo** (launch it from the Image Prep tab, or the *Launch Gizmo* .bat) — it exports clips already on H3's spec — drop them into the training folder next to your images, and caption them on the **Captions tab** like a photo. **Photos, clips and voice recordings all train together in the same folder** — no settings, no separate runs.

**…make clips from my footage?** Open Gizmo, drop a video on it, scrub to a moment, pick a length, *Add to queue* — repeat, then *Export queue*.

**…chop a long video automatically?** Gizmo's **✂ Auto-chop** scene-detects the whole source and offers every segment as a thumbnail — click to keep or skip, and the keepers join the queue.

**…train a voice from a recording?** Gizmo's **Voice** tab: open any audio file (or a video, for its soundtrack), mark segments on the waveform, caption the sound, export — segments come out training-ready with their captions beside them.

**…record a voice dataset from scratch?** Voice tab → **🎙 Record**: read the prompted sentences while holding the button (or the **R** key). Every take arrives trimmed and captioned; ten minutes of reading is a usable dataset.

**…keep a clip's sound out of training?** Mute it in Gizmo — it adds `_mute` to the filename, reversible by renaming. The video still trains.

**…train photos, clips and a voice into one LoRA?** Same folder, one trigger word, one run, any mix. If one category is much smaller, **Finish one category early** on the Training tab lets it finish at a chosen epoch while the rest trains on.

**…get fast previews while training?** Set the **Turbo LoRA** (~780 MB, its own Preferences row): 6-step previews with the Turbo at 75% on top of your training LoRA. Adjustable on the Samples tab.

**…hear what it's generating while training?** Pick a **"with sound"** Sample length on the Samples tab. Each preview carries its generated soundtrack, playable in the gallery.

**…get a clip's spoken words into its caption?** Open it in the caption editor (Captions tab → click the clip): any non-muted video shows an **🎤 Append Transcription** button that Whispers the speech into the caption as `saying "…"` — Gizmo's grammar, without leaving the tab.

**…set it up?** One extra model file for sound: the **audio VAE** (~605 MB), on its own Preferences row. Blank = clips train silent; required only once the folder has voice recordings. Fizgig points out both new files once at startup if your H3 paths are set.

### Training on video clips — and on their sound

Stills teach H3 a look; clips teach it **motion**, and clips with sound teach it **a voice**. Clips cost far more per step than stills — start with a handful. Drop `.mp4` clips into the training folder alongside your images and caption them like photos. A clip has to be on spec, and Fizgig refuses one that isn't rather than quietly fixing it:

| | Requirement |
|---|---|
| Container | `.mp4` |
| Frame rate | exactly 24 fps |
| Frame count | 5, 22, 39, 56, 73, 90, 107 or 124 frames |
| Dimensions | multiples of 32 |
| Audio | 32 kHz stereo, or no track at all |

<p align="center"><img src="assets/gizmo_video.png" alt="Gizmo — Find the moment: first/last frame previews with frame-accurate stepping" width="720"></p>

**Gizmo makes clips that hit it** — mark every section you want (frame-accurate stepping, first/last-frame previews, a ▶ Play of the exact clip), then export the lot in one go. **Crop to the subject**: a clip's cost is its pixels, so drag a rectangle and every token goes on what you want learned — with shape locks (1:1, 16:9, 9:16…) when you want consistent framing. High-frame-rate footage can keep extra frames as slow motion, offered as a choice. Clips are cut at native resolution and resized to your Target Megapixels at training time, so cutting large keeps the choice open.

**What it costs:** 22 frames is the shortest that shows real movement at ~7× a still per step; 124 frames is ~37×. Gizmo says which lengths your card can train, at which megapixels, before you cut anything:

| Clip | 16 GB | 24 GB | 32 GB |
|---|---|---|---|
| up to 56 frames | up to 0.25 MP | up to 0.5 MP | up to 0.5 MP |
| 73–90 frames | — | up to 0.25 MP | up to 0.5 MP |
| 107–124 frames | — | up to 0.25 MP | up to 0.25 MP |

### Training on a voice alone

Drop **`.wav` / `.mp3` / `.flac` / `.m4a`** files into the training folder — alone or mixed with stills and clips. Rate and channels are converted for you; **duration is the strict part**:

| | Requirement |
|---|---|
| Formats | `.wav` `.mp3` `.flac` `.m4a` — any rate or channel count |
| Duration | exactly 0.917, 1.625, 2.333, 3.042, 3.750, 4.458 or 5.167 s (±25 ms) |
| Content | actual sound — digital silence is refused |
| Caption | a `.txt` beside the file, or it silently won't train |
| Audio VAE | required — the ~605 MB Preferences row |

<p align="center"><img src="assets/gizmo_voice.png" alt="Gizmo — Voice tab: waveform with a marked segment, trigger word, transcribed caption and grid lengths" width="720"></p>

**Gizmo's Voice tab cuts them for you** — open a recording (or a video, for its soundtrack), mark segments on the waveform, pick a length, caption, export sample-exact. **Caption the voice, not a picture** — *"a man speaking calmly, low pitch, unhurried"* — with your trigger word leading; the **Transcribe** button (Whisper) appends the spoken words. **Or record the dataset from scratch**: **🎙 Record** prompts sentences across every length and five tonal flavours, rolls a delivery style per take, and every hold-and-release lands trimmed, captioned and ready to queue. **Set Training Structure to Likeness and Style for voices** — tested head-to-head, it converges much faster; Fizgig reminds you when it sees voice files.

### Model files (MiniMax H3)

Each has a **Download link on its row in Preferences**:

| Model | Size | Notes |
|---|---|---|
| DiT — pruned int8 | ~21 GB | The training base — `minimax_h3_fl2va_pruned_int8_convrot.safetensors`, the same file ComfyUI runs. (The ~66 GB bf16 file also works for LoRA training, NF4 at load — but full fine-tuning needs this int8 file) |
| Qwen3-VL-32B text encoder | ~15.7 GB | The **nvfp4** file — same one ComfyUI uses. Loaded once for caching, then freed |
| Video VAE | ~4.9 GB | Caching and preview decode |
| Audio VAE *(optional)* | ~605 MB | Sound training and previews with sound |
| Turbo LoRA *(optional)* | ~780 MB | 6-step previews — `minimax_h3_turbo_v4_step600.safetensors`; you may have it in ComfyUI's loras folder |
| DiT — reference *(optional)* | ~21 GB | Only for reference distillation (`ref2va`) |

**Yes, you train on the pruned file.** "Pruned" here swaps the AdaLN modulation MLP for a curve table — that branch only sees the timestep, so nothing a LoRA learns lives there. You train against the exact weights you deploy on.

### Training-tab controls worth knowing

Every control has a hint in the app; the highlights:

- **Training Structure** (default **Likeness and Style**) — how much of the run trains on nearly-clean images, where likeness *and* style live. **Model default, movement** is the reference trainer's schedule; **Custom** exposes the raw percentage. **Medium to High Noise LR** beside it is best left at 100.
- **Optimised Likeness Learning** (default On) — photo steps train the identity blocks (20-49) only; clips train the full model. The measured best recipe for character and voice work — untick for style or scene training.
- **Blocks to Train** — hand-pick a subset of H3's 50 blocks (disabled while Optimised Likeness Learning owns the choice). The measured recipes: **`20-49` for likeness**, **`0-3, 6-47` for style** (the Style preset sets it), voice core `38-48`. Type ranges (`3-12, 22, 31-33`) to experiment beyond them.
- **Reference distillation** (experimental) — teaches the LoRA to render your subject from the trigger word the way H3 renders them from a *photo*: each image is marked against the model shown *other* photos of the same person, so identity is learned without the scenery. Needs the ref2va model; the LoRA deploys on the ordinary model. **Identity-first** (Auto) trains a teacher-only first phase, then pure photos. A separate, deliberate tick — Multi Concept no longer switches it on for you.
- **Multi Concept** — two subjects, two folders, two trigger words, one LoRA. Each subject's images are only ever compared against their own. Ticking it sets caption dropout to 0.10 (strong) and nothing else; separation rests on the trigger words, which in our tests is what actually does the work.
- **Adapter-relative LR** (default Off) — the LR box becomes a ceiling the run climbs toward, keeping each step proportional to the adapter's size. Worth trying when a run overshoots early.
- **Caption dropout** (default 0.05) and **Weight averaging (EMA)** (default Off) — leave dropout on; switch EMA on when pushing LR hard.
- **Using the Turbo LoRA in ComfyUI? Skip its custom sampler** — current ComfyUI samples H3 audio cleanly with stock Euler; community consensus is 8 steps, with `minimax_h3_turbo_v4_step600_ema` the strongest checkpoint.

Settings are read at launch; Pause → Resume relaunches with your current settings, so a pause is the moment to change them mid-run.

---

## Full fine-tuning (Krea 2 & MiniMax H3) — experimental

Everything above trains a **LoRA**. This trains the **base model itself** — no adapter, no rank
bottleneck — on a single consumer GPU. Tick **⚗ Fine-tune the BASE MODEL instead of training a
LoRA** on the Training tab.

> **A note on where this is at.** I first got fine-tuning working on Krea 2 shortly after
> its release, and I've been deliberately cautious about shipping it — first proving it to
> myself, then refining it through the MiniMax H3 work. This is the point where it needs
> the community to develop further. I don't expect every scenario to work perfectly yet —
> but it works, the numbers below are measured, and there's a solid foundation here to
> build on. Field reports genuinely shape what gets built next. I'm also aware this
> technique is model-agnostic at heart — it opens the door to fine-tuning other models,
> and I'm open to going there. But for that to happen it needs practical community
> support around those models — code, PRs, testing, that kind of thing — so I have the
> time necessary to make it happen. — Peter

New to fine-tuning? The extended **["How do I…?" guide](docs/FINETUNE_HOWDOI.md)** answers
everything this section can't fit — including **five-minute recipes for both families**:
tick Fine-tune, let the settings switch themselves, and change almost nothing.

> **One idea makes everything else here make sense: an "epoch" trains one slice of the
> model.** The trainable window rotates each epoch, so it takes a full cycle — typically
> **4 epochs** — for every part of the model to train once. Rule of thumb: **4 fine-tune
> epochs ≈ 1 true epoch of the whole model.** That's why the epoch defaults look high,
> and why saves land on cycle boundaries — each saved checkpoint is a whole, evenly
> trained model.

> **Note on VRAM:** the "trains on 8 GB" figures elsewhere in this README are for **LoRA**
> training. Full fine-tuning is a different animal — but it now **tiers itself to your card**,
> and fine-tuning defaults to a **4-bit NF4** frozen base that halves the model held on the
> card: on **32 GB and 24 GB** the classic full-depth windows stay resident at full speed, and
> on **16 GB** the frozen blocks stream from system RAM — slower steps, but the same
> component-mode learning. The planner measures your
> free VRAM at launch and prints the plan it chose.

**What can my card fine-tune?** The short answer, at the default training resolution:

| Your card | Krea 2 — photos | MiniMax H3 — photos | H3 — voice | H3 — video, confirmed | H3 — video on likeness blocks, expected |
|---|---|---|---|---|---|
| **16 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** | up to **3.8 s** |
| **24 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** | up to **5.2 s** |
| **32 GB** | ✅ | ✅ | ✅ | ✅ up to **3.8 s** | up to **5.2 s** |

A few things worth knowing about that table: clip lengths follow Gizmo's grid, so **2.3 s
means the 56-frame slot** — cut your clips there and everything fits, **confirmed by
measured runs on every tier**. On **32 GB, 3.8 s is also confirmed**, even with video
training the whole model. Beyond that, the **Restrict video to likeness blocks** tickbox
(on by default with Optimised Likeness Learning — in our tests it trains video just as
well, and it makes clips far lighter) extends the *expected* range: **up to 5.2 s on
24 GB and 32 GB, and 3.8 s on 16 GB** — conservative arithmetic from the measured
constants, not yet individually measured, so treat those as expected rather than
promised. Whole-model 5.2 s clips need more than 32 GB (measured). With the restriction
unticked, one clip anywhere in your folder trains the **whole** model, so a mixed
photos + clips dataset uses the clip column. And 12 GB cards train **LoRAs**, not
fine-tunes — 16 GB is the fine-tune floor.

> **"A full fine-tune of a 12.9B–33B model on 16 GB" sounds like a trick, so here's the
> arithmetic.** Only one slice of the model is ever trainable at a time — the trainable
> window rotates each epoch, so gradients and optimizer state exist for that slice alone.
> The frozen rest is held **4-bit (NF4)** at half size and, on 16 GB, streamed from system
> RAM. The bf16 master copy lives in CPU RAM, never on the card. Those three together are
> the whole magic, and the numbers are measured, not projected: **8.8–12.3 GB peaks on a
> 16 GB card for H3**, **8.4–11.0 GB for Krea 2** — and the console prints your own run's
> peak every epoch, so you can watch the claim hold live. Mechanism, tiers and every
> "how do I" in the extended guide: **[docs/FINETUNE_HOWDOI.md](docs/FINETUNE_HOWDOI.md)**.

**Which model files.** Fine-tuning uses the same training bases you already have — nothing new to
download:

- **Krea 2** fine-tunes the **RAW bf16 model** (`krea2_raw_bf16.safetensors`, ~26 GB), the same
  file LoRA training uses. The fp8 Turbo is the preview model and can't be fine-tuned.
- **MiniMax H3** fine-tunes the **pruned int8 checkpoint**
  (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, ~21 GB) — again the same file the LoRA
  path trains against and ComfyUI runs. The ~66 GB bf16 file, which LoRA training accepts, does
  **not** work for fine-tuning; the trainer refuses it with a clear message.

A finished fine-tune checkpoint is itself a valid base for either family — point the model path
at it to train further (the console prints the exact continuation settings at every save). And —
easy to miss — you can set it as the family's base in **Preferences** and **train LoRAs on top of
your own fine-tuned model**: teach the base your world or cast once, then quick LoRAs for
individual subjects ride on it. Deploy those LoRAs with the same fine-tuned base in ComfyUI. And
**Pause / Resume works on a fine-tune**: Pause saves a full checkpoint even between the regular
save epochs, and Resume continues it — rotation window, checkpoint numbering and the remaining
epoch count all carry over.

**Why bother.** A LoRA constrains every update to a low-rank subspace, so concepts compete for the
same handful of directions. That's why LoRAs tend to drag pose, framing and lighting toward the
training set along with the likeness — they behave a bit like a filter over the model's output. A
full-rank update can change how the model *represents* a concept, so it composes with what the
model already knows. **In our own tests, multi-character and concept teaching seemed to land at
a much deeper level than LoRA training, with much better results** — and the built-in
**Checkpoint to LoRA** converter turns the result into a shareable file, and works very well.
Beyond that, we're deliberately letting the community find the ceiling.

**How it fits.** A naive full fine-tune of Krea 2 (12.9B) needs roughly **78 GB** — bf16 weights,
gradients and optimizer state at once. Rotating windows make only part of the model trainable at a
time, advancing each epoch, so gradients and optimizer state only ever exist for the active slice.
Over a full cycle every weight trains. Around that sit three decisions that do the heavy lifting: a
**CPU-resident bf16 master copy** is the source of truth, so training never round-trips through fp8
and quantisation can't erase the small updates being learned; **optimizer-in-backward** consumes
and frees each gradient the moment it lands (worth 5.2 GB); and **Adafactor**'s factored state is
~10× smaller than AdamW's.

**It sizes itself to your card.** Leave **Window** on **Auto (by VRAM)** and Fizgig measures the
memory actually free at launch, picks the largest window that fits, and prints what it chose and
why. Measured Krea 2 peaks (RTX 5090):

| Window mode | Peak VRAM | Speed | Fits |
|---|---|---|---|
| **component + 4-bit NF4 (the default)** — full-depth windows, resident | ~16 GB (24 GB budget) / ~21–23 GB (32 GB, more headroom held) | ~1.0 s/it | **24 GB and up** |
| component + **4-bit NF4** + streaming | 8.4–11.0 GB | ~2.8 s/it | **16 GB** |
| component on the **fp8 base** (explicit Base-precision pick) — depth-split + streamed | 15.6–17.6 GB | ~3.0 s/it | 24 GB |

**4-bit NF4 is the fine-tune default**, and you don't have to do anything to get it. It halves
the frozen base, which on a 24 GB card is enough to keep the classic full-depth component
windows resident instead of depth-splitting and streaming them: **4 windows instead of 8** — a
full pass over every weight in 4 epochs rather than 8 — at roughly **3× the step speed**
(measured ~1.0 s/it against ~3.0 s/it for the fp8 base, same dataset, same 24 GB budget). On
16 GB it is the only base that fits at all.

The trade is that the *frozen* part of the model is held more coarsely while the trainable
window learns against it. Your saved checkpoint is unaffected either way — it's written in
bf16 from a master copy that never passes through a quantiser. If you want the more accurate
frozen context and have the VRAM, pick **fp8** under Base precision and it will be used.

**Component is the best mode — and Auto now stays in it at every depth.** Every window spans the
model's full depth — attention across all 28 blocks, then each MLP matrix in turn — so a concept
is learned by every layer at once rather than one depth slice at a time. The text-fusion stack
stays trainable throughout: rotation would never reach it, and it's where prompt-to-concept
binding happens. Where the budget used to force a mode change, the planner now **depth-splits**
the windows instead (a fat window trains in slices — more windows per cycle, still full speed),
and below that the frozen out-of-window blocks **stream from system RAM** — slower steps, but
still component-mode learning. The console prints the chosen plan and why.

**Block mode remains an explicit Window-dropdown choice** — contiguous depth slices with frozen
blocks streamed, slower than component at every budget. It's **not yet quality-tested**; every good result so
far came from component runs.

### MiniMax H3 fine-tuning

The same checkbox under the MiniMax H3 family fine-tunes the 33B model, with the recipe adapted
to it: **component windows only** — each window trains one attention or MLP matrix across all 50
blocks (4 windows per cycle), with the token refiner trainable throughout, so every window spans
the model's full depth from the very first epoch.

- **VRAM — it sizes itself to your card.** Measured with **Optimised Likeness Learning** on,
  which is the recommendation on every tier (matched runs came out clearly better on both look
  and prompt adherence than full-model fine-tuning — and it shrinks the windows, so the tiers
  below assume it). On 32 GB the classic 4-window cycle runs at full speed. On **24 GB** the
  planner **depth-splits the fat windows** — `mlp.fc1` trains in two slices, a 5-window cycle,
  still full speed, no offloading; measured peaks 19.1–21.5 GB. On **16 GB** the frozen
  out-of-window blocks also **stream from system RAM** (~7 GB staged, a 9-window cycle):
  measured peaks 8.8–12.3 GB at **~1.5× the step time** — a full fine-tune of a 33B video model
  on a 16 GB card. The console prints the chosen plan and why.
- **Full-model fine-tuning (likeness off, or any dataset with video clips) plans itself too.**
  On stills it fits right down the range — measured peaks 17.3–18.7 GB on a 24 GB budget and
  8.8–11.6 GB streamed on 16 GB. Clip datasets additionally reserve activation memory before
  the windows are sized (clips cost VRAM per frame of length), which is what the card table
  above reflects — and when clips are too long for your card, the trainer says so up front,
  with the fix (cut to the 2.3 s Gizmo slot, or lower Target Megapixels), instead of failing
  mid-run.
- **System RAM:** the bf16 master copy is ~23 GB (likeness) to ~38 GB (full model), and spills
  to disk automatically when RAM is tight — full-model fine-tuning runs on a 64 GB box.
- **Disk:** each save is a full **~21 GB** int8 checkpoint — set the Training tab's **Output
  Directory** to a drive with room *before* the run, or you'll be moving 20 GB files by hand after.
- **Learning rate: 3e-5** — the tested fast-and-reliable rate for H3 fine-tuning. Anything as
  high as 1e-4 will destroy a fine-tune.
- **Use unique trigger tokens** — strongly recommended: an invented token gives the fine-tune
  somewhere clean to bind, where a common word drags its existing meaning along with it.
- **Run length: there's no standard number.** It depends on learning rate, dataset size and
  what you're teaching. The 100-epoch default is a generous scrub-range for a typical small
  dataset — **a large dataset probably needs far fewer epochs** (each epoch is more steps).
  Save once per cycle and compare checkpoints to find where yours peaks; Max epochs and
  Save-every both snap to cycle boundaries so every save ends evenly trained. Total *steps*
  still run well past LoRA habits (only one component trains at a time — a full pass of
  learning costs a full cycle, not an epoch), so budget wall-clock and disk accordingly.
- **Voice and mixed datasets train too** — voice stays confined to its measured blocks (34–49),
  photos to theirs, and the per-category **stop epoch** counts across Pause/Resume: pause a
  mixed run, set the stop to the current epoch, Resume, and it finishes voice-only.

**Saves, previews and numbering run on the rotation cycle, not the Samples tab.** The save
cadence snaps to cycle boundaries — the Save-every box follows the FT controls live in the GUI,
and the trainer snaps it again at launch — so every checkpoint compares like-for-like, with each
window trained equally. Previews ride the saves: one render per saved checkpoint plus the final
one, overriding the Samples tab's "every N epochs" (prompts, resolution, seed and the live
sample override still come from the Samples tab and status bar as usual — every sample in the
gallery maps to a file you can deploy). Checkpoints are numbered by epoch (`-000004`,
`-000008`, …) and the numbering continues across Pause/Resume, so a resumed run never overwrites
an earlier save. Krea 2 fine-tunes behave exactly the same way — saves snap to the cycle,
previews ride them (rendered on the training DiT with the Turbo LoRA), numbering carries over.

The output is a normal H3 checkpoint: load it in ComfyUI directly, or run **Checkpoint to LoRA**
(`run_diff_to_lora.bat` in your Fizgig folder) on it (the extractor decodes the int8 format natively) for a shareable LoRA.

### Learning rates — lower than you're used to

If you're coming from LoRA training, recalibrate before anything else: **fine-tuning wants
much lower learning rates than LoRAs**. A LoRA nudges a small adapter riding on a frozen
model; a fine-tune moves the model's own weights, so the rates you're used to typing land
very differently here — what's a normal LoRA rate can wreck a fine-tune outright.

- **MiniMax H3: use 3e-5.** It's the tested fast-and-reliable rate; **1e-4 will destroy an
  H3 fine-tune** — that one is measured, not folklore.
- **Krea 2: you're welcome to start at 1e-4** — it trains — but realistically the best
  results are found lower. Treat 1e-4 as the top of the experiment range, not the recipe:
  when a run looks almost right but slightly overcooked, the next move is a lower rate,
  not fewer epochs.
- **The regularisation LR × multiplier is part of the same tuning space** (next section).
  It sets how hard the anchor pulls relative to your subject, and it's genuinely worth
  experimenting with per dataset — 0.1–0.3 keeps it a tether, higher trains the reg set
  more like real data.

### Optional: regularisation images

Full fine-tuning moves every weight, so a long run on a handful of subjects drifts the model's
whole notion of people — there's no low-rank bound to limit it the way there is with a LoRA. Point
**Regularisation images** at a folder of ordinary photos of the broader class and they train at a
reduced learning rate (**LR ×**, default 0.2) as an anchor rather than a lesson. That
multiplier is a real dial, not a set-and-forget: **0.1–0.3** tethers the model's prior while
your subject trains; push it toward **1.0** and the reg set trains like a second subject set —
class-balanced training rather than a light anchor, which is a different (valid) thing. If a
fine-tune drifts the broader class, raise it a step; if the subject learns too slowly, lower
it. Worth a little experimentation per dataset.

Use **real photos, not model output** — anchoring a fine-tune to its own samples distils its
artifacts back in, and there's nothing bounding that drift. Caption them normally: anything you
leave unsaid gets attributed to the class word itself. Leave the folder empty to train without
one.

### Then turn it back into a LoRA

A fine-tune produces a **~26 GB checkpoint**, which is not what anyone wants to share. The
**Checkpoint to LoRA** utility — **`run_diff_to_lora.bat` in your Fizgig folder**, which opens
its own small window separate from the main app (Linux/pods: `./run_diff_to_lora.sh`) — takes the base model
you started from and the checkpoint you produced, and extracts the difference as an ordinary
kohya `.safetensors` — at several ranks at once, since one SVD per layer serves them all.

This turned out to work far better than expected: **rank 64 was perceptually
indistinguishable from the full 26 GB checkpoint** at ~0.5 GB, and quality degrades smoothly
at lower ranks rather than falling off a cliff.

The result worth knowing: in our testing, a LoRA **extracted** from a fine-tune came out
better than a LoRA **trained directly** at the same or higher rank on the same dataset. A
low-rank file can *hold* a solution that low-rank training struggles to *find* — so
fine-tune-then-extract isn't a workaround; the full-rank phase is the mechanism, and the
extraction is nearly free.

### What it costs you

Being straight about the trade-offs, because they're real:

- **VRAM tiers itself**: on the default 4-bit NF4 base, **24 GB** runs the classic full-depth
  component cycle at full speed, and **16 GB** adds frozen-block streaming from RAM at ~1.5×
  the step time — both families. (Picking the fp8 base instead costs a 24 GB card depth-split,
  streamed windows at ~3× the step time, and doesn't fit 16 GB at all.) The console prints
  each run's plan; too little VRAM refuses cleanly instead of OOMing.
- **System RAM** for the bf16 master copy, on top of VRAM: ~24 GB on Krea 2, ~23–38 GB on H3.
  H3's spills to disk automatically; **Krea 2's does not, so Krea 2 fine-tuning realistically
  wants 48 GB+ of system RAM** — the trainer warns at launch when RAM looks tight.
- **NVIDIA only, for now.** Fine-tuning is untested on AMD/ROCm — every measured tier is
  NVIDIA, and the NF4 default leans on bitsandbytes 4-bit, the least-travelled part of the
  ROCm stack. Reports welcome either way.
- **Disk — set your save location BEFORE the run.** Every save is a full checkpoint — ~26 GB
  on Krea 2, ~21 GB on H3 — and saving once per 4-epoch cycle is ~260 GB over a 40-epoch run.
  The **Output Directory** on the Training tab defaults to the same folder your LoRAs go to,
  which is often not the drive you want holding a stack of 20+ GB files: change it to a roomy
  drive before you press Start, because afterwards the only fix is moving huge files by hand.
  (A Pause also writes a full checkpoint, on top of the regular cadence.)
- **A low learning rate** — 1e-5 on Krea 2; **3e-5** is the tested rate on H3. LoRA-style rates
  (anything as high as 1e-4) will destroy a fine-tune.
- **Run at least one full cycle** (4 epochs in component mode) or some weights never train at all.
  The console warns you.
- **Adaptive LR is off** — rotation boundaries would read as instability to it. Previews render
  **once per saved checkpoint** on both families (every sample is the rehearsal of a file you can
  deploy); judge those, evaluate checkpoints in ComfyUI, or extract a LoRA and scrub the epochs
  in LoRA Royale.

---

## Training (Klein 9B)

The foundation: fast, light, and tuned for one model.

- **Proven presets** for single subject through multi-character — or roll your own.
- **Context LoRA** — load an existing LoRA as a frozen *active* layer so the new one learns to coexist: a face on top of a style, an outfit on top of a character. No other trainer does this.
- **Adaptive LR** — a bi-directional plateau tracker: set the Min/Max window and it probes up on steady descent, pulls down (with rollback) on plateau or instability.
- **fp8 Base training** — the fp8 Base stays resident at ~9.6 GB, so a full 9B LoRA trains in ~14 GB and fits a 16 GB card. Automatic.
- **Distilled training samples** — 4-step previews that match ComfyUI output closely, multiple prompts (one per line on the Samples tab), and optional **reference-conditioned** samples (Klein is an edit model — previews can edit a real photo).
- **Pause / Resume** — graceful epoch-boundary pause that frees your GPU mid-run and resumes with full state.
- **Model Area targeting** — train only Identity, Style, or Detail blocks, or the full model.
- **Per-dataset caches, cross-checked** — deleted images leave the run; switched datasets can never leak in.

### The sample gallery is an instrument (both families)

- **Live likeness scoring** — pick 3 dataset photos and every sample gets a colour-coded likeness badge (ArcFace, CPU — zero training-speed cost), with a per-epoch trend chart and best-epoch highlight, live while the run goes.
- **Training Run Visualiser** — scrub the run epoch by epoch in the browser, Royale-style, with share-ready WebM/PNG export.
- A **live sample override** in the status bar changes the preview prompt, seed, size or reference mid-run, no restart. The status bar itself carries VRAM/RAM gauges with per-run peak markers.

### Dataset prep

- **AI captioning with the captioner that trains your model** — Krea 2's Qwen3-VL writes viewpoint-aware training captions in five editable preset styles (including **Style**, which describes everything *except* the look so your trigger word binds to it). Every preset's instruction is editable in plain English and persists. Florence-2 remains the zero-setup option. **Bilingual captions** (English + Chinese via Helsinki-NLP) act as text-level augmentation — measurably better skin detail on Klein at identical loss.
- **Image Prep** — batch resize, PNG conversion, InsightFace face-crops with gender targeting. Pairing a tight crop with a full shot adds a lot to a character dataset.
- **Look Consistency Filter** — pick the 3 images that best nail the look and every image is scored against them (ArcFace). Worst matches surface first; mark drifters or let Auto-Suggest flag the outliers, then move them out in one click — nothing is deleted, and the scores feed the Krea 2 trainer's look-outlier warm-up.

### Compatibility

Loads kohya, PEFT, OneTrainer (OMI + legacy), AI-Toolkit, and LyCORIS (LoKR / LoHa) — auto-converted, and LoKR/LoHa run natively everywhere: Repair Studio, Profiler, Extract, Context LoRA. Repair Studio and Explorer save LoKR as LoKR, losslessly. Output is `.safetensors` that drops straight into ComfyUI.

---

## No GPU? Rent one

Fizgig ships as a ready-made cloud image — the **whole app in a browser tab**, not a cut-down web version. Drag datasets in and LoRAs out with a built-in file manager, download models in one click, and optionally have the pod **shut itself down when training finishes**. Your models and datasets persist between sessions.

**[⚡ Deploy on RunPod →](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)**  ·  [Read the guide first](docker/README.md)

---

## Requirements

- **GPU** — NVIDIA RTX 30 / 40 / 50-series, or **AMD Radeon** with ROCm (RDNA1 through RDNA4, Strix Point / Halo, Instinct MI300+). **Klein 9B** needs 16 GB, **Krea 2** trains on 8 GB, **MiniMax H3** on 16 GB — see [VRAM guidance](#vram-guidance). The fp8 Base's VRAM savings apply on NVIDIA Ada+; on AMD, NF4 and INT8 are the primary quant paths.
- **NVIDIA driver** — 555+ on Windows, 550+ on Linux (CUDA 12.8 wheels).
- **AMD ROCm** — **Windows:** `install_fizgig_rocm.bat` (supported path). **Linux:** `./install_fizgig_rocm.sh` — **highly experimental** (newer gfx like RDNA4, desktop compositor + training on the same GPU, and driver resets are common; use Windows ROCm or NVIDIA Linux for production training). Optional system `amdrocm-amdsmi` for accurate status-bar VRAM via `amd-smi`.
- **OS** — Windows 10 / 11 or Linux. macOS handles captioning and image prep, but training needs CUDA or ROCm.
- **Python** — 3.10 – 3.13.
- **System RAM** — 32 GB recommended; 16 GB is workable for **Klein 9B** and **Krea 2**. **MiniMax H3 is the outlier**: its text encoder is a 15.7 GB file that streams from system RAM while captions are cached, and INT8 block streaming stages a similar amount again during training — so 32 GB is comfortable, and 24 GB works only with other apps closed. Worth knowing: when system RAM runs short, the failure arrives dressed as **"CUDA error: out of memory"** even though the GPU is nearly empty. Close what else is running and retry the caching step before suspecting VRAM.
- **Disk** — ~10 GB for the venv, plus ~40 GB for model files.
- **Full fine-tuning** (experimental, Krea 2 & MiniMax H3) asks for more than the above, and
  tiers itself to your card: on the default 4-bit NF4 base, **24 GB** runs the full-depth
  component cycle at full speed and **16 GB** streams the frozen blocks from RAM at ~1.5–3× the
  step time, both families. Add the bf16 master in RAM (spilled to disk
  automatically on H3), and disk for saves — **~26 GB per Krea 2 checkpoint, ~21 GB per
  H3 one**. Each
  family fine-tunes its normal training base — Krea 2 the RAW bf16 model, H3 the pruned int8
  checkpoint (H3's ~66 GB bf16 file works for LoRA training only, not fine-tuning).
- **Visual Studio Build Tools** (Windows only) — for InsightFace and the torch.compile speedup: **[aka.ms/vs/17/release/vs_BuildTools.exe](https://aka.ms/vs/17/release/vs_BuildTools.exe)**, tick **"Desktop development with C++"**. Without it everything still works minus the compile speedup.

---

## Install

Clone the repo:

```bash
git clone https://github.com/shootthesound/Fizgig.git
cd Fizgig
```

**Clone it rather than downloading the ZIP** — `update_fizgig.bat` updates by pulling with git, and a ZIP can't.

<details>
<summary><b>Already installed from a ZIP? Fix it without starting over</b></summary>

Open a terminal in your Fizgig folder and run:

```bash
git init
git remote add origin https://github.com/shootthesound/Fizgig.git
git fetch --depth 1 origin master
git reset --hard FETCH_HEAD
git branch -M master
git branch --set-upstream-to=origin/master master
```

Your model paths, output LoRAs, caches, presets and the venv are all left alone. `update_fizgig.bat` works normally from then on.

</details>

**Windows (NVIDIA, one-click)** — double-click `install_fizgig.bat`. It creates a venv, installs CUDA 12.8 PyTorch and all dependencies, pre-downloads the InsightFace models, and verifies CUDA is visible to PyTorch. Launch with `run_fizgig.bat`; update later with `update_fizgig.bat`.

**Windows (AMD ROCm)** — needs a full **Python 3.12** install first (the ROCm bitsandbytes wheel is cp312-only; Fizgig's GUI needs Tkinter). Do not use the embeddable zip. Install from [Windows downloads](https://www.python.org/downloads/windows/):

- **Recommended (2026)** — [Python Install Manager](https://www.python.org/downloads/latest/pymanager) from the [Microsoft Store](https://apps.microsoft.com/detail/9NQ7512CXL7T), then `py install 3.12`.
- **Alternative** — [python-3.12.10-amd64.exe](https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe); tick **Add python.exe to PATH** and **tcl/tk and IDLE**.

Then double-click `install_fizgig_rocm.bat` (NVIDIA users never run this). It picks 3.12 via `py -3.12` / `python3.12` (not whatever `python` defaults to — e.g. 3.14). GPU detection follows, then pinned multi-arch wheels from **AMD ROCm nightlies** (`https://rocm.nightlies.amd.com/whl-multi-arch/` — not built by Fizgig):

- `torch==2.12.0+rocm7.15.0a20260728`
- `torchvision==0.27.0+rocm7.15.0a20260728`
- `rocm-sdk-devel==7.15.0a20260728`

Override with `TORCH_PIN` / `TORCHVISION_PIN` / `ROCM_SDK_DEVEL_PIN` if needed. **bitsandbytes** is a pinned community Windows ROCm wheel from [0xDELUXA/bitsandbytes_win_rocm](https://github.com/0xDELUXA/bitsandbytes_win_rocm) — built by neither AMD nor Fizgig. Shared deps come from `requirements.txt` with CUDA `torch`/`bitsandbytes` and NVIDIA-only `nvidia-ml-py` filtered out (`filter_requirements_rocm.py`). Launch with `run_fizgig_rocm.bat`; update later with `update_fizgig_rocm.bat` (**not** `update_fizgig.bat` — that script installs CUDA torch and would wipe the ROCm stack).

**`--experimental` (unsupported):** `install_fizgig_rocm.bat --experimental` installs unpinned `torch[device-ARCH]` / `torchvision[device-ARCH]` / `rocm-sdk-devel` from the same multi-arch index and leaves `BNB_ROCM_VERSION` unset so bitsandbytes auto-selects its highest matching DLL (no fallback warning while the resolved torch stays inside the wheel's HIP 7.13-7.16 range). This is **not** the same as Linux `ROCM_CHANNEL=nightly` (which stays on the constrained 7.14 / bitsandbytes 714 lane). Windows already installs from AMD nightlies with pinned versions by default; `--experimental` only drops those pins. Local experimentation only. **Do not open GitHub issues for crashes, install failures, or training problems when `--experimental` was used** — those reports will not be supported. Use the pinned install (no flag) for anything you expect help with.

**Linux (AMD ROCm — highly experimental)** — expect crashes, GPU resets, and incomplete model support on many setups. Best-effort only; Windows ROCm or NVIDIA Linux are the supported training paths. Prerequisites: amdgpu driver loaded (`/dev/kfd`), user in `render`/`video` groups. See [Install ROCm](https://rocm.docs.amd.com/en/latest/install/rocm.html) and [PyTorch for ROCm](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). Then:

```bash
chmod +x install_fizgig_rocm.sh
./install_fizgig_rocm.sh
./run_fizgig_rocm.sh
```

The script detects your gfx target (`detect_gpu_linux.py`). **Nightly is the Linux default** — [TheRock multi-arch RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) index plus a `[device-gfx*]` extra for your GPU (e.g. `gfx1201` → `device-gfx1201`). Unpinned nightly resolves the latest **torch 2.12** + **ROCm 7.14.0a\*** stack (matches `libbitsandbytes_rocm714.so`). Override with `TORCH_PIN=…`, `ROCM_META_PIN=…`, or `TORCH_NIGHTLY_MINOR=…`.

**Stable** (repo.amd.com, no nightly alphas): pin **`torch==2.12.0+rocm7.14.0`** + **`rocm-sdk==7.14.0`** (cp310–cp314):

```bash
ROCM_CHANNEL=stable ./install_fizgig_rocm.sh
```

**Try torch 2.14** (nightly only today — can increase sampling VRAM pressure vs 2.12):

```bash
ROCM_CHANNEL=nightly TORCH_NIGHTLY_MINOR=2.14 ./install_fizgig_rocm.sh
# or an explicit pin, e.g.:
# TORCH_PIN=2.14.0a0+rocm7.14.0a20260625 ROCM_CHANNEL=nightly ./install_fizgig_rocm.sh
# (paired torchvision ~0.29.0a0+rocm7.14.0a… — installer resolves the match)
```

Linux ROCm cache scripts import `fizgig.rocm.cache_exit` only when `FIZGIG_GPU_BACKEND=rocm` (set by `run_fizgig_rocm.sh`); NVIDIA and other platforms call `main()` unchanged. Opt out: `FIZGIG_ROCM_NO_FAST_EXIT=1 ./run_fizgig_rocm.sh`.

Then shared deps from `requirements.txt` (filtered) and `bitsandbytes>=0.50.0` for ROCm.

**Linux / macOS (NVIDIA CUDA path)** — `install_fizgig.py` is CUDA-only (captioning / image prep on macOS; training needs a CUDA or ROCm GPU). On AMD-only Linux hosts it prints a hand-off to the ROCm installer and exits:

```bash
python install_fizgig.py
chmod +x run_fizgig.sh
./run_fizgig.sh
```

**VRAM status bar on AMD:** the existing NVIDIA `pynvml` / `nvidia-smi` path is unchanged; AMD readers (`vram_monitor.read_amd_gpu_vram`) run only as a fallback. Windows ROCm uses `typeperf`; Linux ROCm uses the **`amd-smi`** CLI when available ([AMD SMI / ROCm Core SDK](https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html), e.g. `sudo apt install amdrocm-amdsmi`). Fizgig picks the GPU with the largest VRAM total (skips empty iGPU entries). Legacy `rocm-smi` is a fallback. Do not `pip install amdsmi` — the PyPI package is outdated.

Three small models auto-download on first use: InsightFace `buffalo_l` (~300 MB, during install), Florence-2 (~500 MB–1.5 GB, first AI caption), and Helsinki-NLP `opus-mt-en-zh` (~300 MB, first bilingual translation).
---

## Model downloads

Fizgig doesn't bundle weights. You only need the family you're using — and **Preferences has a ⬇ Download models for me button** under each model card that downloads, verifies, and fills in the paths (Klein needs a free HuggingFace token for BFL's licence; Krea 2 needs no account). Every row also has a manual **Download** link. CLI:

```bash
python -m fizgig.scripts.fetch_models --family krea2   # ~32 GB, no account needed
python -m fizgig.scripts.fetch_models --family klein   # ~34 GB, needs a token
python -m fizgig.scripts.fetch_models --family tools   # Florence-2, face model, translator
```

### Klein 9B

| Model | File | Size | Source |
|---|---|---|---|
| **Base DiT (fp8) — recommended** | `flux-2-klein-base-9b-fp8.safetensors` | ~9.5 GB | [black-forest-labs/FLUX.2-klein-base-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8) |
| Base DiT (bf16) | `flux-2-klein-base-9b.safetensors` | ~17 GB | [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) |
| Distilled DiT | `flux-2-klein-9b-fp8.safetensors` | ~9 GB | [black-forest-labs/FLUX.2-klein-9b-fp8](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8) |
| VAE / AE | `ae.safetensors` | ~320 MB | [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors) (from root, **not** the `vae/` subfolder) |
| Text Encoder | `qwen_3_8b.safetensors` | ~15 GB | [Comfy-Org/vae-text-encorder-for-flux-klein-9b](https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors) |

Training runs on the **Base DiT** — the fp8 version is recommended on every GPU (same quality, half the VRAM). The **Distilled DiT** powers the 4-step previews and the workbench.

### Krea 2

All files live in the one [**Comfy-Org/Krea-2**](https://huggingface.co/Comfy-Org/Krea-2) repo.

| Model | File | Size |
|---|---|---|
| **RAW DiT (bf16) — training** | `krea2_raw_bf16.safetensors` | ~26 GB |
| **Turbo DiT (fp8) — workbench** | `krea2_turbo_fp8_scaled.safetensors` | ~13 GB |
| Turbo LoRA *(auto-downloads)* | `krea2_turbo_lora_rank_64_bf16.safetensors` | ~470 MB |
| Qwen-Image VAE | `qwen_image_vae.safetensors` | ~250 MB |
| **Text Encoder — recommended** | `qwen3vl_4b_fp8_scaled.safetensors` | ~5.2 GB |
| Text Encoder — full precision | `qwen3vl_4b_bf16.safetensors` | ~8.9 GB |

The text-encoder slot is **open**: any Qwen3-VL-4B in the ComfyUI layout loads — fp8_scaled (recommended, captions we couldn't tell apart), bf16, or a community fine-tune/abliterated build, which changes how your dataset gets captioned.

*MiniMax H3's files are listed [in its section above](#model-files-minimax-h3).*

---

## VRAM guidance

### Klein 9B

**Training** — the fp8 Base stays resident at ~9.6 GB, so a 9B LoRA fits **16 GB** (~14 GB observed). Smaller cards: the **4-bit (NF4) base** toggle drops the base to ~5.6 GB — a full LoRA trains in ~7.5 GB, fitting **10–12 GB cards with no swap**.

**Workbench** (Distilled 4-step):

| Block Swap | Min VRAM |
|---|---|
| 0 | 24 GB+ |
| 8 | 16 GB |
| 12 | 14 GB |
| 16 | 12 GB |

On first launch Fizgig auto-detects your VRAM and picks the default; your own choice sticks.

### Krea 2

| Your card | What to do |
|---|---|
| **8 GB** | Everything on **Auto**, batch size 1, stock preset defaults |
| **10–12 GB** | Same — headroom to raise batch size or resolution |
| **16 GB+** | Same — Auto will usually pick the faster INT8 path |

Auto budgets from your *free* VRAM and the console explains its choice. If a preview can't fit, previews auto-disable and **training keeps running and saving**.

### MiniMax H3

See [the Auto table in its section](#minimax-h3--third-model-family) — 16 GB and up trains on the accurate int8 base with streamed block swap; ≤12 GB falls back to 4-bit. On 16 GB-class cards, previews cap themselves at **768×640 and 22 frames** (sound kept) — larger picks in the menus simply clamp, with a console note. On 24 GB cards the int8 base streams ~11 blocks, and clip previews clamp to **22 frames** for the same reason (resolution untouched); lower Target Megapixels if you want the 56-frame preview back on that tier.

**12 GB cards + previews (Windows): leave the paging file system-managed.** Each preview parks the training model and optimizer in system RAM while the decoder runs, and that commit spike is what a fixed small paging file (e.g. 4 GB) can't cover — the run dies with **Windows error 1455** ("paging file is too small"), which says nothing about previews. Settings → System → About → Advanced system settings → Performance → Advanced → Virtual memory → *Automatically manage*. Reported and confirmed on a 12 GB RTX 5070 by David Maybank.

### Desktop feels juddery while training? (Windows)

Turn off **Hardware-accelerated GPU scheduling** (Settings → System → Display → Graphics → *Default graphics settings*), then reboot. With it off, Fizgig runs training at low priority so your desktop stays smooth — training speed is unaffected.

---

## Getting started

Launch Fizgig and work left-to-right through the numbered tabs:

1. **Start** — set your training image folder.
2. **Image Prep** (optional) — resize, face-crop, and run the Look Consistency Filter.
3. **Captions** — trigger-word or AI captions.
4. **Samples** — the preview prompts that render during training.
5. **Training** — pick a preset, click **Start Training**.

The unnumbered tabs are the post-training workbench: **Profiler**, **Repair Studio**, **LoRA the Explorer**, **LoRA Royale**, **Extract**, and **Preferences**.

One tool lives outside the main window: **Checkpoint to LoRA** (`run_diff_to_lora.bat`, or
`python diff_to_lora_gui.py`) — point it at a base model and a fine-tuned checkpoint and it writes
an ordinary LoRA at whichever ranks you tick. Only needed if you use the experimental full
fine-tune above.

**Headless?** Everything the trainer does is also available from the command line — see **[docs/CLI.md](docs/CLI.md)**.

**Community translations** — **Korean (한국어)**: [Fizgig-Korean-Translated-Ver](https://github.com/ssain3d-lgtm/Fizgig-Korean-Translated-Ver) by @ssain3d-lgtm — an unofficial add-on that translates the UI at runtime without touching any Fizgig files, with a one-script uninstall. If you hit a bug while it's installed, uninstall and reproduce before reporting here; layer issues go to that repo.

---

## Support the project

If Fizgig saves you time or helps you make better LoRAs, consider supporting development:

<a href="https://buymeacoffee.com/lorasandlenses"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>

---

## License

Fizgig is open source under the **[Apache License 2.0](LICENSE)** — free to use, modify, and redistribute, including commercially, with attribution and no warranty. Third-party components under compatible permissive licenses (and other terms where noted) are listed in **[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)**.

Copyright © 2026 Peter Neill.

Model weights are **not** covered by this license — each model carries its own terms from its publisher (see the Download links in Preferences).

---

## Working with me

I'm available for consulting on local AI training pipelines, custom workflow tooling, and
private model work — the same engineering that's in Fizgig, applied to your studio's
hardware and IP. Get in touch: **peter@shootthesound.com**.
