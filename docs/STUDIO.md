# Studio

Studio changes a finished LoRA block by block and shows the result live: a **Baseline** render (the LoRA as it is) beside a **Tweaked** render (with your slider changes), same prompt and seed, so the only difference is the sliders. When it looks right, **Save repaired LoRA** writes a new file that renders exactly like the Tweaked preview. The original is never changed.

Use it to:

- **Fix a LoRA instead of retraining it.** A LoRA that overbakes skin, fights the prompt or drags in its training backgrounds can often be rescued by turning one group of blocks down.
- **Keep part of a LoRA.** Turn off everything but the identity blocks, or everything but the style blocks.
- **Mix LoRAs** block by block: identity from one, style from another, or several LoRAs at their own amounts.
- **Explore**: 🎲 I'm feeling lucky rolls random block strengths within limits you set.

Every image and video model has Studio; ACE-Step (songs) has it too, with a song player in place of the pictures.

## Getting started

1. **Model**: the model the LoRA was trained for.
2. **Preview with**: where the model has one, the fast few-step preview model (Turbo / Distilled) or the full model. The fast one is right for slider work; check a final result on the full model.
3. **Primary LoRA**: Choose… the file. **at strength**: the strength you normally use it at (not always 1.0). The sliders work relative to it, and the saved file keeps that scale, so used at this strength it looks exactly like the preview.
4. **Prompt**, **Seed** (↻ for a random one) and **Width / Height** (clips: length, size and steps; songs: length).
5. **Start**. The model loads on the Fizgig machine (the first time takes a minute) and renders the pair. After that every slider change re-renders Tweaked by itself. **Unload model** frees the GPU; leaving the tab does too.

Click either picture to see both full size (← → switches between them). **⧉ Keep in view** puts the pair in a panel that stays on screen while you scroll down to the sliders.

**Reuse unchanged blocks** (on by default where the model has it) keeps the unchanged blocks' results in VRAM and replays them, so a change to a late block renders almost at once. It is the same picture as a full render. Turn it off if VRAM is tight.

Studio can't load while a training run is using the GPU.

## The block sliders

One row per block of the model, grouped as the model is built, and colour-coded by what the Profiler found blocks of that kind do. A greyed row is a block this LoRA doesn't train: moving it would change nothing.

On each row:

- **The tick** turns the block on or off.
- **The slider** sets its strength, from -3 to 3. 1.0 is the block as trained, 0.5 half of it, 0 nothing, a negative value inverts it.
- **The value box** shows the strength. Click it and type a number to set it exactly (Enter sets it, Esc puts it back).
- **0 / 1** set it to 0 or 1, **±** flips the sign, **− / +** step 0.1.
- **🔒** locks the row: presets, bulk buttons and lucky rolls leave it alone.

**↶ Undo / ↷ Redo** (Ctrl+Z / Ctrl+Y) step back and forward through your changes.

Above the rows:

- **Presets**: the model's built-in ones (✨) and yours. **Save preset…** keeps the current sliders under a name for this model.
- **Bulk buttons**: Reset all, All off, All on, Invert, Alternate (even blocks on, odd off), and, where the model has them, Toggle detail blocks.
- **Master controls** (models with a block map, such as Klein): one slider per category (Style + Composition, Identity, Details) that moves all its blocks together, on the primary or the donor.

## A donor LoRA

**Donor LoRA** loads a second LoRA on top. Every block row then gets a second slider, for the donor's block, so you choose block by block how much of each LoRA you get. **⇄** on a row balances it: the other side gets 1 minus this one. Donor: all on / all off and the masters' **Masters affect: Donor** work on the donor's side.

Saving a mix is exact: where both LoRAs are on, the saved file holds both side by side (its rank is the two ranks added), so it renders just like the preview.

## Mixing epochs and LoRAs

**Mix** on the primary's row:

- **Epochs of this run**: the saved epochs beside the primary, blended as a moving average. **Up to** is the newest epoch in the blend, **History** how much of the earlier ones comes with it (0 = that epoch alone). Often smoother than any single epoch.
- **LoRA mix**: the primary plus other LoRAs for this model, each at its own amount (for example, one style at 1.0 and another at 0.5).

The donor has its own **Blend the donor's epochs**. The block sliders shape the whole mix, and Save bakes it into one LoRA.

## 🎲 I'm feeling lucky

Each roll gives the chosen number of sliders (**Roll**: 1, 2, 4 … or All) a random strength between the two limits beside it. **Hold structure + tidy-up** keeps the blocks that hold the picture together near 1, so rolls stay usable; untick it to roll every block across the whole range. With a donor you choose which LoRA a roll changes.

When a roll gives you something you like in part, **🔒 Freeze tweaked** locks every block you've moved, so the next rolls only change the rest. **🔓 Unfreeze** lets them go again (blocks you locked yourself stay locked).

## Reading the result

- **The numbers under the pair** compare the two renders: likeness to a reference photo (📷 Reference…, for a person LoRA), Grid (the model's patch lattice showing through: rising means overbake), Detail (texture in the face, or the whole picture: down is plastic, up is fried), Clipped (blown pixels) and Sat (saturation). They're a guide; trust your eye.
- **Profile found for this LoRA**: when the Profiler has profiled this exact file, its findings show here, with a link to the full report.
- **Model EQ Mode** (models with a Model EQ LoRA set in Preferences): a row holds the model's own weights instead of a LoRA, so the sliders turn the stock model's own blocks up or down. Press it again to get your LoRA back.

## Saving

- **Save repaired LoRA**: name the new file under **Save as** and press it. It's a normal LoRA for ComfyUI and anything else, with the slider settings recorded in its metadata.
- **💾 Download original / Download tweaked**: the two renders as files. A downloaded Tweaked picture keeps Studio's settings inside it (the LoRAs, every slider, the prompt and seed): **drop it on the Studio tab to get them all back**.
- **Save setup**: everything on the tab (the LoRAs, every slider and tick, the prompt, seed, size, photos and source) in one named file. **Load a setup…** restores it exactly, switching model if it needs to, loading its LoRAs and rendering.

## Video (MiniMax H3)

The pair are clips with sound. **Play together** plays both in step. Clip length, size, steps and the built-in Turbo strength are on the Clip row (Turbo 1.0 with 4 steps is the fast loop; set Turbo to 0 to edit a Turbo LoRA itself). **Speed Boost** renders the first half of the steps at half size. **Show early** puts up an estimate in the Tweaked pane while the remaining steps run. **First / Last Frame** takes photos for the clip to start and end on; the Reference model takes up to two photos you name in the prompt as `<Picture 1>` / `<Picture 2>`.

## Songs (ACE-Step)

The song player plays Baseline, Tweaked or No LoRA (the base model, same seed). Prompt B blends a second prompt in (0 = A alone, 1 = B alone, with A's lyrics kept if you like). **Start from a song** follows a song's melody and timing (Follow), starts from its actual sound (Denoise), repaints only a section you drag across the waveform, or extends it. **Sound like a recording** makes the song sound like a reference recording. **⤵ Use as source** makes the Tweaked song the one to start from. **💾 Save Tweaked WAV** saves it.
