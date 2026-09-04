# Fizgig v5.3.0 — DRAFT (unreleased)

## MiniMax H3: 4-bit HQQ moves to group 8 — a third less base error at the same size

HQQ 4-bit now quantises in groups of 8 instead of 16, taking the frozen base from ~6.3% to ~4.8% error against the int8 reference. Plain group 8 would have doubled the per-group overhead to int8's own footprint, so the per-group scales and zeros are themselves stored as 8-bit codes with one affine per output row — measured at 4.83% versus plain group 8's 4.80%, at exactly the group-16 footprint (0.75 bytes per weight, ~15 GB resident on the pruned checkpoint). Group size and the 6% cost on a 16 GB card are **[@rintic-13](https://github.com/rintic-13)**'s numbers (#102).

Also corrected in the docs: HQQ's speed cost is a big-card statement. On a plan that streams blocks — the 12–24 GB tiers it exists for — the dequant hides behind the PCIe transfers and HQQ runs at NF4's speed; only on a large card with nothing streamed does it show as roughly half NF4's step speed.

## Repair Studio sees motion (MiniMax H3)

Repair Studio on H3 used to show one frame of a clip. Now it renders the clip — 22 frames with its sound by default, or a still, or 56 frames — and plays both sides **in the app**: click either preview and baseline and tweaked loop side by side in lockstep, with pause, frame stepping, a scrub bar, slow motion and **S** to swap sides (the left one carries the sound). The metrics strip and the Profiler cross-link still read the middle frame.

**Steps, Turbo, Render size.** Three boxes and two presets instead of a mode switch. Steps and Turbo (the bundled Turbo LoRA's strength) take any numbers; Render size renders every clip at a fraction of your chosen size. **Dial** fills them with the fast loop (4 steps, Turbo 1.0, ⅔ size — ~4 s a move on a 5090), **Confirm** with the render you judge before saving (6 steps, Turbo 0.75, full size, the settings training previews use). Turbo 0 switches the Turbo LoRA off, so a Turbo LoRA loaded as the primary can be edited on its own. **Show early** puts a rough picture up after the second pass while the remaining passes finish. Every preview is an exact render; nothing is approximated. A change to a late block skips the untouched blocks on the first pass — bit-for-bit the same result, up to a fifth faster.

**The block library.** After your first render, every block is rendered switched off in the background at your current settings (about 3½ minutes for a 52-block LoRA at the Dial preset; it pauses the moment you move a slider). From then on the ● beside each slider is live: hover for a thumbnail, click to see the clip with that block removed, instantly. Every render you make is kept as well, so a state you've already seen comes back in a couple of seconds, and the new **History** strip holds all of them — click to view, right-click to pin one as the baseline (compare two tweaks head to head) or save it as an MP4. Library and history live in your cache folder, survive restarts, and go with **Clear cache…**.

**First / Last Frame.** Pin the clip's first and/or last frame to a photo — pick it or drag it from Explorer onto the slot, drag an aspect-locked box over the part you want — and every render starts (or ends) on that picture, so what you compare is the LoRA's effect rather than the shot the seed picked.

**Short clips.** Length now goes down to 5 frames, plus 9 and 13 (marked off-grid — never trained on, render fine). Fewer frames, fewer tokens: a 5-frame Dial move is a fraction of the 22-frame time, for reading a block's effect on the picture before judging motion at 22. The Clip row and every Browse dialog's last folder are remembered across restarts.

**No-LoRA clip.** A tick on the Clip row (and in the player) adds a third pane: the same seed and prompt rendered by the base model with no LoRA, so you see what the LoRA adds rather than only what the sliders changed. Rendered once per setup and cached. The player also gained a 0.1× speed.

**Load strength.** An "at strength" dial per LoRA — the strength it was designed for. Block sliders stay relative to it, the baseline is the LoRA at that strength, and the saved file keeps its original scale so it looks in ComfyUI exactly as it did in the player, at that same strength.

Also fixed on the way: a LoRA carrying AdaLN keys (any run trained with AdaLN on, AI-Toolkit's files) failed to load in Repair Studio on H3; and the first clip decode of a session no longer costs ~10 s in the middle of the first preview.

Klein and Krea 2 Repair Studio are unchanged.

## Multi Concept no longer switches identity-learn on

Ticking Multi Concept now sets caption dropout to 0.10 (strong) and nothing else. It used to switch reference distillation on as well, with its references and identity-first phase; in our runs that wasn't what held two subjects apart — the trigger words do that work — and a data-layout tick quietly enabling a 21 GB-model experiment was the wrong shape. Identity-learn stays its own deliberate tick.
