<!-- DRAFT — unreleased -->
# Fizgig v5.3.0

LoRA surgery on video. Open a MiniMax H3 LoRA and see what every one of its 52 blocks does to a moving clip — the motion, the face, the sound — one click each, every one an exact render. Pin the first and last frame, feed it reference photos, compare against the bare model, and save the fix at the strength it was built for. On 24 GB cards too.

## MiniMax H3: 4-bit HQQ moves to group 8 — a third less base error at the same size

HQQ 4-bit now quantises in groups of 8 instead of 16, taking the frozen base from ~6.3% to ~4.8% error against the int8 reference. Plain group 8 would have doubled the per-group overhead to int8's own footprint, so the per-group scales and zeros are themselves stored as 8-bit codes with one affine per output row — measured at 4.83% versus plain group 8's 4.80%, at exactly the group-16 footprint (0.75 bytes per weight, ~15 GB resident on the pruned checkpoint). Group size and the 6% cost on a 16 GB card are **[@rintic-13](https://github.com/rintic-13)**'s numbers (#102).

On speed: where blocks stream — the 12–24 GB tiers HQQ exists for — the dequant hides behind the PCIe transfers and group 16 measured level with NF4 (**[@rintic-13](https://github.com/rintic-13)**, 16 GB card); plain group 8 cost him 6% there. On a big card with nothing streamed the dequant is exposed: about half NF4's step speed at group 16, and group 8 a further ~15% on a 5090. If group 8's cost doesn't hide on the streamed tiers once measured there, group 16 goes back to being the default.

## LoRA surgery on video: Repair Studio on MiniMax H3

Repair Studio on H3 used to show one frame of a clip. Now it renders the clip — 22 frames with its sound by default, or a still, or 56 frames — and plays both sides **in the app**: click either preview and baseline and tweaked loop side by side in lockstep, with pause, frame stepping, a scrub bar, slow motion and **S** to swap sides (the left one carries the sound). The metrics strip and the Profiler cross-link still read the middle frame.

**Steps, Turbo, Render size.** Three boxes instead of a mode switch. Steps and Turbo (the bundled Turbo LoRA's strength) take any numbers; Render size renders every clip at a fraction of your chosen size. 4 steps at Turbo 1.0 and ⅔ size is the fast loop (under 4 s a move on a 5090); 6 steps at Turbo 0.75 and full size is the render you judge before saving, the settings training previews use. Turbo 0 switches the Turbo LoRA off, so a Turbo LoRA loaded as the primary can be edited on its own. **Show early** puts a rough picture up after the second pass while the remaining passes finish. Every preview is an exact render; nothing is approximated. A change to a late block skips the untouched blocks on the first pass — bit-for-bit the same result, up to a fifth faster.

**The block library.** After your first render, every block is rendered switched off in the background at your current settings (about 3½ minutes for a 52-block LoRA at 4 steps and ⅔ size; it pauses the moment you move a slider). From then on the ● beside each slider is live: hover for a thumbnail, click to see the clip with that block removed, instantly. Every render you make is kept as well, so a state you've already seen comes back in a couple of seconds, and the new **History** strip holds all of them — click to view, right-click to pin one as the baseline (compare two tweaks head to head) or save it as an MP4. Library and history live in your cache folder, survive restarts, and go with **Clear cache…**.

**First / Last Frame.** Pin the clip's first and/or last frame to a photo — pick it or drag it from Explorer onto the slot, drag an aspect-locked box over the part you want — and every render starts (or ends) on that picture, so what you compare is the LoRA's effect rather than the shot the seed picked.

**Reference mode.** A Model picker on the Setup card runs either checkpoint: First/last frame (fl2va) or Reference (ref2va). Under Reference the frame card becomes Reference Images — up to two photos, cropped to the clip's shape and sized to its canvas — and `<Picture 1>` / `<Picture 2>` in the prompt refer to them, the same convention as the r2v workflow. The prompt-plus-pictures encode is paid once per combination and cached; the status line tells you when the encoder is running.

**The text encoder stays parked in RAM.** After the first prompt of a session the 32B encoder stays parked in system RAM when the machine has the room (about 40 GB free — a 64 GB box), so every new or edited prompt after that costs seconds rather than a fresh stream from disk. With less RAM the encoder loads per prompt and the base steps aside for it — unloaded and reloaded from disk on a 32 GB machine rather than copied into RAM, which would page — and returns bit-identical. Released if RAM runs short or when the studio unloads.

**Lengths and canvas.** Length runs from a still and 5 frames up to 124 (about five seconds), plus 9 and 13 (marked off-grid — never trained on, render fine). Width and Height are separate menus, 512 to 1536 each, so tall and wide clips are a pick rather than a list; H3 likes one side at 768 or more, and 768 × 640 is the default. Fewer frames, fewer tokens: a 5-frame move is a fraction of the 22-frame time, for reading a block's effect on the picture before judging motion at 22. The Clip row and every Browse dialog's last folder are remembered across restarts.

**No-LoRA clip.** A tick on the Clip row (and in the player) adds a third pane: the same seed and prompt rendered by the base model with no LoRA, so you see what the LoRA adds rather than only what the sliders changed. Rendered once per setup and cached. The player also gained a 0.1× speed.

**Load strength.** An "at strength" dial per LoRA — the strength it was designed for. Block sliders stay relative to it, the baseline is the LoRA at that strength, and the saved file keeps its original scale so it looks in ComfyUI exactly as it did in the player, at that same strength.

**24 GB cards run the int8 base.** The studio plans its base from free VRAM at Start: 32 GB keeps the int8 base resident; 24 GB runs the same int8 base with its last ~24 blocks streamed from system RAM each pass (0.2% base error, not the NF4 base's 9.5% — the base's own error must not be in the picture when you're judging a block); under ~18 GB free the NF4 base takes over. On a simulated 24 GB card: load 26 s, a slider move at 4 steps and ⅔ size 4.5 s, the No-LoRA clip 3.7 s, a 56-frame 768×640 clip at 6 steps about a minute. The pass-1 resume sits out on the streamed plan. A **Base picker** under the Model picker overrides the plan: **Stream blocks** keeps the exact int8 base on any card with enough of it streamed from RAM for the biggest clips (1024² × 56 frames with a first and last frame, which ran a 5090 out of VRAM with the base resident — with 18 blocks streamed it renders in about 80 s at 6 steps, peak 23 GB), and **NF4** is the smallest base at 9.5% error and the quickest slider loop (2.2 s a move against int8's 3.6 s).

Also fixed on the way: a LoRA carrying AdaLN keys (any run trained with AdaLN on, AI-Toolkit's files) failed to load in Repair Studio on H3; and the first clip decode of a session no longer costs ~10 s in the middle of the first preview.

Klein and Krea 2 Repair Studio are unchanged.

**Small things on the same tab.** A **Reset all / All off / All on / Invert** row above the block sliders, acting on the enable ticks (strengths untouched). **Up to date** / **Pending refresh…** under the tweaked pane's title. **512** and **640** rungs in the Width / Height menus — under spec, a help on smaller cards. A change made while a render is running aborts it within one block and starts the new one; a donor path cleared by hand unloads the donor on the next Update. Any LoRA or photo Browse dialog opens in the folder you last picked from.

## Small things

- The LoRA layer every preview runs through (Repair Studio, Explorer, Profiler, Extract, all three families) does its add in one kernel instead of three — a few percent off every preview, same maths. Training is untouched.
- Fixed in the tests: a stale paused-run sidecar in the output folder blocked the LoRA name retag between families; the retag itself was right to refuse, and the battery now runs in a clean folder.

## Multi Concept no longer switches identity-learn on

Ticking Multi Concept now sets caption dropout to 0.10 (strong) and nothing else. It used to switch reference distillation on as well, with its references and identity-first phase; in our runs that wasn't what held two subjects apart — the trigger words do that work — and a data-layout tick quietly enabling a 21 GB-model experiment was the wrong shape. Identity-learn stays its own deliberate tick.
