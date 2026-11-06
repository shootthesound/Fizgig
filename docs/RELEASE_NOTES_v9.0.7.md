# Fizgig v9.0.7: Vimoo 1.0 Lite, Batch size Auto on big cards, and Studio waits for Update

**Vimoo 1.0 Lite (experimental).** Fizgig now trains Vimoo-AI's 9.5B video model, released under Apache 2.0. One checkpoint does text-to-video and image-to-video at 720p and 24 fps. In our tests its quality and performance sit closer to Wan 2.2 5B than to the newer video models. It's a young model and may improve with later releases.

**Why Lite, and not the full Vimoo 1.0.** If your hardware can run the full Vimoo 1.0, there are probably better video models you can run instead. The full model is two ≈11.7B checkpoints, one for text-to-video and one for image-to-video, and it works on four times as many tokens per clip. Lite's reason to exist is that it's light: one checkpoint for both tasks, each 2×2 patch merged into one token, and it trains on a 16 GB card. That's the case where Vimoo makes sense, so that's the one we support.

Fizgig supports Lite in full: LoRA and LoKR from photos and clips, slider LoRAs, full fine-tunes, masked training, and an image-to-video share that trains part of each clip from its own first frame. Training previews render as clips in 12 steps with Vimoo's distilled LoRA. Studio, the Profiler and LoRA Royale work with it too, and Studio can start a clip from a picture. A 16 GB card trains it: Auto picks 4-bit NF4 there (9.6 GB), or pick INT8 yourself (13.1 GB). Gizmo now asks which video model a clip is for and prepares it to that model's frame counts and sizes. **Download models for me** on Preferences fetches the files (about 40 GB), and LoRAs load in ComfyUI with Vimoo's own nodes. See the [Vimoo guide](https://github.com/shootthesound/Fizgig/blob/master/docs/VIMOO.md).

**Batch size Auto on 40 GB+ cards.** Batch size has a new Auto choice for the models where training several pictures a step measured faster per picture: SDXL, MiniMax H3, Anima, Flux 3 Image Klein and Fibo 2.0. On a 40 GB+ card Auto picks the measured best batch, and the note under the box says what it does on your card. On the same cards, gradient checkpointing now goes off for SDXL, Anima, Flux 3 Image Klein and Klein when the run fits, 1.3–2.3× faster. Move to… uses the same measured batch on the pod. Cards of 32 GB and under train as before.

**Studio waits for Update.** Changing the size, clip length, steps, seed, reference picture, negative prompt, LoRA strengths and the other render settings no longer starts a render at once. The button says **Update**, so you can change several settings first. The block sliders still render as they move. Also in Studio:
- When the Tweaked render is still coming, the Baseline shows (or plays) as soon as it's ready.
- With every slider at its default, the Tweaked picture or clip is the Baseline one: one render instead of two.
- Changing Render with, or the Samples tab's checkpoint, look LoRA or Turbo tick, reloads the model at the next Update. Before, it kept the old model until Unload.
- After each render the worker hands its cached GPU memory back, so a big model doesn't keep the card full between renders.
- **Turbo Preview** is now called **Reuse unchanged blocks**, which is what it does. It moves its stored results to system RAM when the card is tight, so long clips no longer run out of memory with it on.

**Fixes**
- Block swap with a LoRA on some blocks only (Fast Identity Mode, a Model Area) no longer crashes on the second step with "mat2 is on cpu" (#302, thanks @spinolover).
- A LoKR from LyCORIS loads at the right strength. Fizgig multiplied a full-matrix LoKR by the alpha stored in the file, and LyCORIS stores the network dim there, so a Context LoRA played many times too strong and samples came out as static (#304, thanks @Nessa2026691). LoKRs trained in Fizgig are unaffected.
- MOSS-Image: an interrupted tokenizer download no longer stops caching (#305, thanks @MrB4000).
- Prompt sliders train again on every model (a "max() iterable argument is empty" error at the start since 9.0.5).
- MiniMax H3's Slider preset trains on AdamW instead of AdamW8bit, which measured clearly better on H3 LoRAs.
- Fibo 2.0 on a 16 GB card trains INT8 with no block swap at 0.25 MP, 2.1 s a step instead of 3.4 s.
- The training log says which INT8 a run uses.

**Updating:** update Fizgig on your computers, and restart your pods.
