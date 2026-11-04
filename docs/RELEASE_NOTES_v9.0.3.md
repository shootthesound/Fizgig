# Fizgig v9.0.3: transparent pictures, three new optimizers, and masked training for H3 stills

**Transparent pictures train as transparent on Qwen Image 2.1.** A PNG with transparency now trains with its alpha, and the previews keep theirs, shown on a checkerboard (thanks @CognitiveDiffusion, #284). On every other model a transparent PNG is laid on white, as the captioner already saw it, instead of on whatever colour hid behind the transparent pixels. The same goes for face finding, Make masks, the Look filter, Studio, LoRA Royale and the Profiler. Pictures cached the old way are encoded again once, on their next run.

**Three new optimizers: Prodigy, Muon and Automagic v4** (thanks @Murf-Code, #292). Pick them under Optimizer on the Training tab. Prodigy and Automagic v4 set their own learning rate, so Adaptive LR stands down with them. No preset uses them, and every existing optimizer behaves exactly as before. On Qwen Image 2.1 at rank 8, Muon matched AdamW's likeness over 20 epochs; Prodigy and Automagic v4 got there more slowly.

**Masked training for MiniMax H3 stills** (thanks @spaghettified26, #295). H3 gets the Masked training card the picture models have, including prior preservation outside the mask. Masks add about 6% to training time and prior preservation about 39%, with no extra GPU memory. Clips and voice train as before.

**Studio**
- Type a block's value into its box (thanks @tsk5cb-cmyk, #286).
- Download tweaked keeps Studio's settings inside the PNG. Drop that picture on Studio to get them back.
- A guide to Studio: [docs/STUDIO.md](docs/STUDIO.md).

**Also new**
- The loss graph shows the epoch average, a running average and the live learning rate (thanks @Murf-Code, #290).
- Stop asks for a second click, as Pause does (thanks @Murf-Code, #294).
- Right-click works in the app window: Cut, Copy, Paste and Select all in text boxes, and Copy on selected text (thanks @LeonKLAKMUF, #296).

**Fixes**
- On cards of 40 GB and up, an H3 clip run with TREAD off, or on a Training mode other than Default, could go without gradient checkpointing and run out of memory. Those runs now keep it on (thanks @spaghettified26, #282).
- AMD on Linux: RX 6000-series cards (gfx10.3) are detected (thanks @PyroclasticFloe, #281).

**Updating:** update Fizgig on your computers, and restart your pods.
