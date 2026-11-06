# Fizgig v9.0.8: Muon for every picture model's presets

**Improved likeness and training performance across most of the models, thanks to the Muon optimizer.** Every picture model's built-in presets now train on Muon. It steps each weight matrix as a whole and evens out the update across all its directions, so fine detail (skin texture, hair, fabric) learns as fast as the broad shapes instead of waiting behind them. Fizgig scales Muon's step to match AdamW's, so the presets keep their learning rates. Thanks to @Murf-Code, who brought Muon to Fizgig in 9.0.1.

**What changes for you**
- Klein 9B, Flux 3 Image Klein, Fibo 2.0, Krea 2, Qwen Image 2.1, MOSS-Image, Z-Image Turbo and Base, SDXL, Anima and Vimoo 1.0 Lite: every LoRA, slider and edit preset is on Muon. Load the preset again to get it; the Training form and your saved presets keep the optimizer they had.
- Fibo 2.0's main preset trains at a flat 1e-4 instead of Automagic's 1e-6 start. Automagic is still in the Optimizer Type row.
- Full fine-tunes keep Adafactor, MiniMax H3 keeps Automagic (its Slider AdamW), and ACE-Step keeps AdamW.
- Muon keeps one value per trained number where AdamW keeps two, so it uses less memory than AdamW and a little more than AdamW8bit (about 0.2 GB on a rank-32 Krea 2 LoRA).
- With Adaptive LR on, watch for a run that climbs above about 3.5e-4: one Qwen Image 2.1 run trained better once Adaptive LR brought Muon back down to 2e-4. Every epoch is saved, so scrub them in LoRA Royale.

AdamW8bit and AdamW are still in the Optimizer Type row if you prefer them.

**Studio:** if you haven't updated today, [9.0.7](https://github.com/shootthesound/Fizgig/releases/tag/v9.0.7) brought a lot to Studio: settings wait for Update, the Baseline shows while the Tweaked render is still coming, untouched sliders reuse the Baseline instead of rendering twice, and a changed preview model reloads at the next Update.

**Updating:** update Fizgig on your computers, and restart your pods.
