# Fizgig v9.6.0: MiniMax H3 Image

**A new model: MiniMax H3 Image.** MiniMax's picture model from the H3 family, released 3 November: text-to-image and edits with up to 16 reference pictures, rendering in 6 steps with no CFG. Pick it on the Training tab and press **Download models for me** in Preferences; nothing is gated, and if you already have MiniMax H3, its text encoder isn't downloaded again.

**What you can train:** LoRAs, LoKRs, edit LoRAs from before/after pairs, and slider LoRAs from photo pairs, with masked training, Multi Concept, DOP and Fizgig LoRA Isolation, batch size and Compile Blocks. The presets (Standard, Style, Edit and Slider) train on Automagic v3: they start at 1e-4 and the optimizer finds its own learning rate from there. Full fine-tunes aren't offered on this model yet.

**Fizgig's training adapter is on by default.** Without it, people trained on this model came out as caricatures; with it, the same dataset gave photographic skin and a clear likeness, and pictures without the trigger word kept the model's own quality. It stays frozen during training, is off in previews, and is never in your saved LoRA. It downloads with the model, and it's free on [Hugging Face](https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter) for any trainer.

**Clean-end share** (Timestep & Noise) sets how much of training goes to the nearly clean noise level, where fine detail is learned, as MiniMax H3's box does. Leave it empty for the model's own even spread, about 17%.

**Every workbench tool works with it:** Studio with Reuse unchanged blocks and Save Repaired, LoRA Royale with seed and prompt travel, the Profiler and Extract. LoRAs from other trainers load too, in kohya, ComfyUI or diffusers names. Fizgig's LoRAs load in ComfyUI's standard LoRA loader with every layer.

**Sliders from prompts are off on this model for now.** Slider LoRAs from photo pairs work well. A prompt slider (say, smiling vs frowning) moved clothes, styling and light instead of the expression: on H3 Image a change of prompt reshuffles the whole picture, and the slider learns all of it. We tried training on the clean noise levels only, on the noisy ones only, and without the training adapter, and none helped. If you can make prompt sliders work on this model, a pull request is very welcome; start with a post in [New models & fine-tuning](https://github.com/shootthesound/Fizgig/discussions/new?category=new-models-fine-tuning).

**Memory:** about 23 GB with the int8 model, 13 GB in 4-bit; Auto picks for you. A 16 GB card trains and runs Studio. A 12 GB card trains, but Studio there needs about 48 GB of system RAM for the 32B text encoder.

## Also in this release, for every model

**Out-of-memory recovery.** When a run started from the web UI runs out of GPU memory or system RAM, the log now says which, and Fizgig tries to carry the run on by itself, up to three times. With a save, it continues from the newest one; if it runs out again at about the same point, it goes one step lighter: more block swap, then a lower precision, the precision only when it was on Auto. With nothing saved yet, it starts again once, lighter. Each try is in the log and on the run's record, and after the last the log sums up what was tried and what to change.

**A failed run no longer holds the queue.** The Queue box names it and the next run starts, so one failure doesn't stall a night's queue. A stopped run still holds it until you press Start queue.

**Switching models on the Training tab loads that model's first preset.** Settings from before no longer carry over, such as an optimizer the new model doesn't offer. Your folders and the Samples tab are kept. Switching back to the model a run is training, or is paused on, shows that run's own settings.

**Fixes:**
- Studio, opened from a sample preview in the Samples tab, opened a MiniMax H3 Image LoRA as a MiniMax H3 one. A Fizgig LoRA's model now comes from the name stored in it.
- The bitsandbytes warning "No prebuilt binary for CUDA 13.4, loading CUDA 13.2 instead" no longer shows. The 13.2 library it loads works on CUDA 13.4.

Setup, presets, memory and the ComfyUI settings are in the [MiniMax H3 Image guide](https://github.com/shootthesound/Fizgig/blob/master/docs/MINIMAX_IMAGE.md).
