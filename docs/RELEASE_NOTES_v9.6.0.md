# Fizgig v9.6.0: MiniMax H3 Image

**A new model: MiniMax H3 Image.** MiniMax's picture model from the H3 family, released 3 November: text-to-image and edits with up to 16 reference pictures, rendering in 6 steps with no CFG. Pick it on the Training tab and press **Download models for me** in Preferences; nothing is gated, and if you already have MiniMax H3, its text encoder isn't downloaded again.

**What you can train:** LoRAs, LoKRs, edit LoRAs from before/after pairs, and slider LoRAs from photo pairs, with masked training, Multi Concept, DOP and Fizgig LoRA Isolation, batch size and Compile Blocks. The presets are Krea 2's: Ultra Fast, Standard, Style, Edit and Slider.

**Fizgig's training adapter is on by default.** Without it, people trained on this model came out as caricatures; with it, the same dataset gave photographic skin and a clear likeness, and pictures without the trigger word kept the model's own quality. It stays frozen during training, is off in previews, and is never in your saved LoRA. It downloads with the model, and it's free on [Hugging Face](https://huggingface.co/ShootTheSound/Fizgig-MiniMax-H3-Image-Training-Adapter) for any trainer.

**Every workbench tool works with it:** Studio with Reuse unchanged blocks and Save Repaired, LoRA Royale with seed and prompt travel, the Profiler and Extract. LoRAs from other trainers load too, in kohya, ComfyUI or diffusers names. Fizgig's LoRAs load in ComfyUI's standard LoRA loader with every layer.

**Memory:** about 23 GB with the int8 model, 13 GB in 4-bit; Auto picks for you. A 16 GB card trains and runs Studio. A 12 GB card trains, but Studio there needs about 48 GB of system RAM for the 32B text encoder.

Setup, presets, memory and the ComfyUI settings are in the [MiniMax H3 Image guide](https://github.com/shootthesound/Fizgig/blob/master/docs/MINIMAX_IMAGE.md).
