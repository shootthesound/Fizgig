# Fizgig v8.9.95: Z-Image Base, and a bigger captioning model

**Z-Image Base.** Tongyi's undistilled Z-Image is now its own model on the Training tab. It trains on Base itself, with no training adapter since Base isn't distilled: LoRAs, LoKR, sliders from prompts or photo pairs, masked training and full fine-tunes, with the same memory settings as Z-Image Turbo, the big-card ones included. Most people generate with Turbo, so training samples, Studio and LoRA Royale show your Base LoRA on Z-Image Turbo by default (8 steps, no CFG). Untick **Use Z-Image Turbo for samples** on the Samples tab to preview on Base instead, at ComfyUI's Base settings: 25 steps, CFG 4, with a negative prompt. **Download models for me** fetches Base and the Turbo model for the previews; if you already have Z-Image Turbo set up, its VAE, text encoder and Turbo file are used where they are, so only the Base model downloads (thanks @girzai, #261). [Z-Image Base](https://github.com/shootthesound/Fizgig/blob/master/docs/ZIMAGE.md#z-image-base)

**A secondary captioning model.** Preferences has a new **Captioning model** card with an optional **Secondary captioning model**, linking to Qwen3-VL 8B: the bigger model of the same family as the Krea 2 text encoder Fizgig captions with (it's Qwen Image 2.1's text encoder), for more detailed captions. Download the fp8 build (10.6 GB, about 10.5 GB of VRAM) or the bf16 one (17.5 GB, about 17 GB), and choose it there. While it's set, it does all captioning, on the Captions tab and for the trainer's auto-recaption; empty the row to go back to the Krea 2 text encoder. On a card with less VRAM than it needs, it keeps what fits and streams the rest from system RAM, so it still runs, more slowly (thanks @Dwuuds, #270).

**Fixes**

- **Fine-tune and slider previews with a preview-model tick on** (Klein's Distilled, Flux 3's distilled, Qwen Image 2.1's Turbo) rendered on the training model at the preview model's steps and CFG, 4 to 16 steps with no CFG, so they looked unfinished. A fine-tune or slider can't preview on the other model, so these previews now use the training model's own steps, CFG and negative prompt.
- **Z-Image: a LoRA that changes the timestep layer** (for example one extracted from a fine-tune) stopped every render with an error. It renders now.

**Updating:** update Fizgig on your computers, and restart your pods.
