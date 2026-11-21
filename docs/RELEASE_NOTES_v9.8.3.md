# Fizgig v9.8.3: Frankenstein mode, an experiment in blending MiniMax H3 Image with H3 video

**Frankenstein mode is an experimentation tool: it blends MiniMax H3 Image and MiniMax H3 video, block by block, in Studio.** Since [Fizgig H3 Still](https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still), people have been curious how stills from H3 video compare with H3 Image's: H3 video understands motion, while H3 Image is cleaner but tends to render action frozen and pin-sharp. This mode lets you find out for yourself. Choose MiniMax H3 Image in Studio, set **Studio mode** to **Frankenstein** and press Start: each block slider becomes how much of that block comes from H3 video (0 = H3 Image as it is, 1 = all H3 video). Baseline is H3 Image on its own, Tweaked the blend. No LoRA is involved: it works on the two models themselves, using the H3 video file and H3's Turbo LoRA set in Preferences.

![Studio mode: LoRA or Frankenstein, under the model picker](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/assets/frankenstein/studio_mode.png)

**A starting point: ✨Motion.** I swept every block across five motion scenes at 2560 × 2560 (a tennis ball through a window, a car drifting into crates, a wine glass knocked over, a skateboarder grinding a rail, a horse through a river). H3 video's motion seems to live in about blocks 26-36, and **✨Motion** (H3 video's blocks 27-28 and 32-34 at 0.65) does a good job of bringing some of it across: motion blur on fast things, spray, smoke, a curtain whipping in, while keeping H3 Image's look. **✨Motion, gentle** (28-34 at 0.6) is the lighter one. Blending a lot at once tends to break the picture, worst into the last blocks (38-49), so 🎲 I'm feeling lucky rolls from 0 to 0.6 in this mode (raise the limits to go further).

**There's a lot more to find.** My sweep only scratched the surface, and there are surely other interesting combinations of the two models out there: other blocks, other mixes, other kinds of pictures. If you find one, please share it. This mode is here for now, to experiment with; I probably won't keep it in Fizgig for good.

**Save blend as a new base** writes a blend you like as a new H3 Image file (about 21 GB, int8, the original's own format) that ComfyUI and Fizgig load like the original, with the mix recorded in its metadata.

![Save blend as a new base model](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/assets/frankenstein/save_blend.png)

**What it needs: a 16 GB card or bigger, with 48 GB of system RAM (64 GB with a 16 GB card).** The mode always loads H3 Image at int8 and, with block swap on Auto, streams just enough blocks from system RAM for your card: none on a 32 GB card, 11 on a 24 GB card, 32 on a 16 GB card, all measured rendering ✨Motion at 2048 and 2560 and saving a new base. A streamed block blends in system RAM and renders exactly as one on the card; renders on a smaller card take longer. The [Studio guide](https://github.com/shootthesound/Fizgig/blob/master/docs/STUDIO.md#frankenstein-mode-minimax-h3-image) has the full figures.

**MiniMax H3 Image works in Studio on a 16 GB card.** With inference block swap on Auto, Studio streamed nothing on cards of 16 GB and up, so H3 Image's 21 GB model ran out of memory loading on a 16 GB card. When the model can't fit, Studio now streams just enough of it from system RAM; 24 GB cards and up are unchanged.

**Both of Studio's save boxes have a Folder… button** (Save repaired LoRA and Save blend as a new base): it picks where the file goes and keeps its name. Save blend shows a progress bar while the new base is written, then where it went.

**Flux 3 trains on a 16 GB card at bf16.** A text encoder's memory wasn't always freed when it was unloaded (Python held on to it until its collector happened to run), and Flux 3's kept 6.8 GB on the card for the whole run, enough to run out of memory loading the model. Every model's text encoder is now freed as soon as it's unloaded.

**Block swap fixes.** Swapping more than half the blocks with a Batch size above 1 crashed at the first step; Klein with a block swap of 1 crashed at the first step; and MiniMax H3 Image's Auto block swap now counts the two card slots its streamed blocks pass through, so a 12 GB card plans one more block.

**AMD: MOSS-Image, Flux 3, Fibo 2, Z-Image and Vimoo previews decode on ROCm** (#383, thanks @ongziheng).

**Queue several runs in a row:** after Add to queue, the form stays as your setup with Add to queue beside Start, so the next run is a tweak away, and the LoRA name follows the training folder (#386, thanks @kvsh88).

**Choose… opens Fizgig's own chooser on Windows again**, with buttons straight to your training data, trained LoRAs and the Fizgig folder; Preferences can switch it to the computer's own Open window (#387, thanks @axymeus).

**Vimoo downloads its Glyph-ByT5 text encoder again**, from its original source, after the files it used were taken down.

**Studio and the picture viewer.** With no LoRA, the render shows whatever the arrows between the panes were left on, and Keep in view scales it to the panel; Studio opens with both panes each time. Pictures zoom: scroll zooms about the cursor (up to 4× the picture's pixels), drag pans, double-click switches between fit and 1:1, in Studio, the Samples gallery and Captions. The Captions tab's instruction box names the captioner you picked.

**To update:** run the updater as usual.
