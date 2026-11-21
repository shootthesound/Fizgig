# Fizgig v9.8.3: Frankenstein mode, an experiment in blending MiniMax H3 Image with H3 video

**Frankenstein mode is an experimentation tool: it blends MiniMax H3 Image and MiniMax H3 video, block by block, in Studio.** Since [Fizgig H3 Still](https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still), people have been curious how stills from H3 video compare with H3 Image's: H3 video understands motion, while H3 Image is cleaner but tends to render action frozen and pin-sharp. This mode lets you find out for yourself. Choose MiniMax H3 Image in Studio, set **Studio mode** to **Frankenstein** and press Start: each block slider becomes how much of that block comes from H3 video (0 = H3 Image as it is, 1 = all H3 video). Baseline is H3 Image on its own, Tweaked the blend. No LoRA is involved: it works on the two models themselves, using the H3 video file and H3's Turbo LoRA set in Preferences.

![Studio mode: LoRA or Frankenstein, under the model picker](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/assets/frankenstein/studio_mode.png)

**A starting point: ✨Motion.** I swept every block across five motion scenes at 2560 × 2560 (a tennis ball through a window, a car drifting into crates, a wine glass knocked over, a skateboarder grinding a rail, a horse through a river). H3 video's motion seems to live in about blocks 26-36, and **✨Motion** (H3 video's blocks 27-28 and 32-34 at 0.65) does a good job of bringing some of it across: motion blur on fast things, spray, smoke, a curtain whipping in, while keeping H3 Image's look. **✨Motion, gentle** (28-34 at 0.6) is the lighter one. Blending a lot at once tends to break the picture, worst into the last blocks (38-49), so 🎲 I'm feeling lucky rolls from 0 to 0.6 in this mode (raise the limits to go further).

**There's a lot more to find.** My sweep only scratched the surface, and there are surely other interesting combinations of the two models out there: other blocks, other mixes, other kinds of pictures. If you find one, please share it. This mode is here for now, to experiment with; I probably won't keep it in Fizgig for good.

**Save blend as a new base** writes a blend you like as a new H3 Image file (about 21 GB, int8, the original's own format) that ComfyUI and Fizgig load like the original, with the mix recorded in its metadata.

![Save blend as a new base model](https://raw.githubusercontent.com/shootthesound/Fizgig/master/docs/assets/frankenstein/save_blend.png)

**What it needs: a 16 GB card or bigger, with 48 GB of system RAM (64 GB with a 16 GB card).** The mode always loads H3 Image at int8 and, with block swap on Auto, streams just enough blocks from system RAM for your card: none on a 32 GB card, 11 on a 24 GB card, 32 on a 16 GB card, all measured rendering ✨Motion at 2048 and 2560 and saving a new base. A streamed block blends in system RAM and renders exactly as one on the card; renders on a smaller card take longer. The [Studio guide](https://github.com/shootthesound/Fizgig/blob/master/docs/STUDIO.md#frankenstein-mode-minimax-h3-image) has the full figures.

**To update:** run the updater as usual.
