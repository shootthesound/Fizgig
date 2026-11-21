# Fizgig v9.8.3: Frankenstein mode - MiniMax H3 video's motion in H3 Image's stills

**Frankenstein mode in Studio blends two base models block by block: MiniMax H3 Image with MiniMax H3 video.** Many people prefer H3 video for stills because it understands motion; H3 Image is cleaner but renders action frozen and pin-sharp. Choose MiniMax H3 Image in Studio, tick **Frankenstein mode: blend in MiniMax H3 video** and press Start: each block slider becomes how much of that block comes from H3 video (0 = H3 Image as it is, 1 = all H3 video). Baseline is H3 Image on its own, Tweaked the blend. No LoRA is involved: it works on the two models themselves, and uses the H3 video file and H3's Turbo LoRA set in Preferences.

**Start from ✨Motion.** We swept every block across five motion scenes at 2560 × 2560 (a tennis ball through a window, a car drifting into crates, a wine glass knocked over, a skateboarder grinding a rail, a horse through a river). The motion lives in about blocks 26-36: **✨Motion** (H3 video's blocks 27-28 and 32-34 at 0.65) brings back motion blur on fast things, spray, smoke and a curtain whipping in while keeping H3 Image's look, and **✨Motion, gentle** (28-34 at 0.6) is the lighter one. Blending a lot at once breaks the picture, worst into the last blocks (38-49), so 🎲 I'm feeling lucky rolls from 0 to 0.6 in this mode (raise the limits to go further).

**Save blend as a new base** writes the blend as a new H3 Image file (about 21 GB, int8, the original's own format) that ComfyUI and Fizgig load like the original, with the mix recorded in its metadata.

**What it needs: a 32 GB card (RTX 5090 class) and 48 GB or more of system RAM.** The blend is rewritten on the card, so the whole of H3 Image stays there with no block swap: measured on an RTX 5090, it peaks at 23.8 GB on a slider change and 24.9 GB rendering at 3072 × 3072, and the first render's text encoder takes Fizgig's RAM to 31 GB for a moment. A 24 GB card runs out of memory at the first blend. The [Studio guide](https://github.com/shootthesound/Fizgig/blob/master/docs/STUDIO.md#frankenstein-mode-minimax-h3-image) has the full figures.

**To update:** run the updater as usual.
