# Fizgig v9.4.0: Maps in Studio, for any LoRA, and two new themes

**New: Maps in Studio.** The training maps' heatmaps now work on any LoRA Studio can load, one you didn't train included. Under the preview, pick a map and it's drawn over the Tweaked picture in a few seconds, then made again after every render:
- **Where the LoRA acts:** the LoRA as you've set it, against no LoRA. For a person LoRA it should gather on the face and body; spread over clothes or the background, the LoRA changes those too.
- **What your tweaks change:** move any slider and see where your change acts, against the Baseline.
- **Trigger word:** where the trigger word's attention goes (read from the LoRA, or type it and press Show). A made-up trigger word often lands on lettering in the picture, a sign or a logo, as well as the subject.
- **Blocks:** where each group of the LoRA's blocks acts, a button per group. On MOSS-Image, the Identity group (blocks 24–35) lights up the face while the last blocks work on edges and texture across the whole picture: the block map you'd otherwise read in the Profiler's numbers, drawn on the picture. A group that lights up the background or clothes is the one to turn down there.

All levels or one noise level, and Opacity, sit under the buttons. Picture models only; Trigger word on every picture model but SDXL. [Studio guide: Maps](https://github.com/shootthesound/Fizgig/blob/master/docs/STUDIO.md#maps)

**Two new themes: Cosmic Pigeon** (space navy, nebula purple, pigeon-neck green) **and Deep Space** (near-black, a sparing nebula violet), with glowing buttons and star-specked, shimmering progress bars (off when your system asks for reduced motion). From @Murf-Code's Fizgig Advanced. Pick them under Preferences → Theme.

**Updating:** run the updater as usual. Pods you use from your own Fizgig follow your computer's version when they're free.
