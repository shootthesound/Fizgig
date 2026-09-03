# Fizgig v5.2.1

A maintenance release for 24 GB cards training MiniMax H3.

## 24 GB cards: clip previews clamp to 22 frames up front

On a 24 GB card the int8 base streams around 11 blocks, and that plan leaves clip previews roughly 4 GB — where the default 56-frame preview measurably doesn't fit. Until now the trainer found that out the hard way on every run: one failed 56-frame render, then the ladder stepped down to 22 frames and stayed there. It now reads it off the plan instead: when blocks are streamed, clip previews start at 22 frames, with one console line saying so and how to get the longer preview back (pick 22 on the Samples tab to make it the setting, or lower Target Megapixels). Resolution is untouched, sound is kept, and previews on 32 GB cards or the 4-bit base are unaffected.

Verified on a simulated 24 GB card: no OOM, no wasted render, 22-frame previews from epoch 0. If 5.2 felt like sample generation was maxing out your card, this is the fix.
