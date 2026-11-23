# Fizgig v9.8.6: MiniMax H3 Image reference Strength and Speed boost, edit references at their own shape, and a Pillow security update

**MiniMax H3 Image: reference Strength in Studio.** The Strength box beside a reference picture now works on H3 Image, as it does on Klein and Flux 3: 1 uses the picture as it is, lower values steer the render less, and 0 leaves it out. LoRA Royale's sequential reference uses it too.

**MiniMax H3 Image: Speed boost.** On the Samples tab and in Studio, Speed boost On renders the first steps at half size and the rest at full size: 2 steps or a third of them, whichever is more. Off, the default, renders as before.

**Edit models' reference pictures keep their shape.** Studio and LoRA Royale cropped an edit model's reference picture to the render's width and height. It is now used whole, in its own shape, at the Reference MP, and Studio's Reference MP box is no longer greyed for edit models. This covers Klein, Flux 3, Qwen Image, Fibo 2 and MiniMax H3 Image. Training previews were never affected.

**Pillow 13.0.0.** A security fix (GHSA-g4qv-2jxx-769h): a crafted PNG could make reading its EXIF take minutes. ([#398](https://github.com/shootthesound/Fizgig/pull/398), thanks @spaghettified26)

**The Captions tab opens much faster on PNGs.** Fizgig read a PNG's size by decoding the whole picture; it now reads the file's headers, so a folder of a few hundred PNGs opens in a fraction of a second instead of several. ([#399](https://github.com/shootthesound/Fizgig/pull/399), thanks @spaghettified26)

**Studio outputs keep their setup.** Every still saved to the Studio outputs folder, the original as well as the tweaked one, now carries the setup it was rendered with, so dropping either on Studio restores it. Before, a render only kept it when that tab's Save every render box was ticked, and originals never did.

**Studio in two tabs.** A tab another tab has loaded over no longer shows the other tab's loading, progress or pictures as its own. It says another tab has loaded over it, keeps its own pictures, and its Cancel can't stop the other tab's render. ([#395](https://github.com/shootthesound/Fizgig/issues/395), thanks @tsk5cb-cmyk)

**A folder of clips only trains on MiniMax H3.** Start refused it with "Masked training is on" when the tick had been left on from a photo folder, though the section is hidden for clips. ([#402](https://github.com/shootthesound/Fizgig/issues/402), thanks @roadmaus)

**Caching works offline.** Anima's and ACE-Step's tokenizers and SDXL's config load from the downloaded files first, so caching no longer needs the internet once a model is downloaded. The updater fetches any of these helper files that are missing. ([#397](https://github.com/shootthesound/Fizgig/issues/397), thanks @axymeus)

**To update:** run the updater as usual.
