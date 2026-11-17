# Fizgig v9.7.3: cleaner H3 Image pictures with LCM, and Stop for a paused run

**H3 Image renders with LCM.** MiniMax H3 Image has a new **Sampler** choice, on the Samples tab for training previews and separately in Studio (which LoRA Royale and the Profiler share). The default is now **LCM, noise 1.1 → 0.9** instead of Euler: on text-to-image it gives much cleaner skin, hands and faces on the same prompt and seed, including the epoch 0 previews. It takes the same 6 steps in about the same time. Euler, the model's own, is still there. The ⓘ beside each gives the matching ComfyUI settings, so you can render your LoRA the same way there. The recipe comes from u/Obvious-Bad7548's post on r/StableDiffusion (#348, thanks @CognitiveDiffusion).

**Bigger H3 Image pictures.** The Samples tab's preview sizes go up to 3072 and 4096 for H3 Image (up to 16 MP, nearer the size the model was made for); 2048 stays the default, as a bigger preview needs more VRAM. Studio now starts at 1280 × 1280 (was 1024) (#348).

**Stop a paused run.** A paused run held the queue, and the only way past it was Resume, then Stop. While a run is paused, Stop now reads **Stop paused run**: it ends the pause so the queue can start. The run's saves are kept, so **Continue** on its card in Runs & outputs carries it on whenever you want (#348, thanks @CognitiveDiffusion).

**H3 Image's Likeness optimization, explained.** The ⓘ beside it explains the schedule and the share each epoch gets by default, and its rows are now worded like MOSS-Image's Clean-end share: **First share**, **Rising to**, **Until epoch**, **Then slides over**, **Second share** and **Alternate**. What they do is unchanged, and saved runs and presets keep their settings. If hands, faces or bodies come out broken, try an earlier epoch first: H3 Image learns fast, and a late epoch has usually gone past its best (#348).

**To update:** run the updater as usual.
