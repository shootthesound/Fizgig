# Fizgig v9.8.1: MiniMax H3 previews that start from your picture, and leaner Krea 2 on AMD

**MiniMax H3 previews can start and end on a picture.** On the Samples tab, **First frame** and **Last frame** take a picture each: every clip preview starts (and ends) on it, cropped to the preview's shape. A LoRA captioned for what happens next in a clip, an image-to-video dataset, finally shows what it does in its previews. Either can be left empty, and a still preview ignores them. On the **Reference (ref2va)** Training Base the same two rows become **Reference 1** and **Reference 2**: name them `<Picture 1>` and `<Picture 2>` in the preview prompt, as in Studio. The pictures go with a run you move to another computer, and to a previews helper (#376, thanks @axymeus).

**A previews helper sends back whole clips.** A helper rendering MiniMax H3 (or Vimoo) previews sent back only each preview's still, so the gallery showed the middle frame. The clip and its sound now come back with it.

**Krea 2 on AMD: INT8 fast inference is off by default.** On AMD cards the model's stored fp8 file renders Krea 2 previews about twice as fast with a quarter of the RAM (2048 × 2048 on an RX 9070 XT), so INT8 fast inference starts off there and its Preferences label says why. A choice you've already saved is kept; NVIDIA is unchanged (#380, thanks @xGnome).

**Krea 2's text encoder frees its memory without copying.** When Studio or training finished with Krea 2's text encoder, it was copied into system RAM just before being discarded. It's now released in place: on a 5090 the peak RAM during that step fell from 6.6 to 1.8 GB and it takes a quarter of the time, on every card (#380, thanks @xGnome).

**Studio remembers where you pick pictures from.** The Reference picker, every First / Last frame and Reference slot, and the metrics reference each open in the folder you last chose from, across restarts, even after clearing the box.

**Delete run can keep your LoRAs.** On a run's card, **Delete run…** now also offers **Delete, keep the newest 5 epochs**, beside Delete and Cancel: it deletes the run as Delete does, but its newest five epochs (all of them when there are fewer) stay on disk with the finished LoRA, ready to use. **Delete, keep all epochs** keeps every epoch's LoRA file and deletes only the saved states. The **Clear out runs** row has its own box at the top of the list.

**Fixes.**
- **Runs & outputs: Previews & checkpoints opens at once.** With runs saving into a big LoRA folder (a ComfyUI loras folder of thousands of files), it sat on "Loading…" for minutes while every file in the folder was read; now only that run's files are. A run's finished LoRA now shows in its list too.
- **MiniMax H3 Studio keeps its clip settings.** Changing the prompt or another Studio setting could drop H3's saved clip settings and frame photos, so they were gone after a restart.
- **Studio: a landscape First / Last frame or reference photo** no longer runs under the box beside it; a narrow window stacks the slot.

**To update:** run the updater as usual.
