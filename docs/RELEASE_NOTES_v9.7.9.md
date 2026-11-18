# Fizgig v9.7.9: MiniMax H3 Studio twice as fast, long clips in pieces, Continue and Loop back

**MiniMax H3 Studio renders about twice as fast.** On the int8 base, Studio now runs the checkpoint's int8 weights through the int8 kernel ComfyUI uses, instead of converting them for every multiply, and each LoRA adds into the base's output in one pass. On a 5090, a 5-second 512 × 512 clip's six steps went from 24.9 s to 11.2 s. The int8 video VAE (`minimax_h3_video_vae_int8_convrot`) now loads, decodes on the same kernel, and is what Download models fetches. When a longer clip needs more room, a few more of the base's blocks move to system memory in a second or two instead of the model reloading. Training is unchanged; `FIZGIG_NO_INT8_LINEAR=1` before launch goes back to the old way.

**Clips longer than five seconds, In pieces.** Tick **In pieces** on the Clip row and a long clip is made as pieces of 5.2 s (or a length you type beside the tick), each carrying on from the end of the one before, with the joins blended and the lighting evened out. Memory stays at one piece's worth however long the clip. The prompt can change along the way: a line starting with a time, such as `8s: she stops and laughs at the camera`, is the prompt from that point on. The length list also has **Other…** for any length in seconds.

**Continue and Loop back.** Once a clip is rendered, the new **Then** row adds to it without rendering it again: type what happens next and press **Continue** for a new clip that starts on its last frame and goes on the end, or **Loop back** for one that also ends on its first frame, so the whole thing loops with the way back made new rather than played in reverse. Under the clips, **Loop** plays them on repeat or ping-pong.

**Smaller Studio additions.** **Cancel**, between Start and Unload model, stops the render that's going and keeps the model loaded. **⇄** swaps Width and Height, and **Aspect…** sets a shape from 1:1 to 21:9. On MiniMax H3, **Tweaked only** skips the Baseline clip while you explore, for half the render time.

**The big caption view, a line at a time.** Opening a picture big now edits its caption a line at a time, like the cards, so a caption with several lines is easy to read and change. Empty lines are dropped, and **Join lines** turns a caption that was wrapped onto several lines back into one (thanks @mart-hill).

**Fixes.**
- **Studio: some LoRAs failed with "expected mat1 and mat2 to have the same dtype".** A LoRA that also covers layers the base keeps at full precision (MiniMax H3's AdaLN layers) now renders.
- **Studio: a finished render always shows.** One failed status check stopped the page asking, so a render could finish and never appear.
- **Studio clips: changing the donor's strength no longer renders the Baseline clip again.** The donor is off in that clip.
- **MiniMax H3 Studio on 24 GB cards:** the text encoder no longer runs out of memory when the base steps aside for a new prompt.
- **Linux AMD installer:** finding the card works on systems without `rocminfo`.

**To update:** run the updater as usual.
