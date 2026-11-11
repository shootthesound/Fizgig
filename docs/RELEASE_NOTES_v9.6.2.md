# Fizgig v9.6.2: MiniMax H3 Image presets start at 2.8e-4

**MiniMax H3 Image's presets now start Automagic v3 at 2.8e-4**, up from 2e-4, based on early user feedback: it gave better previews at the presets' 0.5 MP. Automagic sets its own learning rate, but it mostly keeps the level it starts from and eases down slowly, so the start matters. At 1 MP or more it may well be fine too; if a run looks overcooked early, try a little lower. Load a ✨ preset to pick it up.

**Element boost: best on pictures of 1 MP or more where the boosted part is small in the frame.** On small pictures, a small face or outfit is only a few pixels wide, and boosting it can teach the LoRA that pixelation. Smaller sizes (0.25 MP and so on) are fine when the boosted part is large in the frame. This applies on MOSS-Image and MiniMax H3 Image, and the tick's hint now says so.
