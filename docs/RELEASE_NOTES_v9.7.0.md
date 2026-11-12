# Fizgig v9.7.0: top-end likeness for MiniMax H3 Image

**MiniMax H3 Image now trains likeness at the very top end.** On our hardest test dataset (few headshots, mostly wider shots), the best previews look more like the person than all but two of their own 45 photos do, scored against the same reference photos. That held on two more datasets, one harder still. It comes from a day of side-by-side runs, and it's all in the Standard preset: load **✨ MiniMax H3 Image Standard** to pick it up.

- **Likeness optimization has a new schedule.** Of every schedule we tried, it brought the person's face shape through earliest, reached the highest likeness, and kept backgrounds from the training pictures out of the results. Its controls are under **Advanced**, inside the Likeness optimization box, closed by default; most people won't need them.
- **Automagic v4**, starting at 2.5e-4, which beat Automagic v3 and Muon by a wide margin in our tests, most of all on a dataset with few headshots.
- **0.25 MP and 30 epochs.** In our tests the best epochs came from the late teens to the late twenties.

**Why back to 0.25 MP.** In 9.6.1 we moved the H3 Image presets up to 0.5 MP, because the model seemed to prefer it. It didn't: what was holding likeness back was the training schedule and the optimizer. With the new schedule and Automagic v4, 0.25 MP reaches the likeness above, and it trains faster on less VRAM. So the presets are back at 0.25 MP.

**Start from the preset, and leave its settings as they are.** Pick **✨ MiniMax H3 Image Standard** (0.25 MP) and you'll see Likeness optimization ticked. The numbers under **Advanced** took a lot of runs to find, so train with them as they are first. Once you've seen what the preset gives you, they're there to experiment with.

**Likeness scores now come from AdaFace ViT instead of ArcFace.** We scored those runs epoch by epoch with ArcFace and two AdaFace models side by side, and AdaFace ViT's best epochs matched the ones we picked by eye every time; ArcFace's often didn't. It now scores likeness everywhere: the Look Consistency Filter, the Samples gallery, LoRA Royale, Studio and the Profiler. It still runs on the CPU, so training keeps the GPU. Scores saved with ArcFace are scored again with AdaFace the first time you open them, and its match / borderline bands are set to keep the same share of real photos of a person as ArcFace's did.

**"Your photos" on the likeness chart.** The Samples gallery now scores your own training photos against the same 3 reference photos and shows where they land: a green band on the likeness chart with its numbers beside it (the middle half of your photos, and their median as a dashed line), and the same in words under it. That's the realistic top: real photos of a person never score 100% against each other, and a preview in that band looks as much like them as a real photo does. The run line says whether your best epoch is within, level with or above your photos, and it now shows the average likeness over the run beside the best epoch.

**Element boost needs Target Megapixels 0.75 or more.** Below that, a face that's small in the frame is only a few pixels across, and boosting it teaches the LoRA pixelation, so the tick greys out and says why. MOSS-Image's 0.25 MP presets (Fast and Turbo Train) no longer turn it on; Standard (1 MP) keeps it.

**Also in this release:**
- **Automagic v4's learning rate stops at 2.5e-4.** Left alone it could climb well past what trains cleanly. Set `max_lr=` in Optimizer Args to change it.
- **The Samples gallery scores only the run you're looking at**, not every run's previews in the samples folder, and a new preview is scored straight away even while older ones are still being scored.
- **MiniMax H3 Image training maps work with Compile Blocks on.** They were switching themselves off at the start of the run.
- **Smaller face downloads.** Fizgig now fetches only InsightFace's face finder (18 MB) instead of its whole 300 MB pack, plus AdaFace ViT (460 MB). The updater fetches them, along with Make masks' person segmenter and your models' tokenizer files, so they're ready before you need them; anything missing still downloads the first time it's used.

**To update:** run the updater as usual. The first time you open the Samples gallery afterwards, it scores the run you're viewing again with AdaFace, which takes a minute or two on the CPU.
