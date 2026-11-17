# Fizgig v9.7.6: LoRA Isolation learns on MiniMax H3 video, Royale renders part of a run, and Gizmo's crop box follows the mouse

**Fizgig LoRA Isolation and Leave the rest of the model alone learn with Automagic.** With Automagic as the optimizer (every MiniMax H3 and H3 Image preset), a run with Fizgig LoRA Isolation or Leave the rest of the model alone learned nothing. The pass that keeps the rest of the model as it was runs on every second step, and Automagic read that back-and-forth as overshooting and lowered its rate to nothing within two epochs. Such a run now trains with AdamW instead, at the Learning Rate box's rate (2e-4 when the box holds Automagic's tiny starting value), or Adaptive LR's when it's ticked, and Start says so. In our testing, a two-woman Isolation run on H3 that learned nothing before learned both women and kept them apart. Two subjects of the same class (two women, two men) take longer to train than one, so give such a run more epochs (#364, thanks @ofung30-sketch).

**Say what your subject is.** With Leave the rest of the model alone on, the Subjects card now shows **Subject 1 - what they are** (man, woman, dog...). It takes the trigger word's place in the captions that keep the model as it was, so captions that don't name the subject keep "a man walking" rather than just "walking". Left empty, it works as before (#365, thanks @Marhalt).

**LoRA Royale renders part of a run.** In the Epochs tool, under Max renders, **Epochs from ... to ...** renders just that stretch, around an epoch you liked: Max renders spreads its picks inside it, with both ends always in (#318, thanks @axymeus). A finished run's folder lists its epochs again: the run's final file and any Save now snapshots made it look like a folder of unrelated LoRAs, compared by file name. The final file now counts as the run's last epoch, and snapshots sit between the epochs they were taken in (#355, thanks @spaghettified26). With Single LoRA as the source, the Epochs tool stays in view, greyed, and says what it needs.

**Gizmo's crop box follows the mouse.** It is drawn from the corner you press, in any direction, and stops flush at the frame's edges, the bottom included. Drag inside it to move it, or drag a corner or side to resize it; the note under the preview says so. Any size works: the export fits the box to the 32-pixel grid, as it does the whole frame. Whole frame also keeps the full picture of a video with non-square pixels (a clip stored 720 wide but shown 640 wide came out stretched, with about 30 px cut off each side) (#367, thanks @axymeus).

**Pictures read as they look.** A photo with a colour profile (an iPhone's Display P3, or an Adobe RGB or ProPhoto export) is converted to sRGB as it is read for training, captioning and Image Prep, so it no longer trains duller than it looks. A 16-bit black-and-white picture no longer trains as pure white. A dataset you've already cached keeps its old colours until you delete its cache (in the cache folder set in Preferences).

**Captioning models can be stopped while they load.** The Captions tab, and the line under the VRAM meter, say whether a captioning model is downloading or loading into memory, and Stop or Unload cancels either. Before, a long load couldn't be stopped (#366, thanks @Aymenbayoss).

**Krea 2 pictures decode with less memory.** A Krea 2 picture decodes with about 2 GB less VRAM at 1024x1024, with identical pixels, in previews and Studio (#369, thanks @xGnome).

**Prompt sliders.** A MOSS-Image prompt slider trains with the Turbo LoRA (it stopped before training), and a slider prompt or anchor starting with "-" reaches the trainer as typed (#361, thanks @pastuh). Any other box starting with "-", such as a title, a negative or a trigger word, no longer stops the run at launch.

**Smaller fixes.**
- The Blocks Swap row only shows for models that can swap blocks (not SDXL, Anima or ACE-Step).
- Start reads captions saved as UTF-16 or Windows-1252 the way training does, warns when two pictures share a name (a.jpg and a.png train as one), and names a Samples box that isn't a number (a CFG of 3,5).
- Auto-recaption finds a camera's `.JPG` files on Linux pods, and the Captions tab lists `.avif` pictures.
- Move a run and helper shares work with a training folder named with brackets (`photos [v2]`).
- The public checks pass on any machine (#359, thanks @whitejt2).

**To update:** run the updater as usual.
