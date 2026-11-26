# Fizgig v9.8.8: auto-recaption corrects your caption, clip captions from several frames, and Royale renders your starred epochs

**Auto-recaption corrects your caption first.** When the per-image loss watch recaptions a stuck picture, its first try now starts from the caption you wrote: it keeps your words, names and style terms and changes only what the picture doesn't show. The second try still writes a new caption. It's on by default; untick **First try corrects the caption you wrote** on the Training tab to start over each time, as before. The caption you had before auto-recaption is kept beside your pictures, and Edit caption in the Problem Images window offers it back with **Put it back**. ([#244](https://github.com/shootthesound/Fizgig/issues/244), thanks @bkrew)

**Clip captions from several frames.** Captioners were handed only a clip's middle frame, so motion and anything that changes over a clip was missed. For models that train on clips, the Captions tab has a **Frames per clip** row (1 / 4 / 8): Qwen3-VL is handed that many frames spread through each clip and describes what happens over it. 1 is the middle frame, as before. ([#351](https://github.com/shootthesound/Fizgig/issues/351), thanks @CognitiveDiffusion)

**LoRA Royale renders the epochs you starred.** The stars you put on previews in the Samples gallery now pick epochs in Royale: the crossfade's Max renders has a **Starred** choice, and the comparison sheet's **★ Starred** button fills the Epochs box with them. A starred epoch whose checkpoint isn't in the folder is named in the message. ([#401](https://github.com/shootthesound/Fizgig/issues/401), thanks @axymeus)

**Studio saves an Epochs of this run mix at its files' size.** Saving a mix put every file's rank side by side, so twenty rank-128 epochs saved as one rank-2560 LoRA of about 4.5 GB. A mix now saves at its largest file's rank, higher only where a mix of unlike LoRAs needs it to keep 99.9% of the change; the save message says the rank and the share kept. On a real 50-epoch run every layer came back to its own rank with at least 99.93% kept. A donor saves as before. ([#411](https://github.com/shootthesound/Fizgig/issues/411), thanks @CognitiveDiffusion)

**Continuing a run picks up edited captions.** A caption changed between runs, in the Problem Images window or on the Captions tab, trained as its old text after Continue. Its text is now cached again before the run resumes, with a line in the log. ([#409](https://github.com/shootthesound/Fizgig/issues/409), thanks @mart-hill)

**Studio says when each pane is rendering.** Baseline, Tweaked and No LoRA each show "Rendering… 45%" under the picture while it renders or the model loads, so you can see it with the sliders scrolled into view.

**Translate captions handles several captions per picture.** Each line of a caption file is translated on its own, trigger kept. Before, the first line stayed English and every line's Chinese landed on the last one.

**Speed-ups for your graphics card.** The Faster block swap tick shows as on after a page reload when it's on (runs always read it correctly). Each setting links to its section of the web UI guide, which says when to try it, the log line that shows it worked, and what it looks like when it's wrong for your machine.

**To update:** run the updater as usual.
