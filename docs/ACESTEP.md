# ACE-Step 1.5 XL (music)

Train a LoRA on your own music and ACE-Step writes new songs in that sound: instrumentals or songs with lyrics, 48 kHz stereo. Fizgig trains on the XL Turbo model by default (8-step previews) or the XL SFT model (30 guided steps, slower, the most detail), and LoRAs load in ComfyUI as they are.

## Your songs

Put the songs in one folder, each with a caption file beside it:

| File | What it holds |
|---|---|
| `song.wav` (or .mp3, .flac, .m4a) | The audio: a whole song of up to 4 minutes, no cutting needed. A longer song trains on its first 4 minutes. |
| `song.txt` | The caption: a trigger word, then the style, e.g. `fzgchip, chiptune, 8-bit, instrumental, square wave lead, arpeggios` |
| `song.lyrics.txt` (optional) | The lyrics. Leave it out for an instrumental. |
| `song.json` (optional) | `bpm`, `timesignature`, `keyscale`, `language`. |

Pick songs that share one sound: one artist, one album or one tight genre. The model learns what the songs have in common.

Two tools help:

- **Caption Every Song** on the Captions tab writes the same caption beside every song: your trigger word, then the common words you type (genre, instruments, mood, vocals). Songs that already have a caption are left alone unless Overwrite is ticked.
- **Gizmo's Music tab** cuts songs into sections, or keeps them whole. Drag the section's ends on the waveform (10 seconds to 4 minutes) and listen to it. Each section exports as a 48 kHz stereo wav with its caption, its lyrics and the song's `.json`. The cuts get a short fade-in and fade-out. Lengths can be mixed in one dataset; longer sections carry more of each song.

## A tested recipe

**10 songs of 2 to 3.5 minutes each, one album, all instrumental, the same caption on every song.** Train with the **✨ ACE-Step Official LoRA (rank 64, 800 epochs)** preset: rank 64, alpha 128, learning rate 1e-4, EMA 0.98, a checkpoint and a preview every 40 epochs. Keep EMA on: the same 800-epoch run sounds far better with it. 800 epochs sounds like a lot, but a song set is small: the whole run took about 2 hours on an RTX 5090, previews included.

The dataset we tested it on, and the LoRA it made, are free to download: the [Fizgig ACE-Step Chiptune Starter](https://huggingface.co/datasets/ShootTheSound/Fizgig-ACE-Step-Chiptune-Starter) (Eric Skiff's *Resistor Anthems*, CC BY 3.0). Train it for your first run, or load its LoRA at strength 0.5 to hear where you're heading.

What to expect:

- **The previews get worse before they get better.** As the model moves from what it started with towards your style, the quality drops, sometimes badly. It can sound like overtraining, but it isn't: once it has moved over to the style it starts to improve again. With the starter dataset the best checkpoints came between epochs 740 and 800.
- **The late epochs hold the detail.** The timbre, the playing style and the production keep sharpening for hundreds of epochs.
- **Strength 0.5 works really well.** Try anything between 0.5 and 1, but a long-run LoRA tends to work better towards 0.5: towards 1 it leans closer to the training songs, and around 0.5 you keep the learned detail and get new music.

The Training tab shows this as a hint once Max Epochs is 400 or more.

## Previews and the workbench

Sample prompts are songs: `caption | lyrics`, with lyric lines separated by ` / `, e.g.

```
fzgchip, chiptune, 8-bit, upbeat, robotic vocals | [verse] / Jump jump, coins in the air / [chorus] / Power up, press play
```

Leave out the lyrics for an instrumental. The Samples tab sets the preview length (60 seconds by default).

![Repair Studio with a song LoRA and a donor](assets/ace_repair_studio.png)

Repair Studio plays songs in place of pictures: a baseline, the tweaked version and the song with no LoRA, side by side, keeping your place when you switch. Its built-in mixes (**✨Demo Slider Adjustment**, **✨Early Epoch Improver**, **✨Make it more full**) are good starting points. Extract reduces a LoRA's rank as it does for every model.

## Sliders

A slider is a LoRA whose strength is a dial between two kinds of music: +1 one way, -1 the other, 0 the song as it was. Pick **Slider** as the kind of training (or the **✨ ACE-Step Slider (rank 8, 20 epochs)** preset) and fill in three lines:

1. **What the music is**, e.g. `instrumental, synth pop,`
2. **The +1 end adds**, e.g. `energetic, fast tempo, loud drums`
3. **The -1 end adds**, e.g. `calm, slow tempo, soft`
4. **Leave these alone** (optional): captions of music the dial must not change, one a line, e.g. `instrumental, solo piano, classical` for a heavier-guitar dial meant for rock. The previews play the first one beside the dial, so you can hear it stay put. **How firmly** (default 1) sets how much that counts against the dial.

No songs are needed: Fizgig teaches the dial from the model's own songs. Each checkpoint's preview plays -1, 0 and +1 one after another. Change one thing per slider: words that also change the genre or the instruments turn the dial into a genre switch.

While it trains, the Training tab's **Slider prompts** panel lets you change the three lines, the push strength and the captions to leave alone: they take over at the next epoch, from where it has got to.

## Full fine-tuning

The **✨ ACE-Step Fine-tune** preset trains the whole model (both attentions and the MLP of all 32 layers) a part at a time on a 4-bit base, at 1e-5, for 800 rotations with a checkpoint and a preview every 50. On a 32 GB card a rotation is one epoch; smaller cards split the model into more parts, so a rotation takes several. Each checkpoint is a complete model file of about 10 GB that ComfyUI loads like the original, so the full run needs about 160 GB of disk.
