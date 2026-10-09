# Fizgig v7.2.0: Qwen Image 2.1 Turbo

Qwen's own Turbo for Qwen Image 2.1 came out a few hours ago, and Comfy-Org has published it as a single file plus a matching Turbo LoRA. Fizgig v7.2.0 uses both: Repair Studio, LoRA the Explorer and LoRA Royale preview on the Turbo model, and training previews render with the new Turbo LoRA, both at 8 steps with no CFG.

Training stays on Qwen Image 2.1, the base model, which is where it's best done: Turbo is for previews and for generating. LoRAs you train on the base work on Turbo as they are. In our tests with a trained character LoRA, the likeness on Turbo was better than on Qwen Image 2.1, and it renders about four times faster:

| Previews on | Steps / CFG | Time per 1024 image (RTX 5090) |
|---|---|---|
| Qwen Image 2.1 | 25 / CFG 3 | ~22 s |
| Qwen Image 2.1 Turbo | 8 / CFG 1 | ~5 s |

Crucially, Turbo performs at CFG 1 as well as base did at CFG 3 - so the issues that base had are resolved.

## Headlines

- **The workbench previews on Qwen Image 2.1 Turbo.** Repair Studio, LoRA the Explorer and LoRA Royale render on the Turbo model by default once it's set in Preferences: 8 steps, CFG 1, Turbo's own schedule. It's the 7.3 GB int8 file, half the size of the bf16 one, and Fizgig runs it as it is.
- **A new Turbo LoRA for training previews.** Comfy-Org's Qwen Image 2.1 Turbo LoRA replaces Viggle's. With it, the Samples tab's default for Qwen is strength 1, 8 steps, CFG 1.
- **Training samples on the Turbo model, if you want them.** A new tick on the Samples tab, **Use the Turbo model for samples**, renders each epoch's previews on the Turbo model instead, with the model being trained parked beside it. It's off by default. Measured with no out-of-memory errors on 12, 16 and 24 GB cards; 12 GB cards stream part of the Turbo model, which makes each preview round slower.
- **The Samples tab remembers each model's settings.** CFG, steps, size and flow shift are now kept per model across restarts, like the negative prompt already was. Before, a restart put each model back on its defaults.

## Getting the new files

- **Updating:** `update_fizgig.bat` downloads the new Turbo LoRA (910 MB) and points Qwen Image 2.1's Turbo LoRA row at it. Viggle's LoRA stays on your disk; delete it if you no longer use it.
- **The Turbo model** (7.3 GB) is too big for the updater to fetch unasked, so if you use Qwen Image 2.1 the updater ends with a reminder: open Fizgig, and on the Preferences tab press **Download models for me** in the Qwen Image 2.1 section, or point the **Qwen 2.1 Turbo DiT** row at your own copy of `qwen_image_2.1_turbo_int8_convrot.safetensors` from [Comfy-Org/Qwen-Image-2.1](https://huggingface.co/Comfy-Org/Qwen-Image-2.1). New installs get both files with the rest of the Qwen download.
- **Until the files are there,** previews work as before: the plain model at 25 steps, CFG 3, and the Turbo tick stays greyed out.

Nothing else is needed for offline use: Turbo shares Qwen Image 2.1's text encoder and tokenizer.

## Qwen Image 2.1 preview defaults

| Where | With the Turbo files | Without them |
|---|---|---|
| Training previews | Turbo LoRA, strength 1, 8 steps, CFG 1 | plain model, 25 steps, CFG 3 |
| Training previews, Turbo tick on | Turbo model, 8 steps, CFG 1 | — (tick greyed out) |
| Repair Studio, Explorer, Royale | Turbo model, 8 steps, CFG 1 | the Samples tab's settings |

Settings you've typed on the Samples tab are never replaced by these defaults.

## Also in this release

- **MiniMax H3 trains long captions on cards older than the RTX 30 series, such as the V100.** The text encoder processes long captions in chunks there, and the model's attention takes a lighter path, so captions that ran out of memory on those cards now train. RTX 30 and newer cards are unchanged. By @Hell-Bent-Fox (#172), tested on 4×V100.
- **Batch Size stays at 1** on the Training tab, with a hint to use Gradient Accumulation for a larger effective batch. Every model trains one image at a time, and a higher value stopped some runs at the start. Reported by @twe00074 (#181).
- **Each model keeps its own LoRA output folder.** Switching models no longer saves one model's folder under another, and Browse opens in the folder the field points at.
- **Pause works with its pop-up still open.** A run that reached its pause while the "Pause Requested" message was still on screen ended instead of pausing; it now pauses.
- **Each model keeps its own Turbo strength** on the Samples tab; switching models could carry one model's value to another.
- **A tip for large datasets:** a standard LoRA at a learning rate of 2e-4 or more, on more than 125 steps an epoch, gets a card beside Min / Max LR suggesting a cooler range, with a one-click button, so the best point doesn't fall between two saved epochs.
- **Override next sample** is ignored on slider runs, whose previews always show the slider's own prompt.
- **Z-Image Turbo's Repair Studio** starts at 768, its default size, instead of 1024.
