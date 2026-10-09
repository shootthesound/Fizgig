# Qwen Image 2.1

[← Back to the README](../README.md)

Qwen Image 2.1 trains LoRAs and LoKRs in Fizgig, including edit LoRAs learned from before/after pairs, with turbo previews during training, the per-image loss watch, and all five workbench tools (Repair Studio, LoRA the Explorer, Profiler, Extract, LoRA Royale). Saved LoRAs, LoKRs, Repair Studio saves and Extract outputs load in ComfyUI's standard LoRA loader. It's the first model on Fizgig's driver system.

## Getting set up

1. On the **Preferences** tab, press **Download models for me** in the Qwen Image 2.1 section. It fetches the DiT, VAE, text encoder, the Fizgig training adapter, the Turbo LoRA and the Turbo DiT (about 41 GB), plus Krea 2's Qwen3-VL-4B captioner (about 5 GB) if you don't have it, and the tokenizer files so training works offline.
2. On the **Training** tab, pick **Qwen Image 2.1** as the Base Model. The Fast preset loads on your first visit.

## The Fizgig training adapter

Qwen 2.1 LoRAs tend to collapse into texture or wobble part-way through training, and a lower loss doesn't warn you. The training adapter fixes this: a small LoRA that stays frozen and active while you train, switched off for previews and never saved into your LoRA, so the LoRA works on the plain model.

Fizgig's adapter is trained at a higher resolution than the existing Qwen 2.1 training assistant, and LoRAs trained with it come out much sharper. It's on in every Qwen preset and downloads with the models. It's also on Hugging Face for any trainer: [ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter](https://huggingface.co/ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter).

## Presets

Qwen renders very sharp images. A LoRA pulls fine detail such as skin texture toward your dataset, so a long run on softer photos trades Qwen's sharpness for theirs. Training faster keeps more of it, and in our testing **0.5 MP is the sweet spot**: quicker than 1 MP, and it keeps more sharpness than 0.25 MP.

All three presets train at 0.5 MP with adamw8bit, EMA 0.98 and the training adapter, for 30 epochs with every epoch saved:

| Preset | Rank | Learning rate | For |
|---|---|---|---|
| **Qwen 2.1 Fast** (default) | 8 | Adaptive LR 2e-4 to 4e-4 | Most subjects. The quickest to likeness in our tests, and the best at holding skin detail. |
| **Qwen 2.1 Standard** | 16 | Adaptive LR 1e-4 to 2e-4 | Bigger or mixed datasets. Rank 16 at Fast's rates overtrains. |
| **Qwen 2.1 Style** | 16 | Flat 1.5e-4 | Styles. Adaptive LR climbs on style datasets, which is where styles overbake. |

If a later epoch looks softer than you'd like, an earlier one is often the better pick; scrub them in LoRA Royale.

### 0.5 MP is not 512×512

0.5 MP is half a million pixels, about 704×704 for a square image. Fizgig buckets each image by its shape at that pixel count:

| Aspect | Training size |
|---|---|
| 1:1 | 704×704 |
| 4:5 | 624×784 |
| 2:3 | 576×848 |
| 9:16 | 528×928 |
| 3:2 | 848×576 |
| 16:9 | 928×528 |

512×512 is 0.25 MP, the lowest option in the Target MP box.

## Memory: 10 GB and up

**Base precision: Auto** picks bf16, INT8 or 4-bit NF4 at launch and sizes Blocks Swap to match, from your free VRAM and training resolution. It picks the most precise option that fits and quantises before it swaps, because swapping costs far more speed. At the presets' 0.5 MP:

| Card | Auto picks |
|---|---|
| 24 GB and up | bf16 |
| 12–16 GB | INT8, no block swap |
| 10 GB | 4-bit NF4 |

The text encoder only encodes, and loads in 8-bit below about 20 GB free (about 8 GB in all), which is what sets the 10 GB floor. On an RTX 5090 limited to 12 GB, the Fast preset trained on INT8 with a 9.9 GB peak, previews included; limited to 10 GB, on NF4 with a 7.3 GB peak.

## Previews and Turbo

Training previews render with the **Turbo LoRA** at strength 1, **8 steps, CFG 1** (both set in Preferences; the download button fetches them). Ticking **Use the Turbo model for samples** on the Samples tab renders them on the Turbo DiT instead, with the model being trained parked beside it. Without the Turbo LoRA they render on the plain model at **25 steps, CFG 3**. The training adapter is off for previews; a Context LoRA stays on.

Repair Studio, LoRA the Explorer and LoRA Royale preview on the Turbo DiT (8 steps, CFG 1) when it's set; without it they follow the Samples tab.

## Edit LoRAs

Qwen Image 2.1 is one model for text-to-image and editing, so a LoRA can learn an edit: a grade, a look, a relight, a retouch. You train it on pairs of the same photo, the original and your edited version, and it learns to make that edit to new photos.

A film grade trained on 40 pairs: the original, the edit, and Fizgig's epoch 9 training preview.

![Before edit](../assets/qwen_edit/1_before_edit.jpg)
![After edit](../assets/qwen_edit/2_after_edit.jpg)
![Fizgig's epoch 9 preview](../assets/qwen_edit/3_fizgig_epoch9_preview.png)

**What you need:** about 40 pairs (20 at least; more if your photos vary a lot). Each photo 1 MP or larger, e.g. 1200×800; bigger is fine, Fizgig resizes them. An original and its edited version must have the same crop and shape.

1. On the Training tab, pick an Edit preset (each ticks **Edit LoRA** under Training Parameters): **Qwen 2.1 Edit** (rank 8, Adaptive LR 2e-4 to 4e-4, 12 epochs) for most edits, or **Qwen 2.1 Edit Strong** (rank 16, Adaptive LR 1e-4 to 2e-4) for trickier ones.
2. **Originals folder (before editing):** your original, unedited photos. They need no captions.
3. **Edited folder (after editing):** the same photos after your edit, with the same file names as the originals (`IMG_0001.jpg` in both), one edited photo per original. This is the same folder as on the Start tab.
4. **Captions for the edited photos:** type what the edit is, e.g. "Apply my concert grade.", and press **Write captions**. It saves that text as the caption of every photo in the Edited folder.
5. **Test photo for previews (optional):** an original photo that is in neither folder. The previews during training show the edit applied to it. Any size: it's fitted to the preview size automatically. Left empty, previews use the first original. Edit previews use your edit instruction as their prompt; the Samples tab's prompts aren't used while Edit LoRA is on.

Start refuses to run if an edited photo has no original with the same file name, has no caption, or has a different crop or shape from its original.

In ComfyUI, load the LoRA as usual and use **Text Encode Qwen Image 2.1**: connect the VAE, plug the photo to edit into its first image input, write the instruction as the prompt, and sample from the node's **latent** output so the result keeps the photo's shape. The node's **resolution** works best near the size the LoRA trained at: in our tests an edit LoRA trained at 0.5 MP matched its Fizgig previews at resolution 768 and came out slightly less exact at the default 1024.

In our tests, 40 pairs of a colour grade at 0.59 MP with rank 16 learned the grade on held-out photos within 6 epochs, about 8 minutes on an RTX 5090. Faces, poses and framing came through unchanged. Turbo previews score the same as 25-step ones, so they are a fair guide to an edit LoRA too.

## Slider LoRAs

A slider LoRA is a dial between two looks: strength +1 moves a picture toward one look (happy, warm), -1 toward the opposite (sad, cool), and anything in between gives a bit of either. Sliders are trained at +1 and -1, but most will also go further, so try 1.5 or 2 (or -1.5, -2) for a stronger effect; how far a slider goes before the picture breaks down varies from one to the next.

On the Training tab, pick **Kind of LoRA: Slider** under Training Parameters, or load **✨ Qwen 2.1 Slider (rank 8, 2e-4)**, which selects it (rank 8, learning rate 2e-4, 30 epochs). Then choose where the two ends come from.

**From photo pairs:** two folders of the same shots, one for each end of the dial. 4 to 10 pairs is enough.

1. **+1 end folder:** the photos at the +1 end, e.g. smiling. This is the same folder as on the Start tab, and the captions go here.
2. **-1 end folder:** the same shots at the -1 end, each with the same file name as its +1 photo. No captions.
3. **Captions:** describe what the two photos of a pair have in common and leave out the difference, e.g. "a portrait photo of a woman", not "a smiling woman". **Write captions** saves it as the caption of every +1 photo.

Frame each pair the same way. Handheld shots from the same spot are fine; what matters is that the only change every pair has in common is the one you want the dial to learn. Start refuses to run if a photo has no partner with the same file name, has no caption, or has a different shape from its partner.

**From prompts:** no photos. Fizgig renders its own practice pictures from your description and trains on them.

1. **What the picture is:** the start of the prompt. Each end's words are added after it with a space, so end it with a comma if you want one.
2. **The +1 end adds** and **The -1 end adds:** e.g. "happy" and "sad".
3. **Push strength:** how hard the two ends are pushed apart. 2 is a good start (values 2 to 9 tested); higher gives a stronger dial but can change more than the one thing you asked for.

Describe one person in the first line ("a close-up photo of a young woman with short dark hair,") and the dial changes her expression and keeps her the same. Keep it general ("a close-up photo of a person who is") and each practice picture shows a different person, so the dial learns only the change and works on anyone. Words like "sad" can bring more than a face (grey light, rain); if the whole scene changes, try a narrower word such as "unsmiling".

**Previews** are strips of three pictures on one seed, at strength -1, 0 and +1. They use the slider's own prompt (the shared caption, or the "What the picture is" line); the Samples tab sets the seed, size and steps.

Sliders are always a plain LoRA (a LoKR setting is switched to LoRA for the run), and Adaptive LR, EMA and the per-image loss watch are off for them. A Context LoRA works: the dial is learned on top of it, so use the slider with the same context LoRA at the same strength. In ComfyUI, load the slider like any LoRA and set its strength.

## Licence

Qwen Image 2.1 is released under the Qwen Research License: non-commercial use only unless you get a commercial licence from the Qwen team, and a LoRA or fine-tune you share must say "Built with Qwen" or "Improved using Qwen". Read the [licence](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) before publishing or selling anything made with it.

## The per-image loss watch

The four Training-tab toggles work on Qwen as on Krea 2 (see [Krea 2](KREA2.md)): problem-image detection, per-image LR, look-outlier warm-up, and auto-recaption. Auto-recaption uses the same Qwen3-VL-4B captioner as the Captions tab. Picking Automagic as the optimizer turns Adaptive LR, per-image LR and the look warm-up off, since Automagic sets its own rate; detection keeps running.

## Command line

Qwen trains headless through `src/fizgig/families/cache.py` and `src/fizgig/families/train.py` with `--family qwen_image21`. See [the CLI guide](CLI.md#qwen-image-21-training) for a full example and the flags.
