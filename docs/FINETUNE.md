# Full fine-tuning (experimental)

[← Back to the README](../README.md)

Fine-tuning trains the **base model itself** — no adapter, no rank bottleneck — on a single consumer GPU, for **Flux 3 Image Klein**, **Fibo 2.0**, **Krea 2**, **Qwen Image 2.1** and **MiniMax H3**. On Flux 3, Krea 2 and Qwen, pick **Fine-tune the whole model** under **Kind of training** on the Training tab; on MiniMax H3, tick **⚗ Fine-tune the BASE MODEL instead of training a LoRA**. The window plan sizes itself to your card either way.

For step-by-step answers, including five-minute recipes for both families, see **["How do I…?"](FINETUNE_HOWDOI.md)**.

> **Status.** It works and the numbers below are measured, but not every scenario is covered yet; field reports on GitHub shape what gets built next. The technique is model-agnostic, and extending it to other models depends on community support for those models — code, PRs and testing. — Peter

## Epochs and cycles

**An epoch trains one slice of the model.** The trainable window rotates each epoch, so a full cycle — a **rotation**, typically **4 epochs** — trains every part of the model once. Rule of thumb: **one rotation ≈ 1 true epoch of the whole model.** Saves land on rotation boundaries, so each saved checkpoint is a whole, evenly trained model. Some plans split windows and lengthen the rotation; the console prints the plan at launch.

On Krea 2 and Qwen the fine-tune card counts in rotations: **Train for N rotations**, **checkpoint + preview every K rotations**, and **each part trains for E epochs** before the next. A live line under the card says how many windows your card gets and how many epochs that makes. On MiniMax H3 you set epochs, and **Run at least one full cycle** or some weights never train; the console warns you.

## What card do I need?

The "trains on 8 GB" figures elsewhere apply to **LoRA** training. Fine-tuning tiers itself to your card: the planner measures free VRAM at launch, picks the largest plan that fits, and prints what it chose and why. At the default training resolution:

| Your card | MOSS-Image — photos | Flux 3 Image Klein — photos | Krea 2 — photos | MiniMax H3 — photos | H3 — voice | H3 — video, confirmed | H3 — video on likeness blocks, expected |
|---|---|---|---|---|---|---|---|
| **16 GB** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ up to **2.3 s** | up to **3.8 s** |
| **24 GB** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ up to **2.3 s** | up to **5.2 s** |
| **32 GB** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ up to **3.8 s** | up to **5.2 s** |

- Clip lengths follow Gizmo's grid: **2.3 s is the 56-frame slot**, confirmed by measured runs on every tier. **3.8 s is confirmed on 32 GB**, even with video training the whole model.
- The expected column assumes **Training mode: Default** (the default), which confines clips to the likeness blocks; in our tests that trains video as well as the whole model and makes clips far lighter. Those figures are conservative arithmetic from measured constants, not individually measured.
- Whole-model 5.2 s clips need more than 32 GB (measured).
- With Training mode set to Off, one clip anywhere in the folder trains the **whole** model, so a mixed photos + clips dataset uses the clip column.
- **16 GB is the fine-tune floor** for these families; 12 GB cards train LoRAs only. MOSS-Image is lighter: see below.
- Measured 16 GB peaks: **8.8–12.3 GB for H3**, **up to 11.4 GB for Krea 2** (at 1 MP). Flux 3 Image Klein trains the whole model at once with a **14.0 GB** peak at 1 MP (12.6 GB at 0.25 MP), measured on a 32 GB card, so it fits 16 GB without splitting. The console prints your run's peak every epoch.
- Too little VRAM refuses cleanly at launch instead of running out of memory mid-run.

## How it fits

A naive full fine-tune of Krea 2 (12.9B) needs roughly **78 GB** — bf16 weights, gradients and optimizer state at once. Fizgig closes the gap with:

- **Rotating windows.** Only one slice is trainable at a time, so gradients and optimizer state exist for that slice alone. Over a full cycle every weight trains.
- **A 4-bit NF4 frozen base** (the default). The frozen rest of the model is held at half size; on 16 GB it streams from system RAM.
- **A CPU-resident bf16 master copy** as the source of truth. Training never round-trips through a quantiser, so small updates aren't erased, and checkpoints are written in bf16 from the master.
- **Optimizer-in-backward**, which consumes and frees each gradient as it lands (saves 5.2 GB).
- **Adafactor**, whose factored state is ~10× smaller than AdamW's.

## Model files

Fine-tuning uses the same training bases as LoRA training; nothing new to download.

- **Krea 2** fine-tunes the **RAW bf16 model** (`krea2_raw_bf16.safetensors`, ~26 GB). The fp8 Turbo is the workbench preview model and can't be fine-tuned.
- **Qwen Image 2.1** fine-tunes its **bf16 DiT** (`qwen_image_2.1_bf16.safetensors`); the training adapter stays on, as for LoRAs.
- **Flux 3 Image Klein** fine-tunes its **bf16 Base** (`flux3-image-klein-base-bf16.safetensors`), every block's Linears; the distilled model can't be fine-tuned. It peaked at 12.6 GB at 0.25 MP and 14.0 GB at 1 MP, the whole model at once on a 32 GB card.
- **Fibo 2.0** fine-tunes its **bf16 Dev DiT** (`fibo2_dev_bf16.safetensors`): the 44 main layers in five parts (attention, cross-attention and each FFN matrix), with the refiners, the Perceiver and the modulation frozen. The checkpoint is written in ComfyUI's layout and loads there in place of the Dev file. A 32 GB card trains it in two windows, a 24 GB card in three, and a 16 GB card streams the frozen layers from system RAM.
- **MiniMax H3** fine-tunes the **pruned int8 checkpoint** (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, ~21 GB), the same file ComfyUI runs. The ~66 GB bf16 file works for LoRA training only; the trainer refuses it for fine-tuning with a clear message.

A finished fine-tune checkpoint is a valid base for its family:

- **Keep training it** — point the model path at the checkpoint; the console prints the exact continuation settings at every save.
- **Train LoRAs on top of it** — set it as the family's base in **Preferences**. Teach the base your world or cast once, then train quick LoRAs for individual subjects. Deploy those LoRAs with the same fine-tuned base in ComfyUI.

**Pause / Resume** works on a fine-tune. On Krea 2 and Qwen, Pause **finishes the rotation it is in**, then saves a full checkpoint and exits, so a checkpoint is never a partial rotation; Resume continues from that checkpoint for the rotations left. On MiniMax H3, Pause saves a full checkpoint even between regular save epochs; Resume carries over the rotation window, checkpoint numbering and remaining epoch count.

## Why fine-tune

A LoRA constrains every update to a low-rank subspace, so concepts compete for the same few directions. That is why LoRAs tend to drag pose, framing and lighting toward the training set along with the likeness. A full-rank update can change how the model *represents* a concept, so it composes with what the model already knows. In our tests, multi-character and concept teaching landed at a much deeper level than LoRA training, with much better results. [Checkpoint to LoRA](#turn-it-back-into-a-lora) turns the result into a shareable file.

## MOSS-Image

Measured at 1 MP: training the whole model at once peaks at **12.6 GB** (all parts) or **8.3 GB** with **Train attention only**, so a 16 GB card fine-tunes the whole model with nothing rotating, and the planner fits an attention-only fine-tune on 12 GB. Fine-tune Base, not SFT. Details: [MOSS_IMAGE.md](MOSS_IMAGE.md#fine-tuning).

## Krea 2 and Qwen Image 2.1

Both fine-tune through the same card. The frozen base is **4-bit NF4**; the trainable window is held in bf16 from a bf16 master copy, and **biases stay frozen**. The windows are **components across the full depth** — attention across all the blocks, then each MLP matrix in turn — so a concept is learned by every layer at once. On Krea 2 the text-fusion stack stays trainable throughout: rotation would never reach it, and it is where prompt-to-concept binding happens.

Measured on Krea 2 at 0.25 MP:

| Card | Plan | Peak VRAM | Speed |
|---|---|---|---|
| **32 GB / 24 GB** | 4 windows, resident | ~15.6 GB | ~0.85 s/step |
| **16 GB** (with 32 GB of system RAM) | 8 windows, frozen blocks streamed from RAM, master on disk | under 16 GB | ~5.1 s/step |

At 1 MP a 16 GB card plans 12 streamed windows (peak 11.4 GB, ~9.8 s/step). When the budget is tight the planner splits windows, and below that the frozen out-of-window blocks **stream from system RAM** — slower steps, the same learning. Too little VRAM refuses cleanly at launch.

## MiniMax H3

The same checkbox under the MiniMax H3 family fine-tunes the 33B model. It uses **component windows only**: each window trains one attention or MLP matrix across all 50 blocks (4 windows per cycle), so every window spans the full depth from the first epoch.

- **Training mode: Default** is recommended on every tier: matched runs came out clearly better on look and prompt adherence than full-model fine-tuning, and it shrinks the windows. The VRAM figures below assume it.
  - **32 GB:** the 4-window cycle at full speed.
  - **24 GB:** the planner depth-splits `mlp.fc1` into two slices — a 5-window cycle, full speed, no offloading. Measured peaks 19.1–21.5 GB.
  - **16 GB:** frozen out-of-window blocks also stream from system RAM (~7 GB staged, a 9-window cycle). Measured peaks 8.8–12.3 GB at **~1.5× the step time**.
- **Full-model fine-tuning** (Training mode Off, or any dataset with video clips) plans itself too. On stills: measured peaks 17.3–18.7 GB on a 24 GB budget and 8.8–11.6 GB streamed on 16 GB. Clip datasets reserve activation memory before windows are sized (clips cost VRAM per frame), which is what the card table reflects. When clips are too long for your card, the trainer says so before training starts, with the fix: cut to the 2.3 s Gizmo slot or lower Target Megapixels.
- **Voice and mixed datasets** train too. Voice trains the same blocks as photos. The per-category **stop epoch** counts across Pause/Resume: pause a mixed run, set the stop to the current epoch, Resume, and it finishes voice-only.
- **Use unique trigger tokens.** An invented token gives the fine-tune somewhere clean to bind; a common word drags its existing meaning along.
- **Run length has no standard number.** It depends on learning rate, dataset size and what you're teaching. The 100-epoch default is a generous scrub range for a typical small dataset; **a large dataset probably needs far fewer epochs** (each epoch is more steps). Save once per cycle and compare checkpoints to find where yours peaks. Total *steps* still run well past LoRA habits, so budget wall-clock and disk accordingly.

Video, voice, Training mode and Gizmo are covered in [MINIMAX_H3.md](MINIMAX_H3.md).

## Saves, previews and numbering

These follow the rotation cycle, not the Samples tab, on every family.

- On Krea 2 and Qwen the card's **checkpoint + preview every K rotations** is the cadence, and the epoch boxes are greyed out. On MiniMax H3, **Max epochs** and **Save every** snap to cycle boundaries — the Save-every box follows the fine-tune controls live in the GUI and the trainer snaps it again at launch. Either way every checkpoint has each window trained equally.
- **Previews ride the saves:** one render per saved checkpoint plus the final one, overriding the Samples tab's "every N epochs". Prompts, resolution, seed and the live preview change still come from the Samples tab. Every sample in the gallery maps to a deployable file. Krea 2 previews render on the training DiT with the Turbo LoRA.
- **Checkpoints are numbered by epoch** (`-000004`, `-000008`, …) and numbering continues across Pause/Resume, so a resumed run never overwrites an earlier save.
- **Adaptive LR is off** under fine-tune: rotation boundaries would read as instability to it. Judge the per-save previews, evaluate checkpoints in ComfyUI, or extract a LoRA and scrub the epochs in LoRA Royale.

An H3 fine-tune is a normal H3 checkpoint: load it in ComfyUI directly, or extract a LoRA from it (the extractor reads the int8 format natively).

## Learning rates

**Fine-tuning wants much lower learning rates than LoRAs.** A LoRA nudges a small adapter on a frozen model; a fine-tune moves the model's own weights, so a normal LoRA rate can wreck a fine-tune.

- **MiniMax H3: 3e-5**, the tested fast-and-reliable rate. **1e-4 destroys an H3 fine-tune** (measured).
- **Fibo 2.0: 1e-5** (choosing Fine-tune sets it): a 12-epoch run on 16 photos showed the training at that rate.
- **Flux 3 Image Klein: 5e-5** (choosing Fine-tune sets it). Its weights are larger than Krea 2's and Qwen's, so it needs a higher rate to move.
- **Krea 2 and Qwen Image 2.1: 1e-5** (choosing Fine-tune sets it, and turns Adaptive LR off). On Krea 2, 1e-4 trains, but treat it as the top of the experiment range; the best results are found lower. When a run looks almost right but slightly overcooked, lower the rate rather than the epoch count.
- The regularisation **LR ×** multiplier (below) is part of the same tuning space.

## Subjects

A fine-tune can train up to four subjects from their own folders, and **Leave the rest of the model alone** on the Subjects card keeps them out of prompts that don't use their trigger words. A fine-tune has no LoRA to switch off, so Fizgig saves what the original model makes for each training picture before the first step and holds the trigger-free pass to that. **Fizgig LoRA Isolation** is for LoRAs.

## Regularisation images (optional)

Full fine-tuning moves every weight, so a long run on a few subjects drifts the model's whole notion of people; there is no low-rank bound to limit it. Point **Regularisation images** at a folder of ordinary photos of the broader class and they train at a reduced rate as an anchor. Leave the folder empty to train without one.

- **LR ×** (default 0.2): **0.1–0.3** tethers the model's prior while your subject trains. Toward **1.0** the reg set trains like a second subject set — class-balanced training rather than an anchor, which is a different, valid thing.
- If the fine-tune drifts the broader class, raise it a step; if the subject learns too slowly, lower it. Worth tuning per dataset.
- Use **real photos, not model output**: anchoring a fine-tune to its own samples distils its artifacts back in.
- Caption them normally. Anything left unsaid gets attributed to the class word.

## Turn it back into a LoRA

On the **Extract** tab, choose **From a fine-tune**. Point it at the base model you started from and your fine-tuned checkpoint, and it extracts the difference as an ordinary kohya `.safetensors` at whichever ranks you tick — several at once, since one SVD per layer serves them all. It reads H3's int8 format natively.

- **Rank 64 was perceptually indistinguishable from the full ~26 GB checkpoint** at ~0.5 GB. Quality degrades smoothly at lower ranks.
- In our testing, a LoRA **extracted** from a fine-tune came out better than a LoRA **trained directly** at the same or higher rank on the same dataset. A low-rank file can hold a solution that low-rank training struggles to find; the full-rank phase does the work and extraction is nearly free.

## What it costs

- **VRAM:** on the NF4 base, **24 GB** runs the full-depth component cycle at full speed and **16 GB** streams frozen blocks from RAM at a slower step (see the tables above).
- **System RAM** for the bf16 master copy, on top of VRAM: ~24 GB on Krea 2, ~23 GB (likeness) to ~38 GB (full model) on H3. The master spills to disk automatically when RAM is short (Krea 2 and Qwen when it would take over 40% of the available RAM), so a 32 GB box fine-tunes Krea 2, more slowly. The trainer warns at launch when RAM looks tight.
- **Disk — set the save location before the run.** Every save is a full checkpoint: ~26 GB on Krea 2, ~21 GB on H3. Saving once per 4-epoch cycle over a 40-epoch run is ~260 GB, and a Pause writes an extra full checkpoint. The Training tab's **Output Directory** defaults to your LoRA folder; point it at a roomy drive before you press Start, because afterwards the only fix is moving huge files by hand.
- **NVIDIA only.** Fine-tuning is untested on AMD/ROCm: every measured tier is NVIDIA, and the NF4 default relies on bitsandbytes 4-bit, the least-travelled part of the ROCm stack. Reports welcome either way.
- **Low learning rates and long step counts** — see [Learning rates](#learning-rates).
