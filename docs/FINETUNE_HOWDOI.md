# Full fine-tuning — "How do I…?"

The extended companion to the README's fine-tuning section, for **Flux 3 Image Klein**, **Krea 2**,
**Qwen Image 2.1** and **MiniMax H3**. Everything here is measured or field-tested, not aspirational. If you're
new to fine-tuning and coming from LoRA training, read the first four answers in order —
they're the ones that save you a wasted overnight run.

> **One idea makes everything else here make sense: an "epoch" trains one slice of the
> model.** The trainable window rotates each epoch, so it takes a full cycle — a
> **rotation**, typically **4 epochs** — for every part of the model to train once. Rule of
> thumb: **one rotation ≈ 1 true epoch of the whole model.** That's why saves land on
> rotation boundaries — each saved checkpoint is a whole, evenly trained model. Krea 2 and
> Qwen count the run in rotations; MiniMax H3 counts it in epochs.

---

## How do I start a fine-tune?

On Flux 3 Image Klein, Krea 2 or Qwen Image 2.1, pick **Fine-tune the whole model** under **Kind of training** on
the Training tab; on MiniMax H3, tick **⚗ Fine-tune the BASE MODEL instead of training a
LoRA**. Then press Start. That's genuinely it — the planner
measures the memory actually free on your card at launch, picks the plan that fits, and
prints what it chose and why. Everything else on this page is about making the run *good*
rather than making it *go*.

Before your first run, do two things:

1. **Set the Output Directory to a drive with room** (Training tab). Every save is a full
   checkpoint — ~26 GB on Krea 2, ~21 GB on H3 — and the box defaults to the same folder
   your LoRAs go to. Change it before you press Start; afterwards the only fix is moving
   20 GB files by hand.
2. **Check your learning rate** (next answer). This is the single most common way to ruin
   a fine-tune.

## What's the quickest way to fine-tune MiniMax H3?

The whole recipe, using what the GUI already sets for you:

1. **Load the ✨ MiniMax H3 Fast preset** (Training tab, Load Preset).
2. **Tick ⚗ Fine-tune the BASE MODEL.** The moment you tick it, the learning rate
   switches to **3e-5** and the epochs and save cadence move to fine-tune values — the
   Save-every box suggests a save every second cycle (~8–10 epochs; previews ride the
   saves) and follows your card's plan live, so trust its guidance.
3. **Set Max epochs and Save every N epochs** to taste — the GUI guides both. Save-every
   snaps to full cycles so every checkpoint compares like-for-like.
4. **Leave the learning rate at 3e-5** — the tested rate, and the most you should use. If a
   run looks too eager (drift, over-sharpening between checkpoints), come down to **1e-5**.
5. **Leave Optimised Likeness Learning on.**
6. **Change the Output Directory** to a drive with room (each save is ~21 GB).
7. **Make sure your captions use a trigger token** — an invented word, not a common one.
8. Press Start. Leave everything else alone.

That's it — the planner does the VRAM thinking, previews ride the checkpoint saves, and
every save is a deployable model.

## What's the quickest way to fine-tune Krea 2 or Qwen Image 2.1?

Even shorter, because picking Fine-tune sets everything that matters:

1. **Pick Fine-tune the whole model** under Kind of training. The moment you pick it:
   learning rate → **1e-5**, and **Adaptive LR switches off** (rotation boundaries read as
   instability to it). The card starts at **Train for 10 rotations** with a **checkpoint +
   preview every rotation**, and the line under it says how many windows your card gets.
2. **Set the rotations to taste.** Nobody has a canonical number for a diffusion DiT
   fine-tune yet — ten rotations give you ten comparable checkpoints to scrub through;
   find where yours peaks rather than trusting a number.
3. **Leave the learning rate at 1e-5.** Experimenting higher is allowed (on Krea 2 up to
   1e-4 — it trains) but the best results are realistically found lower.
4. **Optional but recommended for long runs: regularisation images** (a folder of real
   photos of the broader class) with **LR ×** at the default 0.2, on the same card.
5. **Change the Output Directory** to a drive with room (each Krea 2 save is ~26 GB).
6. **Make sure your captions use a trigger token** — an invented word, not a common one.
7. Press Start. Leave everything else alone.

## What learning rate should I use?

**Lower than you're used to.** A LoRA nudges a small adapter riding on a frozen model; a
fine-tune moves the model's own weights. The rates you know from LoRA training land very
differently here.

- **MiniMax H3: ticking Fine-tune sets 3e-5** — the tested rate and the most you should
  ever use. It starts there rather than lower so you can judge real results and come down to
  **1e-5** if a run looks too eager, instead of waiting on a rate that is too slow to tell.
  **1e-4 will destroy an H3 fine-tune** — that's measured, not folklore.
- **Krea 2: 1e-5 is the safe recommendation.** You're welcome to *start experimenting* at
  1e-4 — it trains — but realistically the best results are found lower. Treat 1e-4 as
  the top of the experiment range, not the recipe. When a run looks almost right but
  slightly overcooked, the next move is a lower rate, not fewer epochs.

If you use regularisation images, their **LR ×** multiplier is part of the same tuning
space — see the regularisation answer below.

## What card do I need?

At the default training resolution:

| Your card | Flux 3 Image Klein — photos | Krea 2 — photos | H3 — photos | H3 — voice | H3 — video, confirmed | H3 — video on likeness blocks, expected |
|---|---|---|---|---|---|---|
| **16 GB** | ✅ | ✅ | ✅ | ✅ | ✅ up to **2.3 s** | up to **3.8 s** |
| **24 GB** | ✅ | ✅ | ✅ | ✅ | ✅ up to **2.3 s** | up to **5.2 s** |
| **32 GB** | ✅ | ✅ | ✅ | ✅ | ✅ up to **3.8 s** | up to **5.2 s** |

12 GB cards train **LoRAs**, not fine-tunes — 16 GB is the tested fine-tune floor. Every
confirmed cell comes from a measured run; the expected column is conservative arithmetic
from the measured constants, with the restriction tickbox on.

## How does it fit on my card at all?

A naive full fine-tune of Krea 2 needs ~78 GB. Three things close the gap: **rotating
windows** (only one slice of the model is trainable at a time — over a full cycle every
weight trains, but gradients and optimizer state only ever exist for the active slice), a
**4-bit NF4 frozen base** (the fine-tune default — the frozen part of the model is held
at half size while the trainable window runs bf16), and a **CPU-resident bf16 master
copy** that is the source of truth, so your saved checkpoint never passes through a
quantiser regardless of what the card holds.

On 32 GB and 24 GB the full-depth windows stay resident at full speed (~0.85 s/step on
Krea 2). On 16 GB the frozen blocks stream from system RAM — slower steps, same learning.
The console always prints the plan.

## How long should I train?

**At least one full cycle, or some weights never train at all.** A cycle is one pass of
the rotation — 4 epochs in the standard component plan, more when your card's plan splits
windows (the console prints your cycle length at launch). Both your **Max epochs** and
**Save every** snap up to cycle boundaries automatically, so every checkpoint — including
the final one — ends with every component evenly trained.

Beyond that, **there is no standard number**: the right length depends on your learning
rate, your dataset size, and what you're teaching. The H3 default of 100 epochs is a
generous scrub-range for a typical small dataset, not a target — **a large dataset
probably needs far fewer epochs** (each epoch is more steps, so the model sees more per
cycle). The honest method is the one the defaults are built for: save once per cycle,
then compare checkpoints and find where yours peaks. One warning stands either way:
only one component trains at a time, so total *steps* run well past LoRA habits —
budget wall-clock and disk accordingly.

## How do I fine-tune on video clips? (H3)

1. **Cut your clips with Gizmo** — it produces exactly the spec H3 trains on (24 fps, the
   right frame counts) from any footage.
2. **Leave "Restrict video to likeness blocks" ticked** (it's on by default with
   Optimised Likeness Learning). In our tests it trains video just as well as the whole
   model, and it makes clips far lighter on VRAM. Untick it only if you specifically
   want video training the entire model.
3. **Clip length — what's confirmed and what's expected.** The **2.3 s (56-frame) slot
   is confirmed by measured runs on every tier**, whole model included, and **3.8 s is
   confirmed on 32 GB** even whole-model. With the restriction ticked, the *expected*
   range extends to **5.2 s on 24 GB and 32 GB, and 3.8 s on 16 GB** — conservative
   arithmetic from the measured constants, not yet individually measured. Whole-model
   5.2 s clips need more than 32 GB (measured).
4. Mind the mix: with the restriction **unticked**, one clip anywhere in your folder
   trains the whole model, so a mixed photos + clips dataset runs at the clip tiers.

If your clips are too long for your card, the trainer tells you **before training
starts** — with the fix (cut to the 2.3 s slot, or lower Target Megapixels) — instead of
failing mid-run. Clips train with their sound automatically when the audio VAE is set.

## How do I fine-tune voices? (H3)

Drop voice recordings (or record a dataset with Gizmo's built-in recorder) into the
training folder like any other item. Voice stays confined to its measured blocks (34–49)
automatically, so audio gradients never touch the visual blocks. Mixed datasets work:
photos, clips and voice in one folder train one fine-tune in one run, each modality
confined to its zone.

The per-category **stop epoch** counts across Pause/Resume — pause a mixed run, set the
visual stop to the current epoch, Resume, and it finishes voice-only.

## Should I leave Optimised Likeness Learning on? (H3)

**Yes.** Matched runs came out clearly better on both look and prompt adherence than
full-model fine-tuning, and it shrinks the windows so every VRAM tier gets easier. And
as of this release video follows it too: the **Restrict video to likeness blocks**
sub-tick (on by default) routes clips to the same identity blocks — in our tests that
trains video just as well, and it makes clips far lighter on VRAM. Untick it for
whole-model video.

## Where did the EMA setting go? (H3)

It is a LoRA-run control and the row hides when Fine-tune is ticked. EMA keeps a smoothed shadow copy of whatever is training and previews and saves from that copy. For a LoRA that shadow is a few hundred megabytes. Under fine-tune the thing training is the model itself, and it trains in rotating windows, so an honest shadow would be a second copy of the whole model in system RAM with the live window folded in as it trains, and the checkpoint would have to be written from it. That is a real feature with a real memory cost, not a tick, and it has not been built or measured yet. Until it is, a fine-tune saves the trained weights as they stand.

## Should I train the text token refiner on a fine-tune? (H3)

No, unless you are testing it. The refiner is the model's bridge from the text encoder into the DiT and sets how every prompt is read. Fine-tunes used to train it on every epoch, four times as often as any block matmul, and it was the tensor that moved most in every checkpoint. LoRA runs already leave it frozen because training it softened output and made previews judder without helping likeness; fine-tune now does the same. The trigger word is learned in the blocks' attention either way. The tickbox in Other Options ("Train the text token refiner") puts it back for an A/B.

## Should I use the training adapter on a fine-tune? (H3)

You can. The training adapter (a de-distillation LoRA, Circlestone by default for every H3 LoRA run) is available under Fine-tune with the same contract: it rides frozen at 1.0 for every training step so the gradient is about your subject rather than about undoing H3's distillation, it switches off for previews, and it is never written into the checkpoint — the file you get is a plain H3 fine-tune. It is on by default under Fine-tune as it is for LoRA runs; the LoRA-mode numbers (faster likeness, higher peak, no frying with other Turbo LoRAs, measured with Ostris's adapter; Circlestone's trained sharper LoRAs again in later A/Bs on stills) are the reason, and an A/B on your own dataset — same seed, adapter on versus off — is how to check it for yourself. The file is the same one your LoRA runs use: the **Training adapter** dropdown picks Circlestone (one file for both bases — best for photos), Ostris (fl2va or ref2va file to match the base — best for videos) or Off; for a mixed dataset, choose by whether the photos or the videos are the priority.

## Do the per-image tools work on a fine-tune? (Krea 2)

Mostly, yes. **Track how each image trains**, **per-image adaptive LR** and **look-outlier
warmup** all work under a fine-tune — their throttles ride the same per-step loss scaling
the regularisation multiplier uses, and detection judges each image against the rest of
the dataset at the same epoch, so the rotation's epoch-to-epoch shifts cancel out.

Two exceptions, both deliberate: **(global) Adaptive LR** switches off when you pick
Fine-tune — rotation boundaries look like instability to its plateau watcher — and
**auto-recaption** is off under a fine-tune (its between-epoch caption re-encode isn't
fine-tune-safe yet). Caption edits queued from the per-image loss view (Runs & outputs) during a
fine-tune are held and apply in your next LoRA-mode run on that dataset.

## What are regularisation images, and what does LR × do?

Full fine-tuning moves every weight, so a long run on a handful of subjects can drift the
model's whole notion of people — there's no low-rank bound to stop it like a LoRA has.
Point **Regularisation images** at a folder of ordinary **real photos** of the broader
class (not model output — anchoring a fine-tune to its own samples distils its artifacts
back in), captioned normally.

They train at the **LR ×** multiplier beside the folder box, and that multiplier is a
real dial worth experimenting with per dataset:

- **0.1–0.3** (default 0.2): a tether — the reg set nudges the model's prior back while
  your subject trains.
- **Toward 1.0**: the reg set trains like a second subject set — class-balanced training
  rather than a light anchor. A different, valid thing.
- Class drifting toward your subject? **Raise it a step.** Subject learning too slowly?
  **Lower it.**

## Where do my checkpoints go, and how much disk will this eat?

Checkpoints land in the Training tab's **Output Directory** — set it to a roomy drive
*before* the run. Sizes: **~26 GB per Krea 2 save, ~21 GB per H3 save**. Saving once per
4-epoch cycle over a 40-epoch run is **~260 GB**. A Pause writes an extra full checkpoint
on top of the regular cadence. Delete the intermediate saves you don't need once you've
picked your best epoch — each file is a complete, standalone checkpoint.

## How do I pause and resume?

Press **Pause Training**: the trainer finishes the epoch, writes a full checkpoint (even
between the regular save epochs), and exits cleanly. **Resume** continues it — rotation
window, checkpoint numbering and the remaining epoch count all carry over, so a resumed
run never overwrites an earlier save and the cycle stays evenly trained.

## How do I keep training a finished fine-tune?

Every save prints the exact continuation settings in the console — point the model path
(**--dit**) at your checkpoint and use the printed start-window. A finished fine-tune
checkpoint is a valid training base for its family, so you can chain runs indefinitely.

## Can I train LoRAs on top of my fine-tune?

Yes — and this is one of the most useful things about it, even if it's easy to miss.
Your fine-tuned checkpoint is a normal base model, so point the family's model path in
**Preferences** at it and train LoRAs against it exactly as you would the stock base.
Teach the base your world, style or cast with a fine-tune, then train quick LoRAs on top
for individual subjects and concepts. One deployment note: a LoRA trained on your
fine-tuned base pairs best with that same base at inference — load your checkpoint in
ComfyUI and stack the LoRA on it, just like you trained it.
## How do I see previews during training?

Previews **ride the checkpoint saves** — one render per save plus the final one, each
sample the rehearsal of a checkpoint you can actually deploy. You need three things set:
**sample prompts** (Samples tab), a **sample cadence** (Sample every N epochs, or
sample-at-first), and on Krea 2 the **Turbo LoRA** (the standalone Turbo checkpoint can't
show fine-tuned weights, so previews render on the training DiT with the Turbo LoRA
applied fresh). On Krea 2 and Qwen, the card's **checkpoint + preview every K rotations**
sets the cadence. If prompts are set but previews stay off, the console now tells you
exactly which ingredient is missing.

## How do I share a fine-tune? It's 26 GB!

On the **Extract** tab, choose **From a fine-tune**. It diffs
your fine-tune against the
base model and extracts the difference as an ordinary kohya `.safetensors` at several
ranks at once. Measured: **rank 64 was perceptually indistinguishable from the full
checkpoint** at ~0.5 GB, and quality degrades smoothly at lower ranks. In our testing an
extracted LoRA came out better than one trained directly at the same rank. The result
loads anywhere a normal LoRA does, ComfyUI included.

## NF4 or fp8 Base precision?

On Krea 2 and Qwen the fine-tune base is always **NF4** — there's nothing to pick. It's what
makes 16 GB fit at all and keeps 24 GB at full speed, and your saved checkpoint is written
in bf16 from the master copy either way, so it only affects the frozen context the
trainable window learns against.

## The trainer refused to start, or I hit CUDA out of memory — now what?

If it **refused up front**, read the message — it names the actual problem and the fix
(usually: close other GPU apps, shorten clips to the 2.3 s slot, or lower Target
Megapixels). A refusal is the trainer doing its job: it beats an OOM three hours in.

If you hit a genuine **mid-run OOM**, the usual suspects in order: something else grabbed
VRAM after launch (browsers on hardware acceleration are the classic), clips longer than
your card's tier, or Target Megapixels above the default. And know the Windows quirk:
when **system RAM** runs out, the error still arrives dressed as "CUDA out of memory"
with the GPU nearly empty — close other apps and retry before suspecting VRAM.

## What model files do I need?

The same training bases you already have — nothing new to download:

- **Krea 2** fine-tunes the **RAW bf16 model** (`krea2_raw_bf16.safetensors`, ~26 GB).
  The fp8 Turbo is the workbench preview model and can't be fine-tuned.
- **Qwen Image 2.1** fine-tunes its **bf16 DiT** (`qwen_image_2.1_bf16.safetensors`).
- **Flux 3 Image Klein** fine-tunes its **bf16 Base** (`flux3-image-klein-base-bf16.safetensors`), not the
  distilled model.
- **MiniMax H3** fine-tunes the **pruned int8 checkpoint**
  (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, ~21 GB) — the same file ComfyUI
  runs. The ~66 GB bf16 file works for LoRA training only; the trainer refuses it for
  fine-tuning with a clear message.

Plus system RAM for the bf16 master copy: ~24 GB on Krea 2, ~23–38 GB on H3. The master
**spills to disk automatically when RAM is tight** — a 32 GB box fine-tunes Krea 2, more
slowly (and remember the Windows quirk: running out of RAM surfaces as "CUDA error: out of
memory" with the GPU nearly empty). The trainer warns at launch when RAM looks tight.

One more honesty note: **fine-tuning is untested on AMD/ROCm** — every measured tier is
NVIDIA, and the NF4 default depends on bitsandbytes 4-bit, the least-travelled part of
the ROCm stack. It may work; either way, a field report on GitHub genuinely helps.

## Why fine-tune instead of training a LoRA?

A LoRA constrains every update to a low-rank subspace, so concepts compete for the same
handful of directions — which is why LoRAs tend to drag pose, framing and lighting toward
the training set along with the likeness. A full-rank update can change how the model
*represents* a concept, so it composes with what the model already knows. **In our own
tests, multi-character and concept teaching seemed to land at a much deeper level than
LoRA training, with much better results.** And with Checkpoint to LoRA at the end — which
works very well — you don't have to choose between fine-tune quality and a shareable
file. Beyond that: this feature is young, and we're deliberately letting the community
find the ceiling. Field reports genuinely shape what gets built next.
