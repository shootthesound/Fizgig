# Krea 2

[← Back to the README](../README.md)

Krea 2 is a from-scratch native port: 12.9B single-stream MMDiT, Qwen-Image VAE, Qwen3-VL-4B text encoder. Train on the **RAW model**; previews render on the training model itself with the official Turbo LoRA (auto-downloads) applied for the render only. Pick Krea 2 from the **Base Model selector** on the Training tab and the **✨ Krea 2 Ultra Fast (rank 8, adaptive LR)** preset applies on your first visit; the other built-in presets, including **✨ Krea 2 Standard (rank 32, full model)** and **✨ Krea 2 Style (rank 16, gentle LR)**, are in Load Preset.

## What works

Everything works on Krea 2:

- All five workbench tools ([TRAINING.md](TRAINING.md#the-workbench)), **Pause/Resume**, **Context LoRA**, **Adaptive LR**, reference images and the live sample override.
- **Weight averaging (EMA)**, on by default at 0.98 — checkpoints and previews come from a running average of the adapter's recent steps. Measured on Krea 2 and MiniMax H3, it lifts late-epoch likeness and steadies the epochs.
- **LoKR training** — pick it from Network Type, factor 8 or below, and drop the learning rate to 5e-5 whichever preset you started from (or, with Adaptive LR, set Min 5e-5 and Max 1e-4). Standard LoRA is ~20% faster and is the default.
- **Automagic v3** (`automagic3`) and **v4** (`automagic4`) in the Optimizer Type row. Each sets its own learning rate, so the scheduler and Adaptive LR stand down while it owns the rate; the Learning Rate box is only its start. See the optimizer notes in [MINIMAX_H3.md](MINIMAX_H3.md).
- **Full fine-tuning** of the base model itself — see [FINETUNE.md](FINETUNE.md).

Output is ComfyUI-ready.

## VRAM: 8 GB is enough

Users train full Krea 2 LoRAs on 8 GB with everything on **Auto** and batch size 1. Auto reads your *free* VRAM and picks INT8 or NF4 plus the right block swap — the console explains its choice. On longer runs the transformer blocks **torch.compile** automatically for roughly 2× faster steps.

| Your card | What to do |
|---|---|
| **8 GB** | Everything on **Auto**, batch size 1, stock preset defaults |
| **10–12 GB** | Same — headroom to raise batch size or resolution |
| **16 GB+** | Same — Auto will usually pick the faster INT8 path |

If a preview can't fit, previews auto-disable and **training keeps running and saving**.

## SageAttention image generation on AMD

With a compatible SageAttention build installed in Fizgig's Python environment, an AMD launch offers
**Preferences → Inference performance → Force SageAttention for Krea 2 image generation**.
It takes effect on the next image in training previews, Studio, Royale and Profiler. Training keeps its own backend.

Generation follows ComfyUI's fixed-length attention calls: expanded K/V heads, explicit HND layout and
`smooth_k=False` (the RDNA4 package can override smoothing for accuracy). Equal-length samples share one call;
ragged samples are split so padding never enters attention. Sage generation also skips the extra 256-token
shape padding, including Studio's cached forwards. A failing forced kernel stops generation.
Reference: [ComfyUI attention](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/modules/attention.py)
and [Krea 2 model](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/krea2/model.py).

For a ComfyUI comparison, match the checkpoint, precision, resolution, prompt, steps and CFG.
Compare warmed-up renders; weight offloading and VAE decoding can still differ.

## High-resolution generation

For the Turbo checkpoint, use **Blocks Swap: Auto**. Krea 2 starts with its block weights mostly on CPU,
then chooses how many can stay on the GPU using available VRAM and the requested resolution. If sampling
still runs out of memory, it retries the same seed with more blocks offloaded, up to the model's swap limit.
The log reports the activation reserve and block count. Manual block counts stay fixed; NF4 does not support
block swapping. Auto preserves stored FP8 weights and uses INT8 for dense checkpoints.

The DiT parks on CPU during text encoding and VAE decoding, including when block swapping is enabled.
On ROCm, Krea 2 leaves the DiT and VAE on CPU after rendering and releases unused GPU allocations. The next render
restores the streaming layout; Auto also budgets for CPU-resident embedders, adapters and buffers that must
return to the GPU. Keeping weights in system RAM does not match ComfyUI's fast-disk storage policy.
ROCm renders cap the PyTorch allocator at available memory with no additional desktop reserve.
Auto plans against this cap; a stricter existing cap is preserved, and the previous cap is restored after
the DiT parks. The previous 1.5 GiB margin passed a 2048×2048 RX 9070 XT run and subsequent user-reported
repeated runs without crashes, with slightly higher RAM usage than ComfyUI. The reduced 500 MiB margin
also passed subsequent user-reported repeated-run testing on the RX 9070 XT. The zero-headroom setting
passed user testing and four consecutive local renders with the AMD launcher's allocator garbage collection
enabled (`garbage_collection_threshold:0.8`). Disabling it caused desktop framebuffer allocation failures
and slower sampling (69.75 s versus 45.70–46.63 s with collection enabled). The sampling activation budget remains.
Two local ROCm inference renders using the existing pinned-memory weight exchange path took 46.04 / 46.29 s
for sampling, versus 45.70–46.63 s for the staged path. Faster logged weight exchanges did not improve sampling
time, so that pinned-memory policy was not retained. Output was repeatable within each setting, with a small pixel
difference between settings (RMSE 0.00124 on a 0–1 scale); its cause was not isolated.
A subsequent H2D-only candidate used six streaming blocks and two reusable GPU buffers. Its two renders
sampled in 46.22 / 46.08 s, with identical pixels to the staged baseline and no new kernel GPU faults.
Sampling transfer volume fell from an estimated 64.69 GiB to a counted 19.41 GiB, excluding stage placement,
but sampling time did not improve and peak allocated VRAM rose by about 0.4 GiB. The candidate was not retained.
The Linux ROCm workbench keeps stored FP8 DiT Linear weights in private, file-backed CPU
views. Parking rebinds those frozen weights instead of copying the GPU model into anonymous RAM; block swaps
reuse their CPU masters with one pinned staging pool and no return copy. Adapters and mutable biases still move
normally, and the existing GPU block budget is unchanged. Clean mapped pages are released after transfers.
The text encoder is deleted without a preceding copy to CPU in this path. Other backends, training and
additional quantization retain their existing loading policy. Keep the base checkpoint unchanged while loaded.
The first file-backed comparison completed two images per policy with identical pixels and no kernel faults.
Peak RSS fell from 22.66 to 14.72 GiB, but sampling slowed from 49.82 / 51.49 s to 59.76 / 59.03 s, and the
warm render rose from 61.43 to 76.14 s. A small ROCm copy diagnostic showed private file pages becoming anonymous
during direct uploads. The current refinement uploads through pinned staging first; its 256 KiB copy check
preserved values and left the source mapping with zero resident anonymous pages. Two subsequent 2048×2048
RX 9070 XT renders validated the refinement: peak RSS was 5.67 GiB (75% below the 22.66 GiB control), and
sampling took 47.58 / 49.00 s. The warm render fell from 61.43 to 56.97 s; initial pipeline loading fell from
13.02 to 1.49 s, without flushing the system file cache. Whole-GPU peak VRAM remained 15.65 GiB. Both
outputs matched the control pixels exactly, with no worker failures or kernel GPU faults. Both VAE decodes
recovered through tiled fallback. These two runs retain the existing five swapped blocks and allocator GC 0.8.
The saved ComfyUI comparison remains faster (37.41 s warm sampling, 41.03 s warm total) with 4.50 GiB peak RSS.
Raw measurements and comparison limits are in `user_data/profiles/file_backed_refined_2026-11-17/report.json`.
A four-image repeat peaked at 6.02 GiB RSS (6% above the two-image test, 73% below the old memory policy),
with render times of 64.15 / 57.16 / 56.05 / 56.00 s and whole-GPU peak VRAM of 15.34 GiB. All four completed
without worker failures or kernel GPU faults and used tiled VAE fallback. Studio's retained images from runs
three and four matched control pixels exactly; both preview panes were checked equal during every render.
The control uses this ROCm branch with zero additional desktop reserve and file-backed loading disabled,
not original main or the unchanged 500 MiB commit. Both policies preserve the same FP8 DiT and FP8 text
encoder with BF16 VAE; the `bf16` workbench label means no additional DiT quantization. See
`user_data/profiles/file_backed_four_2026-11-17/report.json` for the repeat measurements.
If ordinary VAE decoding runs out of memory, it retries with the existing tiled decoder and releases its
feature cache. Single-frame VAE cache removal is proposed separately. The temporal convolution shortcut is deferred. BF16 rounding and tiled decoding
can change pixels slightly. More offloading also means more CPU RAM and
transfer time; this does not guarantee that every resolution fits.

The Turbo checkpoint preset retains **8 steps, CFG 1, Euler / simple**. The beta schedule is proposed separately.
Other step counts retain the existing simple schedule. These changes apply to the workbench engine used
by Studio, Royale and Profiler; they do not change training's memory policy.

The memory reserve follows [ComfyUI's Krea 2 estimate](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/model_base.py),
using Fizgig's existing block offloader.

RX 9070 XT measurements before idle CPU parking (16 November 2026, PyTorch 2.14.0+rocm10.1.0,
SageAttention, stored FP8 Turbo, BF16 Qwen-Image VAE, eight steps, CFG 1, Auto swap):

| Check | Before single-frame convolution shortcut | After |
| --- | ---: | ---: |
| Warm 1024×1024 VAE decode, identical latents | 1.14 s | 0.24 s |
| VAE peak allocated VRAM | 6.69 GiB | 4.44 GiB |

Full 1184×1776 generation completed in 29.2 s without a LoRA (21.4 s sampling, 1.0 s decoding),
and 32.1 s with the rank-64 Turbo LoRA attached (23.4 s sampling, 1.0 s decoding). These totals include
offloader reconfiguration and stage transfers, excluding initial model loading and text encoding.
Both runs also completed 1024×1024 renders; no new kernel GPU faults were logged. The separate VAE
comparison had an RMS pixel difference of 0.00147 on a 0–1 scale from BF16 convolution rounding.

These are bounded hardware checks, not speed parity or a guarantee against resets. The reference ComfyUI
sampling time was 14.7 s; its prompt and INT8 text encoder differ from this test's neutral teapot prompt
and FP8 text encoder.

The 17 November 2048×2048 validation completed two seed-42 images with a rank-32 primary, no donor and
unchanged sliders, but **failed desktop stability**. Sampling took 83.4 / 55.0 s and decoding 5.0 / 2.8 s;
both decodes required the tiled fallback. There were 748 allocator OOM warnings. Idle GPU allocation fell
to 0.078 GiB, while process RSS reached 17.7 / 20.1 GiB after the two images.

During the first sampling pass, the kernel reported insufficient memory for command submission and
`Failed to pin framebuffer: -ENOMEM`. Xwayland aborted and ChatGPT exited. No GPU reset was logged, so
the absence of a reset does not establish desktop stability. This run predates the hard allocator cap above.
The original Studio worker exit remains unreproduced.

With the corrected cap and its original 1.5 GiB headroom, one further gfx1201 / RX 9070 XT run completed
at 2048×2048, seed 42, with the same
rank-32 primary and neutral prompt. Auto offloaded 8 blocks instead of 5. Sampling took 46.7 s, with 11.95 GiB
peak allocated / 12.44 GiB peak reserved and no allocator OOM warnings. Decode took 5.1 s including its
recoverable switch to tiled decoding. Idle GPU allocation was 0.078 GiB; process RSS was 17.8 GiB.
No new GPU fault, framebuffer allocation failure or desktop coredump was logged. The previous allocator
limit was restored after rendering, and the worker exited normally. The user subsequently confirmed
repeated-run stability on the RX 9070 XT. Exact-prompt quality, speed parity and system-RAM parity with
ComfyUI remain unverified.

Profiler logs include the total render count, Auto's memory budget, sampling backend, VAE decode and image
completion. Two seeds run sequentially for every ablation condition, not as a two-image batch. PyTorch's
allocator can print OOM warnings and recover by releasing cached allocations; a warning alone is not a failed
render. Check whether image completion and progress continue, or whether the job reports an exception.
An unexpected Studio worker exit now includes its process exit code in the error and log. Native crashes
also print a Python fault traceback to the worker log where the runtime can provide one.

## The trainer curates your dataset while it trains (experimental)

Four Training-tab toggles no other trainer has. They work on Krea 2 and [Qwen Image 2.1](QWEN_IMAGE.md).

Images don't start out as problems: early on, the hard ones are often the most valuable in the set. The watch follows every image's loss across epochs, and only an image that has stopped teaching the model (stuck without improving, or mined out after a good run) gets treated as a problem.

- **Track how each image trains** — per-image loss is tracked across epochs (noise-normalised). Most images do their work early and are simply learned by the end; the few that stay hard without improving get flagged in the **Per-image loss** view (on the run card while it trains, and in Runs & outputs), with thumbnails and trends. In real runs the top flags were all caption/image mismatches.
- **Per-image adaptive LR** — flagged images are throttled so one bad caption can't yank the weights all run; healthy images get a gentle boost. Matched-epoch A/Bs: faster likeness *and* a higher ceiling.
- **Auto-recaption stuck images** — between epochs, Krea 2's Qwen3-VL-4B *looks at* each stuck image and rewrites its caption from what's visible. Still stuck after two attempts and the image is excluded for the run (remembered per-dataset; fix the caption and it's re-admitted).
- **Warm up look outliers** — real-but-unusual shots (tight angles, profiles) ease in at reduced LR while the identity forms, then release to full. It reads the scores saved by the [Look Consistency Filter](TRAINING.md#dataset-prep).

Edit any caption yourself mid-run from that view — no restart. When nothing is improving any more, a plateau banner names the best-checkpoint window to scrub in LoRA Royale. Pause, resume, restart: a resumed run replays its own loss log and loses nothing.

## Captioning

Fizgig's AI captioning — the Captions tab's Qwen3-VL option and auto-recaption — uses Krea 2's Qwen3-VL-4B for every model. Florence-2 is the zero-setup alternative on the Captions tab.

## Help map Krea 2's blocks

Krea 2's per-block roles aren't charted yet, which is why Repair Studio's colour-coded sliders and Model Area targeting are Klein-only. The Profiler's weight-only report is the instrument — [open an issue](https://github.com/shootthesound/Fizgig/issues) with what you find and it drives the presets and Repair Studio colour-coding to come.

The four **text fusion** blocks have more to give. Load a Krea 2 LoRA in Repair Studio and pick **✨Text fusion ×2** or **×3** (experimental) — measured across several LoRAs, ×3 lifted the detail meter and likeness on every one with the composition unchanged; an overtrained LoRA is the one case it works against. Save Repaired LoRA bakes it in, so the boosted file works anywhere at strength 1.0.

## Model files

All files live in the one [**Comfy-Org/Krea-2**](https://huggingface.co/Comfy-Org/Krea-2) repo. **Preferences → ⬇ Download models for me** under the Krea 2 card downloads, verifies and fills in the paths, with no account needed; every row also has a manual **Download** link. From the command line (~32 GB):

```bash
python -m fizgig.scripts.fetch_models --family krea2
```

| Model | File | Size |
|---|---|---|
| **RAW DiT (bf16) — training** | `krea2_raw_bf16.safetensors` | ~26 GB |
| **Turbo DiT (fp8) — workbench** | `krea2_turbo_fp8_scaled.safetensors` | ~13 GB |
| Turbo LoRA *(auto-downloads)* | `krea2_turbo_lora_rank_64_bf16.safetensors` | ~470 MB |
| Qwen-Image VAE | `qwen_image_vae.safetensors` | ~250 MB |
| **Text Encoder — recommended** | `qwen3vl_4b_fp8_scaled.safetensors` | ~5.2 GB |
| Text Encoder — full precision | `qwen3vl_4b_bf16.safetensors` | ~8.9 GB |

The text-encoder slot is **open**: any Qwen3-VL-4B in the ComfyUI layout loads — fp8_scaled (recommended, captions we couldn't tell apart), bf16, or a community fine-tune/abliterated build, which changes how your dataset gets captioned.

The two DiT slots are open too: point them at your own Krea 2 checkpoints from ComfyUI's `diffusion_models` folder, fine-tunes included. fp8 builds (fp8_scaled, fp8mixed) run as they are. int8 builds (int8 ConvRot, int8mixed) and older scaled_fp8 files are decoded to bf16 as they load, and then Fizgig's own precision applies, so they take a couple of minutes longer to load. A Turbo build goes in the Turbo slot, since previews render it in 8 steps without CFG.

General install steps are in [INSTALL.md](INSTALL.md).

Review scope: custom low-memory mode, Auto precision and resolution-aware block swapping apply only on ROCm.
NVIDIA retains the generic workbench defaults and Studio/Profiler block-swap preferences. Historical timings
above include the separately proposed beta schedule and VAE optimizations; they do not validate this split.
