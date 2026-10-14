# Contributors

Fizgig is written by [Peter Neill (shootthesound)](https://github.com/shootthesound).

## Linkram

[@Linkram](https://github.com/Linkram) reported and tested Krea 2 ROCm stability and
performance on an RX 6800. Their measurements guided the RDNA2 attention and NF4 paths,
and their training logs helped identify startup GPU probes that stalled the Windows GUI.

## scryptio

[scryptio](https://github.com/scryptio) built the **AMD ROCm support** that headlines v4.3.0
([#53](https://github.com/shootthesound/Fizgig/pull/53)): the Windows and Linux ROCm install
paths and launchers, GPU architecture detection, cross-platform VRAM monitoring for the status
bar — a lot of building and testing on real hardware.
Along the way the same investigation pinned down the torch.compile recompile cost on
ROCm, now handled automatically. The PR thread was a group effort:
[tsubasasora](https://github.com/tsubasasora) ran repeated from-scratch Linux installs on an
R9700 that shook out real install bugs, [FNGarvin](https://github.com/FNGarvin) pushed for the
wheel pinning that made the install materially safer, and
[taisunyoung](https://github.com/taisunyoung) contributed expert ROCm performance diagnostics.

They then diagnosed and fixed the **captioning slowdown**
([#90](https://github.com/shootthesound/Fizgig/issues/90) →
[#93](https://github.com/shootthesound/Fizgig/pull/93)): AI captioning held GPU memory inside
the GUI process, which silently poisoned the next training run's memory planning (measured
~4x slower steps). Their fix moves captioning into a persistent worker subprocess — the same
architecture training and caching already use — so the VRAM genuinely returns, with a warm
worker keeping single-image Regenerate instant. Validated on AMD by them and on NVIDIA here.

## rintic-13

[rintic-13](https://github.com/rintic-13) designed and prototyped the **async H2D-only int8
block streaming** that headlines v4.0.0
([#73](https://github.com/shootthesound/Fizgig/issues/73)): the frozen base's swapped blocks
stream host-to-GPU through a pinned ring buffer on a copy stream and never travel back —
measured **6.4× faster** than round-trip swap at the same depth, which is what lets 16 GB and
24 GB cards train MiniMax H3 on the accurate int8 base instead of 4-bit. Landed with
`Co-authored-by` credit (ab90dda). They then extended the idea to the **text encoder**
([#79](https://github.com/shootthesound/Fizgig/pull/79), merged in v4.3.0): layer-streamed
reference-mode encoding drops the peak from ~26 GB to ~12.7 GB with bit-for-bit identical
output, bringing identity distillation to 16 GB cards
([#74](https://github.com/shootthesound/Fizgig/issues/74)).

## dewwwey

[dewwwey](https://github.com/dewwwey) diagnosed and fixed a subtle crash on 24 GB cards
([#84](https://github.com/shootthesound/Fizgig/pull/84), shipped in v4.2.1): parking the DiT
for a preview corrupted the H2D ring buffer's shared slot storage, killing training at step 0
with a misleading CUDA error far from the cause. The instrumented diagnosis was exact and the
fix minimal — verified and merged the same day.

## mabseyuk

**u/mabseyuk** proved MiniMax H3 LoRA training runs on a **12 GB card** — an RTX 5070 field
report with instrumented diagnosis of the two crashes in the way, both fixed in v4.3.1
([#92](https://github.com/shootthesound/Fizgig/issues/92)): the checkpoint-save MemoryError
in the optional hash metadata (landed with co-author credit, e54b0d3) and the post-preview
fragmentation OOM. Confirmed training otherwise fully stable at 12 GB — epoch 14 with
checkpoints throughout when they first reported.

## Hell-Bent-Fox

[Hell-Bent-Fox](https://github.com/Hell-Bent-Fox) diagnosed and fixed the **still-preview OOM
on 16 GB cards** running the int8 streamed plan
([#134](https://github.com/shootthesound/Fizgig/issues/134) →
[#109](https://github.com/shootthesound/Fizgig/pull/109)): the tail-block parking that lets a
clip preview's decoder fit ran only for clips, so a still preview loaded the 4.85 GB decoder
into 4 GB of free VRAM, failed twice and switched previews off for the rest of the run. Their
fix parks for stills too and takes the decoder off the card before the parked blocks return.
The PR was first offered in August with the same exact write-up, closed here on a promise that
did not get delivered, and merged as theirs in September once they came back with the log.

## marduk191

[marduk191](https://github.com/marduk191) contributed two fixes from their fork, shipped in
v5.7.1 with authorship intact: **caption files that are not UTF-8** (Windows-1252, UTF-16) now
load with a warning that names the file instead of aborting the whole caching run on a single
curly apostrophe, and the **`expandable_segments` allocator option is no longer requested on
Windows**, where PyTorch rejects it and warned on every launch.

## johndpope

[johndpope](https://github.com/johndpope) landed three MiniMax H3 loader fixes in one go
(merged for v4.0.0's run-up, August 2026): the bundled **tokenizer learned H3's own special
tokens** ([#56](https://github.com/shootthesound/Fizgig/pull/56)) so dialogue and cutoff markup
resolve to single ids instead of being shredded into byte pairs; the **H3 base loads straight
from a directory of Hub shards** ([#57](https://github.com/shootthesound/Fizgig/pull/57)), no
66 GB merge step; and the **video VAE encodes a clip, not just a still**
([#58](https://github.com/shootthesound/Fizgig/pull/58)), removing the two guards that pinned
encode to a single frame.

## 0xDELUXA

[0xDELUXA](https://github.com/0xDELUXA) keeps the **ROCm path current with AMD's stack**: the
measured removal of the gfx12 batched-GEMM workaround
([#91](https://github.com/shootthesound/Fizgig/pull/91)), benchmarked on real gfx1200 hardware
where the old setting cost ~7.5x on the batched path and did nothing for the trainer, and the
**bitsandbytes wheel bump to the HIP 7.16 build**
([#103](https://github.com/shootthesound/Fizgig/pull/103)) so a current nightly install stops
falling back a minor version on every launch.

## rocketsvm

[rocketsvm](https://github.com/rocketsvm) fixed the **GPU selection mismatch on multi-GPU
Windows machines** ([#104](https://github.com/shootthesound/Fizgig/issues/104) →
[#105](https://github.com/shootthesound/Fizgig/pull/105)): NVML and CUDA can enumerate cards
in different orders, so picking "GPU 0" could train on a different card. The GPU chooser now
carries each card's UUID, which is the same across NVML, nvidia-smi and torch.

## FNGarvin

[FNGarvin](https://github.com/FNGarvin) has contributed a string of high-quality features and
fixes, including:

- **The Metadata tab and rich safetensors metadata** ([#34](https://github.com/shootthesound/Fizgig/pull/34)) —
  trigger phrase, auto-embedded thumbnails, and full ModelSpec blocks on every checkpoint;
  the headline feature of v3.0.5.
- **Fully-offline captioning via a local tokenizer folder** ([#38](https://github.com/shootthesound/Fizgig/pull/38)),
  which became the path to zero-internet captioning, plus the Captions-tab path hints.
- **Florence-2 supply-chain hardening** ([#32](https://github.com/shootthesound/Fizgig/pull/32)) —
  pinning `trust_remote_code` revisions, including the subtle cross-repo `auto_map` case.
- **Pod entrypoint robustness** ([#33](https://github.com/shootthesound/Fizgig/pull/33)) — loud
  failures instead of silently running stale code on a bad `FIZGIG_REF`.
- **Optional SSH on pods** ([#30](https://github.com/shootthesound/Fizgig/pull/30)) — off by
  default, wired to RunPod's own `PUBLIC_KEY` convention.
- **`HF_TOKEN` handling in the model downloader** ([#29](https://github.com/shootthesound/Fizgig/pull/29)),
  and the UI truthfulness fix in [#6](https://github.com/shootthesound/Fizgig/pull/6).

**A note on the record:** the PRs above (except #6) show as *closed* rather than *merged* — a
back-end merge script applied them to master under the maintainer's authorship and closed them.
That was wrong, and it erased FNGarvin's contribution record. Retroactive `Co-authored-by`
attribution commits were added for each one (512e6cc…2e9ba2d), and contributor PRs are merged
normally — authorship intact — from here on.
