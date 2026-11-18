# Contributors

Fizgig is written by [Peter Neill (shootthesound)](https://github.com/shootthesound).

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

Since then they've kept the AMD path working: a safe ROCm update ([#107](https://github.com/shootthesound/Fizgig/pull/107)), the nightly
wheels behind `--experimental` ([#129](https://github.com/shootthesound/Fizgig/pull/129)), RDNA1/2 attention overrides and the MIOpen /
rocBLAS paths ([#130](https://github.com/shootthesound/Fizgig/pull/130)), and the ROCm settings and VRAM readout for the web UI
([#192](https://github.com/shootthesound/Fizgig/pull/192)). In November they made AMD installs default to AMD's stable ROCm index ([#319](https://github.com/shootthesound/Fizgig/pull/319)) with the experimental and nightly indexes overridable ([#340](https://github.com/shootthesound/Fizgig/pull/340)), added Preferences' **Disable GPU for the app window** ([#325](https://github.com/shootthesound/Fizgig/pull/325)), and bisected a Krea 2 LoKR out-of-memory on AMD ([#341](https://github.com/shootthesound/Fizgig/issues/341)).

## Murf-Code

[Murf-Code](https://github.com/Murf-Code) brought **MOSS-Image** to Fizgig
([#195](https://github.com/shootthesound/Fizgig/pull/195), v8.5): OpenMOSS's 5B image model as a training family, with LoRAs, previews on
MOSS-Image SFT and the workbench tools. Since then: a **pluggable face finder and look match**
([#190](https://github.com/shootthesound/Fizgig/pull/190)) so furry, anime and drawn characters get likeness scores too; Resume taking the
learning rate from the form ([#214](https://github.com/shootthesound/Fizgig/pull/214)); three bug fixes in one ([#259](https://github.com/shootthesound/Fizgig/pull/259)); the
**loss graph views** ([#290](https://github.com/shootthesound/Fizgig/pull/290)); **Prodigy, Muon and Automagic v4** ([#292](https://github.com/shootthesound/Fizgig/pull/292)); and
Stop asking for a second click ([#294](https://github.com/shootthesound/Fizgig/pull/294)). From their Fizgig Advanced fork, with credit: Adaptive LR judging plateaus on a 3-epoch average, the Samples gallery's
**Style score**, the **Cosmic Pigeon** and **Deep Space** themes (v9.4.0, their own commits) and
**out-of-memory recovery**, which tries a failed run again lighter (v9.6.0), and **Choose…** opening your system's own file window on the computer running Fizgig (v9.7.8, their commit).

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
checkpoints throughout when they first reported. They later added the **live Progress section**
to the Training tab ([#138](https://github.com/shootthesound/Fizgig/pull/138)) and keep sending precise bug reports.

## Hell-Bent-Fox

[Hell-Bent-Fox](https://github.com/Hell-Bent-Fox) diagnosed and fixed the **still-preview OOM
on 16 GB cards** running the int8 streamed plan
([#134](https://github.com/shootthesound/Fizgig/issues/134) →
[#109](https://github.com/shootthesound/Fizgig/pull/109)): the tail-block parking that lets a
clip preview's decoder fit ran only for clips, so a still preview loaded the 4.85 GB decoder
into 4 GB of free VRAM, failed twice and switched previews off for the rest of the run. Their
fix parks for stills too and takes the decoder off the card before the parked blocks return.
The PR was first offered in August with the same exact write-up, closed here on a promise that
did not get delivered, and merged as theirs in September once they came back with the log. In October they added
**chunked prefill for long captions** in H3's text encoder on cards older than RTX 30
([#172](https://github.com/shootthesound/Fizgig/pull/172)).

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

## spaghettified26

[spaghettified26](https://github.com/spaghettified26) added **masked training for MiniMax H3
stills** ([#295](https://github.com/shootthesound/Fizgig/pull/295), v9.0.3), with prior preservation outside the mask, and fixed the clip
size row for clips kept in sub-folders ([#278](https://github.com/shootthesound/Fizgig/pull/278)). Their careful memory measurements on a
96 GB card ([#282](https://github.com/shootthesound/Fizgig/pull/282)) found that H3 clip runs with TREAD off could go without gradient
checkpointing and run out of memory; those runs now keep it on.

Since then: **Fibo 2.0** as a trainable family, with measured planner figures ([#250](https://github.com/shootthesound/Fizgig/pull/250), v9.0.4); **Purge run history** on Runs & outputs ([#345](https://github.com/shootthesound/Fizgig/pull/345), v9.7.2); and LoRA Royale treating a run's Save now snapshots and last checkpoint as its own epochs ([#355](https://github.com/shootthesound/Fizgig/pull/355), v9.7.6). Their exact reports found the H3 Context LoRA crash at step 0 ([#298](https://github.com/shootthesound/Fizgig/issues/298)) and MOSS-Image block swap leaving weights on the CPU at a batch size above 1 ([#335](https://github.com/shootthesound/Fizgig/issues/335)). In v9.7.7, a resume at a new size caches again whatever Enable Cache says, and its record waits until training starts ([#370](https://github.com/shootthesound/Fizgig/pull/370)).

## DmitryAkDev

[DmitryAkDev](https://github.com/DmitryAkDev) fixed the **app window on Linux**: closing it no
longer hangs ([#197](https://github.com/shootthesound/Fizgig/pull/197)), and Ctrl+= / Ctrl+- / Ctrl+0 zoom works under WebKitGTK
([#198](https://github.com/shootthesound/Fizgig/pull/198)), both in v8.9.0.

## vishnuteja7054

[vishnuteja7054](https://github.com/vishnuteja7054) built the first **web UI for headless cloud
machines** ([#182](https://github.com/shootthesound/Fizgig/pull/182), from [#165](https://github.com/shootthesound/Fizgig/issues/165)): the schema extractor, the server with its
login, live log and pause, and Start running through the same launch plan as the app. Fizgig's
web UI was built on top of their commit.

## PyroclasticFloe

[PyroclasticFloe](https://github.com/PyroclasticFloe) added **AMD RX 6000-series (gfx10.3)
detection on Linux** ([#281](https://github.com/shootthesound/Fizgig/pull/281), v9.0.3).

## Linkram

[Linkram](https://github.com/Linkram) spotted that a failed Krea 2 preview decode could leave the
model parked ([#152](https://github.com/shootthesound/Fizgig/pull/152)); the fix restores it whatever happens. Their RDNA2 speed-ups for
Krea 2 on the RX 6800 ([#158](https://github.com/shootthesound/Fizgig/pull/158)) were offered and are not merged yet.

## xGnome

[xGnome](https://github.com/xGnome) works on **AMD RDNA4 under Linux**: an optional pinned **SageAttention-RDNA4** build with a
Force SageAttention preference for Krea 2 ([#344](https://github.com/shootthesound/Fizgig/pull/344), v9.7.2, tested with [daxter93](https://github.com/daxter93)), a Training tab that
lists optimizers without loading their libraries ([#346](https://github.com/shootthesound/Fizgig/pull/346), v9.7.5), and Krea 2's VAE decoding a still picture
with about 2 GB less VRAM ([#369](https://github.com/shootthesound/Fizgig/pull/369), v9.7.6). They also found three helpers missing from the Linux AMD installer
(v9.7.2) and where Arch-based systems keep libnuma ([#356](https://github.com/shootthesound/Fizgig/issues/356)). In v9.7.8, Krea 2 previews fit 16 and 24 GB AMD cards at high resolutions
([#363](https://github.com/shootthesound/Fizgig/pull/363)), and their investigation found that expandable memory segments corrupt memory on Radeon cards with current ROCm, which is why Fizgig turns them off there.

## pastuh

[pastuh](https://github.com/pastuh) fixed two **prompt slider** bugs ([#361](https://github.com/shootthesound/Fizgig/pull/361), v9.7.6): a MOSS-Image slider with the Turbo LoRA
stopped before training, and a slider prompt or anchor starting with "-" didn't reach the trainer as typed.

## Ideas and bug reports

Features and fixes that started as someone's issue, credited in the release notes:
[driqeks](https://github.com/driqeks) (slider anchor prompts, [#252](https://github.com/shootthesound/Fizgig/issues/252)),
[zer0signa1](https://github.com/zer0signa1) (LoRA Royale likeness from several photos,
[#263](https://github.com/shootthesound/Fizgig/issues/263)), [VRAM-Hoarder](https://github.com/VRAM-Hoarder) (sample prompt sets,
[#267](https://github.com/shootthesound/Fizgig/issues/267)), [girzai](https://github.com/girzai) (Z-Image Base, [#261](https://github.com/shootthesound/Fizgig/issues/261)),
[Dwuuds](https://github.com/Dwuuds) (the 8B captioning model, [#270](https://github.com/shootthesound/Fizgig/issues/270)),
[bkrew](https://github.com/bkrew) (wording for style, object and concept LoRAs, [#244](https://github.com/shootthesound/Fizgig/issues/244)),
[mart-hill](https://github.com/mart-hill) (the mask painter's brush, [#271](https://github.com/shootthesound/Fizgig/issues/271)),
[crazyscale-dotcom](https://github.com/crazyscale-dotcom) (ComfyUI checkpoints for Krea 2,
[#273](https://github.com/shootthesound/Fizgig/issues/273)), [Klow666](https://github.com/Klow666) (the auto-recaption stall, [#275](https://github.com/shootthesound/Fizgig/issues/275)),
[CognitiveDiffusion](https://github.com/CognitiveDiffusion) (transparent pictures,
[#284](https://github.com/shootthesound/Fizgig/issues/284)), [tsk5cb-cmyk](https://github.com/tsk5cb-cmyk) (Studio's typed values and saved
setups, [#286](https://github.com/shootthesound/Fizgig/issues/286)), [LeonKLAKMUF](https://github.com/LeonKLAKMUF) (the app window's
right-click menu, [#296](https://github.com/shootthesound/Fizgig/issues/296)), [gradyhouston](https://github.com/gradyhouston) (the H3
reference distillation crash, [#264](https://github.com/shootthesound/Fizgig/issues/264)) and [amoszach](https://github.com/amoszach)
(taking a folder off the training data list, [#266](https://github.com/shootthesound/Fizgig/issues/266)). [Nynxz](https://github.com/Nynxz)'s ZenTrainer gave Fizgig the ideas for
Start at a smaller size, the run summary, the sidebar meters and the captions table.

Since November:

- [tsk5cb-cmyk](https://github.com/tsk5cb-cmyk): the logs and OneTrainer settings behind 9.2's speed work ([#221](https://github.com/shootthesound/Fizgig/issues/221)), and the case for repeats from folder names ([#300](https://github.com/shootthesound/Fizgig/issues/300))
- [pomoe136](https://github.com/pomoe136): Extract on MOSS-Image ([#301](https://github.com/shootthesound/Fizgig/issues/301))
- [spinolover](https://github.com/spinolover): a Fast Identity run error ([#302](https://github.com/shootthesound/Fizgig/issues/302))
- [Nessa2026691](https://github.com/Nessa2026691): Context LoRAs in LoKR format ([#304](https://github.com/shootthesound/Fizgig/issues/304)) and tracking down the Context LoRA slowdown ([#307](https://github.com/shootthesound/Fizgig/issues/307)), and the diagnosis and fix for MiniMax H3 helper previews on 16 GB cards ([#372](https://github.com/shootthesound/Fizgig/issues/372))
- [MrB4000](https://github.com/MrB4000): a MOSS-Image LoRA that wouldn't start ([#305](https://github.com/shootthesound/Fizgig/issues/305))
- [arkinson9](https://github.com/arkinson9): MOSS-Image preview quality on small cards ([#309](https://github.com/shootthesound/Fizgig/issues/309))
- [josm2026](https://github.com/josm2026): the installer's "from was unexpected" error, with the fix ([#310](https://github.com/shootthesound/Fizgig/issues/310))
- [RitschRatsch666](https://github.com/RitschRatsch666): fine-tunes training with the optimizer you pick ([#312](https://github.com/shootthesound/Fizgig/issues/312))
- [mart-hill](https://github.com/mart-hill): JoyCaption ([#313](https://github.com/shootthesound/Fizgig/issues/313)) and the big caption view a line at a time
- [axymeus](https://github.com/axymeus): the WD tagger ([#315](https://github.com/shootthesound/Fizgig/issues/315)), the regularisation dataset ([#332](https://github.com/shootthesound/Fizgig/issues/332)), LoRA Royale on a range of epochs ([#318](https://github.com/shootthesound/Fizgig/issues/318)) and Gizmo's crop box ([#367](https://github.com/shootthesound/Fizgig/issues/367))
- [AlexWolfgangDRG](https://github.com/AlexWolfgangDRG): JoyCaption in 4-bit on ROCm ([#316](https://github.com/shootthesound/Fizgig/issues/316)) and a Continue training problem ([#333](https://github.com/shootthesound/Fizgig/issues/333))
- [CognitiveDiffusion](https://github.com/CognitiveDiffusion): Resume Training's place on the page ([#317](https://github.com/shootthesound/Fizgig/issues/317)), long runs and the queue ([#348](https://github.com/shootthesound/Fizgig/issues/348)), Gizmo's crop box ([#350](https://github.com/shootthesound/Fizgig/issues/350)) and Automagic v4's braked-weights readout ([#349](https://github.com/shootthesound/Fizgig/issues/349))
- [Kaoru8](https://github.com/Kaoru8): every setting the previews use, on the Samples tab ([#330](https://github.com/shootthesound/Fizgig/issues/330))
- [daxter93](https://github.com/daxter93): ROCm crashes on tight VRAM ([#321](https://github.com/shootthesound/Fizgig/issues/321)), high-resolution renders on AMD ([#343](https://github.com/shootthesound/Fizgig/issues/343)) and testing SageAttention-RDNA4
- [Ferocit](https://github.com/Ferocit): download progress ([#327](https://github.com/shootthesound/Fizgig/issues/327))
- [jordanjay29](https://github.com/jordanjay29): steps remaining in the epoch ([#328](https://github.com/shootthesound/Fizgig/issues/328))
- [DaRoamer](https://github.com/DaRoamer): H3 Image edit LoRAs' text caching on 24 GB cards ([#334](https://github.com/shootthesound/Fizgig/issues/334))
- [8-lab1627](https://github.com/8-lab1627): Krea 2 LoKR speed on 16 GB cards ([#337](https://github.com/shootthesound/Fizgig/issues/337))
- [raymondgp](https://github.com/raymondgp): helper previews ([#339](https://github.com/shootthesound/Fizgig/issues/339))
- [johnstephendonahue-ux](https://github.com/johnstephendonahue-ux): which card an AMD PC trains on ([#342](https://github.com/shootthesound/Fizgig/issues/342))
- [Aleksandr-AI90](https://github.com/Aleksandr-AI90): ACE-Step's song encoding memory ([#347](https://github.com/shootthesound/Fizgig/issues/347))
- [RandomTurtle](https://github.com/RandomTurtle): Resume Training folder mix-ups ([#352](https://github.com/shootthesound/Fizgig/issues/352))
- [prookyon](https://github.com/prookyon): Automagic v4's real rate ([#353](https://github.com/shootthesound/Fizgig/issues/353)) and several captions per picture, as OneTrainer writes them ([#300](https://github.com/shootthesound/Fizgig/issues/300))
- [howDoILearnPython](https://github.com/howDoILearnPython): H3 Image on 12 GB cards ([#354](https://github.com/shootthesound/Fizgig/issues/354))
- [Aymenbayoss](https://github.com/Aymenbayoss): H3 at int8 on pruned files ([#357](https://github.com/shootthesound/Fizgig/issues/357)) and stopping a captioning model while it loads ([#366](https://github.com/shootthesound/Fizgig/issues/366))
- [girzai](https://github.com/girzai): big datasets on a pod, now downloaded onto it from the Start tab ([#308](https://github.com/shootthesound/Fizgig/issues/308))
- [CrypticCherry](https://github.com/CrypticCherry): repeats from folder names and several captions per picture ([#300](https://github.com/shootthesound/Fizgig/issues/300))
- [whitejt2](https://github.com/whitejt2): checks that failed on smaller cards ([#359](https://github.com/shootthesound/Fizgig/pull/359))
- [ofung30-sketch](https://github.com/ofung30-sketch): LoRA Isolation with Automagic ([#364](https://github.com/shootthesound/Fizgig/issues/364))
- [Marhalt](https://github.com/Marhalt): saying what your subject is ([#365](https://github.com/shootthesound/Fizgig/issues/365))
- WiseDuck (by email): Gizmo's **Prepare a folder** and clips at any audio sample rate
- u/Obvious-Bad7548 on r/StableDiffusion: the LCM sampler recipe for H3 Image

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
- **A model-family check before Royale, Explorer and Studio load a pipeline** ([#63](https://github.com/shootthesound/Fizgig/pull/63)), the
  installer's uv hardlinks with a fallback ([#66](https://github.com/shootthesound/Fizgig/pull/66)), and the Pillow 12.3 bump ([#68](https://github.com/shootthesound/Fizgig/pull/68)).

**A note on the record:** the PRs above (except #6, #63, #66 and #68) show as *closed* rather than *merged* — a
back-end merge script applied them to master under the maintainer's authorship and closed them.
That was wrong, and it erased FNGarvin's contribution record. Retroactive `Co-authored-by`
attribution commits were added for each one (512e6cc…2e9ba2d), and contributor PRs are merged
normally — authorship intact — from here on.
