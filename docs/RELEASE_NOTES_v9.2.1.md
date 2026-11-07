# Fizgig v9.2.1: Python 3.11 or newer, for large speed gains on PyTorch 2.15

**Fizgig now needs Python 3.11 or newer.** PyTorch 2.15, which 9.2 moved to, has no Python 3.10 build, so updating an install made with Python 3.10 couldn't install it. Installs on Python 3.11, 3.12 or 3.13 aren't affected, and AMD ROCm installs keep their own PyTorch.

**If your install uses Python 3.10,** the updater now says so and stops before changing anything. To fix it:
1. Install Python 3.12 from https://www.python.org/downloads/ (on Windows, tick "Add Python to PATH").
2. Run `install_fizgig.bat` (Linux: `python3.12 install_fizgig.py`). It rebuilds Fizgig's Python environment on 3.12. Your models, datasets, LoRAs and settings are kept.

If the 9.2 update stopped with "Failed to install/update dependencies", this is why: follow the two steps above.

**Training is much faster, uses far less memory, and likeness is unchanged.** On an RTX 5090, a MOSS-Image LoRA at batch 1 (94 pictures, Muon, 70 epochs) now takes about 20 minutes instead of 49. On batch training, Fizgig now matches OneTrainer's speed on the same card with less memory: MOSS-Image at batch 3 runs at 1.07 s a step in 6.9 GB, against OneTrainer's 1.07 s in 7.4 GB. Likeness comes out the same as 9.0.8 or better. Thanks to @tsk5cb-cmyk, whose logs and OneTrainer settings in #221 showed where the time was going.

**What changes for you**
- **PyTorch 2.15.** With gradient checkpointing on, training uses much less memory, at batch 1 too: Krea 2 22.9 → 14.9 GB, MOSS-Image at batch 3 21.7 → 7.1 GB. Every model trains at the same speed or faster: on an RTX 5090, MOSS-Image 22% and Klein 23% faster; on a 16 GB laptop 4090, Fibo 2.0 22%, Krea 2 11% and Klein 13% faster.
- **Muon is much faster.** Its step now handles a LoRA's matrices together instead of one at a time: 12× faster, with exactly the same results. That's most of the batch-1 speed-up above.
- **Faster batches.** With Batch size above 1, captions of different lengths no longer slow attention down. A batched step is 18% faster on MOSS-Image, 12% on Krea 2, 9% on Qwen Image 2.1 and 8% on Z-Image.
- **LoKR is 4–6× faster** on every model: training with a LoKR Context LoRA (which made training about 5× slower), LoKR training, and LoKR files in Studio and the other tools. Same results. Thanks to @Nessa2026691 for tracking it down (#307).

**Requirements**
- **Python 3.11 or newer** (see the top of these notes).
- **An NVIDIA driver that supports CUDA 13** (version 580 or newer). If Fizgig can't see your GPU after the update, update the driver. RunPod pods are picked to match.
- AMD ROCm installs keep their own PyTorch, so nothing changes there.

**Also in this release**
- **Adaptive LR learns a ceiling.** If a step up in learning rate is followed by a cut within three epochs, later steps up stop below that rate, so a run doesn't climb back to a rate that already failed.
- **Studio:** the ↻ New seed button renders straight away. On MiniMax H3, Show early's mid-render look now uses taeh3, @madebyollin's tiny H3 decoder (22.7 MB): about 10× faster than the full 4.5 GB decoder and using a fraction of the memory, so it no longer stalls a render on a nearly full card. It downloads with the H3 models, or the first time Show early needs it. Until then, or with no connection, Show early uses the full decoder when there's room for it, and says so when there isn't. The finished clip still decodes through the full VAE.
- **No telemetry.** Recent onnxruntime builds for Linux, which Fizgig uses for face detection, report a device ID and machine details to Microsoft unless told not to. Fizgig now switches that off, along with Hugging Face's usage reporting, on every platform. Thanks to the user who emailed about it.
- **Switching optimizer:** changing a preset from Muon or Prodigy to AdamW or AdamW8bit now clears their Optimizer Args. Before, AdamW couldn't use those args and quietly fell back to plain, slower AdamW. Args you typed yourself are kept.
- **Run summary card:** shows the run's real Batch size (it always said 1).

**How it was tested.** A PyTorch upgrade touches everything, so 9.2 was checked more thoroughly than any release before it. The testing ran for close to a full day: from late on 6 November, through the night with pods running, and all of 7 November, with me hands-on throughout: directing each test and checking every result.
- **The core:** RunPod A6000 and A40 pods ran OneTrainer and Fizgig side by side, and found the causes of the speed and memory gap with one-change-at-a-time runs scored for likeness.
- **Every model:** trained on the old and new PyTorch with the same code (Klein, Flux 3 Image Klein, Krea 2, Qwen Image 2.1, MOSS-Image, Z-Image Turbo and Base, SDXL, Anima, Fibo 2.0, MiniMax H3, Vimoo, ACE-Step).
- **Every kind of training:** LoRA, full fine-tune, slider, masked training, loss watch, video clips and songs.
- **The tools:** Studio, Extract, the Profiler, captioning and caption translation.
- **Small cards:** a 16 GB laptop 4090 ran the big models with block swap.
- **Installs:** fresh installs and the updater, on Windows and Linux.

That pod time was paid for by the community: when you rent a pod through Fizgig's RunPod template, my RunPod account gets usage credits, and those credits funded all of this testing. Thank you.

**Updating:** run the updater as usual. It now checks your Python environment first and shows a 10-second countdown: press Space (or wait) to update as normal. This update downloads PyTorch 2.15 (about 3 GB) and the updater says so. If Fizgig won't start or train afterwards, run the updater again and press **R**: it deletes the Python environment and installs it fresh. Your models, datasets, LoRAs and settings are kept.

**Pods:** the pod image moves to Ubuntu 24.04 (Python 3.12) to carry PyTorch 2.15. Restart your pods to pick up 9.2.
