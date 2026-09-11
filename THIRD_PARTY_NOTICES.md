# Third-Party Notices

Fizgig is licensed under the Apache License, Version 2.0 (see `LICENSE`).

It includes code derived from the third-party projects listed below. Each
component remains under its upstream license. Permissive components (Apache-2.0,
MIT) are compatible with Fizgig's Apache-2.0 license; where those files were
modified, Fizgig's changes are released under Apache-2.0. Copyleft components
(GPL-3.0) are marked separately and stay under GPL-3.0 — see the comfyui-rocm
section below.

---

## musubi-tuner — Apache License 2.0

Upstream: https://github.com/kohya-ss/musubi-tuner
Copyright the musubi-tuner authors (kohya-ss and contributors).

The following Krea 2 modules are adapted (and modified) from musubi-tuner:

- `src/fizgig/krea2/offloading.py` (block-swap offloader)
- `src/fizgig/krea2/fp8_optimization_utils.py` (fp8 quantization)
- `src/fizgig/krea2/safetensors_utils.py` (mmap safetensors I/O)
- `src/fizgig/krea2/lora_utils.py`
- `src/fizgig/krea2/attention.py` (attention dispatch)
- `src/fizgig/krea2/vae_loader.py` (Qwen-Image VAE loader/converter)
- training hooks (gradient checkpointing, block-swap wiring) in
  `src/fizgig/krea2/model.py` and the flow-matching training recipe in
  `src/fizgig/krea2/trainer.py`

Licensed under the Apache License, Version 2.0:
http://www.apache.org/licenses/LICENSE-2.0

---

## ai-toolkit (Ostris, LLC) — MIT License

Upstream: https://github.com/ostris/ai-toolkit

The Krea 2 single-stream MMDiT backbone (`src/fizgig/krea2/model.py`) and the
functional flow-matching sampler (`src/fizgig/krea2/sampling.py`) are ported
from ai-toolkit's `extensions_built_in/diffusion_models/{krea2,flux2}/src`,
then adapted for Fizgig.

```
MIT License

Copyright (c) 2024 Ostris, LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## FLUX (Black Forest Labs) — Apache License 2.0

Upstream: https://github.com/black-forest-labs/flux

The Klein 9B DiT (`src/fizgig/klein/model.py`) is a Fizgig-native implementation
based on the FLUX reference model code. Licensed under the Apache License,
Version 2.0. (Model weights are distributed separately under their own license —
see "Note on model weights" below.)

---

## Diffusers / Qwen-Image VAE — Apache License 2.0

Upstream: https://github.com/huggingface/diffusers

`src/fizgig/krea2/vae.py` is copied and modified from the Diffusers
`AutoencoderKLQwenImage` implementation.
Copyright 2025 The Qwen-Image Team, Wan Team, and The HuggingFace Team.
All rights reserved. Licensed under the Apache License, Version 2.0.

---

## comfyui-rocm — GNU General Public License v3.0

Upstream: https://github.com/patientx/comfyui-rocm
Copyright the comfyui-rocm authors (patientx and contributors).

comfyui-rocm is a Windows ROCm build of ComfyUI. Fizgig's AMD ROCm Windows
installer uses code from that project as follows:

- `detect_gpu.py` — copied verbatim from comfyui-rocm's `detect_gpu.py`
  (AMD GPU → gfx architecture detection on Windows). **Unmodified.** This file
  is **GPL-3.0 only**; it is not relicensed under Apache-2.0.
- `install_fizgig_rocm.bat` — Fizgig-authored installer that **calls**
  `detect_gpu.py` and follows the same ROCm-wheel install patterns as
  comfyui-rocm's `install.bat` (adapted for Fizgig's venv and
  requirements). The batch file itself is Apache-2.0; the bundled
  `detect_gpu.py` remains GPL-3.0.
- `run_fizgig_rocm.bat` / `run_fizgig_rocm.sh` — Fizgig-authored launchers.
  Legacy RDNA1/2 SDP overrides, aotriton gating, and MIOpen/rocBLAS tensile
  DB path setup are adapted from comfyui-rocm's `comfyui-rocm.bat`. The
  launchers themselves are Apache-2.0; Windows detection still goes through
  the GPL-3.0 `detect_gpu.py` above.

`detect_gpu.py` is free software: you may redistribute and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. A copy of the GPL-3.0 text is available at
https://www.gnu.org/licenses/gpl-3.0.html and in comfyui-rocm's `LICENSE` file.

To receive source for `detect_gpu.py`, use this repository or the upstream
comfyui-rocm repository linked above.

---

## bitsandbytes Windows ROCm wheel (0xDELUXA) — downloaded at install time

The Windows AMD installer (`install_fizgig_rocm.bat`) downloads a **pinned**
community bitsandbytes wheel from:

https://github.com/0xDELUXA/bitsandbytes_win_rocm

That wheel is **not** built by AMD or Fizgig; it is only fetched on the AMD
install path. Official ROCm PyTorch wheels similarly come from AMD's nightly
index (`https://rocm.nightlies.amd.com/whl-multi-arch/`), also disclosed by the
installer before download.

---

## Note on model weights

The third-party notices above cover **source code** only. Krea 2 / FLUX.2 model
weights, the Qwen-Image VAE weights, and the Qwen3-VL text-encoder weights are
distributed by their respective publishers under their own model licenses, which
the user accepts when downloading them. Fizgig does not redistribute any model
weights.
