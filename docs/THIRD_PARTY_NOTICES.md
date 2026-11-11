# Third-Party Notices

Fizgig is licensed under the Apache License, Version 2.0 (see `LICENSE`).

It includes code derived from the third-party projects listed below. Each
component remains under its upstream license. Permissive components (Apache-2.0,
MIT) are compatible with Fizgig's Apache-2.0 license; where those files were
modified, Fizgig's changes are released under Apache-2.0. Copyleft components
(GPL-3.0) are marked separately and stay under GPL-3.0 — see the comfyui-rocm
and ComfyUI sections below.

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

## sd-scripts — Apache License 2.0

Upstream: https://github.com/kohya-ss/sd-scripts
Copyright the sd-scripts authors (kohya-ss and contributors); the model code it
carries is NVIDIA CORPORATION & AFFILIATES' (Apache-2.0).

- `src/fizgig/anima/model.py` (Anima's DiT: Cosmos-Predict2 MiniTrainDIT and the
  LLM adapter), adapted from `library/anima_models.py` and modified: PyTorch SDPA
  attention, plain non-reentrant gradient checkpointing, no block swap / CPU
  offload.

Licensed under the Apache License, Version 2.0:
http://www.apache.org/licenses/LICENSE-2.0

---

## ai-toolkit (Ostris, LLC) — MIT License

Upstream: https://github.com/ostris/ai-toolkit

The Krea 2 single-stream MMDiT backbone (`src/fizgig/krea2/model.py`) and the
functional flow-matching sampler (`src/fizgig/krea2/sampling.py`) are ported
from ai-toolkit's `extensions_built_in/diffusion_models/{krea2,flux2}/src`,
then adapted for Fizgig.

The Automagic v3 optimizer (`src/fizgig/training/automagic3.py`) is kept
byte-for-byte as upstream's `toolkit/optimizers/automagic3.py`, under a Fizgig
header describing how it is constructed. Offered as an optimizer choice on the
MiniMax H3 and Krea 2 LoRA paths.

The Automagic v4 optimizer (`src/fizgig/training/automagic4.py`) is upstream's
`toolkit/optimizers/automagic4.py` (commit 534d462f), unchanged except that its
base-class import points at Fizgig's vendored Automagic v3.

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

## FLUX 3 Image Klein (Black Forest Labs) — Apache License 2.0

Upstream: https://github.com/black-forest-labs/flux-3-image-klein

`src/fizgig/flux3/` (the DiT, autoencoder, text-encoder wrapper and sampler) is adapted from the FLUX 3 Image Klein reference implementation; its licence is in `src/fizgig/flux3/LICENSE`. Licensed under the Apache License, Version 2.0. (Model weights are distributed separately under the FLUX Kommunity License — see "Note on model weights" below.)

---

## Diffusers / Qwen-Image VAE — Apache License 2.0

Upstream: https://github.com/huggingface/diffusers

`src/fizgig/krea2/vae.py` is copied and modified from the Diffusers
`AutoencoderKLQwenImage` implementation.
Copyright 2025 The Qwen-Image Team, Wan Team, and The HuggingFace Team.
All rights reserved. Licensed under the Apache License, Version 2.0.

---

## Diffusers / Fibo 2.0 — Apache License 2.0

Upstream: https://github.com/huggingface/diffusers

`src/fizgig/fibo2/model.py` is ported from the Diffusers `BriaFibo2Transformer2DModel`
(`models/transformers/transformer_bria_fibo2.py`, v0.42.1) and the text recipe and edit-prompt builder of
`BriaFibo2Pipeline` (`pipelines/bria_fibo2/pipeline_bria_fibo2.py`). Copyright 2026 Bria AI and The HuggingFace
Team. Licensed under the Apache License, Version 2.0. Fizgig's changes are described in the file's header. (The
Fibo 2.0 weights are distributed separately by BRIA under CC BY-NC 4.0.)

---

## Z-Image (Tongyi-MAI / Alibaba) — Apache License 2.0

Upstream: https://github.com/Tongyi-MAI/Z-Image

`src/fizgig/zimage/model.py` (the S3-DiT transformer) and `src/fizgig/zimage/pipeline.py` (text encoding and
sampling) are adapted from the Z-Image reference implementation (`src/zimage/transformer.py`,
`src/zimage/pipeline.py`). Copyright 2025 Alibaba Z-Image Team. Licensed under the Apache License, Version 2.0.
Fizgig's changes are described in each file's header. (Model weights are distributed separately under their own
license — see "Note on model weights" below.)

---

## comfyui-rocm — GNU General Public License v3.0

Upstream: https://github.com/patientx/comfyui-rocm
Copyright the comfyui-rocm authors (patientx and contributors).

comfyui-rocm is a Windows ROCm build of ComfyUI. Fizgig's AMD ROCm Windows
installer uses code from that project as follows:

- `src/install/detect_gpu.py` — copied verbatim from comfyui-rocm's `detect_gpu.py`
  (AMD GPU → gfx architecture detection on Windows). **Unmodified.** This file
  is **GPL-3.0 only**; it is not relicensed under Apache-2.0.
- `install_fizgig_rocm.bat` — Fizgig-authored installer that **calls**
  `detect_gpu.py` and follows the same ROCm-wheel install patterns as
  comfyui-rocm's `install.bat` (adapted for Fizgig's venv and
  requirements). The batch file itself is Apache-2.0; the bundled
  `detect_gpu.py` remains GPL-3.0.
- `Launch Fizgig AMD.bat` / `launch_fizgig_amd.sh` — Fizgig-authored launchers.
  Legacy RDNA1/2 SDP overrides, aotriton gating, and MIOpen/rocBLAS tensile
  DB path setup are adapted from comfyui-rocm's `comfyui-rocm.bat`. The
  launchers themselves are Apache-2.0; Windows detection still goes through
  the GPL-3.0 `detect_gpu.py` above.

`detect_gpu.py` is free software: you may redistribute and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. A copy of the GPL-3.0 text is available at
`LICENSES/GPL-3.0.txt` in this repository (also at
https://www.gnu.org/licenses/gpl-3.0.html and in comfyui-rocm's `LICENSE` file).

To receive source for `detect_gpu.py`, use this repository or the upstream
comfyui-rocm repository linked above.

---

## ComfyUI — GNU General Public License v3.0

Upstream: https://github.com/comfyanonymous/ComfyUI
Copyright (C) comfyanonymous and the ComfyUI contributors.

Fizgig's MiniMax H3 support includes code derived from ComfyUI's MiniMax H3
implementation. These files are **GPL-3.0**; they are not relicensed under
Apache-2.0, and each carries a header saying so:

- `src/fizgig/minimax/model.py` — the H3 DiT, ported from
  `comfy/ldm/minimax/model.py` (ComfyUI's plumbing replaced by plain PyTorch).
- `src/fizgig/minimax/vae.py` — the H3 video VAE, ported from
  `comfy/ldm/minimax/vae.py`.
- `src/fizgig/minimax/audio_vae.py` — the H3 audio VAE, ported from
  `comfy/ldm/minimax/audio_vae.py`.
- `src/fizgig/minimax/reference.py` — the reference-to-video conditioning,
  transcribed from `comfy_extras/nodes_minimax_h3.py`
  (`MiniMaxH3ReferenceToVideo`).
- `src/fizgig/minimax/sampling.py` — the `_res_multistep_coeffs` function only,
  ported from `comfy/k_diffusion/sampling.py` (`res_multistep`). The rest of
  that file is Fizgig's own and Apache-2.0.
- `src/fizgig/minimax_image/vae.py` — the MiniMax-H3-Image VAE's tiling, its 2D
  decoder settings and its encode / decode steps, ported from
  `comfy/ldm/minimax/vae.py` (`split_tiles`, `blend`, `tiled_encode`,
  `tiled_decode`, `H3ImageVAE`, `MiniMaxH3ImageVAE`).

Fizgig's SAM 3.1 masks (Image Prep's Make masks from words) are ported from
ComfyUI's SAM 3 support at commit 2035799. These files are **GPL-3.0** too, each
with a header saying so:

- `src/fizgig/sam3/sam.py`, `tracker.py`, `detector.py` — copied from
  `comfy/ldm/sam3/` with ComfyUI's helpers taken from `_comfy.py`.
- `src/fizgig/sam3/_comfy.py` — the ComfyUI helpers they use (layer classes,
  attention, the FLUX rope functions, LayerNorm2d, a resize) in plain PyTorch.
- `src/fizgig/sam3/segment.py` — the checkpoint key conversion
  (`comfy/supported_models.py`), the text encoder settings
  (`comfy/text_encoders/sam3_clip.py`) and the detect and refine steps
  (`comfy_extras/nodes_sam3.py`).

These files are free software: you may redistribute and/or modify them under
the terms of the GNU General Public License v3.0. A copy of the GPL-3.0 text is
in `LICENSES/GPL-3.0.txt` in this repository (also at
https://www.gnu.org/licenses/gpl-3.0.html and in ComfyUI's
`LICENSE` file). Their source is this repository.

The MiniMax H3 model weights are not part of this code; MiniMax's own licence
applies to them (see the note on model weights below).

---

## bitsandbytes Windows ROCm wheel (0xDELUXA) — downloaded at install time

The Windows AMD installer (`install_fizgig_rocm.bat`) downloads a **pinned**
community bitsandbytes wheel from:

https://github.com/0xDELUXA/bitsandbytes_win_rocm

That wheel is **not** built by AMD or Fizgig; it is only fetched on the AMD
install path. Official ROCm PyTorch wheels similarly come from AMD's stable
index (`https://stable.repo.amd.com/rocm/whl-next/`), also disclosed by the
installer before download.

---

## QR Code Generator for JavaScript (Kazuhiko Arase) — MIT License

`src/fizgig/web/static/qrcode.js` is qrcode-generator 1.4.4, unchanged: <https://github.com/kazuhikoarase/qrcode-generator>. It draws the Start tab's **Open on your phone** code.

```
Copyright (c) 2009 Kazuhiko Arase

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

The word "QR Code" is a registered trademark of DENSO WAVE INCORPORATED.

## LC Speed Boost (lonecatone23) — MIT License

`src/fizgig/minimax/speed_boost.py` (the Studio's H3 **Speed Boost**) adapts the spectral shrink / grow from LC Speed Boost in ComfyUI_LC123_nodes: <https://github.com/lonecatone23/ComfyUI_LC123_nodes>. It implements SPEED ("Spectral Progressive Diffusion for Efficient Image and Video Generation", Xiao, Chao, Yariv and Wetzstein, 2026).

```
Copyright (c) 2026 lonecatone23

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## LiteRT (Google, ai-edge-litert) and the MediaPipe Selfie Multiclass model — Apache License 2.0

Image Prep's **Make masks** (masked training) runs Google's MediaPipe Selfie Multiclass segmentation model
(`selfie_multiclass_256x256.tflite`) on the LiteRT runtime, installed from PyPI as `ai-edge-litert`. The model file is
**not** shipped with Fizgig: it is downloaded from Google's own address the first time masks are made, and checked
against a pinned SHA-256.

https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter

---

## CLIP tokenizer (OpenAI) — MIT License

`src/fizgig/assets/clip_tokenizer/` (the SAM 3.1 text encoder's vocabulary) is
the tokenizer of OpenAI's CLIP ViT-L/14 (`openai/clip-vit-large-patch14`), as
ComfyUI ships it. Copyright (c) 2021 OpenAI, MIT License.

---

## Prodigy (`prodigyopt`, Konstantin Mishchenko) — MIT License

Upstream: https://github.com/konstmish/prodigy

`src/fizgig/training/prodigy.py` is vendored from `prodigyopt` 1.1.2 (`prodigyopt/prodigy.py`) under the MIT License:

```
MIT License

Copyright (c) 2023 Konstantin Mishchenko and Prodigy contributors

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

## taehv (Ollin Boer Bohan, madebyollin) — MIT License

Upstream: https://github.com/madebyollin/taehv

`src/fizgig/minimax/taehv.py` is vendored from taehv (commit 011dfc2, `taehv.py`), and its `taeh3` weights (MiniMax H3's quick-preview decoder) are downloaded from the same repository (commit 62f7591), under the MIT License:

```
MIT License

Copyright (c) 2025 Ollin Boer Bohan

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

## PyTorch (`torch.optim.Muon`) — BSD 3-Clause License

Upstream: https://github.com/pytorch/pytorch

`src/fizgig/training/muon.py` is adapted from PyTorch's `torch/optim/_muon.py` (2.10), used when `torch.optim.Muon` itself can't take the parameters. PyTorch's licence:

```
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

## Note on model weights

The third-party notices above cover **source code** only. Krea 2 / FLUX.2 model
weights, the Qwen-Image VAE weights, the Qwen3-VL text-encoder weights, and the
MiniMax H3 training adapters (circlestone-labs and Ostris), Meta's SAM 3.1 weights (the SAM License) and the taeh3 weights (MIT, above) are distributed by their respective publishers under their own model licenses, which
the user accepts when downloading them. Fizgig does not redistribute any model
weights.
