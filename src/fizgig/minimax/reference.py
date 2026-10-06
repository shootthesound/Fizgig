# This file is derived from ComfyUI (https://github.com/comfyanonymous/ComfyUI), Copyright (C) comfyanonymous and
# the ComfyUI contributors, which is licensed under the GNU General Public License v3.0. This file therefore stays
# under GPL-3.0 and is not covered by Fizgig's Apache-2.0 licence. See THIRD_PARTY_NOTICES.md (ComfyUI).
"""MiniMax H3 r2v reference images: sizing and encoding.

Transcribed from comfy_extras/nodes_minimax_h3.py::MiniMaxH3ReferenceToVideo.execute, which is
what produces the reference blocks the r2v workflow feeds the DiT. The sizing is not cosmetic:
the reference's latent grid sets how many condition rows it contributes and what its
area-normalized rope coordinates are, so a reference resized differently is a different
conditioning signal even though the picture looks the same.

Two modes, both DOWN-ONLY (a reference is never upscaled):

  match — scale the reference to the GENERATION's pixel area. Fewer tokens, faster.
          scale = min(1, sqrt((gen_w * gen_h) / (ref_w * ref_h)))
  max   — scale so the short edge is at most 2048. The node's note: "best identity fidelity",
          at the cost of speed, since reference tokens ride through every sampling step.

Both then round each side to a multiple of 32 (CANVAS_MULTIPLE) with a floor of one multiple.
"""

import math

import torch

CANVAS_MULTIPLE = 32          # every canvas dimension is a multiple of this
REF_IMAGE_SHORT_EDGE = 2048   # the "max" mode's short-edge cap
VAE_SPATIAL = 16              # the video VAE's spatial downscale


def reference_canvas(ref_w: int, ref_h: int, gen_w: int, gen_h: int, mode: str = "match"):
    """-> (target_w, target_h) for a reference image. Down-only, 32-px multiples."""
    if mode == "match":
        scale = min(1.0, math.sqrt((gen_w * gen_h) / float(ref_w * ref_h)))
    elif mode == "max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / float(min(ref_w, ref_h)))
    else:
        raise ValueError(f"reference sizing mode must be 'match' or 'max', got {mode!r}")
    tw = max(CANVAS_MULTIPLE, round(ref_w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(ref_h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return int(tw), int(th)


def resize_reference(image, gen_w: int, gen_h: int, mode: str = "match"):
    """PIL image -> PIL image on the reference canvas.

    THE SAME resized image must go to BOTH the VAE (for the condition rows) and the Qwen3-VL
    vision blocks (for the conditioning). The node does exactly that — `resized` is passed to
    vae.encode AND appended to ref_items — and it matters: handed the original, the image
    processor picks its own grid from its own pixel budget, so the vision block describes a
    different-sized picture than the condition rows do. Both still run, and the conditioning is
    simply wrong."""
    from PIL import Image
    tw, th = reference_canvas(image.width, image.height, gen_w, gen_h, mode)
    return image.convert("RGB").resize((tw, th), Image.LANCZOS)


def reference_to_tensor(image):
    """PIL image -> [1, 3, H, W] float tensor in [-1, 1], the range the VAE encoder expects
    (it rescales to [0, 1] and applies the imagenet normalization itself)."""
    arr = torch.from_numpy(_to_array(image.convert("RGB"))).float()
    return (arr.permute(2, 0, 1).unsqueeze(0) / 255.0) * 2.0 - 1.0


def prepare_reference_image(image, gen_w: int, gen_h: int, mode: str = "match"):
    """PIL image -> [1, 3, H, W] float tensor in [-1, 1] on the reference canvas."""
    return reference_to_tensor(resize_reference(image, gen_w, gen_h, mode))


def _to_array(img):
    import numpy as np
    return np.asarray(img, dtype="uint8")


def reference_image_for_latent(path_or_image, latent):
    """The reference image, framed EXACTLY as the picture its cached latent was encoded from.

    The latent came from the training pipeline, which fits an image to its bucket with
    resize_image_to_bucket: aspect-preserving scale, then a CENTRE CROP. A plain resize to the
    same width/height instead STRETCHES, so for any photo whose aspect differs from its bucket
    the vision block would describe a squashed picture while the latent describes a cropped one —
    the teacher shown one framing and conditioned on another, with nothing to reveal it.

    So this goes through the same shared helper the trainer uses, sized from the latent itself
    (latent h,w x 16), rather than re-deriving the geometry.
    """
    from PIL import Image

    from fizgig.dataset.image_dataset import resize_image_to_bucket
    img = (Image.open(path_or_image).convert("RGB")
           if isinstance(path_or_image, str) else path_or_image.convert("RGB"))
    w, h = int(latent.shape[-1]) * VAE_SPATIAL, int(latent.shape[-2]) * VAE_SPATIAL
    return Image.fromarray(resize_image_to_bucket(img, (w, h)))


@torch.no_grad()
def encode_reference(vae, image, gen_w: int, gen_h: int, mode: str = "match", device="cuda",
                     dtype=torch.float32):
    """PIL image -> normalized reference latent [1, 24, 1, h, w], ready to pack as condition rows."""
    px = prepare_reference_image(image, gen_w, gen_h, mode).to(device=device, dtype=dtype)
    return vae.encode(px)
