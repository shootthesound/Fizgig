"""Vimoo's Studio engine: the shared video engine plus image-to-video from a start picture (Studio's First frame).

A start picture becomes what Vimoo's image-to-video takes: its latent (the clip's first latent frame, anchored while
the clip denoises) and SigLIP's 729 tokens of it. Vimoo has no last-frame conditioning, so only the First frame slot
is offered."""
import gc

import numpy as np
import torch

from fizgig.families.video_workbench import VideoWorkbenchEngine


class VimooWorkbenchEngine(VideoWorkbenchEngine):
    supports_keyframes = True
    keyframe_slots = ("first",)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._siglip = None

    def ensure_pipeline(self, *a, precision="auto", **kw):
        """Auto on a card under 20 GB loads NF4 (as training's Auto there): the room it leaves matters more on the
        smaller machines this model is expected to run on than INT8's slightly closer preview. Up to 32 GB it loads
        INT8: bf16 (19.3 GB) plus a clip's decode filled a 32 GB card (31.4 GB at 768x768 x 49), leaving nothing for
        the page to play the clip with. bf16 from 40 GB up."""
        from fizgig.families import quant
        from fizgig.families.train import _small_card_previews
        if precision == "auto" and self.pipeline is None:
            if _small_card_previews():
                precision = "nf4"
            elif quant.card_gb() < quant.NOGC_MIN_CARD_GB:
                precision = "int8"
        return super().ensure_pipeline(*a, precision=precision, **kw)

    @torch.no_grad()
    def encode_keyframe(self, image, width, height):
        """A cropped PIL picture -> {"latent": (1, 48, 1, h, w), "vision": (729, 1152)} at the clip's canvas, on CPU."""
        from PIL import Image
        from fizgig.utils.image import flatten
        w, h = (int(width) // 32) * 32, (int(height) // 32) * 32
        img = flatten(image)
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)
        if self.lowmem:
            self._park_dit("cpu")
            self.vae.to(self.device)
        try:
            z = self.driver.encode_images(self.vae, [np.asarray(img)])[0]            # 48, h, w
            vision = self._siglip_encoder().encode(img, w, h)
        finally:
            if self.lowmem:
                self.vae.to("cpu")
                self._free_siglip()
                self._park_dit(self.device)
        return {"latent": z[None, :, None].float().cpu(), "vision": vision.to(torch.float16)}

    def _siglip_encoder(self):
        if self._siglip is None:
            import os
            from fizgig.vimoo.text import SigLIP
            path = getattr(self.driver, "_siglip_path", "")
            if not (path and os.path.isfile(path)):
                raise ValueError("A start frame needs SigLIP so400m (Preferences, Vimoo's model files)")
            self._siglip = SigLIP(path, self.device)
        return self._siglip

    def _free_siglip(self):
        self._siglip = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _clip_conditioning(self, state, cond, width, height, frames):
        first = next((hit for idx, hit in (getattr(state, "keyframes", None) or []) if int(idx) == 0), None)
        if first is None or int(frames) <= 1:
            return cond, {}
        return {**cond, "vision": first["vision"]}, {"refs": [first["latent"]]}
