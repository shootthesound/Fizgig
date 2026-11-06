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
        smaller machines this model is expected to run on than INT8's slightly closer preview. From 20 GB up the
        shared rule loads bf16 where the file fits with room: INT8 mottles plain backgrounds (measured 6 Nov 2026), and
        with the VAE parked after each decode and the render's cache handed back, a 32 GB card keeps room for playing
        the clips."""
        from fizgig.families.train import _small_card_previews
        if precision == "auto" and self.pipeline is None and _small_card_previews():
            precision = "nf4"
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
