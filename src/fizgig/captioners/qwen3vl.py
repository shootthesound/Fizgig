"""Qwen3-VL: Krea 2's text encoder (4B), or the secondary captioning model (an 8B of the same architecture), following
the Captions tab's instruction. The same file auto-recaption uses inside a run - which loads it directly, never
through here (the memory rule in fizgig/captioners/__init__.py)."""
import torch

from fizgig.captioners import Captioner, CaptionRequest, CaptionResult


class Qwen3VLCaptioner(Captioner):
    def load(self):
        from fizgig.krea2.utils import load_krea2_text_encoder
        self.encoder = load_krea2_text_encoder(self.files["path"], dtype=torch.bfloat16, device=self.device)

    def caption(self, image, request: CaptionRequest) -> CaptionResult:
        from fizgig.krea2.embedder import generate_caption
        return CaptionResult(text=generate_caption(self.encoder, image, max_new_tokens=request.max_new_tokens,
                                                   instruction=request.instruction or None))

    def unload(self):
        self.encoder = None
