"""The MiniMax-H3-Image DiT: MiniMax H3's DiT (minimax/model.py, loaded by minimax/loader.py) with three changes the
checkpoint makes, applied after the load:

* no audio stream: the file has no audio_patch_proj / final_layer.audio_out, so they are dropped and no silence rows
  are packed (ComfyUI's MiniMaxH3ImageModel: audio_latents_dim=None, PackedLayout(..., audio_t=0));
* one output head per sampling step: final_layer.video_out is [6 x 96, 5376]. Row block 0 is a full head and blocks
  1-5 are offsets from it; a step from grid sigma k to grid sigma m uses head 0 plus the dt-weighted mean of the offsets
  it spans (ComfyUI comfy/ldm/minimax/model.py FinalLayer / _pdd_head), so the 6-step schedule uses head k at step k;
* its own sigma grid (the file's pdd_sigmas), not H3's shifted uniform one.

The head for a forward comes from its timestep, as ComfyUI picks it from the sigma: the grid point nearest the sigma
starts the span and the next one ends it (`span_for`). A sampler running another step count sets `span_override` for
each step (sampling.py), the same span ComfyUI's sampler gives.
"""
import json
import struct

import torch
import torch.nn.functional as F

from fizgig.minimax.model import FinalLayer, MiniMaxH3DiT

# MiniMaxAI/MiniMax-H3-Image pdd_config.json, and the Comfy-Org file's metadata (config.transformer.pdd_sigmas)
PDD_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)


def file_pdd_sigmas(path):
    """The sigma grid a checkpoint's heads were distilled on, from its safetensors metadata; PDD_SIGMAS when the file
    does not say (the official shards keep it in pdd_config.json beside them)."""
    try:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n)).get("__metadata__") or {}
        cfg = json.loads(meta.get("config") or "{}").get("transformer") or {}
        s = cfg.get("pdd_sigmas")
        if s and len(s) >= 2:
            return tuple(float(v) for v in s)
    except Exception:
        pass
    return PDD_SIGMAS


def nearest_grid(grid, sigma):
    return min(range(len(grid)), key=lambda k: abs(grid[k] - float(sigma)))


class PDDFinalLayer(FinalLayer):
    """FinalLayer with the checkpoint's bank of per-step heads (set up by `convert`)."""

    def span_for(self, t):
        """(start, stop) grid intervals for a forward at flow time t (= 1 - sigma)."""
        if self.span_override is not None:
            return self.span_override
        n = len(self.pdd_sigmas) - 1
        sigma = 1.0 - float(t.reshape(-1)[0]) if torch.is_tensor(t) else 1.0 - float(t)
        start = min(nearest_grid(self.pdd_sigmas, sigma), n - 1)
        return start, start + 1

    def head(self, start, stop):
        """The weight and bias of the step from grid[start] to grid[stop] (ComfyUI _pdd_head)."""
        g = self.pdd_sigmas
        n = len(g) - 1
        W, b = self.video_out.weight, self.video_out.bias
        rows, brows = W.view(n, -1, W.shape[1]), b.view(n, -1)
        first = max(start, 1)
        if first >= stop:
            return rows[0], brows[0]
        dt = torch.tensor([g[k] - g[k + 1] for k in range(start, stop)], dtype=W.dtype, device=W.device)
        w = (dt / dt.sum())[first - start:]
        return (rows[0] + torch.einsum("n,noi->oi", w, rows[first:stop]),
                brows[0] + torch.einsum("n,no->o", w, brows[first:stop]))

    def forward(self, x_video, t_emb, t_index: int = 0):
        shift, scale = self.adaln_proj(t_emb)
        hv = self.norm(x_video) * (1.0 + scale[t_index]) + shift[t_index]
        span = self._spans.pop(0) if self._spans else (0, 1)
        W, b = self.head(*span)
        return F.linear(hv.to(W.dtype), W, b)


class MiniMaxH3ImageDiT(MiniMaxH3DiT):
    """MiniMaxH3DiT whose forwards queue each prediction's head span for the PDD final layer."""

    def forward(self, video_latent, t, text_embeds, *args, **kwargs):
        if kwargs.get("_pack_only"):
            return super().forward(video_latent, t, text_embeds, *args, **kwargs)
        self.final_layer._spans = [self.final_layer.span_for(t)]
        try:
            return super().forward(video_latent, t, text_embeds, *args, **kwargs)
        finally:
            self.final_layer._spans = []

    def forward_packed(self, items):
        self.final_layer._spans = [self.final_layer.span_for(it[1]) for it in items]
        try:
            return super().forward_packed(items)
        finally:
            self.final_layer._spans = []

    def forward_cached(self, video_latent, t, text_embeds, *args, **kwargs):
        self.final_layer._spans = [self.final_layer.span_for(t)]
        try:
            return super().forward_cached(video_latent, t, text_embeds, *args, **kwargs)
        finally:
            self.final_layer._spans = []


def convert(dit, sigmas=PDD_SIGMAS):
    """A loaded MiniMaxH3DiT holding an H3-Image checkpoint -> MiniMaxH3ImageDiT, in place. The loader leaves the two
    audio Linears on the meta device (the file has no weights for them) and puts the file's [576, 5376] head bank in
    final_layer.video_out."""
    n = len(sigmas) - 1
    d = dit.latents_dim * dit.patch_size[0] * dit.patch_size[1] * dit.patch_size[2]
    if tuple(dit.final_layer.video_out.weight.shape) != (n * d, dit.hidden_size):
        raise ValueError(f"not a MiniMax-H3-Image checkpoint: final_layer.video_out is "
                         f"{tuple(dit.final_layer.video_out.weight.shape)}, expected ({n * d}, {dit.hidden_size}) "
                         f"for {n} per-step heads")
    dit.__class__ = MiniMaxH3ImageDiT
    del dit.audio_patch_proj
    fl = dit.final_layer
    del fl.audio_out
    fl.__class__ = PDDFinalLayer
    fl.pdd_sigmas = tuple(float(s) for s in sigmas)
    fl.span_override = None
    fl._spans = []
    dit.pack_audio_rows = False
    return dit


def load_dit(path, device, *, base_quant="int8", blocks_to_swap=0, compute_dtype=torch.bfloat16):
    """The H3-Image DiT through H3's loader (int8 ConvRot kept, or NF4 / HQQ), converted."""
    from fizgig.minimax.loader import load_minimax_h3_dit
    dit = load_minimax_h3_dit(path, device=device, compute_dtype=compute_dtype, quantize=True,
                              blocks_to_swap=blocks_to_swap, base_quant=base_quant, adaln_fp32=True)
    return convert(dit, file_pdd_sigmas(path)).requires_grad_(False)
