"""Exponential moving average of a trainable adapter — shared by the MiniMax H3 and Krea 2
trainers. Checkpoints and previews are saved from the average; training always runs on the
raw weights (swap_in / swap_out bracket a save or a preview).

Measured on MiniMax H3 (9 Sep 2026, 4-way A/B, 50 epochs, same seed): decay 0.98 sat five
likeness points above EMA-off on the late epochs with half the epoch-to-epoch spread; 0.99
smoothed without lifting the level; 0.995 lagged the run and finished below off."""
import torch


class EMAWeights:
    """Exponential moving average of the trainable adapter — the smooth center of a rough
    trajectory.

    High static LRs take big Adam strides that zigzag around the good solution; the raw
    weights at any single step are one corner of the zigzag, and that roughness reads as
    distortion in samples. The EMA is the running average of the path, so what gets SAVED
    (and previewed) is the center the strides orbit — the standard diffusion-training cure
    for exactly this. Training itself always runs on the raw weights: swap_in/swap_out
    bracket saves and previews only.

    Decay ramps in as min(decay, (1+n)/(10+n)) so the first steps track the weights closely
    instead of anchoring to the zero init. Shadow is fp32 (the adapter is small)."""

    def __init__(self, network, decay: float):
        self.decay = float(decay)
        self.n = 0
        self.params = [p for p in network.parameters() if p.requires_grad]
        self.shadow = [p.detach().clone().float() for p in self.params]
        self._backup = None

    @torch.no_grad()
    def update(self):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for s, p in zip(self.shadow, self.params):
            s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def swap_in(self):
        """Put the averaged weights into the live network (for a save or a preview).

        The raw-weight backup lives on CPU: swap_in brackets previews, and a clip preview
        is exactly when GPU headroom is scarcest — a GPU-resident backup (~0.6 GB at LoKR
        factor 8) was part of what tipped 32 GB cards back into the Windows VRAM spill."""
        self._backup = [p.detach().to("cpu", copy=True) for p in self.params]
        for s, p in zip(self.shadow, self.params):
            p.data.copy_(s.to(p.device, p.dtype))

    @torch.no_grad()
    def swap_out(self):
        """Restore the raw training weights. Must always pair with swap_in."""
        for b, p in zip(self._backup, self.params):
            p.data.copy_(b.to(p.device, p.dtype))
        self._backup = None

    def state_dict(self):
        return {"n": self.n, "decay": self.decay,
                "shadow": [s.detach().cpu() for s in self.shadow]}

    def load_state_dict(self, sd):
        self.n = int(sd["n"])
        if len(sd["shadow"]) != len(self.shadow):
            raise ValueError(f"EMA state has {len(sd['shadow'])} tensors, network has "
                             f"{len(self.shadow)} — different run configuration?")
        self.shadow = [t.to(s.device, torch.float32) for t, s in zip(sd["shadow"], self.shadow)]
