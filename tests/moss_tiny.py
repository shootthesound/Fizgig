"""Tiny random MOSS-Image configs shared by the MOSS tests (CPU, fp32)."""
import torch

TINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=4, text_dim=48, freq_dim=32, latent_channels=8)


def randomise(model, seed=0):
    """Non-trivial values everywhere (norm weights near 1, small Linears) so a wrong index or rope shows up."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") or "norm" in name:
                p.copy_(1 + 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) / max(1, p.shape[-1]) ** 0.5)
    return model


def pair(seed=0, **over):
    """(fizgig model, reference model) with identical weights."""
    import reference
    from fizgig.moss_image.model import MossImageDiT
    cfg = dict(TINY, **over)
    ref = randomise(reference.load().MOSSImageTransformer(**cfg), seed).eval()
    fz = MossImageDiT(**cfg).eval()
    sd = {k: v for k, v in ref.state_dict().items() if k != "rope.bands"}
    fz.load_state_dict(sd, strict=True)
    return fz, ref


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))
