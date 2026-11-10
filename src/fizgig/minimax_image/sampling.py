"""MiniMax-H3-Image sampling, as ComfyUI runs the checkpoint (comfy/ldm/minimax/sampling.py
MiniMaxH3ImageModelSampling, the `simple` scheduler, the `euler` sampler; comfy/ldm/minimax/model.py FinalLayer):

* sigma(timestep) is linear in the PDD grid index, so `simple` at 6 steps lands on (about) the 6 grid points the heads
  were distilled for; other step counts land between them and a step takes the dt-weighted mean of the heads it spans;
* the start is pure noise (sigma 1), drawn on the CPU from the seed, [1, 24, 1, h, w];
* Euler on the raw head: the DiT returns x0 - noise (minimax/sampling.py's convention), so
  x_next = x + (sigma - sigma_next) * out.
CFG-free: the API refuses guidance other than 1 and any negative prompt.
"""
import torch

from fizgig.minimax_image.model import PDD_SIGMAS, nearest_grid

_TIMESTEPS = 1000


def grid_sigma(grid, timestep):
    """MiniMaxH3ImageModelSampling.sigma: timestep in [0, 1000] -> sigma, linear in the grid index."""
    asc = list(reversed(grid))
    n = len(asc) - 1
    pos = min(max(timestep / _TIMESTEPS, 0.0), 1.0) * n
    i = min(int(pos), n - 1)
    return asc[i] + (asc[i + 1] - asc[i]) * (pos - i)


def schedule(steps, grid=PDD_SIGMAS):
    """ComfyUI's `simple` scheduler over the model's 1000-entry sigma table: steps + 1 sigmas, the last 0."""
    table = [grid_sigma(grid, t) for t in range(1, _TIMESTEPS + 1)]
    stride = len(table) / steps
    return [table[-(1 + int(x * stride))] for x in range(steps)] + [0.0]


def step_span(grid, sigma, sigma_next):
    """The head span ComfyUI's FinalLayer gives a step from sigma to sigma_next."""
    n = len(grid) - 1
    start, stop = nearest_grid(grid, sigma), nearest_grid(grid, sigma_next)
    start = min(start, n - 1)
    return start, max(stop, start + 1)


def initial_noise(seed, width, height):
    """The seed's start, as ComfyUI draws it: CPU generator, [1, 24, 1, H/16, W/16]."""
    return torch.randn((1, 24, 1, height // 16, width // 16), generator=torch.Generator("cpu").manual_seed(int(seed)),
                       dtype=torch.float32)


@torch.no_grad()
def sample(dit, text, width, height, *, steps=6, seed=0, sigmas=None, noise=None, on_step=None, refs=None,
           text_token_tags=None, forward=None):
    """One picture -> its normalised latent [1, 24, 1, h, w] (fp32, on the DiT's device). `forward(x, t, txt, **kw)`
    replaces the DiT call (Reuse unchanged blocks); refs are normalised reference latents [1, 24, 1, h, w]."""
    device = next(p for p in dit.parameters() if p.device.type != "meta").device
    grid = dit.final_layer.pdd_sigmas
    sig = list(sigmas) if sigmas is not None else schedule(steps, grid)
    if sig[-1] != 0.0:
        sig = sig + [0.0]
    x = (noise if noise is not None else initial_noise(seed, width, height)).to(device, torch.float32)
    x = x * sig[0]                              # CONST noise scaling of an empty latent: sigma * noise
    txt = text.to(device, torch.bfloat16)
    txt = txt[None] if txt.dim() == 2 else txt
    kw = {"seed": int(seed)}
    if refs:
        kw["ref_latents"] = [r.to(device, torch.float32) for r in refs]
    if text_token_tags is not None:
        kw["text_token_tags"] = text_token_tags
    call = forward or dit
    fl = dit.final_layer
    n = len(sig) - 1
    for i in range(n):
        if on_step is not None:
            on_step(i, n)
        s, s_next = float(sig[i]), float(sig[i + 1])
        fl.span_override = step_span(grid, s, s_next)
        try:
            t = torch.tensor([1.0 - s], device=device, dtype=torch.float32)
            out = call(x.to(torch.bfloat16), t, txt, **kw).float()
        finally:
            fl.span_override = None
        x = x + (s - s_next) * out
    return x
