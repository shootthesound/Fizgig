"""MOSS-Image schedule, sampler, timestep draw and VAE packing (T6, T7, T8): CPU, against the reference pipeline."""
import math

import pytest
import torch

from moss_tiny import TINY, pair, rel

RESULTS = {}


# ---- T6: schedule and sampler -------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [1, 2, 8, 25, 28, 50, 100, 1000])
def test_t6_sigmas_equal_reference_scheduler(n):
    import reference
    from fizgig.moss_image import sampling as S
    sch = reference.load().MOSSImageScheduler()
    sch.set_timesteps(n)
    got = S.sigmas(n)
    assert torch.equal(got, sch.sigmas.float()), n
    assert torch.equal(got[:-1] * 1000, sch.timesteps.float())


def test_t6_noise_is_fp32_and_seeded():
    from fizgig.moss_image import sampling as S
    a, b = S.initial_noise(5, 64, 48), S.initial_noise(5, 64, 48)
    assert a.dtype == torch.float32 and a.shape == (1, 128, 4, 3) and torch.equal(a, b)
    with pytest.raises(ValueError):
        S.initial_noise(1, 60, 64)


def _ref_pipe():
    import reference
    R = reference.load()
    fz, ref = pair(seed=11)
    return R.MOSSImagePipeline(transformer=ref, scheduler=R.MOSSImageScheduler()), fz


@pytest.mark.parametrize("kw", [
    dict(guidance_scale=3.0, uncond_skip_layers=(2,)),
    dict(guidance_scale=4.5, uncond_skip_layers=()),
    dict(guidance_scale=1.0, uncond_skip_layers=(), slg_layers=[1], slg_scale=3.0, slg_start=0.01, slg_stop=1.0),
    dict(guidance_scale=3.0, uncond_skip_layers=(2,), slg_layers=[3], slg_scale=2.0, slg_start=0.2, slg_stop=0.7),
])
def test_t6_sample_equals_reference_pipeline(kw):
    from fizgig.moss_image import sampling as S
    pipe, fz = _ref_pipe()
    torch.manual_seed(12)
    pos = torch.randn(1, 6, TINY["text_dim"])
    neg = torch.randn(1, 3, TINY["text_dim"])
    noise = torch.randn(1, TINY["latent_channels"], 3, 4)
    steps = 8
    with torch.no_grad():
        want = pipe(prompt_embeds=pos, negative_prompt_embeds=neg, latents=noise.clone(), height=48, width=64,
                    num_inference_steps=steps, output_type="latent", **kw).images
        got = S.sample(fz, pos, 48, 64, steps=steps, cfg=kw["guidance_scale"], neg=neg, noise=noise.clone(),
                       uncond_skip_layers=kw.get("uncond_skip_layers"), slg_layers=kw.get("slg_layers", ()),
                       slg_scale=kw.get("slg_scale", 0.0), slg_start=kw.get("slg_start", 0.0),
                       slg_stop=kw.get("slg_stop", 1.0), device=torch.device("cpu"))
    RESULTS[f"T6 sample {sorted(kw.items())}"] = rel(got, want)
    assert rel(got, want) <= 1e-6


def test_t6_sample_padded_negative_equals_unpadded():
    """A negative padded to the positive's length (batched CFG later) renders as the unpadded one."""
    from fizgig.moss_image import sampling as S
    _, fz = _ref_pipe()
    pos = torch.randn(1, 6, TINY["text_dim"])
    neg = torch.randn(1, 3, TINY["text_dim"])
    padded = torch.cat([neg, torch.full((1, 3, TINY["text_dim"]), 5.0)], 1)
    mask = torch.tensor([[True] * 3 + [False] * 3])
    a = S.sample(fz, pos, 32, 32, steps=4, seed=1, neg=neg, uncond_skip_layers=(1,), device=torch.device("cpu"))
    b = S.sample(fz, pos, 32, 32, steps=4, seed=1, neg=padded, neg_mask=mask, uncond_skip_layers=(1,),
                 device=torch.device("cpu"))
    assert rel(b, a) <= 1e-6


# ---- T7: training timesteps -----------------------------------------------------------------------------------------
def test_t7_paper_timesteps_match_eq4():
    from fizgig.moss_image import sampling as S
    g = torch.Generator().manual_seed(0)
    n = 200_000
    counts = torch.zeros(1000, dtype=torch.float64)
    for _ in range(n):
        s = S.sample_t(g)
        j = round(s * 1000)
        assert 1 <= j <= 1000 and abs(s * 1000 - j) < 1e-9
        counts[j - 1] += 1
    pmf = S.paper_pmf()
    assert abs(float(pmf.sum()) - 1) < 1e-9
    # chi-square over 20 bins of 50 grid points each
    obs = counts.view(20, 50).sum(1)
    exp = pmf.view(20, 50).sum(1) * n
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    RESULTS["T7 chi2 (19 dof)"] = chi2
    assert chi2 < 43.8                         # p = 0.001 for 19 dof
    mean = float((counts * torch.arange(1, 1001, dtype=torch.float64) / 1000).sum() / n)
    RESULTS["T7 mean sigma"] = mean
    assert abs(mean - float((pmf * torch.arange(1, 1001, dtype=torch.float64) / 1000).sum())) < 3e-3


def test_t7_other_kinds_and_range():
    from fizgig.moss_image import sampling as S
    g = torch.Generator().manual_seed(1)
    for kind in ("uniform", "logit_normal"):
        v = [S.sample_t(g, kind) for _ in range(2000)]
        assert all(0.001 <= x <= 1.0 for x in v)
    assert S.scale_t(0.5, 0.0, 1.0) == 0.5
    assert S.scale_t(0.5, 0.2, 0.6) == 0.4
    assert S.scale_t(0.0001, 0.0, 0.5) == 0.001
    with pytest.raises(ValueError):
        S.sample_t(g, "nope")


# ---- T8: VAE packing ----------------------------------------------------------------------------------------------
def test_t8_flux3_encode_then_reference_unpack_is_exact():
    """flux3 encode's (128, h, w) latents, de-normalised and unpacked the reference pipeline's way, give back the
    encoder's 32-channel mean: same channel order (c*4 + py*2 + px) and the same BatchNorm (eps 1e-4)."""
    from fizgig.flux3 import sampling as F3
    from fizgig.flux3.autoencoder import AutoEncoder
    from fizgig.moss_image import sampling as S
    torch.manual_seed(13)
    ae = AutoEncoder().eval()
    h2, w2 = 6, 10                                  # 32-channel latent at /8: (2h, 2w) with h=3, w=5
    mean = torch.randn(1, 32, h2, w2)

    class Enc(torch.nn.Module):
        def forward(self, x):
            return torch.cat([mean, torch.zeros_like(mean)], 1)
    ae.encoder = Enc()
    ae.bn.running_mean.copy_(torch.randn(128))
    ae.bn.running_var.copy_(torch.rand(128) + 0.5)
    tokens, ids = ae.encode(torch.zeros(3, 8 * h2, 8 * w2))
    lat = F3.to_latents(tokens, h2 // 2, w2 // 2)[None]          # what the driver caches, (1, 128, h, w)
    z = S.unpack_for_vae(lat, ae.bn.running_mean, ae.bn.running_var, ae.bn_eps)
    assert rel(z, mean) <= 1e-6
    # the reference's own expression (pipeline __call__, decode branch), verbatim
    zz = lat * (ae.bn.running_var + 1e-4).sqrt().view(1, -1, 1, 1) + ae.bn.running_mean.view(1, -1, 1, 1)
    b, c, h, w = zz.shape
    zz = zz.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    assert torch.equal(z, zz)
    # channel order spelled out
    c, py, px = 5, 1, 0
    assert torch.allclose(lat[0, c * 4 + py * 2 + px] * (ae.bn.running_var[c * 4 + 2] + 1e-4).sqrt()
                          + ae.bn.running_mean[c * 4 + 2], mean[0, c, py::2, px::2], atol=1e-5)


def test_zz_print_results():
    for k, v in RESULTS.items():
        print(f"{k[:60]:60s} {v:.3e}")
