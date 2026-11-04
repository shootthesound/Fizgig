"""Masked training on MiniMax H3: a still's mask weights its video loss per cell, the crop x0 takes is the crop the
mask takes, the prior term lands outside the mask, and a clip or voice step - which the dataset never gives a mask -
trains exactly as it did. All against a stand-in DiT: H3's own returns the latent grid unchanged in shape, so there
is nothing to repack (unlike Krea 2, whose tokens need patchify)."""
import pytest

torch = pytest.importorskip("torch")

from fizgig.dataset.masks import mask_weight  # noqa: E402
from fizgig.families import registry  # noqa: E402
from fizgig.minimax.common import compute_loss  # noqa: E402
from fizgig.minimax.driver import MiniMaxDriver  # noqa: E402

DTYPE = torch.bfloat16
_TXT = torch.zeros(1, 6, 8, dtype=DTYPE)          # the stand-in DiT ignores its conditioning


class _DiT:
    """H3's DiT's contract: a bf16 latent in, the same-shaped latent out - compute_loss's own MSE requires it.
    `mult` stands in for the LoRA's strength, so a preservation forward can differ from a training one."""

    patch_size = (1, 2, 2)
    pack_audio_rows = True

    def __init__(self, mult=1.0):
        self.mult = mult

    def __call__(self, x, t, txt, *, audio_rows=None, return_audio=False, ref_latents=None, **kw):
        out = x.float() * self.mult
        if not return_audio:
            return out.to(x.dtype)
        return out.to(x.dtype), (audio_rows.float() * 0.25 if audio_rows is not None else None)


class _Prior:
    """Stands in for train._MaskPrior: records that the preservation forward ran with the LoRA off."""

    def __init__(self, dit, weight=1.0):
        self.dit, self.weight, self.calls = dit, float(weight), 0

    def base(self, forward):
        self.calls += 1
        m, self.dit.mult = self.dit.mult, 0.0
        try:
            with torch.no_grad():
                return forward().detach().float()
        finally:
            self.dit.mult = m


def _step(seed=5, hw=(16, 16), sigma=0.35):
    """A step's latent, its noise and its sigma - reproducible, and never all-white by accident."""
    g = torch.Generator().manual_seed(seed)
    lat = torch.randn(1, 24, 1, *hw, generator=g)
    noise = torch.randn(lat.shape, generator=g)
    return lat, noise, torch.tensor([sigma])


def _grid(dit, lat, noise, sigma):
    """compute_loss's inputs after its crop to the patch multiple: (x0, noised, target) on the grid the model sees."""
    _pt, ph, pw = dit.patch_size
    h, w = lat.shape[-2], lat.shape[-1]
    # the trainer hands the driver a bf16 latent (minimax/driver.py's lat.to(DTYPE)), so the reference has to round it
    # the same way - a latent left in fp32 here shifts the loss in the last bits and the two never agree exactly
    x0 = lat.to(DTYPE).float()[..., :(h // ph) * ph, :(w // pw) * pw]
    nz = noise.float()[..., :(h // ph) * ph, :(w // pw) * pw]
    s = sigma.reshape(1, 1, 1, 1, 1).float()
    return x0, (1.0 - s) * x0 + s * nz, x0 - nz


def _pred(dit, noised, mult):
    m, dit.mult = dit.mult, mult
    try:
        return dit(noised.to(DTYPE), None, None).float()
    finally:
        dit.mult = m


def _cells(dit, lat, noise, sigma, mult=1.0):
    """The per-position squared error the loss is the mean of: (1, T, H, W), channels averaged out."""
    _x0, noised, target = _grid(dit, lat, noise, sigma)
    return (_pred(dit, noised, mult) - target.to(DTYPE).float()).pow(2).mean(dim=1)


def _base_cells(dit, lat, noise, sigma):
    """The error against the model's own picture: the LoRA at 0, which is what prior preservation aims at."""
    _x0, noised, _t = _grid(dit, lat, noise, sigma)
    return (_pred(dit, noised, 1.0) - _pred(dit, noised, 0.0)).pow(2).mean(dim=1)


def _weighted(se, mw):
    """The step's loss under a mask: the per-cell errors weighted by the mask and rescaled to the same total, so a
    masked picture counts for as much as an unmasked one (batch 1, so the per-image mean is the plain mean)."""
    return (se * mw / mw.mean()).mean()


def test_minimax_opts_into_masked_training():
    assert registry.get("minimax").masked_training is True


def test_an_unmasked_step_is_the_flat_mean():
    dit = _DiT()
    lat, noise, sigma = _step()
    loss, s = compute_loss(dit, lat.to(DTYPE), _TXT, sigma=sigma, noise=noise)
    assert torch.allclose(loss.float(), _cells(dit, lat, noise, sigma).mean(), atol=1e-5)
    assert s == pytest.approx(0.35)


def test_a_flat_mask_weights_nothing():
    """One constant value across a mask is a uniform weight: rescaled to mean 1, it changes the step not at all.
    This is what makes a clip step's all-white mask, and a fully-painted picture, safe to hand a mask to."""
    dit = _DiT()
    lat, noise, sigma = _step()
    plain = compute_loss(dit, lat.to(DTYPE), _TXT, sigma=sigma, noise=noise)[0].float()
    for v in (1.0, 0.37, 0.1):
        m = torch.full((1, 1, 16, 16), v, dtype=torch.float32)
        got = compute_loss(dit, lat.to(DTYPE), _TXT, sigma=sigma, noise=noise, loss_mask=m)[0].float()
        assert torch.allclose(got, plain, atol=1e-5), f"a constant mask of {v} changed the step"


def test_a_half_mask_weights_exactly():
    dit = _DiT()
    lat, noise, sigma = _step()
    m = torch.zeros(1, 1, 16, 16)
    m[..., :8, :] = 1.0                                 # the top half white, the bottom half black
    mw = mask_weight(m, 0.1)                           # the floor the trainer applies before the driver sees it
    loss = compute_loss(dit, lat.to(DTYPE), _TXT, sigma=sigma, noise=noise, loss_mask=mw)[0].float()
    assert torch.allclose(loss, _weighted(_cells(dit, lat, noise, sigma), mw), atol=1e-5)


def test_an_off_grid_latent_crops_the_mask_with_itself():
    # a 15-px latent edge against a 2-px patch: compute_loss crops x0, and the mask has to follow it exactly
    dit = _DiT()
    lat, noise, sigma = _step(hw=(16, 15))
    mw = mask_weight(torch.rand(1, 1, 16, 15, generator=torch.Generator().manual_seed(2)), 0.1)
    loss = compute_loss(dit, lat.to(DTYPE), _TXT, sigma=sigma, noise=noise, loss_mask=mw)[0].float()
    _pt, ph, pw = dit.patch_size
    cropped = mw[..., : (16 // ph) * ph, : (15 // pw) * pw]
    assert torch.allclose(loss, _weighted(_cells(dit, lat, noise, sigma), cropped), atol=1e-5)


def test_the_cropped_edge_contributes_nothing():
    """Whatever compute_loss crops away must contribute nothing to the loss - with a mask on, the same as without."""
    dit = _DiT()
    lat, noise, sigma = _step(hw=(16, 15))
    other = lat.clone()
    other[..., -1, -1] = 1000.0                        # the column the crop drops
    mw = mask_weight(torch.rand(1, 1, 16, 15, generator=torch.Generator().manual_seed(3)), 0.1)
    a = compute_loss(dit, lat.to(DTYPE), _TXT, sigma=sigma, noise=noise, loss_mask=mw)[0].float()
    b = compute_loss(dit, other.to(DTYPE), _TXT, sigma=sigma, noise=noise, loss_mask=mw)[0].float()
    assert torch.allclose(a, b, atol=1e-5)


def test_the_prior_term_lands_outside_the_mask():
    dit = _DiT()
    lat, noise, sigma = _step()
    m = torch.zeros(1, 1, 16, 16)
    m[..., :8, :] = 1.0
    mw = mask_weight(m, 0.1)
    p = _Prior(dit, 0.7)
    loss = compute_loss(dit, lat.to(DTYPE), _TXT, sigma=sigma, noise=noise, loss_mask=mw, prior=p)[0].float()
    assert p.calls == 1                                # one extra forward, taken before the training one
    want = _weighted(_cells(dit, lat, noise, sigma), mw) + 0.7 * _weighted(_base_cells(dit, lat, noise, sigma),
                                                                         1.0 - mw)
    assert torch.allclose(loss, want, atol=1e-5)


def test_an_all_white_mask_preserves_nothing():
    """with_prior weights by 1 - mask: an all-white mask leaves no area outside it, so the preservation forward buys
    an exactly zero term (and the driver skips that forward outright - see its own test below)."""
    dit = _DiT()
    lat, noise, sigma = _step()
    ones = torch.ones(1, 1, 16, 16)
    p = _Prior(dit, 1.0)
    with_p = compute_loss(dit, lat.to(DTYPE), _TXT, sigma=sigma, noise=noise, loss_mask=ones, prior=p)[0].float()
    without = compute_loss(dit, lat.to(DTYPE), _TXT, sigma=sigma, noise=noise, loss_mask=ones)[0].float()
    assert p.calls == 1 and torch.allclose(with_p, without, atol=1e-6)


def test_a_clip_step_with_an_all_white_mask_is_unchanged():
    """A mixed folder hands every item a mask entry, so a clip's is all-white: the video term is a no-op weighted,
    and the sound term is never weighted at all."""
    dit = _DiT()
    g = torch.Generator().manual_seed(9)
    lat = torch.randn(1, 24, 5, 16, 16, generator=g)   # a clip: 5 frames on H3's 17n+5 grid
    noise = torch.randn(lat.shape, generator=g)
    audio = torch.randn(10, 32, generator=g)
    # the sound rows get their own noise drawn inside compute_loss, so both calls need the same seeded generator
    plain = compute_loss(dit, lat.to(DTYPE), _TXT, sigma=torch.tensor([0.5]), noise=noise, audio_latent=audio,
                         generator=torch.Generator().manual_seed(21))[0].float()
    masked = compute_loss(dit, lat.to(DTYPE), _TXT, sigma=torch.tensor([0.5]), noise=noise, audio_latent=audio,
                          generator=torch.Generator().manual_seed(21), loss_mask=torch.ones(1, 1, 16, 16))[0].float()
    assert torch.allclose(plain, masked, atol=1e-5)


def test_the_driver_takes_the_mask_and_the_prior():
    """The crash the family flag alone would have caused: train.py passes loss_mask (and prior) into every step of a
    masked run, and H3's training_loss used to declare neither."""
    dit = _DiT()
    lat, _noise, _sigma = _step(seed=4)
    drv, gen = MiniMaxDriver(), torch.Generator().manual_seed(2)
    mw = mask_weight(torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(3)), 0.1)
    p = _Prior(dit, 0.5)
    loss, info = drv.training_loss(dit, lat.to(DTYPE), {"hidden_states": _TXT[0]}, gen, loss_mask=mw, prior=p)
    assert torch.isfinite(loss) and "t" in info and p.calls == 1


def test_the_driver_skips_the_preservation_forward_for_an_all_white_mask():
    dit = _DiT()
    lat, _noise, _sigma = _step(seed=6)
    drv, gen = MiniMaxDriver(), torch.Generator().manual_seed(1)
    p = _Prior(dit, 1.0)
    drv.training_loss(dit, lat.to(DTYPE), {"hidden_states": _TXT[0]}, gen, loss_mask=torch.ones(1, 1, 16, 16),
                      prior=p)
    assert p.calls == 0


def test_a_masked_slider_step_runs():
    """A photo-pair slider step carries a mask too (as on every family that takes masks), and prior is not passed
    to a slider step by the trainer at all."""
    dit = _DiT()
    lat, _noise, _sigma = _step(seed=8)
    drv, gen = MiniMaxDriver(), torch.Generator().manual_seed(4)
    ref = torch.randn(1, 24, 1, 16, 16, generator=torch.Generator().manual_seed(6)).to(DTYPE)
    mw = mask_weight(torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(7)), 0.1)
    loss, info = drv.training_loss(dit, lat.to(DTYPE), {"hidden_states": _TXT[0]}, gen,
                                   diff_ref=ref, diff_weight=0.5, loss_mask=mw)
    assert torch.isfinite(loss) and "t" in info


def test_the_masked_note_logs_once_per_run(caplog):
    """The driver-side proof a mask reached the loss: one [masks] line per run, and none at all on an unmasked step."""
    dit, gen = _DiT(), torch.Generator().manual_seed(3)
    lat, _noise, _sigma = _step(seed=21)
    mw = torch.ones(1, 1, 16, 16)
    mw[..., :8] = 0.1
    drv = MiniMaxDriver()
    with caplog.at_level("INFO", logger="fizgig.minimax.driver"):
        for _ in range(3):
            drv.training_loss(dit, lat.to(DTYPE), {"hidden_states": _TXT[0]}, gen, loss_mask=mw)
        notes = [r.getMessage() for r in caplog.records if "[masks]" in r.getMessage()]
        caplog.clear()
        MiniMaxDriver().training_loss(dit, lat.to(DTYPE), {"hidden_states": _TXT[0]}, gen)
        unmasked = [r.getMessage() for r in caplog.records if "[masks]" in r.getMessage()]
    assert len(notes) == 1 and "mask on (16, 16), x0.10..x1.00, prior off" in notes[0]
    assert unmasked == []


def test_a_flat_mask_changes_no_slider_step_either():
    dit = _DiT()
    lat, _noise, _sigma = _step(seed=12)
    drv = MiniMaxDriver()
    ref = torch.randn(1, 24, 1, 16, 16, generator=torch.Generator().manual_seed(13)).to(DTYPE)
    kw = {"diff_ref": ref, "diff_weight": 0.5}
    plain = drv.training_loss(dit, lat.to(DTYPE), {"hidden_states": _TXT[0]},
                              torch.Generator().manual_seed(14), **kw)[0].float()
    masked = drv.training_loss(dit, lat.to(DTYPE), {"hidden_states": _TXT[0]},
                               torch.Generator().manual_seed(14), loss_mask=torch.ones(1, 1, 16, 16), **kw)[0].float()
    assert torch.allclose(plain, masked, atol=1e-5)
