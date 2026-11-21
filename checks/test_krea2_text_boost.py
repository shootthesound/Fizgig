"""The ✨Prompt Enhancer end to end: the maths (krea2/text_boost.py), the DiT's derivation, the live
settings' rules, what each epoch records, and the declaration itself.

CPU-only and small on purpose - a real 12x2560 text fusion is hundreds of MB and is not a CI
fixture. `scripts/check_text_boost.py` is the instrument for the measurements that need one.
"""
import torch

from fizgig.krea2 import text_boost as tb

#: The donor's 12-tap profile. At s=1 the gains ARE the profile (1 + 1*(p-1) == p), tiled twice.
DONOR_PROFILE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.5, 5.0, 1.1, 4.0, 1.0)


def test_strength_is_clamped_to_the_donor_range():
    for bad in (None, "", "abc", 0, -1, -0.0, float("nan")):
        assert tb.clamped(bad) == 0.0
    assert tb.clamped("0.6") == 0.6
    assert tb.clamped(1.0) == 1.0
    assert tb.clamped(99) == tb.MAX_STRENGTH == 2.0


def test_text_scale_is_neutral_by_default_and_bounded():
    for bad in (None, "", "x", float("nan")):
        assert tb.text_scale(bad) == 1.0
    assert tb.text_scale(0.01) == 0.25
    assert tb.text_scale(100) == 4.0
    assert tb.text_scale(1.5) == 1.5


def test_applies_only_to_the_krea2_conditioning():
    assert tb.applies(torch.zeros(1, 4, 12, 2560))
    for shape in ((1, 4, 2560), (1, 4, 12, 2048), (1, 4, 24, 1280), (12, 2560), (1, 4, 12, 2560, 1)):
        assert not tb.applies(torch.zeros(shape))


def test_gains_are_the_donors_profile():
    assert torch.allclose(tb.gains(1.0, "cpu", torch.float32),
                          torch.tensor(list(DONOR_PROFILE) * 2, dtype=torch.float32))
    half = tb.gains(0.5, "cpu", torch.float32)
    assert abs(half[8].item() - 3.0) < 1e-6       # 1 + 0.5*(5.0 - 1)
    assert abs(half[9].item() - 1.05) < 1e-6      # 1 + 0.5*(1.1 - 1)
    assert abs(tb.gains(2.0, "cpu", torch.float32)[8].item() - 9.0) < 1e-6
    assert torch.allclose(tb.gains(0.0, "cpu", torch.float32), torch.ones(24))


def test_scaled_context_lands_the_gains_and_the_global_lift():
    # The donor tiles its 12-tap profile over 24 chunks of 1280, and the conditioning flattens
    # 12 taps x 2560 into exactly those chunks - so the emphasis lands where the tiling puts it
    # (chunks 7-11 and 19-23), not where the profile's own indices would suggest. Reproducing the
    # node is the port; a tidied-up reading of it would be a different feature.
    x = torch.ones(1, 2, 12, 2560)
    chunks = tb.scaled_context(x, 1.0)[0, 0].reshape(tb.CHUNKS, tb.CHUNK_DIM)
    assert abs(chunks[8][0].item() - 75.0) < 1e-3     # gain 5.0 x global 15.0
    assert abs(chunks[0][0].item() - 15.0) < 1e-3     # gain 1.0 x global 15.0
    assert abs(chunks[23][0].item() - 15.0) < 1e-3
    assert tb.scaled_context(x, 0.0) is x             # off: the same tensor back, not a copy
    assert tb.scaled_context(torch.ones(1, 2, 2048), 1.0).shape == (1, 2, 2048)


def test_residual_cap_holds_across_every_scale():
    g = torch.Generator().manual_seed(7)
    for scale in (1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0):
        ref = torch.randn(2, 3, 2560, generator=g)
        cand = ref + scale * torch.randn(2, 3, 2560, generator=g)
        out = tb.capped_residual(ref, cand)
        base = ref.pow(2).mean(-1, keepdim=True).sqrt()
        moved = (out - ref).pow(2).mean(-1, keepdim=True).sqrt()
        assert torch.all(moved <= 0.75 * base + 1e-5), scale           # never past the cap
        assert torch.all(moved <= (cand - ref).pow(2).mean(-1, keepdim=True).sqrt() + 1e-5), scale


def test_a_small_residual_passes_through_untouched():
    ref = torch.ones(1, 1, 2560)
    cand = ref + 1e-3
    assert torch.allclose(tb.capped_residual(ref, cand), cand, atol=1e-6)


def test_the_result_keeps_the_input_dtype():
    ref = torch.ones(1, 1, 2560, dtype=torch.bfloat16)
    cand = torch.full((1, 1, 2560), 3.0, dtype=torch.bfloat16)
    assert tb.capped_residual(ref, cand).dtype == torch.bfloat16


def test_off_runs_the_fusion_once():
    calls = []
    out = tb.enhance(lambda x: (calls.append(1), x.sum(-2))[1], torch.ones(1, 2, 12, 2560), 0.0)
    assert len(calls) == 1
    assert out.shape == (1, 2, 2560)


def test_on_runs_the_fusion_twice_and_stays_inside_the_cap():
    calls = []

    def run(x):
        calls.append(x.clone())
        return x.sum(-2)

    x = torch.ones(1, 1, 12, 2560)
    out = tb.enhance(run, x, 1.0)
    assert len(calls) == 2
    assert torch.equal(calls[0], x)             # the plain pass sees the conditioning itself
    assert not torch.equal(calls[1], x)         # the second sees the gain-scaled one
    assert out.shape == (1, 1, 2560)
    ref = calls[0].sum(-2)
    assert (out - ref).pow(2).mean(-1).sqrt() <= 0.75 * ref.pow(2).mean(-1).sqrt() + 1e-5


def test_once_per_preview_equals_per_step():
    """The whole shortcut in one test: the tensor the step loop reuses is the tensor the donor
    recomputes inside every model call. Deterministic maths, so this is exact rather than close."""
    x = torch.randn(1, 3, 12, 2560, generator=torch.Generator().manual_seed(3))
    w = torch.randn(1, 1, 12, 1, generator=torch.Generator().manual_seed(4))
    run = lambda c: (c * w).sum(-2)         # a fusion with real per-tap weighting
    once = tb.enhance(run, x, 1.0)
    assert all(torch.equal(once, tb.enhance(run, x, 1.0)) for _ in range(5))


def test_the_diT_derivation_is_the_numerical_twin_of_the_plain_pass():
    """`SingleStreamDiT.enhanced_text` has to build its text mask and attention params exactly as
    `forward` does - the enhancement is the difference between two runs of that fusion, so a drift
    here shows up as a wrong picture rather than as an error. A duck-typed self keeps it CPU-small."""
    from fizgig.krea2.model import SingleStreamDiT

    class Stub:
        attn_mode, split_attn = "torch", False

        def __init__(self):
            self.contexts, self.params = [], []

        def txtfusion(self, c, nomask, params):
            self.contexts.append(c.clone())
            self.params.append((nomask, params))
            return c.sum(-2)

    stub = Stub()
    mask = torch.ones(1, 6 + 4, dtype=torch.bool)
    x = torch.ones(1, 4, 12, 2560)
    assert SingleStreamDiT.enhanced_text(stub, x, mask, 6, 0.0).shape == (1, 4, 2560)
    assert len(stub.contexts) == 1                       # off: the plain pass, once, nothing else
    stub.contexts.clear()
    stub.params.clear()
    out = SingleStreamDiT.enhanced_text(stub, x, mask, 6, 1.0)
    assert len(stub.contexts) == 2
    assert torch.equal(stub.contexts[0], x)              # the plain pass sees the conditioning itself
    assert not torch.equal(stub.contexts[1], x)          # the second sees the gain-scaled one
    assert stub.params[0][0] is stub.params[1][0]        # both runs share one set of attention params,
    assert stub.params[0][1] is stub.params[1][1]        # so the two outputs are comparable
    assert out.shape == (1, 4, 2560)


def test_the_preview_path_carries_the_settings_only_when_said():
    from fizgig.families.train import _enh_options
    assert _enh_options(None, None) == ()                                   # never said: nothing sent
    assert _enh_options(0.0, None) == (("krea2_text_boost", 0.0),)          # off is said, not absent
    assert _enh_options(0.6, 1.5) == (("krea2_text_boost", 0.6), ("krea2_text_scale", 1.5))


def test_the_tick_decides_and_the_boxes_set_the_strengths():
    from fizgig.families.train import _enh_settings
    assert _enh_settings(None, {}) == (None, None)                          # never ticked: plain render
    assert _enh_settings(None, {"krea2_text_enhance": "1"}) == (1.0, 1.0)   # ticked, blank boxes = neutral 1
    assert _enh_settings(None, {"krea2_text_enhance": "1",
                                "krea2_text_boost": "0.4"}) == (0.4, 1.0)
    assert _enh_settings(None, {"krea2_text_enhance": "0"}) == (None, None)  # an explicit 0 is not ticked
    assert _enh_settings({"krea2_text_enhance": 0}, {"krea2_text_enhance": "1"}) == (None, None)  # unticked
    assert _enh_settings({"krea2_text_enhance": 1}, None) == (1.0, 1.0)                # ticked mid-run
    assert _enh_settings({"krea2_text_enhance": 1,
                          "krea2_text_boost": 0.6, "krea2_text_scale": 1.5}, None) == (0.6, 1.5)
    # an override file from before the tick exists, or from the desktop panel: the launch values stand
    assert _enh_settings({"krea2_text_enhance": None,
                          "krea2_text_boost": 0.8}, {"krea2_text_enhance": "1"}) == (0.8, 1.0)
    assert _enh_settings({"krea2_text_enhance": 1, "krea2_text_boost": "nonsense"}, None) == (1.0, 1.0)


def test_the_record_shows_either_knob_or_both():
    """What each epoch's previews were rendered with - including a render whose only enhancement is the text
    scale, which the row left out entirely until now."""
    from fizgig.families.train import _enh_summary
    assert _enh_summary({}) == ""                                        # never said: claims nothing
    assert _enh_summary({"boost": 0, "text_scale": 1}) == ""             # both at neutral
    assert _enh_summary({"boost": 1.0}) == "1"                           # strength alone
    assert _enh_summary({"boost": 0, "text_scale": 1.8}) == "×1.8"       # scale alone
    assert _enh_summary({"boost": 1.0, "text_scale": 1.8}) == "1 ×1.8"   # both, the usual case


def test_krea2_declares_the_donors_three_controls_on_the_samples_tab():
    from fizgig.families.registry import FAMILIES
    opts = {o.key: o for o in FAMILIES["krea2"].options}
    assert set(opts) == {"KREA2_TEXT_ENHANCE", "KREA2_TEXT_BOOST", "KREA2_TEXT_SCALE"}
    assert all(o.tab == "samples" for o in opts.values())
    assert opts["KREA2_TEXT_ENHANCE"].kind == "check"
    assert opts["KREA2_TEXT_ENHANCE"].tokens == "krea2_text_enhance=1"      # nothing sent until it is ticked
    assert opts["KREA2_TEXT_BOOST"].tokens == "krea2_text_boost={}"
    assert opts["KREA2_TEXT_SCALE"].tokens == "krea2_text_scale={}"
    assert opts["KREA2_TEXT_BOOST"].default == "" and opts["KREA2_TEXT_SCALE"].default == ""


def test_the_override_file_round_trips_the_enhancer(tmp_path):
    """The live channel's file boundary: what the Samples tab writes is what the trainer reads back. Every
    other check here hands a function a dict it built, so nothing else ever touches the file - which means a
    key renamed at one end would keep the whole suite green while the previews quietly stopped following the
    tab. This is the check that notices."""
    import json
    from fizgig.families.train import _read_sample_override

    def read(**fields):
        (tmp_path / ".sample_override.json").write_text(
            json.dumps({"prompts": ["a photo"], **fields}), encoding="utf-8")
        return _read_sample_override(str(tmp_path))

    ov = read(krea2_text_enhance=1, krea2_text_boost=0.6, krea2_text_scale=1.8)
    assert (ov["krea2_text_enhance"], ov["krea2_text_boost"], ov["krea2_text_scale"]) == (1, 0.6, 1.8)
    assert read()["krea2_text_enhance"] is None                       # the tab never said
    assert read(krea2_text_enhance=0)["krea2_text_enhance"] == 0      # an untick, which is not the same
