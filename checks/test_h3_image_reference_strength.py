"""MiniMax H3 Image's reference Strength (Studio, Royale): the box sets the noise the model's reference rows carry and
the timestep they sit at (visual_cond_noise_aug; 1 = the model's own 0.999), not a scaling of the reference latents.
0 leaves the reference out, as for every family with the box."""
from types import SimpleNamespace

import torch
from torch import nn

from fizgig.families.registry import get
from fizgig.families.workbench import WorkbenchEngine
from fizgig.minimax_image import sampling as S
from fizgig.minimax_image.model import PDD_SIGMAS


def test_the_box_maps_to_the_models_reference_noise():
    assert S.reference_aug(None) is None
    assert abs(S.reference_aug(1) - 0.999) < 1e-9 and abs(S.reference_aug(3) - 0.999) < 1e-9
    assert abs(S.reference_aug("0.5") - 0.4995) < 1e-9 and S.reference_aug(0) == 0.0 and S.reference_aug(-1) == 0.0
    assert S.reference_aug("x") is None


class _Head:
    pdd_sigmas = list(PDD_SIGMAS)
    span_override = None


class _DiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.final_layer = _Head()
        self.kw = []

    def forward(self, x, t, txt, **kw):
        self.kw.append(kw)
        return -0.5 * x.float()


def test_the_sampler_hands_it_to_the_model_with_a_reference():
    ref = [torch.zeros(1, 24, 1, 4, 4)]
    dit = _DiT()
    S.sample(dit, torch.zeros(4, 8), 64, 64, steps=2, seed=1, refs=ref, ref_aug=0.5)
    assert all(k["visual_cond_noise_aug"] == 0.5 for k in dit.kw)
    for refs, aug in ((ref, None), (None, 0.5)):                 # no box, or no reference: the model's own
        dit = _DiT()
        S.sample(dit, torch.zeros(4, 8), 64, 64, steps=2, seed=1, refs=refs, ref_aug=aug)
        assert all("visual_cond_noise_aug" not in k for k in dit.kw)


def _eng(desc):
    eng = WorkbenchEngine.__new__(WorkbenchEngine)
    eng.desc = desc
    eng._reference_latents = lambda path, w, h: torch.ones(1, 24, 4, 4)
    eng._ref_size = lambda path, mp, w, h: (w, h)
    eng._state_reference = staticmethod(lambda state: ("r.png", 1.0))
    return eng


def test_the_workbench_passes_the_box_on_unscaled():
    d = get("minimax_image")
    assert d.reference_strength and d.reference_strength_noise
    eng = _eng(d)
    refs = eng._edit_refs(SimpleNamespace(ref_strength=0.4), "r.png", 64, 64)
    assert torch.equal(refs[0], torch.ones(1, 24, 4, 4))            # the latents as they are
    assert eng._ref_options((("sampler", "lcm"),), refs) == (("sampler", "lcm"), ("ref_strength", 0.4))
    assert eng._ref_options((), refs) == ()                          # used up by that render
    # 0: no reference rows, but the prompt still holds the picture - the box still reaches the driver
    assert eng._edit_refs(SimpleNamespace(ref_strength=0), "r.png", 64, 64) is None
    assert eng._ref_options((), None) == (("ref_strength", 0.0),)
    assert eng._edit_refs(SimpleNamespace(ref_strength=1), "", 64, 64) is None and eng._ref_options((), None) == ()


def test_the_pictures_prompt_rows_are_scaled():
    from fizgig.minimax.embedder import TEXT_TAG, VIDEO_TAG
    from fizgig.minimax_image.driver import picture_tokens_scaled
    tags = torch.tensor([TEXT_TAG, VIDEO_TAG, VIDEO_TAG, TEXT_TAG])
    cond = {"hidden_states": torch.ones(4, 3), "token_tags": tags}
    out = picture_tokens_scaled(cond, 0.25)["hidden_states"]
    assert out[:, 0].tolist() == [1.0, 0.25, 0.25, 1.0]
    assert picture_tokens_scaled(cond, 0)["hidden_states"][1:3].abs().sum() == 0
    for s in (None, 1, 5, "x"):                                       # no box, full, past full, unreadable
        assert picture_tokens_scaled(cond, s) is cond
    assert picture_tokens_scaled({"hidden_states": torch.ones(4, 3)}, 0.5)["hidden_states"].sum() == 12
    assert picture_tokens_scaled({**cond, "token_tags": tags[:3]}, 0.5)["hidden_states"].sum() == 12   # misaligned


def test_latent_scaling_families_are_unchanged():
    d = get("klein")
    assert d.reference_strength and not d.reference_strength_noise
    eng = _eng(d)
    refs = eng._edit_refs(SimpleNamespace(ref_strength=0.5), "r.png", 64, 64)
    assert torch.equal(refs[0], torch.full((1, 24, 4, 4), 0.5)) and eng._ref_options((), refs) == ()


def _encoding_engine(desc, tmp_path):
    from PIL import Image
    ref = tmp_path / "r.png"
    Image.new("RGB", (64, 64), (250, 250, 250)).save(ref)
    eng = WorkbenchEngine.__new__(WorkbenchEngine)
    eng.desc, eng._prompt_cache, eng.lowmem, eng.te_path, eng.device = desc, {}, False, "te", "cpu"
    eng._park_dit = lambda where: None
    seen = []
    eng.driver = SimpleNamespace(
        text_encoder_gb=lambda p: 0.0, text_cache_key=lambda: (), unload_text_encoder=lambda te: None,
        load_reference_text_encoder=lambda p, d: "te", load_text_encoder=lambda p, d: "te",
        encode_text_with_references=lambda te, caps, refs: seen.append(refs[0][0]) or [{"c": i} for i in caps])
    return eng, str(ref), seen


def test_the_encoder_sees_the_picture_faded_by_the_strength(tmp_path):
    from fizgig.families.workbench import REF_GREY
    eng, ref, seen = _encoding_engine(get("minimax_image"), tmp_path)
    assert eng._ref_fade(SimpleNamespace(ref_strength=0.25)) == 0.25
    assert eng._ref_fade(SimpleNamespace(ref_strength=3)) == 1.0 and eng._ref_fade(SimpleNamespace(ref_strength=-1)) == 0
    eng.encode(["a"], ref=ref, size=(64, 64), fade=1.0)
    eng.encode(["a"], ref=ref, size=(64, 64), fade=0.0)
    eng.encode(["a"], ref=ref, size=(64, 64), fade=0.5)
    eng.encode(["a"], ref=ref, size=(64, 64), fade=0.5)          # the same strength again: from the cache
    assert len(seen) == 3
    assert int(seen[0][..., :3].min()) == 250 and (seen[1][..., :3] == REF_GREY).all()
    assert abs(int(seen[2][0, 0, 0]) - round(250 * 0.5 + REF_GREY * 0.5)) <= 1


def test_other_families_see_the_whole_picture(tmp_path):
    eng, ref, seen = _encoding_engine(get("klein"), tmp_path)
    assert eng._ref_fade(SimpleNamespace(ref_strength=0.25)) == 1.0          # Klein scales its latents instead
