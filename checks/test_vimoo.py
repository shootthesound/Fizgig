"""Vimoo 1.0 Lite (fizgig/vimoo): the transformer's names and size against Vimoo's ComfyUI file, the VAE's names, the
distilled LoRA's q/k/v fusion, the schedule, quoted-text glyphs, prompt-travel padding and image-to-video's first
frame - all CPU, no model files (full-size models are built on the meta device)."""
import torch

from fizgig.families.registry import get


def _tiny(**kw):
    from fizgig.vimoo.model import VimooLite
    torch.manual_seed(0)
    cfg = dict(hidden_size=64, heads=2, depth=2, rope_dims=(8, 12, 12), text_dim=32, byt5_dim=16, vision_dim=8,
               mod_rank=4, in_channels=6)
    cfg.update(kw)
    m = VimooLite(**cfg)
    for p in m.parameters():
        p.data.normal_(0, 0.02)
    return m


def test_the_transformer_has_the_comfyui_files_names_and_size():
    """vimoo_v1_lite_bf16.safetensors: 1031 tensors, ~9.5B parameters (read from its header, 5 Nov 2026)."""
    from fizgig.vimoo.model import VimooLite
    with torch.device("meta"):
        m = VimooLite()
    sd = m.state_dict()
    assert len(sd) == 1031
    assert abs(sum(v.numel() for v in m.parameters()) / 1e9 - 9.526) < 0.001
    for k, shape in (("double_blocks.0.img_attn.qkv.weight", (9216, 3072)), ("double_blocks.39.img_mod.lin.weight",
                     (18432, 16)), ("img_in.proj.weight", (3072, 97, 1, 1, 1)), ("img_in.patch_merge.weight",
                     (3072, 12288)), ("final_layer.patch_split.weight", (12288, 3072)), ("mod_projection.weight",
                     (16, 3072)), ("vision_in.proj.3.weight", (3072, 1152)), ("byt5_in.fc1.weight", (3072, 1472))):
        assert tuple(sd[k].shape) == shape, k


def test_the_vae_has_wan_22_ti2vs_names():
    from fizgig.vimoo import vae as V
    cfg = dict(dim=160, z_dim=48, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
               temperal_downsample=[False, True, True], dropout=0.0)
    with torch.device("meta"):
        m = V.WanVAE_(**cfg)
    assert len(m.state_dict()) == 196 and len(V.LATENT_MEAN) == len(V.LATENT_STD) == 48


def test_the_distilled_loras_split_qkv_fuses_exactly():
    drv = get("vimoo").load_driver()
    g = torch.Generator().manual_seed(1)
    sd = {}
    for p in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj",
              "attn.to_out.0.projection", "img_mlp.in_proj.projection"):
        sd[f"transformer.double_blocks.3.{p}.lora_A.weight"] = torch.randn(4, 12, generator=g)
        sd[f"transformer.double_blocks.3.{p}.lora_B.weight"] = torch.randn(12, 4, generator=g)
    out = drv.convert_lora_state_dict(sd)
    k = "lora_unet_double_blocks_3_img_attn_qkv"
    fused = out[k + ".lora_up.weight"] @ out[k + ".lora_down.weight"]
    for i, p in enumerate(("to_q", "to_k", "to_v")):
        split = sd[f"transformer.double_blocks.3.attn.{p}.lora_B.weight"] @ sd[f"transformer.double_blocks.3.attn.{p}.lora_A.weight"]
        assert torch.allclose(fused[i * 12:(i + 1) * 12], split, atol=1e-6)
    # alpha / rank stays the adapter's 0.5 on the fused (3r) and plain (r) pairs
    assert float(out[k + ".alpha"]) / out[k + ".lora_down.weight"].shape[0] == 0.5
    k2 = "lora_unet_double_blocks_3_img_mlp_0"
    assert float(out[k2 + ".alpha"]) / out[k2 + ".lora_down.weight"].shape[0] == 0.5
    # a file in the family's own format passes through untouched
    own = {"lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight": torch.zeros(2, 3)}
    assert drv.convert_lora_state_dict(own) is own


def test_the_lite_schedule():
    from fizgig.vimoo.driver import lognorm_sigmas
    s = lognorm_sigmas(50)
    assert len(s) == 51 and float(s[0]) == 1.0 and float(s[-1]) == 0.0
    assert bool((s[1:] <= s[:-1]).all())


def test_glyphs_are_the_quoted_texts_once_each():
    from fizgig.vimoo.text import glyph_texts
    assert glyph_texts('a sign that says "OPEN" and “SALE”, then "OPEN" again') == ["OPEN", "SALE"]
    assert glyph_texts("no quotes here") == []


def test_prompt_travel_padding_renders_as_the_prompt_itself():
    drv = get("vimoo").load_driver()
    m = _tiny()
    a = {"text": torch.randn(5, 32)}
    b = {"text": torch.randn(9, 32), "glyph": torch.randn(4, 16)}
    pa, _pb = drv.pad_conditioning([a, b])
    x = torch.randn(1, 13, 2, 4, 4)
    o1 = m(x, torch.tensor([400.0]), a["text"][None])
    o2 = m(x, torch.tensor([400.0]), pa["text"][None], pa["text_mask"][None], pa["glyph"][None], pa["glyph_mask"][None])
    assert (o1 - o2).abs().max() < 1e-5


def test_image_to_video_keeps_the_first_frame_and_moves_the_rest():
    drv = get("vimoo").load_driver()
    m = _tiny(in_channels=48, vision_dim=1152, text_dim=3584, byt5_dim=1472)
    cond = {"text": torch.randn(6, 3584), "vision": torch.randn(729, 1152)}
    ref = torch.randn(1, 48, 4, 4)
    out = drv.generate(m, cond, 64, 64, steps=3, seed=7, frames=9, refs=[ref])["latent"]
    assert out.shape == (1, 48, 3, 4, 4)
    anchor = drv._anchor(drv._ref5(ref, "cpu"), drv.initial_noise(7, 64, 64, 9)[:, :, :1].to(torch.bfloat16))
    assert torch.allclose(out[:, :, :1], anchor.float(), atol=1e-2)
    assert (out[:, :, 1:] - out[:, :, :1]).abs().mean() > 0.1


def test_the_image_to_video_share_trains_a_clip_from_its_own_first_frame():
    """With i2v_share on, a clip step conditions on its own first latent frame and the cached SigLIP tokens, and scores
    only the frames after it; off, it trains text-to-video on every frame."""
    drv = get("vimoo").load_driver()
    m = _tiny(in_channels=48, vision_dim=1152, text_dim=3584, byt5_dim=1472)
    calls = []
    real = m.forward
    m.forward = lambda x, t, *a, **kw: (calls.append((x[:, 48:].abs().sum().item(), kw.get("observed_t") is not None,
                                                      kw.get("vision") is not None)), real(x, t, *a, **kw))[1]
    lat = torch.randn(1, 48, 3, 4, 4)
    cond = {"text": torch.randn(1, 6, 3584), "first_vision": torch.randn(1, 729, 1152)}
    drv.set_options({"i2v_share": "1"})
    loss, _ = drv.training_loss(m, lat, cond, torch.Generator().manual_seed(0))
    assert calls[-1][0] > 0 and calls[-1][1] and calls[-1][2] and torch.isfinite(loss)
    drv.set_options({})
    drv.training_loss(m, lat, cond, torch.Generator().manual_seed(0))
    assert calls[-1] == (0.0, False, False)


def test_studio_start_frame_conditioning():
    """Studio's First frame reaches generate() as Vimoo's image-to-video: the latent as refs, SigLIP's tokens in
    the prompt's conditioning; no last-frame slot is offered, and a still ignores the start frame."""
    from types import SimpleNamespace

    from fizgig.vimoo.workbench import VimooWorkbenchEngine
    assert VimooWorkbenchEngine.keyframe_slots == ("first",)
    assert get("vimoo").workbench_engine_class() is VimooWorkbenchEngine
    hit = {"latent": torch.zeros(1, 48, 1, 4, 4), "vision": torch.ones(729, 1152)}
    st = SimpleNamespace(keyframes=[(0, hit)])
    eng = VimooWorkbenchEngine.__new__(VimooWorkbenchEngine)
    cond, extra = eng._clip_conditioning(st, {"text": torch.zeros(3, 8)}, 64, 64, 33)
    assert cond["vision"] is hit["vision"] and extra["refs"][0] is hit["latent"]
    assert eng._clip_conditioning(st, {"text": 1}, 64, 64, 1) == ({"text": 1}, {})
    assert VimooWorkbenchEngine.keyframe_signature(st)
