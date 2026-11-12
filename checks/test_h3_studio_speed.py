"""H3 Studio render speed: the int8 linear switch stays out of everything but a CUDA render, a LoRA's one-pass add is
the same maths as the two-pass one, and the base plan for a long clip (no reload to NF4, no over-asking past 56 frames)."""
import torch
import torch.nn as nn

from fizgig.modules import int8_linear
from fizgig.networks.lora import LoRAInfModule
from fizgig.repair_studio.h3_engine import H3RepairEngine as E


def test_int8_linear_switch_is_scoped_and_declines_off_the_card():
    assert not int8_linear.wanted()
    x, q, s = torch.randn(4, 16), torch.zeros(8, 16, dtype=torch.int8), torch.ones(8, 1)
    with torch.no_grad(), int8_linear.renders(True):
        assert int8_linear.wanted()
        assert int8_linear.linear(x, q, s, None, 1, torch.float32) is None      # CPU tensors: the caller's own path
    assert not int8_linear.wanted()
    with int8_linear.renders(True):
        assert int8_linear.linear(x, q, s, None, 1, torch.float32) is None      # a gradient is being taken


def _lora(lin, mult=0.7):
    m = LoRAInfModule("lora_unet_x", lin, multiplier=mult, lora_dim=4, alpha=2.0)
    nn.init.normal_(m.lora_up.weight)
    m.apply_to()
    return m


def test_lora_one_pass_add_matches_the_two_pass_one():
    torch.manual_seed(0)
    lin = nn.Linear(32, 48)
    m = _lora(lin)
    x = torch.randn(2, 5, 32)
    want = m.org_forward(x) + m.lora_up(m.lora_down(x)) * m.multiplier * m.scale
    with torch.no_grad():
        got = lin(x)
    assert torch.allclose(got, want, atol=1e-5)
    assert torch.allclose(lin(x), want, atol=1e-5)                              # with a gradient: the old path
    m.enabled = False
    with torch.no_grad():
        assert torch.equal(lin(x), m.org_forward(x))


def test_lora_one_pass_add_chains_two_loras():
    torch.manual_seed(1)
    lin = nn.Linear(16, 24)
    a = _lora(lin, 1.0)
    b = _lora(lin, -0.5)
    x = torch.randn(3, 16)
    want = (a.org_forward(x) + a.lora_up(a.lora_down(x)) * a.scale
            + b.lora_up(b.lora_down(x)) * -0.5 * b.scale)
    with torch.no_grad():
        assert torch.allclose(lin(x), want, atol=1e-5)


def _old_need(w, h, f, k=2, r=0):
    return (4.5 + (w * h / (768 * 640)) * (0.92 + 0.081 * f + 0.4 * k + 0.75 * r)) * 2 ** 30 / 1e9


def test_clip_need_is_unchanged_up_to_56_frames_and_lower_past_it():
    for c in [(768, 640, 22, 2, 0), (768, 640, 56, 2, 0), (768, 640, 22, 0, 2), (512, 512, 22, 0, 0),
              (1024, 1024, 56, 2, 0)]:
        assert abs(E.clip_need_gb(*c) - _old_need(*c)) < 1e-6
    long_clip = E.clip_need_gb(512, 512, 124, 0, 0)
    assert E.clip_need_gb(512, 512, 56, 0, 0) < long_clip < _old_need(512, 512, 124, 0, 0)
    assert E.clip_need_gb(512, 512, 124, 0, 0) < E.clip_need_gb(512, 512, 226, 0, 0)


def test_a_reload_for_room_keeps_the_int8_base():
    need = E.clip_need_gb(512, 512, 124, 0, 0)
    assert E.plan_base(12.0, "auto", need) == ("nf4", 0)                        # a first load on a small card
    quant, n = E.plan_base(12.0, "auto", need, keep_int8=True)                  # a reload mid-session
    assert quant == "int8" and 0 < n <= 40
    assert E.plan_base(31.0, "auto", need, keep_int8=True)[0] == "int8"
    assert E.plan_base(12.0, "nf4", need, keep_int8=True) == ("nf4", 0)         # the user's own pick stands


def _engine(have, free=30.5):
    e = E.__new__(E)
    e.base_mode, e.dit, e.base_plan, e._plan_free = "auto", object(), ("int8", have), free
    e.calls = []
    e._stream_more = lambda want: e.calls.append(want) or True
    return e


def test_more_room_streams_blocks_in_place_and_stops_at_the_most_a_plan_gives():
    e = _engine(6)
    e.ensure_room(768, 640, 124)                     # needs more than 6 blocks leave
    assert e.calls and 6 < e.calls[0] <= E._MAX_STREAMED
    e = _engine(E._MAX_STREAMED)
    e.ensure_room(1280, 960, 124)                    # asks past the most: nothing more to do, no reload each render
    assert e.calls == []
    e = _engine(20)
    e.ensure_room(512, 512, 22)                      # a smaller clip keeps the plan
    assert e.calls == []


def test_updater_moves_the_fp16_video_vae_to_the_int8_one_and_leaves_other_choices(tmp_path, monkeypatch):
    import json
    import os
    from fizgig.scripts import fetch_turbo_lora as F
    monkeypatch.setattr(F, "_valid_safetensors", lambda *a, **k: True)
    monkeypatch.setattr(F, "_download", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download")))
    models = tmp_path / "models"
    models.mkdir()
    (tmp_path / "user_data").mkdir()
    prefs = tmp_path / "user_data" / "prefs.json"
    fp16, int8 = models / F.H3_VAE_FP16, models / F.H3_VAE_INT8
    fp16.write_bytes(b"x")
    int8.write_bytes(b"x")
    prefs.write_text(json.dumps({"minimax_vae": str(fp16), "other": 1}))
    assert F.ensure_h3_int8_vae(str(tmp_path), log=lambda *_: None) == str(int8)
    got = json.loads(prefs.read_text())
    assert got == {"minimax_vae": str(int8), "other": 1} and os.path.isfile(fp16)      # fp16 file left in place
    prefs.write_text(json.dumps({"minimax_vae": str(models / "my_own_vae.safetensors")}))
    assert F.ensure_h3_int8_vae(str(tmp_path), log=lambda *_: None) is None             # someone's own file: kept
    prefs.write_text(json.dumps({}))
    assert F.ensure_h3_int8_vae(str(tmp_path), log=lambda *_: None) is None             # H3 not set up: nothing fetched


def test_pingpong_file_is_made_once_beside_the_clip(tmp_path):
    import os
    import pytest
    from fizgig.minimax.common import _find_ffmpeg, write_preview_mp4
    from fizgig.web import studio
    assert studio.pingpong(str(tmp_path / "x.png")) == str(tmp_path / "x.png")          # not a clip: itself
    if not _find_ffmpeg():
        pytest.skip("no ffmpeg here")
    clip = str(tmp_path / "tweaked_3.mp4")
    write_preview_mp4(clip, torch.rand(3, 9, 64, 64), None, fps=24)
    out = studio.pingpong(clip)
    assert out == str(tmp_path / "tweaked_3_pp.mp4") and os.path.getsize(out) > 0
    made = os.path.getmtime(out)
    assert studio.pingpong(clip) == out and os.path.getmtime(out) == made               # the second ask reuses it
