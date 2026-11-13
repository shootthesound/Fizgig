"""H3 Studio's In pieces clips (minimax/continuation.py): the piece plan stays on the model's frame grid, timed prompt
lines land on a piece, a later piece's call pins what it is shown and adds its guide rows, and the join deflicker
leaves the rest of the clip alone."""
import numpy as np
import torch

from fizgig.minimax import continuation as co


def test_pieces_cover_the_clip_on_the_frame_grid():
    assert co.piece_lengths(124) == [124] and co.piece_lengths(56) == [56]
    for n in range(8, 40):
        frames = 17 * n + 5
        lengths = co.piece_lengths(frames)
        assert all((x - 5) % 17 == 0 and x <= 124 for x in lengths)
        assert lengths[0] + sum(x - 39 for x in lengths[1:]) == frames
        assert co.joins(lengths) == [124 + 85 * k for k in range(len(lengths) - 1)]


def test_a_piece_length_of_your_own():
    assert co.piece_frames(None) == 124 and co.piece_frames(240) == 243 and co.piece_frames(10) == 56
    lengths = co.piece_lengths(549, 243)
    assert lengths == [243, 243, 141] and co.joins(lengths) == [243, 447]
    assert co.piece_lengths(243, 243) == [243]


def test_timed_prompt_lines():
    assert co.timed_prompts("a beach") == [(0.0, "a beach")]
    assert co.timed_prompts("a\nb\n8s: c\n 12.5 s: d") == [(0.0, "a\nb"), (8.0, "c"), (12.5, "d")]
    # four pieces start at 0, 5.2, 8.7 and 12.2 s: a change lands on the nearest start
    assert co.piece_prompts("a\n8s: c\n12.5s: d", co.piece_lengths(379)) == ["a", "a", "c", "d"]
    assert co.piece_prompts("a\n6s: c", co.piece_lengths(294)) == ["a", "c", "c"]


def test_a_later_piece_is_shown_the_piece_before():
    seen = {}

    class Model:
        def __call__(self, x, t, text, audio_rows=None, return_audio=False, **kw):
            seen.update(x=x, audio=audio_rows, kf=kw.get("keyframes"))
            return x

    class Engine:
        dit = Model()

    lat, aud = torch.randn(1, 24, co.CTX, 4, 4), torch.randn(2, 65, 32)
    mine = (0, torch.zeros(1, 24, 1, 4, 4))
    model = co.Continue(Engine(), lat, aud, seed=1)
    x, rows = torch.randn(1, 24, 37, 4, 4), torch.randn(2 * 207, 32)
    model(x, torch.tensor([1.0]), None, audio_rows=rows, keyframes=[mine])          # t = 1: clean
    assert torch.allclose(seen["x"][:, :, :co.PIN], lat[:, :, :co.PIN], atol=1e-5)
    assert torch.equal(seen["x"][:, :, co.PIN:], x[:, :, co.PIN:])
    assert [f for f, _z in seen["kf"]] == [0, 1, 5, 9, 13, 17, 18, 22, 26, 30, 34, 35, 0] and seen["kf"][-1] is mine
    got = seen["audio"].reshape(2, -1, 32)
    assert torch.allclose(got[:, :65], aud, atol=1e-4) and torch.equal(got[:, 65:], rows.reshape(2, -1, 32)[:, 65:])


def test_deflicker_touches_only_the_frames_near_a_join():
    rng = np.random.default_rng(0)
    frames = [np.full((48, 64, 3), 120.0, np.float32) + rng.normal(0, 2, (48, 64, 3)).astype(np.float32)
              for _ in range(120)]
    for f in frames[60:]:
        f *= 1.06                                       # a lighting step at the join
    before = [f.copy() for f in frames]
    co.deflicker(frames, [60])
    assert all(np.array_equal(a, b) for a, b in zip(frames[:36], before[:36]))
    assert all(np.array_equal(a, b) for a, b in zip(frames[84:], before[84:]))
    step = lambda fs: abs(float(fs[60].mean() - fs[59].mean()))
    assert step(frames) < 0.35 * step(before)


def test_the_donor_strength_does_not_render_the_baseline_again():
    from types import SimpleNamespace as NS
    from fizgig.families.video_workbench import ClipContract

    class Engine(ClipContract):
        primary_path, donor_path = "p.safetensors", "d.safetensors"

    key = lambda st, **k: Engine().clip_key(st, frames=22, steps=4, turbo_strength=1.0, with_audio=True, **k)
    st = lambda **k: NS(seed=1, prompt="a", preview_width=512, preview_height=512, primary_scale=1.0, donor_scale=1.0,
                        **k)
    a, b = st(), st()
    b.donor_scale = 0.4
    assert key(a) == key(b)                             # the donor is off in the Baseline clip
    b.primary_scale = 0.8
    assert key(a) != key(b) and key(a, primary_scale=False) == key(b, primary_scale=False)


def test_loop_back_is_a_first_last_frame_clip_from_the_clips_end_to_its_start():
    from types import SimpleNamespace as NS
    from PIL import Image
    from fizgig.repair_studio.h3_engine import H3RepairEngine
    calls = {}

    class Engine:
        regime_params = lambda self, regime, steps, turbo: (steps, turbo)
        _int8_tag = lambda self: False
        encode_keyframe = lambda self, im, w, h: im.getpixel((0, 0))[0]          # which picture it was

        def render_latent(self, st, *, frames, **_k):
            calls.update(frames=frames, prompt=st.prompt, keyframes=st.keyframes)
            return torch.zeros(1, 24, 2, 2, 2), None

        def decode_clip_frames(self, lat):
            return [Image.new("RGB", (8, 8), (200, 0, 0)) for _ in range(calls["frames"])]

        def decode_audio(self, aud):
            return torch.ones(2, 100 * calls["frames"])

    class State(NS):
        def copy(self):
            return State(**self.__dict__)

    clip = {"frames": [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(124)], "wav": torch.zeros(2, 12400)}
    st = State(prompt="back", seed=1, preview_width=8, preview_height=8, keyframes=None, references=None, blocks={})
    got = H3RepairEngine.loop_back_clip(Engine(), st, clip, frames_new=0, steps=4, turbo_strength=1.0)
    assert calls["frames"] == 124 and calls["prompt"] == "back"
    assert calls["keyframes"] == [(0, 123), (123, 0)]            # from the clip's last frame to its first
    assert len(got["frames"]) == 124 + 123 and got["frames"][123].getpixel((0, 0))[0] == 123
    assert got["wav"].shape[1] == 12400 + 12300
    H3RepairEngine.loop_back_clip(Engine(), st, clip, frames_new=60, steps=4, turbo_strength=1.0)
    assert calls["frames"] == 56                                 # a length of your own goes to the frame grid
    got = H3RepairEngine.loop_back_clip(Engine(), st, clip, frames_new=0, steps=4, turbo_strength=1.0, loop=False)
    assert calls["keyframes"] == [(0, 123)] and len(got["frames"]) == 247        # Continue: no last frame


def test_upscale_sections_share_a_frame_and_end_on_the_clips_last_frame():
    from fizgig.minimax import upscale as up
    assert up.sections(124) == [(0, 124)]
    assert up.sections(370) == [(0, 124), (123, 124), (246, 124)]
    assert up.sections(383, 192) == [(0, 192), (191, 192)]
    for total in (22, 100, 140, 247, 379, 396, 500):
        for sec in (0, 56, 192):
            plan = up.sections(total, sec)
            done = 0
            for a, n in plan:
                assert n >= 22 and (n - 5) % 17 == 0                   # a length the model makes
                assert a <= max(0, done - 1)                           # starts on a frame already finished
                done = a + n
            assert done == total                                       # every frame of the clip, none past it
    assert all(n >= 56 for _a, n in up.sections(379))                  # no leftover of a few frames
    assert up.section_frames(0) == 124 and up.section_frames(190) == 175


def test_upscale_noise_levels_start_at_the_denoise_and_end_at_zero():
    from fizgig.minimax import upscale as up
    sig = up.low_sigmas(0.4, 3)
    assert len(sig) == 4 and abs(sig[0] - 0.4) < 1e-6 and sig[-1] == 0.0
    assert all(a > b for a, b in zip(sig, sig[1:]))
    assert abs(up.low_sigmas(1.0, 2)[0] - 1.0) < 1e-6


def test_upscale_step_weighting_and_noise_range():
    from fizgig.minimax import upscale as up
    even, low_end, high_end = up.low_sigmas(0.4, 6), up.low_sigmas(0.4, 6, 1.0), up.low_sigmas(0.4, 6, -1.0)
    assert low_end[0] == even[0] == high_end[0] and low_end[-1] == high_end[-1] == 0.0
    assert low_end[3] < even[3] < high_end[3]                  # the levels crowd to one end or the other
    rng = up.low_sigmas(0.6, 4, 0.0, 0.3)                      # a range: 4 steps between 0.6 and 0.3, then clean
    assert len(rng) == 5 and abs(rng[0] - 0.6) < 1e-6 and abs(rng[3] - 0.3) < 1e-6 and rng[4] == 0.0
    assert all(a > b for a, b in zip(rng, rng[1:]))
    assert up.low_sigmas(0.4, 1, 0.0, 0.2) == [0.4, 0.0]


def test_upscale_section_hint_stays_under_the_measured_limits():
    from fizgig.minimax import upscale as up
    # a 5090 (33.1 GB plannable): 1536x896 took 192 frames and failed at 260; 1920x1088 took 124, failed at 192
    assert 192 <= up.room_frames(33.1, 1536, 896) < 260 and 124 <= up.room_frames(33.1, 1920, 1088) < 192
    # held to 24 GB: 1280x736 took 192 and failed at 260; 1536x896 took 124 and failed at 192
    assert 192 <= up.room_frames(23.35, 1280, 736) < 260 and 124 <= up.room_frames(23.35, 1536, 896) < 192
    assert up.room_frames(8.0, 1920, 1088) == 0


def test_upscale_encodes_in_tiles_that_add_up_to_the_whole_frame():
    """An encoder that only averages 16-px cells gives the same latent whole or in tiles; 17 + 5 frames give 5 + 2."""
    from PIL import Image
    from fizgig.minimax import upscale as up

    class Enc:
        def encode(self, x):                                           # [1, 3, F, h, w] -> [1, 24, 5 or 2, h/16, w/16]
            f = x.shape[2]
            t = torch.stack([x[:, :, i].mean(1) for i in ([0, 4, 8, 12, 16] if f == 17 else [0, 4])], 1)
            return torch.nn.functional.avg_pool2d(t, 16).unsqueeze(1).repeat(1, 24, 1, 1, 1)

    rng = np.random.default_rng(0)
    frames = [Image.fromarray(rng.integers(0, 255, (160, 224, 3), dtype=np.uint8)) for _ in range(22)]
    whole = up.encode_tiled(Enc(), frames, "cpu", tile=512)
    tiled = up.encode_tiled(Enc(), frames, "cpu", tile=96, overlap=32)
    assert tuple(whole.shape) == (1, 24, 7, 10, 14)
    assert float((whole - tiled).abs().max()) < 1e-5
