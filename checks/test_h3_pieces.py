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
