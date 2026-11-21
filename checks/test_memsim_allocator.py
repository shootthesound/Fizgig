"""The simulated GPU's allocator model and cuDNN workspace rule (checks/memsim.py), on the CPU: no card needed."""
import math
import random
import types

import pytest

pytest.importorskip("torch")
import memsim as M  # noqa: E402

GB, MB, PAGE = M.GB, M.MB, M.PAGE


def _blocks(p):
    b, out = p.tail, []
    while b is not None:
        out.append(b)
        b = b.prev
    return out[::-1]


def _check(p, reserved):
    off = mapped = 0
    bl = _blocks(p)
    for i, b in enumerate(bl):
        assert b.off == off
        off += b.size
        if b.mapped:
            mapped += b.size
        else:
            assert b.free and b.off % PAGE == 0 and b.size % PAGE == 0     # holes are whole pages
        if i:
            assert not (bl[i - 1].free and b.free and bl[i - 1].mapped == b.mapped)   # neighbours merged
    assert off == p.end and mapped == reserved
    assert {id(e[4]) for e in p.free} == {id(b) for b in bl if b.free and b.mapped}
    assert {id(e[2]) for e in p.holes} == {id(b) for b in bl if not b.mapped}


def test_expandable_empty_cache_gives_back_free_pages_anywhere():
    """PyTorch's expandable segments unmap every whole free page on empty_cache, not only those at the end (a freed
    block in the middle of the segment stayed reserved in the old model: Flux 3's epoch peak read 5 GB high)."""
    p, res = M._Pool(False, True, lambda n: None), 0
    blocks = []
    for _ in range(3):
        b, g = p.alloc(GB)
        res += g
        blocks.append(b)
    p.release(blocks[1])
    res -= p.empty_cache()
    _check(p, res)
    assert res == 2 * GB
    b, g = p.alloc(GB // 2)                 # the lowest hole first, mapping only what it needs
    res += g
    assert (b.off, g) == (GB, GB // 2)
    b, g = p.alloc(GB)                      # too big for what is left of the hole: the end
    res += g
    assert (b.off, g) == (3 * GB, GB)
    _check(p, res)


def test_allocator_invariants_random():
    rng = random.Random(7)
    for trial in range(60):
        p, res, live = M._Pool(trial % 3 == 0, True, lambda n: None), 0, []
        for _ in range(250):
            r = rng.random()
            if r < 0.5 or not live:
                s = M._round(rng.randint(1, M.SMALL) if p.small else
                             rng.choice([rng.randint(1, 50) * MB + rng.randint(0, 4095), rng.randint(1, 3000) * MB]))
                b, g = p.alloc(s)
                res += g
                live.append(b)
                assert not b.free and b.mapped and b.size >= s
            elif r < 0.9:
                p.release(live.pop(rng.randrange(len(live))))
            else:
                res -= p.empty_cache()
            _check(p, res)
        for b in live:
            p.release(b)
        res -= p.empty_cache()
        assert res == 0 and p.end == 0


def _t(shape, size, dev="cuda", layout="standard"):
    def contiguous(memory_format=None):
        return layout == ("standard" if memory_format is None else "last")
    return types.SimpleNamespace(shape=shape, device=types.SimpleNamespace(type=dev), element_size=lambda: size,
                                 numel=lambda: math.prod(shape), dim=lambda: len(shape), is_contiguous=contiguous)


@pytest.mark.parametrize("x, w, out, size, measured_mb", [
    ((1, 256, 1024, 1024), (256, 256, 3, 3), (1, 256, 1024, 1024), 4, 1026.3),   # SDXL / Z-Image decode, fp32
    ((1, 512, 256, 256), (512, 512, 3, 3), (1, 512, 256, 256), 4, 137.0),
    ((1, 128, 1024, 1024), (3, 128, 3, 3), (1, 3, 1024, 1024), 4, 524.1),       # 3 channels out: the output too
    ((1, 256, 1024, 1024), (128, 256, 1, 1), (1, 128, 1024, 1024), 4, 0.0),     # 1x1: none
    ((2, 640, 128, 128), (640, 640, 3, 3), (2, 640, 128, 128), 2, 87.0),        # SDXL UNet, bf16
])
def test_conv_workspace_matches_measured(x, w, out, size, measured_mb):
    got = M.conv_workspace((_t(x, size), _t(w, size), None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1),
                           _t(out, size)) / MB
    assert abs(got - measured_mb) <= max(1.0, 0.02 * measured_mb)
    assert M.conv_workspace((_t(x, size, "cpu"), _t(w, size), None), _t(out, size)) == 0     # only on the card


@pytest.mark.skipif(not M.CAN_RUN, reason="the simulated card needs a torch built with CUDA and an NVIDIA card")
def test_mover_threads_run_inline_for_modules_loaded_later():
    """Krea 2's offloader is imported when its DiT loads, inside the simulation: its mover thread must run inline too
    (a real thread shared the fake mode - an 'in kernel' assertion and half-moved weights in its swapped preview)."""
    import concurrent.futures
    with M.simulated_card(16, 1.3):
        from concurrent.futures import ThreadPoolExecutor
        assert ThreadPoolExecutor is M._Inline
    assert concurrent.futures.ThreadPoolExecutor is not M._Inline


def test_conv_workspace_follows_the_input_layout():
    """The Qwen-Image VAE's encode (Anima / Krea 2), measured: its first convolutions take a picture permuted into
    channels-last order - only the weight's size; a 1x1 kernel over attention's non-contiguous output copies its input."""
    w = (96, 96, 3, 3, 3)
    got = M.conv_workspace((_t((1, 96, 3, 994, 994), 2, layout="last"), _t(w, 2), None), _t((1, 96, 1, 992, 992), 2))
    assert abs(got / MB - 0.5) < 0.1                                   # measured 0.5 MB
    got = M.conv_workspace((_t((1, 384, 128, 128), 2, layout="other"), _t((384, 384, 1, 1), 2), None),
                           _t((1, 384, 128, 128), 2))
    assert abs(got / MB - 12.0) < 0.6                                  # measured 12.0 MB
    got = M.conv_workspace((_t((1, 96, 3, 1026, 1026), 2), _t(w, 2), None), _t((1, 96, 1, 1024, 1024), 2))
    assert abs(got / MB - 770.7) < 2                                   # the decode, standard layout: measured 770.7
    # fp32 copies its input whatever the layout: SDXL's convolution after an upsample (channels-last, as the residual
    # add before it leaves it), measured 137.0 MB
    got = M.conv_workspace((_t((1, 512, 256, 256), 4, layout="last"), _t((512, 512, 3, 3), 4), None),
                           _t((1, 512, 256, 256), 4))
    assert abs(got / MB - 137.0) < 1
