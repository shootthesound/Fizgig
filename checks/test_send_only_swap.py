"""Send-only block swap (fizgig.krea2.offloading.LoRAStreamOffloader) for Krea 2 and MOSS-Image LoRA runs.

A LoRA never changes the base model, so the CPU copy of a swapped block is always current: the classic swap's copy back
(device -> host) is pure overhead. Send-only keeps each streamed block's weights on the CPU, copies them host -> device
into a small GPU ring for use, and drops the GPU copy by re-pointing the module at the CPU copy - nothing comes back.

These checks drive the real offloader, the real Krea 2 / MOSS-Image block loops, swap switches, drivers and Fizgig's own
LoRA / LoKR adapters on CPU tiny models (the copy engine there is a plain synchronous copy - the ring residency, the
evictions, the backward hooks and gradient checkpointing's recompute are all the real code). Against the same model with
no block swap at all, the losses, LoRA gradients and LoRA weights after several AdamW steps - with validation-style
forward-only passes and MOSS's preview park in between - come out bit-identical. They also count the weight bytes each
swap moves per training step: send-only moves the same host -> device bytes as the classic swap and nothing back.
Real speed needs a GPU run."""
import os
import sys
import types

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.families.lora import LoRALinear  # noqa: E402
from fizgig.krea2 import offloading  # noqa: E402
from fizgig.krea2.driver import Krea2Driver  # noqa: E402
from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402
from fizgig.moss_image.driver import MOSSImageDriver  # noqa: E402
from fizgig.moss_image.model import MOSSImageDiT  # noqa: E402
from fizgig.modules import offloading as shared_offloading  # noqa: E402

CPU = torch.device("cpu")
KCFG = SingleMMDiTConfig(features=64, tdim=32, txtdim=32, heads=4, multiplier=2, layers=8, patch=1, channels=4,
                         kvheads=2, txtlayers=2, txtheads=4, txtkvheads=4)
MTINY = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=6, text_dim=48, freq_dim=32, latent_channels=16)
STEPS = 4


@pytest.fixture(autouse=True)
def _one_thread():
    """One CPU thread: the int8 path's multi-threaded CPU kernels are not run-to-run deterministic (two runs of the same
    unswapped model differ by ~1e-3 in a gradient now and then), and these checks compare bit for bit."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ---- models -------------------------------------------------------------------------------------------------------
def _wrap(blocks, names, kind):
    """Fizgig's own adapters on the block Linears (families/lora.py), seeded so every model gets the same ones."""
    torch.manual_seed(123)
    for block in blocks:
        for name, child in list(block.named_modules()):
            if isinstance(child, nn.Linear) and name.split(".")[-1] in names:
                parent = block.get_submodule(name.rpartition(".")[0]) if "." in name else block
                w = LoRALinear(child)
                if kind == "lokr":
                    w.add_lokr("lora", True, factor=4)
                    nn.init.normal_(w.adapters["lora"].lokr_w2, std=0.02)
                else:
                    w.add("lora", 4, 4, True)
                    nn.init.normal_(w.adapters["lora"][1].weight, std=0.02)
                setattr(parent, name.split(".")[-1], w)


def _int8(model, blocks):
    from fizgig.families import quant
    drv = types.SimpleNamespace(int8_fp32_scales=True)
    for b in blocks:
        for m in b.modules():
            if type(m) is nn.Linear:
                quant._quantize_one(m, drv, "int8", CPU)


def _krea(mode, kind="lora", int8=False, swap=4):
    torch.manual_seed(0)
    m = SingleStreamDiT(KCFG)
    for p in m.parameters():
        p.data.normal_(0, 0.05)
    m.requires_grad_(False)
    if int8:
        _int8(m, m.blocks)
    if mode != "none":
        Krea2Driver().enable_block_swap(m, swap, "cpu", True, send_only=(mode == "send"))
    _wrap(m.blocks, {"qkv", "q", "k", "v", "proj", "o", "out", "w1", "w2", "w3", "down", "up", "gate"}, kind)
    m.enable_gradient_checkpointing()
    m.train()
    return m


def _krea_batch(step, batch=2):
    g = torch.Generator().manual_seed(100 + step)
    L, T = 6, 5
    img = torch.randn(batch, L, 4, generator=g)
    ctx = torch.randn(batch, T, 2, 32, generator=g)
    t = torch.rand(batch, generator=g)
    pos = torch.randn(batch, L + T, 3, generator=g)
    mask = torch.ones(batch, L + T, dtype=torch.bool)
    mask[1, L + 3:] = False
    return img, ctx, t, pos, mask


def _krea_fwd(m, step):
    img, ctx, t, pos, mask = _krea_batch(step)
    return (m(img, ctx, t, pos, mask) - img).square().mean()


def _moss(mode, kind="lora", swap=3):
    torch.manual_seed(0)
    m = MOSSImageDiT(**MTINY)
    for p in m.parameters():
        p.data.normal_(0, 0.05)
    m.requires_grad_(False)
    if mode != "none":
        MOSSImageDriver().enable_block_swap(m, swap, CPU, True, send_only=(mode == "send"))
        m.switch_block_swap_for_training()
    _wrap(m.layers, {"q", "v", "ffn_down"}, kind)
    m.enable_gradient_checkpointing(True)
    m.train()
    return m


def _moss_fwd(m, step, batch=2):
    g = torch.Generator().manual_seed(200 + step)
    x = torch.randn(batch, 16, 4, 6, generator=g)
    ctx = torch.randn(batch, 9, 48, generator=g)
    mask = torch.zeros(batch, 9, dtype=torch.bool)
    mask[0, :9] = True
    if batch > 1:
        mask[1, :4] = True
    t = torch.tensor([900.0, 300.0][:batch])
    return (m(x, t, ctx, mask) - x).square().mean()


def _lora_params(m):
    return {n: p for n, p in m.named_parameters() if p.requires_grad}


def _train(m, fwd, driver, between=None):
    """STEPS AdamW steps; `between(step)` runs between steps (a validation forward, a preview park)."""
    params = _lora_params(m)
    opt = torch.optim.AdamW(list(params.values()), lr=1e-2)
    losses, grads = [], []
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss = fwd(m, step)
        loss.backward()
        losses.append(loss.detach().clone())
        grads.append({n: p.grad.clone() for n, p in params.items()})
        opt.step()
        if between is not None:
            between(step)
    return losses, grads, {n: p.detach().clone() for n, p in params.items()}


def _same(a, b):
    la, ga, wa = a
    lb, gb, wb = b
    assert all(torch.equal(x, y) for x, y in zip(la, lb)), (la, lb)
    for x, y in zip(ga, gb):
        assert x.keys() == y.keys()
        for n in x:
            assert torch.equal(x[n], y[n]), n
    assert wa.keys() == wb.keys() and all(torch.equal(wa[n], wb[n]) for n in wa)


def _val(driver, m, fwd):
    def between(step):
        driver.block_swap_mode(m, inference=True)        # the fixed validation / a preview: forward-only streaming
        with torch.no_grad():
            m.eval()
            fwd(m, 50 + step)
            m.train()
        driver.block_swap_mode(m, inference=False)
    return between


# ---- parity -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["lora", "lokr"])
def test_krea2_send_only_matches_no_swap(kind):
    ref = _train(_krea("none", kind), _krea_fwd, Krea2Driver())
    m = _krea("send", kind)
    assert isinstance(m.offloader, offloading.LoRAStreamOffloader)
    assert m.offloader.S == 6 and m.offloader.B == 2         # 4 swapped + the 2-slot ring: the classic swap's VRAM
    _same(_train(m, _krea_fwd, Krea2Driver(), _val(Krea2Driver(), m, _krea_fwd)), ref)


def test_krea2_send_only_int8_base():
    ref = _train(_krea("none", int8=True), _krea_fwd, Krea2Driver())
    m = _krea("send", int8=True)
    assert isinstance(m.offloader, offloading.LoRAStreamOffloader)
    streamed = [mm for mm in m.blocks[m.offloader.stream_idx[0]].modules() if getattr(mm, "_is_int8", False)]
    assert streamed and all(mm.weight.dtype == torch.int8 for mm in streamed)    # int8 codes stream as they are
    _same(_train(m, _krea_fwd, Krea2Driver(), _val(Krea2Driver(), m, _krea_fwd)), ref)


@pytest.mark.parametrize("kind", ["lora", "lokr"])
def test_moss_send_only_matches_no_swap(kind):
    ref = _train(_moss("none", kind), _moss_fwd, MOSSImageDriver())
    m = _moss("send", kind)
    assert isinstance(m.offloader, offloading.LoRAStreamOffloader)
    drv = MOSSImageDriver()
    val = _val(drv, m, _moss_fwd)

    def between(step):
        val(step)
        if step == 1:            # the SFT preview's park and unpark: re-enters enable_block_swap bare, both ways
            tok = drv.park_for_preview(m, CPU)
            assert isinstance(m.offloader, offloading.LoRAStreamOffloader)
            drv.unpark_after_preview(m, CPU, tok)
            assert isinstance(m.offloader, offloading.LoRAStreamOffloader) and m.blocks_to_swap == 3
    _same(_train(m, _moss_fwd, drv, between), ref)


def test_base_weights_never_change():
    m = _krea("send")
    before = {i: [w.clone() for w in m.offloader.cpu_master[i]] for i in m.offloader.stream_idx}
    _train(m, _krea_fwd, Krea2Driver())
    for i, ws in before.items():
        assert all(torch.equal(a, b.data) for a, b in zip(ws, m.offloader.cpu_master[i]))
        assert all(not p.requires_grad for p in m.offloader.cpu_master[i])


# ---- copies per step ------------------------------------------------------------------------------------------------
def _block_bytes(block):
    return sum(mm.weight.numel() * mm.weight.element_size() for mm in block.modules()
               if hasattr(mm, "weight") and mm.weight is not None and mm.__class__.__name__.endswith("Linear"))


def _classic_tally(monkeypatch, cls, m, fwd):
    """The classic offloader's real schedule, counted: each swap = one block device -> host + one host -> device. The
    move itself is skipped (on CPU there is nowhere to move to), so the step runs the plain model's maths."""
    tally = {"h2d_bytes": 0, "d2h_bytes": 0, "swaps": 0}

    def submit(self, blocks, to_cpu, to_cuda):
        tally["swaps"] += 1
        tally["d2h_bytes"] += _block_bytes(blocks[to_cpu])
        tally["h2d_bytes"] += _block_bytes(blocks[to_cuda])
    monkeypatch.setattr(cls, "_submit_move_blocks", submit)
    monkeypatch.setattr(cls, "prepare_block_devices_before_forward", lambda self, blocks: None)
    fwd(m, 0).backward()                       # warm-up step, then count one steady-state step
    for k in tally:
        tally[k] = 0
    fwd(m, 1).backward()
    return tally


def _send_tally(m, fwd):
    fwd(m, 0).backward()
    t0 = dict(m.offloader.transfers)
    fwd(m, 1).backward()
    return {k: m.offloader.transfers[k] - t0[k] for k in t0}


@pytest.mark.parametrize("family", ["krea2", "moss"])
def test_copies_per_step(monkeypatch, family, capsys):
    if family == "krea2":
        classic, send, fwd, cls = _krea("classic"), _krea("send"), _krea_fwd, offloading.ModelOffloader
    else:
        classic, send, fwd, cls = _moss("classic"), _moss("send"), lambda m, s: _moss_fwd(m, s, batch=1), shared_offloading.ModelOffloader
    assert type(classic.offloader) is cls
    s = _send_tally(send, fwd)
    c = _classic_tally(monkeypatch, cls, classic, fwd)
    swap = classic.blocks_to_swap
    assert c["swaps"] == 2 * swap                            # S out-and-in exchanges forward, S backward
    assert s["d2h_bytes"] == 0 and s["d2h_copies"] == 0      # nothing ever comes back
    assert s["h2d_copies"] == 2 * swap                       # the same blocks' worth sent, one coalesced copy each
    # the same host -> device payload (the flat block buffer adds only 256-byte alignment padding)
    assert c["h2d_bytes"] <= s["h2d_bytes"] <= c["h2d_bytes"] * 1.05
    with capsys.disabled():
        print(f"\n[{family}] per step: classic {c['h2d_bytes'] + c['d2h_bytes']} B "
              f"(H2D {c['h2d_bytes']} + D2H {c['d2h_bytes']}, {c['swaps']} swaps); "
              f"send-only {s['h2d_bytes']} B (H2D only, {s['h2d_copies']} copies)")


# ---- when send-only is used ---------------------------------------------------------------------------------------
def test_plan():
    from fizgig.families.train import send_only_plan
    assert send_only_plan(True, 24, "int8", True, 0, False) is True
    assert send_only_plan(False, 24, "int8", True, 0, False) is False      # switched off
    assert send_only_plan(True, 0, "int8", True, 0, False) is False        # no swap
    assert send_only_plan(True, 24, "nf4", True, 0, False) is False        # NF4 never swaps
    assert send_only_plan(True, 24, "int8", False, 0, False) is False      # un-checkpointed: the ring would be stale
    assert send_only_plan(True, 24, "int8", True, 3, False) is False       # some blocks un-checkpointed
    assert send_only_plan(True, 24, "int8", True, 0, True) is False        # a fine-tune trains the base


def test_falls_back_to_classic():
    torch.manual_seed(0)
    m = SingleStreamDiT(KCFG)
    m.requires_grad_(False)
    m.blocks[6].attn.requires_grad_(True)                      # a trained base weight: must never stream one-way
    assert "trainable" in offloading.send_only_unfit(list(m.blocks), 8, 6)
    Krea2Driver().enable_block_swap(m, 4, "cpu", True, send_only=True)
    assert type(m.offloader) is offloading.ModelOffloader
    m.blocks[6].attn.requires_grad_(False)
    blocks = list(m.blocks)
    blocks[3] = SingleStreamDiT(SingleMMDiTConfig(**{**KCFG.__dict__, "features": 128, "heads": 8})).blocks[0].requires_grad_(False)
    assert "differ" in offloading.send_only_unfit(blocks, 8, 6)
    assert offloading.send_only_unfit(list(m.blocks), 8, 6) is None


def test_moss_classic_by_default_and_on_request():
    assert type(_moss("classic").offloader) is shared_offloading.ModelOffloader
    m = _moss("send")
    m.enable_block_swap(2, CPU, True)                          # bare re-entry keeps the run's mode
    assert isinstance(m.offloader, offloading.LoRAStreamOffloader)
    m.enable_block_swap(2, CPU, True, send_only=False)
    assert type(m.offloader) is shared_offloading.ModelOffloader


def test_load_base_passes_it_only_to_drivers_that_offer_it(monkeypatch):
    from fizgig.families import quant
    seen = {}

    class Drv:
        send_only_block_swap = False

        def load_planned(self, *a):
            return None

        def max_blocks_to_swap(self, dit=None):
            return 10

        def load_dit(self, path, device):
            return nn.Linear(2, 2)

        def enable_block_swap(self, dit, n, device, supports_backward):
            seen["classic"] = n
    monkeypatch.setattr(quant, "quantize", lambda *a, **k: 0)
    quant.load_base(Drv(), "x", "cpu", "int8", 4, send_only=True)
    assert seen == {"classic": 4}

    class Drv2(Drv):
        send_only_block_swap = True

        def enable_block_swap(self, dit, n, device, supports_backward, send_only=False):
            seen["send"] = send_only
    quant.load_base(Drv2(), "x", "cpu", "int8", 4, send_only=True)
    assert seen["send"] is True


def test_flag():
    from fizgig.families.train import setup_parser
    req = ["--family", "krea2", "--dit", "x", "--output_dir", "o", "--output_name", "n"]
    assert setup_parser().parse_known_args(req)[0].send_only_swap is True
    assert setup_parser().parse_known_args(req + ["--no-send_only_swap"])[0].send_only_swap is False


def test_the_parity_check_catches_a_stale_ring(monkeypatch):
    """Control: with the host -> device copy skipped (the ring keeps whatever block was there before), the same
    comparison fails - so the bit-identical results above mean the right weights were in place at every use."""
    ref = _train(_krea("none"), _krea_fwd, Krea2Driver())
    m = _krea("send")
    monkeypatch.setattr(offloading._SyncCopier, "submit", lambda self, key, dst, src, gate_event=None: None)
    with pytest.raises(AssertionError):
        _same(_train(m, _krea_fwd, Krea2Driver()), ref)


def test_why_it_needs_checkpointing():
    """Without gradient checkpointing a streamed block's weights would be saved for the backward from a ring slot the
    next block then overwrites: autograd refuses it (the plan never asks for send-only there - test_plan)."""
    m = _krea("send")
    m.disable_gradient_checkpointing()
    with pytest.raises(RuntimeError, match="inplace"):
        _krea_fwd(m, 0).backward()

