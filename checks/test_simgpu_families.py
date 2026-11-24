"""Each family's own block swap on the simulated GPU (checks/simgpu.py): a tiny model of the family, set up as the
trainer sets it up (the base with block swap on, gradient checkpointing, then Fizgig's LoRA wrapped round its Linears)
and trained a few steps through the family driver's own training loss - one picture a step, or the driver's batch
path (several forwards, one backward) - with an optimizer step between, so each step starts from where the last left
the blocks. Against the same model with nothing swapped: the same losses and gradients, and no operation ever meets a
block left on the CPU."""
import copy

import pytest
import torch

pytest.importorskip("fizgig.families.lora")
from simgpu import on_card, simulated_gpu  # noqa: E402

from fizgig.families.lora import FamilyLoRA  # noqa: E402
from fizgig.families.registry import get as get_family  # noqa: E402

N = 6                                         # blocks in each tiny model: swaps of 1 .. N-2


def _moss(monkeypatch):
    from fizgig.moss_image.model import MOSSImageDiT
    dit = MOSSImageDiT(dim=64, ffn_dim=96, num_heads=4, num_layers=N, text_dim=24, freq_dim=32, latent_channels=8)

    def item(g):
        return torch.randn(1, 8, 4, 6, generator=g), {"hidden_states": torch.randn(1, 5, 24, generator=g)}
    return dit.float(), item


def _fibo(monkeypatch):
    from fizgig.fibo2.model import Fibo2DiT
    dit = Fibo2DiT(dim=64, n_layers=N, n_refiner_layers=1, n_heads=4, cap_feat_dim=24,
                   injection_layer_ids=tuple(range(N)), perceiver_num_layers=1, min_num_gist_tokens=4,
                   max_num_gist_tokens=8, gist_step=2, gist_min_text_len=2, gist_max_text_len=12, axes_dims=(4, 6, 6),
                   axes_lens=(64, 32, 32))

    def item(g):
        n = 6
        return (torch.randn(1, 32, 8, 8, generator=g),
                {"tokens": torch.randn(1, 1, N + 1, n, 24, generator=g), "mask": torch.ones(1, 1, n, dtype=torch.bool)})
    return dit.float(), item


def _flux3(monkeypatch):
    from fizgig.flux3 import model as M
    for k, v in (("HIDDEN", 64), ("HEADS", 4), ("DEPTH", 1), ("DEPTH_SINGLE", N), ("AXES_DIM", (4, 4, 4, 4))):
        monkeypatch.setattr(M, k, v)
    monkeypatch.setitem(M.IN_CHANNELS, "text", 24)
    dit = M.Flux3DiT()

    def item(g):
        return torch.randn(1, 128, 4, 6, generator=g), {"tokens": torch.randn(1, 5, 24, generator=g)}
    return dit.float(), item


def _qwen(monkeypatch):
    from fizgig.qwen_image21.model import QwenImage21DiT
    dit = QwenImage21DiT(num_layers=N, attention_head_dim=16, num_attention_heads=2, context_in_dim=24,
                         axes_dims_rope=(4, 6, 6))

    def item(g):
        return torch.randn(1, 64, 4, 4, generator=g), {"hidden_states": torch.randn(1, 5, 24, generator=g)}
    return dit.to(torch.bfloat16), item                  # the driver feeds it bf16, as the real base is


def _zimage(monkeypatch):
    from fizgig.zimage.model import ZImageTransformer2DModel
    dit = ZImageTransformer2DModel(in_channels=16, dim=64, n_layers=N, n_refiner_layers=1, n_heads=4, n_kv_heads=4,
                                   cap_feat_dim=24, axes_dims=(8, 4, 4), axes_lens=(64, 32, 32))
    for prm in dit.parameters():                         # its pad tokens are torch.empty until a checkpoint loads
        torch.nn.init.normal_(prm, std=0.05)

    def item(g):
        return torch.randn(1, 16, 8, 8, generator=g), {"cap": torch.randn(1, 5, 24, generator=g)}
    return dit.to(torch.bfloat16), item                  # the driver runs it in bf16


def _klein(monkeypatch):
    from fizgig.klein.model import Flux2Params, KleinDiT
    # Klein 9B's 1:3 double / single ratio (8 / 24), so a swap splits as it does there - a swap of 1 is double 0,
    # single 1, the case that once left every double block on the CPU
    dit = KleinDiT(Flux2Params(in_channels=16, context_in_dim=24, hidden_size=64, num_heads=4, depth=4,
                               depth_single_blocks=12, axes_dim=[4, 4, 4, 4]), attn_mode="torch")

    def item(g):
        return torch.randn(1, 16, 4, 4, generator=g), {"ctx_vec": torch.randn(1, 5, 24, generator=g)}
    return dit.float(), item


def _h3(monkeypatch):
    """MiniMax H3's classic swap (parking): each swapped block onto the card just in time for its forward and its
    checkpoint recompute (_run_block), then back - set up by its own model, as its loader does."""
    from fizgig.minimax import driver as HD
    from fizgig.minimax.model import MiniMaxH3Config, MiniMaxH3DiT
    monkeypatch.setattr(HD, "DTYPE", torch.float32)
    dit = MiniMaxH3DiT(MiniMaxH3Config(hidden_size=128, num_layers=N, token_refiner_num_layers=1, num_attention_heads=1,
                                       attention_head_dim=128, ffn_hidden_size=128, text_dim=24,
                                       time_embed_hidden_size=64, time_embed_dim=32))

    def item(g):
        return torch.randn(1, 24, 8, 6, generator=g), {"hidden_states": torch.randn(1, 5, 24, generator=g)}

    def swap(dit, k):
        dit.to("cuda")
        dit.enable_block_swap(k, h2d_only=False)
    return dit.float(), item, swap


def _h3_ring(kind):
    """MiniMax H3 on a quantised base with its H2D ring, built as minimax/loader.py builds it: the block Linears become
    `kind` shells (int8 ConvRot / bnb NF4 / HQQ 4-bit), filled on the CPU here, and a swapped block's packed tensors
    parked on the CPU; enable_block_swap(h2d_only=True) then picks the ring by module type, which streams them host
    -> device into ring_size card slots - the blocks never move, their tensors are re-pointed at ring slots."""
    def build(monkeypatch):
        if kind == "hqq":
            pytest.importorskip("hqq.core.quantize")
        if kind == "nf4":
            pytest.importorskip("bitsandbytes")
        if kind == "int8":
            from fizgig.minimax import convrot      # its rotation matrix is cached per device name: start fresh, as a
            monkeypatch.setattr(convrot, "_hadamard_cache", {})   # process does (a "cuda" one would outlive the run)
        dit, item, _ = _h3(monkeypatch)
        for b in dit.blocks:
            for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
                parent, _, leaf = name.rpartition(".")
                host = b.get_submodule(parent)
                lin = getattr(host, leaf)
                setattr(host, leaf, _quantised(kind, lin))

        def swap(dit, k):
            dit.to("cuda")
            for blk in dit.blocks[len(dit.blocks) - k:]:      # the loader parks a swapped block's packed tensors
                blk.to("cpu")
            assert dit.enable_block_swap(k, h2d_only=True) == k
            assert getattr(dit, "_h2d_offloader", None) is not None, f"the {kind} ring wasn't built (fell back)"
        return dit, item, swap
    return build


def _quantised(kind, lin):
    w = lin.weight.detach().float()
    bias = None if lin.bias is None else lin.bias.detach().clone()
    if kind == "int8":
        from fizgig.minimax.convrot import ConvRotInt8Linear, quantize_int8_convrot
        q = ConvRotInt8Linear(lin.in_features, lin.out_features, bias=bias is not None, rot=64,
                              compute_dtype=torch.float32)
        qd, sc = quantize_int8_convrot(w, rot=64)
        q.qdata, q.wscale = qd, sc.reshape(-1, 1)
    elif kind == "nf4":
        import bitsandbytes as bnb
        q = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=bias is not None, compute_dtype=torch.float32,
                              quant_type="nf4")
        q.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4")
        q.weight = q.weight._quantize("cpu") if hasattr(q.weight, "_quantize") else q.weight
    else:
        from fizgig.minimax.hqq4 import HQQ4bitLinear
        q = HQQ4bitLinear(lin.in_features, lin.out_features, bias=bias is not None, compute_dtype=torch.float32)
        q.load_dense(w, "cpu")
    if bias is not None:
        q.bias = torch.nn.Parameter(bias, requires_grad=False)
    return q


def _krea2(monkeypatch):
    from fizgig.krea2 import driver as KD
    from fizgig.krea2.offloading import ModelOffloader  # noqa: F401 - loaded so the simulated card catches its mover
    from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT
    monkeypatch.setattr(KD, "DTYPE", torch.float32)
    dit = SingleStreamDiT(SingleMMDiTConfig(features=64, tdim=32, txtdim=24, heads=4, multiplier=2, layers=N, patch=2,
                                            channels=16, txtlayers=3, txtheads=4, txtkvheads=4))

    def item(g):
        m = torch.zeros(1, 20, dtype=torch.bool)
        m[0, :6] = True
        m[0, -2:] = True
        return (torch.randn(1, 16, 8, 8, generator=g),
                {"hidden_states": torch.randn(1, 20, 3, 24, generator=g), "attention_mask": m})
    return dit.float(), item


def _krea2_stream(monkeypatch):
    from fizgig.krea2 import driver as KD
    dit, item = _krea2(monkeypatch)

    def swap(dit, k):
        drv = KD.Krea2Driver()
        drv.enable_block_swap(dit, k, "cuda", send_only=True)
    return dit, item, swap


FAMILIES = {"minimax": _h3, "krea2": _krea2, "moss_image": _moss, "fibo2": _fibo, "flux3": _flux3, "qwen_image21": _qwen, "zimage": _zimage,
            "klein": _klein}


def _streamed(block):
    """The tensors a swap streams for a block: HQQ codes (W_q), int8 ConvRot codes (qdata), else Linear weights - never
    a LoRA adapter's, which stay on the card."""
    out = []
    for name, m in block.named_modules():
        if "adapters" in name.split("."):
            continue
        for attr in ("W_q", "qdata"):
            if isinstance(getattr(m, attr, None), torch.Tensor):
                out.append(getattr(m, attr))
                break
        else:
            w = m._parameters.get("weight") if "Linear" in m.__class__.__name__ else None     # Linear4bit too
            if w is not None:
                out.append(w)
    return out


def _train(key, build, monkeypatch, swap, batch, steps=3, precision="bf16"):
    """Losses and LoRA gradients of `steps` training steps, swapping `swap` blocks (0: all on the card). precision
    "int8": the base quantised as quant.load_base does it (on the card, then kept on the CPU when swapped)."""
    drv = get_family(key).load_driver()
    torch.manual_seed(0)
    built = build(monkeypatch)
    dit, item = built[0], built[1]
    dit.requires_grad_(False)                              # frozen, as every driver's loader returns the base
    own_swap = built[2] if len(built) > 2 else None        # a family whose model sets its swap up itself (H3)
    g = torch.Generator().manual_seed(1)
    data = [[item(g) for _ in range(batch)] for _ in range(steps)]
    out = []
    with simulated_gpu():
        if precision != "bf16":
            from fizgig.families import quant
            assert quant.quantize(dit, drv, precision, "cuda", store_device="cpu" if swap else None) > 0
        if swap and own_swap:
            own_swap(dit, swap)
        elif swap:
            drv.enable_block_swap(dit, swap, "cuda")
        else:
            dit.to("cuda")
        drv.enable_gradient_checkpointing(dit, True)
        torch.manual_seed(2)
        net = FamilyLoRA(dit, drv, device="cuda")
        net.add_trainable(4, 4)
        for p in net.parameters():
            p.data = p.data.to("cuda")
            torch.nn.init.normal_(p, std=0.05)          # both poles non-zero, so every gradient path is live
        opt = torch.optim.SGD(net.parameters(), lr=0.05)
        dit.train()
        # residency: at every block's forward (its recompute too), how many of the streamed list's blocks hold their
        # weights on the card - exactly what each mechanism allows, no more: the exchange offloaders (the shared one,
        # Krea 2's) swap blocks in pairs, so the resident count; H3's parking one in flight on top (it once piled
        # parked blocks up); an H2D ring its ring_size slots on top
        blocks = max((m for m in dit.modules() if isinstance(m, torch.nn.ModuleList)), key=len)
        streamed = swap
        if swap and getattr(dit, "offloader_single", None) is not None and blocks is getattr(dit, "single_blocks", None):
            streamed = dit.offloader_single.blocks_to_swap        # Klein splits its swap between two lists
        ring = getattr(dit, "_h2d_offloader", None)
        offloader = getattr(dit, "offloader", None)
        ring_b = ring.ring_size if ring is not None else getattr(offloader, "B", 0)
        allowed = len(blocks) - streamed + (ring_b if ring_b else 1 if own_swap else 0)
        seen = []

        hooks = []
        for b in blocks:
            hooks.append(b.register_forward_pre_hook(
                lambda m, a: seen.append(sum(1 for blk in blocks
                                             if (ts := _streamed(blk)) and all(on_card(t) for t in ts)))))
        for s, items in enumerate(data):
            gen = torch.Generator().manual_seed(100 + s)
            lats = [lat.to("cuda") for lat, _ in items]
            conds = [{k: v.to("cuda") for k, v in c.items()} for _, c in items]
            if batch > 1:
                loss, _ = drv.training_loss_batch(dit, torch.cat(lats), drv.stack_conds(conds), gen)
            else:
                loss, _ = drv.training_loss(dit, lats[0], conds[0], gen)
            loss.backward()
            out.append((float(loss.detach().to("cpu")),
                        [None if p.grad is None else p.grad.detach().to("cpu").clone() for p in net.parameters()]))
            opt.step()
            opt.zero_grad(set_to_none=True)
    for h in hooks:
        h.remove()
    if swap:
        assert max(seen) <= allowed, (key, swap, max(seen), allowed)
    return out


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("swap", [1, 3, 4])
@pytest.mark.parametrize("key", sorted(FAMILIES))
def test_family_block_swap_trains_as_with_no_swap(key, swap, batch, monkeypatch):
    want = _train(key, FAMILIES[key], monkeypatch, 0, batch)
    got = _train(key, FAMILIES[key], monkeypatch, swap, batch)
    for step, ((wl, wg), (gl, gg)) in enumerate(zip(want, got)):
        assert wl == pytest.approx(gl, rel=1e-5, abs=1e-6), (key, step)
        for i, (a, b) in enumerate(zip(wg, gg)):
            assert (a is None) == (b is None), (key, step, i)       # an adapter with no gradient has none either way
            if a is not None:
                assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), (key, step, i, (a - b).abs().max())


def test_klein_swap_of_one_puts_every_double_block_on_the_card(monkeypatch):
    """Found by these checks (20 Nov 2026), confirmed on a 5090: a swap of 1 splits as double 0 / single 1 on Klein 9B,
    and moving the model left both block lists off the card for their offloaders - but an offloader with 0 blocks does
    nothing, so every double block stayed on the CPU and the first step met "mat2 is on cpu"."""
    drv = get_family("klein").load_driver()
    torch.manual_seed(0)
    dit, _ = _klein(monkeypatch)
    with simulated_gpu():
        drv.enable_block_swap(dit, 1, "cuda")
        assert dit.offloader_double.blocks_to_swap == 0 and dit.offloader_single.blocks_to_swap == 1
        assert all(on_card(b.img_attn.qkv.weight) for b in dit.double_blocks)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("swap", [1, 4])
@pytest.mark.parametrize("key", ["moss_image", "qwen_image21", "zimage"])
def test_int8_base_block_swap_trains_as_with_no_swap(key, swap, batch, monkeypatch):
    """The int8 base through the swap: the int8 weights stream, their scales stay with the block."""
    want = _train(key, FAMILIES[key], monkeypatch, 0, batch, precision="int8")
    got = _train(key, FAMILIES[key], monkeypatch, swap, batch, precision="int8")
    for step, ((wl, wg), (gl, gg)) in enumerate(zip(want, got)):
        assert wl == pytest.approx(gl, rel=1e-5, abs=1e-6), (key, step)
        for i, (a, b) in enumerate(zip(wg, gg)):
            assert (a is None) == (b is None), (key, step, i)
            if a is not None:
                assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), (key, step, i, (a - b).abs().max())


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("swap", [1, 3, 4])
@pytest.mark.parametrize("kind", ["int8", "nf4", "hqq"])
def test_h3_ring_trains_as_with_no_swap(kind, swap, batch, monkeypatch):
    """H3's three H2D rings (2 card slots each): the same losses and gradients as the same quantised base fully on the
    card, and never more streamed blocks on the card than the ring has slots."""
    want = _train("minimax", _h3_ring(kind), monkeypatch, 0, batch)
    got = _train("minimax", _h3_ring(kind), monkeypatch, swap, batch)
    for step, ((wl, wg), (gl, gg)) in enumerate(zip(want, got)):
        assert wl == pytest.approx(gl, rel=1e-5, abs=1e-6), step
        for i, (a, b) in enumerate(zip(wg, gg)):
            assert (a is None) == (b is None), (step, i)
            if a is not None:
                assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), (step, i, (a - b).abs().max())


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("swap", [1, 3, 4])
def test_krea2_stream_trains_as_with_no_swap(swap, batch, monkeypatch):
    """Krea 2 fast stream offloader (H2D ring with B slots): the same losses and gradients as no swap, and never more
    streamed blocks on the card than the ring has slots."""
    want = _train("krea2", _krea2, monkeypatch, 0, batch)
    got = _train("krea2", _krea2_stream, monkeypatch, swap, batch)
    for step, ((wl, wg), (gl, gg)) in enumerate(zip(want, got)):
        assert wl == pytest.approx(gl, rel=1e-5, abs=1e-6), step
        for i, (a, b) in enumerate(zip(wg, gg)):
            assert (a is None) == (b is None), (step, i)
            if a is not None:
                assert torch.allclose(a, b, atol=1e-5, rtol=1e-4), (step, i, (a - b).abs().max())


def _card_weight_bytes(dit):
    """Frozen bytes on the card: every unique storage the model reaches (parameters, buffers, packed codes, bnb's
    quantisation state) plus an H2D ring's own slots - what block swap is there to save."""
    found = []
    ring = getattr(dit, "_h2d_offloader", None)
    found += [t for t in (getattr(ring, "gpu_data", None) or []) if torch.is_tensor(t)]
    offloader = getattr(dit, "offloader", None)
    found += [t for t in (getattr(offloader, "ring_flat", None) or []) if torch.is_tensor(t)]
    for m in dit.modules():
        for v in list(m._parameters.values()) + list(m._buffers.values()):
            if v is None:
                continue
            found.append(v)
            qs = getattr(v, "quant_state", None)
            if qs is not None:
                found += [qs.absmax, qs.code]
        found += [v for k, v in m.__dict__.items() if torch.is_tensor(v)]
    st = {}
    for t in found:
        if on_card(t) and not t.requires_grad:
            us = t.untyped_storage()
            st[us.data_ptr()] = us.nbytes()
    return sum(st.values())


def _peak_card_bytes(key, build, monkeypatch, swap):
    """The most frozen bytes on the card at any block's forward (recomputes included) over one training step."""
    drv = get_family(key).load_driver()
    torch.manual_seed(0)
    built = build(monkeypatch)
    dit, item, own_swap = built[0], built[1], (built[2] if len(built) > 2 else None)
    dit.requires_grad_(False)
    blocks = max((m for m in dit.modules() if isinstance(m, torch.nn.ModuleList)), key=len)
    lat, cond = item(torch.Generator().manual_seed(1))
    with simulated_gpu():
        if swap and own_swap:
            own_swap(dit, swap)
        elif swap:
            drv.enable_block_swap(dit, swap, "cuda")
        else:
            dit.to("cuda")
        drv.enable_gradient_checkpointing(dit, True)
        net = FamilyLoRA(dit, drv, device="cuda")
        net.add_trainable(4, 4)
        for p in net.parameters():
            p.data = p.data.to("cuda")
        seen = []
        hooks = [b.register_forward_pre_hook(lambda m, a: seen.append(_card_weight_bytes(dit))) for b in blocks]
        dit.train()
        loss = drv.training_loss(dit, lat.to("cuda"), {k: v.to("cuda") if torch.is_tensor(v) else v
                                                        for k, v in cond.items()}, torch.Generator().manual_seed(2))
        (loss[0] if isinstance(loss, tuple) else loss).backward()
        for h in hooks:
            h.remove()
    return max(seen)


@pytest.mark.parametrize("mech", ["exchange", "krea2", "ring-krea2", "parking", "ring-int8", "ring-nf4", "ring-hqq"])
def test_card_memory_saved_per_swapped_block(mech, monkeypatch):
    """What Auto's planner relies on (peak minus GB per swapped block), at the mechanism level: the exchange offloaders
    (the shared one, Krea 2's) free a whole block per swapped block; H3's parking frees one fewer (a block is always
    in flight); an H2D ring frees nothing for its first ring_size (2) swapped blocks - their card slots cost a block
    each - then the same for each one after. (H3 Image's planner figures are a straight line through 10 / 20 swapped
    blocks, so 1-9 swapped plan on memory a ring doesn't give back: noted for the planner, not changed here.)"""
    key, build = {"exchange": ("moss_image", _moss), "krea2": ("krea2", _krea2),
                  "ring-krea2": ("krea2", _krea2_stream),
                  "parking": ("minimax", _h3),
                  "ring-int8": ("minimax", _h3_ring("int8")), "ring-nf4": ("minimax", _h3_ring("nf4")),
                  "ring-hqq": ("minimax", _h3_ring("hqq"))}[mech]
    base = _peak_card_bytes(key, build, monkeypatch, 0)
    saved = {k: base - _peak_card_bytes(key, build, monkeypatch, k) for k in (1, 2, 3, 4)}
    if mech in ("exchange", "krea2", "ring-krea2", "parking"):
        lost = 1 if mech == "parking" else 0
        unit = saved[4] / (4 - lost)
        assert unit > 0
        for k, v in saved.items():
            assert v == pytest.approx((k - lost) * unit, rel=0.05, abs=unit * 0.05), (mech, saved)
    else:
        unit = saved[4] - saved[3]                         # a slot can cost a little more than its block (padding)
        assert unit > 0
        assert saved[1] <= 0.05 * unit and saved[2] <= 0.05 * unit, (mech, saved)
        assert saved[3] - saved[2] == pytest.approx(unit, rel=0.05), (mech, saved)
