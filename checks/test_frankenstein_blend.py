"""Frankenstein mode's blend: int8 ConvRot layers blended in their stored (rotated) basis equal the blend of the
decoded weights, to the int8 grid's error; the ends are the sources' own codes, byte for byte; DiTBlend writes a block's
blend into the live model and puts the loaded model's own tensors back at 0, under a LoRA wrapper too (Studio's
engine wraps every block Linear: the int8 layers were silently left out), with the other side's LoRA baked in."""
import types

import pytest

torch = pytest.importorskip("torch")
from fizgig.families.frankenstein import DiTBlend, blend_dense, blend_int8_rows  # noqa: E402
from fizgig.minimax.convrot import (ConvRotInt8Linear, dequantize_int8_convrot,  # noqa: E402
                                    quantize_int8_convrot, rotate)

CONF = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}


def _pair():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(64, 512, generator=g)
    b = a + 0.5 * torch.randn(64, 512, generator=g)
    return a, b, quantize_int8_convrot(a, 256), quantize_int8_convrot(b, 256)


def test_blend_in_the_stored_basis_matches_the_decoded_blend():
    a, b, (qa, sa), (qb, sb) = _pair()
    for w in (0.25, 0.5, 0.8):
        q, s = blend_int8_rows(qa, sa, qb, sb, w)
        got = dequantize_int8_convrot(q, s, CONF, torch.float32)
        want = torch.lerp(dequantize_int8_convrot(qa, sa, CONF, torch.float32),
                          dequantize_int8_convrot(qb, sb, CONF, torch.float32), w)
        rel = (got - want).norm() / want.norm()
        assert rel < 0.01, (w, float(rel))                  # one int8 step of error, as the files themselves carry


def test_ends_are_the_sources_own_codes():
    _, _, (qa, sa), (qb, sb) = _pair()
    q0, s0 = blend_int8_rows(qa, sa, qb, sb, 0.0)
    q1, s1 = blend_int8_rows(qa, sa, qb, sb, 1.0)
    assert torch.equal(q0, qa) and torch.equal(s0, sa.reshape(-1, 1))
    assert torch.equal(q1, qb) and torch.equal(s1, sb.reshape(-1, 1))


def test_dense_blend():
    a = torch.randn(8, 8, dtype=torch.bfloat16)
    b = torch.randn(8, 8, dtype=torch.bfloat16)
    assert blend_dense(a, b, 0.0) is a and torch.equal(blend_dense(a, b, 1.0), b)
    mid = blend_dense(a, b, 0.5)
    assert mid.dtype == torch.bfloat16 and torch.allclose(mid.float(), (a.float() + b.float()) / 2, atol=1e-2)


class _Wrap(torch.nn.Module):                       # a LoRA wrapper, as families/lora.py's LoRALinear holds .base
    def __init__(self, base):
        super().__init__()
        self.base = base


def test_dit_blend_live_model(tmp_path):
    from safetensors.torch import save_file
    _, _, (qa, sa), (qb, sb) = _pair()
    lin = ConvRotInt8Linear(512, 64)
    lin.qdata.copy_(qa)
    lin.wscale.copy_(sa)
    blk = torch.nn.Module()
    blk.lin = _Wrap(lin)
    blk.norm = torch.nn.LayerNorm(64)
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([blk])
    other_norm = torch.randn(64)
    other = str(tmp_path / "other.safetensors")
    marker = torch.frombuffer(bytearray(b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'),
                              dtype=torch.uint8)
    save_file({"blocks.0.lin.weight": qb, "blocks.0.lin.weight_scale": sb, "blocks.0.lin.comfy_quant": marker,
               "blocks.0.norm.weight": other_norm, "blocks.0.norm.bias": torch.zeros(64)}, other)
    lora = str(tmp_path / "lora.safetensors")
    A, B = 0.01 * torch.randn(4, 512), 0.01 * torch.randn(64, 4)
    save_file({"blocks.0.lin.lora_A.weight": A, "blocks.0.lin.lora_B.weight": B}, lora)
    groups = [types.SimpleNamespace(blocks=[types.SimpleNamespace(id="b0", modules=["blocks.0.lin"])])]
    bl = DiTBlend(dit, groups, other, other_lora=(lora, 1.0))
    assert not bl.skipped_names and bl.lora_used == 1
    bl.set({"b0": 1.0})
    want = qb.float() * sb.reshape(-1, 1) + rotate(B @ A, 256)          # the other side, LoRA in the stored basis
    got = lin.qdata.float() * lin.wscale
    assert ((got - want).norm() / want.norm()) < 0.01
    assert torch.allclose(blk.norm.weight, other_norm)
    bl.set({"b0": 0.0})
    assert torch.equal(lin.qdata, qa) and torch.equal(lin.wscale, sa.reshape(-1, 1))   # the loaded model's own again
    assert torch.allclose(blk.norm.weight, torch.ones(64))


def _one_block_dit(qa, sa):
    lin = ConvRotInt8Linear(512, 64)
    lin.qdata.copy_(qa)
    lin.wscale.copy_(sa)
    blk = torch.nn.Module()
    blk.lin = lin
    blk.norm = torch.nn.LayerNorm(64)
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([blk])
    groups = [types.SimpleNamespace(blocks=[types.SimpleNamespace(id="b0", modules=["blocks.0.lin"])])]
    return dit, lin, groups


def _conf(d):
    import json
    return torch.frombuffer(bytearray(json.dumps(d).encode()), dtype=torch.uint8)


def test_other_file_in_plain_int8_is_turned_into_the_convrot_basis(tmp_path):
    # an H3 video file quantised per row WITHOUT the rotation (s3_int8-style): its layers were skipped, so a block
    # blended only its norms / AdaLN and Studio rendered noise
    from safetensors.torch import save_file
    _, b, (qa, sa), _ = _pair()
    sp = b.abs().amax(dim=1, keepdim=True) / 127.0
    qp = torch.round(b / sp).clamp(-127, 127).to(torch.int8)
    other = str(tmp_path / "plain.safetensors")
    save_file({"blocks.0.lin.weight": qp, "blocks.0.lin.weight_scale": sp,
               "blocks.0.lin.comfy_quant": _conf({"format": "int8_tensorwise", "per_row": True}),
               "blocks.0.norm.weight": torch.randn(64), "blocks.0.norm.bias": torch.zeros(64)}, other)
    dit, lin, groups = _one_block_dit(qa, sa)
    bl = DiTBlend(dit, groups, other)
    assert not bl.skipped_names
    bl.set({"b0": 1.0})
    got = dequantize_int8_convrot(lin.qdata, lin.wscale, CONF, torch.float32)
    # the other model's own weights, in this model's basis: two int8 roundings (its file's, then the rotated grid's)
    # ~1%; read in the wrong basis it would be ~100% off
    assert ((got - b).norm() / b.norm()) < 0.02


def test_a_layer_it_cant_read_refuses_instead_of_half_blending(tmp_path):
    from safetensors.torch import save_file
    _, _, (qa, sa), (qb, sb) = _pair()
    other = str(tmp_path / "rot128.safetensors")
    save_file({"blocks.0.lin.weight": qb, "blocks.0.lin.weight_scale": sb,
               "blocks.0.lin.comfy_quant": _conf({"format": "int8_tensorwise", "convrot": True,
                                                   "convrot_groupsize": 128}),
               "blocks.0.norm.weight": torch.randn(64), "blocks.0.norm.bias": torch.zeros(64)}, other)
    dit, _, groups = _one_block_dit(qa, sa)
    with pytest.raises(ValueError, match="can't be blended"):
        DiTBlend(dit, groups, other)


def test_other_files_int8_adaln_is_read_with_its_scale(tmp_path):
    # an H3 file that also stores its pruned AdaLN projections as int8 + weight_scale: the raw codes were read as the
    # weights (~135x off), so every blended block's modulation was garbage and Studio rendered noise
    from safetensors.torch import save_file
    from fizgig.minimax.model import AdalnProj
    _, _, (qa, sa), (qb, sb) = _pair()
    dit, _, groups = _one_block_dit(qa, sa)
    dit.blocks[0].adaln_proj = AdalnProj(8, 16, 2, 1, apply_silu=False)
    groups[0].blocks[0].modules.append("blocks.0.adaln_proj.linear")
    dit.pruned_adaln = True
    dit.register_buffer("adaln_t_table", torch.randn(33, 8))
    dit._time_embedding = lambda t: t
    W = 0.05 * torch.randn(32, 8)
    s = W.abs().amax(dim=1, keepdim=True) / 127.0
    other = str(tmp_path / "int8_adaln.safetensors")
    save_file({"blocks.0.lin.weight": qb, "blocks.0.lin.weight_scale": sb, "blocks.0.lin.comfy_quant": _conf(CONF),
               "blocks.0.norm.weight": torch.randn(64), "blocks.0.norm.bias": torch.zeros(64),
               "adaln_t_table": torch.randn(33, 8),
               "blocks.0.adaln_proj.linear.weight": torch.round(W / s).to(torch.int8),
               "blocks.0.adaln_proj.linear.weight_scale": s,
               "blocks.0.adaln_proj.linear.comfy_quant": _conf({"format": "int8_tensorwise"}),
               "blocks.0.adaln_proj.linear.bias": torch.zeros(32)}, other)
    bl = DiTBlend(dit, groups, other)
    Wv = next(it[2] for it in bl.items["b0"] if it[0] == "a")
    assert ((Wv - W).norm() / W.norm()) < 0.01                # the weight itself, to its int8 grid


def test_export_writes_the_blend_and_copies_the_rest(tmp_path):
    """Export: the loaded model's file with the blended blocks' current values in place, everything else byte for
    byte, the mix in its metadata."""
    import json

    from safetensors import safe_open
    from safetensors.torch import save_file
    _, _, (qa, sa), (qb, sb) = _pair()
    lin = ConvRotInt8Linear(512, 64)
    lin.qdata.copy_(qa)
    lin.wscale.copy_(sa)
    blk = torch.nn.Module()
    blk.lin = lin
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([blk])
    marker = torch.frombuffer(bytearray(b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'),
                              dtype=torch.uint8)
    own = str(tmp_path / "own.safetensors")
    save_file({"blocks.0.lin.weight": qa, "blocks.0.lin.weight_scale": sa, "blocks.0.lin.comfy_quant": marker,
               "head.weight": torch.arange(6.0)}, own, metadata={"source": "own"})
    other = str(tmp_path / "other.safetensors")
    save_file({"blocks.0.lin.weight": qb, "blocks.0.lin.weight_scale": sb, "blocks.0.lin.comfy_quant": marker}, other)
    groups = [types.SimpleNamespace(blocks=[types.SimpleNamespace(id="b0", modules=["blocks.0.lin"])])]
    bl = DiTBlend(dit, groups, other)
    bl.set({"b0": 0.5})
    out = bl.export(own, str(tmp_path / "mix.safetensors"))
    with safe_open(out, "pt") as f:
        assert torch.equal(f.get_tensor("blocks.0.lin.weight"), lin.qdata)
        assert torch.equal(f.get_tensor("blocks.0.lin.weight_scale").reshape(-1, 1), lin.wscale)
        assert torch.equal(f.get_tensor("head.weight"), torch.arange(6.0))         # outside the blocks: as it was
        assert torch.equal(f.get_tensor("blocks.0.lin.comfy_quant"), marker)
        meta = f.metadata()
    assert meta["source"] == "own" and json.loads(meta["fizgig_frankenstein"])["weights"] == {"b0": 0.5}


def test_reads_codes_from_the_files_instead_of_holding_them(tmp_path):
    """With the loaded model's own file given, nothing of the int8 layers is held in RAM: the loaded codes come back
    from that file at 0, the other file's are read when the block changes."""
    from safetensors.torch import save_file
    _, _, (qa, sa), (qb, sb) = _pair()
    lin = ConvRotInt8Linear(512, 64)
    lin.qdata.copy_(qa)
    lin.wscale.copy_(sa)
    blk = torch.nn.Module()
    blk.lin = lin
    dit = torch.nn.Module()
    dit.blocks = torch.nn.ModuleList([blk])
    marker = torch.frombuffer(bytearray(b'{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}'),
                              dtype=torch.uint8)
    own, other = str(tmp_path / "own.safetensors"), str(tmp_path / "other.safetensors")
    save_file({"blocks.0.lin.weight": qa, "blocks.0.lin.weight_scale": sa, "blocks.0.lin.comfy_quant": marker}, own)
    save_file({"blocks.0.lin.weight": qb, "blocks.0.lin.weight_scale": sb, "blocks.0.lin.comfy_quant": marker}, other)
    groups = [types.SimpleNamespace(blocks=[types.SimpleNamespace(id="b0", modules=["blocks.0.lin"])])]
    bl = DiTBlend(dit, groups, other, own_path=own)
    assert bl.own_f is not None and bl.items["b0"][0][2] is None          # no copy of the codes kept
    bl.set({"b0": 1.0})
    assert torch.equal(lin.qdata, qb)
    bl.set({"b0": 0.0})
    assert torch.equal(lin.qdata, qa) and torch.equal(lin.wscale, sa.reshape(-1, 1))
