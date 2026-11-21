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
