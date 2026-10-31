"""Krea 2 DiT files from ComfyUI builds (#273): int8 with or without ConvRot, fp8 in the older `.scale_weight`
layout, a "model.diffusion_model." prefix and tensors Krea 2 doesn't have all read as the model's own weights, and
fp8 in the current layout stays on the fp8 path."""
import json

import torch
from safetensors.torch import save_file

from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks
from fizgig.krea2.utils import ComfyDecodeReader, _make_krea2_comfy_fp8_split_hook, dit_layout
from fizgig.minimax.convrot import quantize_int8_convrot

P = "model.diffusion_model."


def _marker(conf):
    return torch.frombuffer(bytearray(json.dumps(conf).encode()), dtype=torch.uint8)


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def _weights():
    g = torch.Generator().manual_seed(0)
    return {n: torch.randn(256, 512, generator=g) * 0.02 for n in "abcd"}


def test_int8_convrot_plain_int8_and_fp8_decode_to_the_weights(tmp_path):
    w = _weights()
    sd = {}
    q, s = quantize_int8_convrot(w["a"], 256)
    sd |= {P + "a.weight": q, P + "a.weight_scale": s,
           P + "a.comfy_quant": _marker({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256})}
    q, s = quantize_int8_convrot(w["b"], 1)                       # int8 with no rotation and no marker
    sd |= {P + "b.weight": q, P + "b.weight_scale": s}
    sc = w["c"].abs().max() / 448
    sd |= {P + "c.weight": (w["c"] / sc).to(torch.float8_e4m3fn), P + "c.weight_scale": sc.reshape(())}
    sd |= {P + "d.weight": w["d"].to(torch.bfloat16), P + "last.up.weight": torch.zeros(4), P + "norm.scale": torch.ones(8)}
    path = str(tmp_path / "mixed.safetensors")
    save_file(sd, path)
    assert dit_layout(path) == "decode"
    want = ["a.weight", "b.weight", "c.weight", "d.weight", "norm.scale"]
    with ComfyDecodeReader(path, want, calc_device="cpu") as f:
        assert sorted(f.keys()) == want                           # prefix gone, last.up skipped
        for n in "abcd":
            assert _rel(f.get_tensor(n + ".weight", dtype=torch.float32), w[n]) < 0.04
        assert f.get_tensor("a.weight").dtype == torch.bfloat16


def test_older_scaled_fp8_layout_decodes(tmp_path):
    w = _weights()["c"]
    sc = w.abs().max() / 448
    path = str(tmp_path / "old.safetensors")
    save_file({"c.weight": (w / sc).to(torch.float8_e4m3fn), "c.scale_weight": sc.reshape(1),
               "scaled_fp8": torch.zeros(0, dtype=torch.float8_e4m3fn)}, path)
    assert dit_layout(path) == "decode"
    with ComfyDecodeReader(path, ["c.weight"], calc_device="cpu") as f:
        assert f.keys() == ["c.weight"]
        assert _rel(f.get_tensor("c.weight", dtype=torch.float32), w) < 0.04


def test_current_fp8_layout_stays_fp8_and_drops_what_the_model_lacks(tmp_path):
    w = _weights()["c"]
    sc = w.abs().max() / 448
    path = str(tmp_path / "fp8.safetensors")
    save_file({"diffusion_model.c.weight": (w / sc).to(torch.float8_e4m3fn), "diffusion_model.c.weight_scale": sc.reshape(()),
               "diffusion_model.c.comfy_quant": _marker({"format": "float8_e4m3fn"}),
               "diffusion_model.last.up.weight": torch.zeros(3), "diffusion_model.norm.scale": torch.ones(2)}, path)
    assert dit_layout(path) == "fp8"
    hook = _make_krea2_comfy_fp8_split_hook(torch.bfloat16, expected={"c.weight", "norm.scale"})
    with MemoryEfficientSafeOpen(path) as f:
        a = TensorWeightAdapter(WeightTransformHooks(split_hook=hook), f)
        assert sorted(a.keys()) == ["c.scale_weight", "c.weight", "norm.scale"]
        assert a.get_tensor("c.weight").dtype == torch.float8_e4m3fn
        assert a.get_tensor("c.scale_weight").shape == (1,)
