"""MiniMax H3's int8 on any pruned file (#357): a pruned int8 file keeps its own int8 ConvRot codes; a pruned bf16 or
fp8_scaled file (Comfy-Org ships both, and fine-tunes or merges come that way) is converted to the same storage at
load, so Auto plans int8 for it and a hand-picked int8 starts. Only the full unpruned checkpoint can't take int8
(its AdaLN projections would stay dense, ~26 GB): Auto plans 4-bit there and Start refuses int8 picked by hand. The
fp8 builds mark their weights with the same `.comfy_quant` as the int8 ones, so the weight's dtype decides. Fake
files: these checks only read the safetensors header."""
import json
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fizgig.families import launch, quant  # noqa: E402
from fizgig.families.registry import get  # noqa: E402
from fizgig.minimax import common as T  # noqa: E402
from fizgig.minimax.convrot import dequantize_comfy, dequantize_int8_convrot, quantize_int8_convrot  # noqa: E402


def _file(path, entries):
    head = json.dumps({k: {"dtype": dt, "shape": [0], "data_offsets": [0, 0]} for k, dt in entries.items()}).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head)
    return str(path)


def _files(tmp_path):
    w = "blocks.0.attn.qkv_proj"
    table = {"adaln_t_table": "F32"}
    return {
        "int8": _file(tmp_path / "pruned_int8_convrot.safetensors",
                      {**table, f"{w}.weight": "I8", f"{w}.weight_scale": "F32", f"{w}.comfy_quant": "U8"}),
        "bf16": _file(tmp_path / "pruned_bf16.safetensors", {**table, f"{w}.weight": "BF16"}),
        "fp8": _file(tmp_path / "pruned_fp8_scaled.safetensors",
                     {**table, f"{w}.weight": "F8_E4M3", f"{w}.weight_scale": "F32", f"{w}.comfy_quant": "U8"}),
        "full": _file(tmp_path / "full_bf16.safetensors", {"time_embedder.linear_1.weight": "BF16",
                                                           f"{w}.weight": "BF16"}),
    }


def test_header_tells_the_files_apart(tmp_path):
    f = _files(tmp_path)
    assert T.has_int8_codes(f["int8"])
    assert not T.has_int8_codes(f["bf16"]) and not T.has_int8_codes(f["fp8"])     # fp8's marker is not int8
    assert all(T.int8_loadable(f[k]) for k in ("int8", "bf16", "fp8"))
    assert not T.int8_loadable(f["full"])
    assert not T.has_int8_codes(str(tmp_path / "missing.safetensors"))


def test_h3_image_plans_int8_for_any_pruned_file(tmp_path):
    f = _files(tmp_path)
    desc = get("minimax_image")
    drv = desc.load_driver()
    for k in ("int8", "bf16", "fp8"):
        assert quant.plan(desc, drv, "auto", -1, free_gb=30.0, megapixels=0.25, dit_path=f[k])[:2] == ("int8", 0), k
        assert drv.precision_problem("int8", f[k]) == "", k
    assert quant.plan(desc, drv, "auto", -1, free_gb=30.0, megapixels=0.25, dit_path=f["full"])[:2] == ("nf4", 0)
    why = drv.precision_problem("int8", f["full"])
    assert "unpruned" in why and "minimax_h3_image_pruned_int8_convrot.safetensors" in why
    assert drv.precision_problem("nf4", f["full"]) == "" and drv.precision_problem("hqq", f["full"]) == ""


def test_h3_video_plans_int8_for_any_pruned_file(tmp_path):
    f = _files(tmp_path)
    assert T.plan_base_quant(30.0, True)[0] == "int8"
    mode, n, _c, why = T.plan_base_quant(30.0, False, int8=False)
    assert (mode, n) == ("nf4", 0) and "unpruned" in why
    drv = get("minimax").load_driver()
    assert "minimax_h3_fl2va_pruned_int8_convrot.safetensors" in drv.precision_problem("int8", f["full"])
    assert all(drv.precision_problem("int8", f[k]) == "" for k in ("int8", "bf16", "fp8"))


def test_start_refuses_int8_only_for_the_unpruned_file(tmp_path):
    f = _files(tmp_path)
    for key, pref in (("minimax_image", "minimax_image_dit"), ("minimax", "minimax_dit")):
        desc = get(key)
        labels = {**launch.PRECISION_LABELS, **desc.precision_labels}
        pick = {"FAMILY_PRECISION": labels["int8"], "models": {pref: f["full"]}}
        out = launch.precision_problems(desc, pick)
        assert len(out) == 1 and "4-bit" in out[0], (key, out)
        for k in ("int8", "bf16", "fp8"):
            assert launch.precision_problems(desc, {**pick, "models": {pref: f[k]}}) == [], (key, k)
        assert launch.precision_problems(desc, {**pick, "FAMILY_PRECISION": labels["auto"]}) == []
    klein = get("klein")                                   # a family without the hook: nothing to say
    assert launch.precision_problems(klein, {"FAMILY_PRECISION": "int8", "models": {}}) == []


def test_conversion_matches_the_int8_storage():
    """What the loader does to a pruned bf16 / fp8 weight: encode it into int8 ConvRot (rotate, per-row absmax,
    nearest). Encoding a weight decoded from int8 codes gives those codes back exactly, and an fp8 weight decodes as
    weight x its one scale."""
    g = torch.Generator().manual_seed(0)
    w = torch.randn(64, 512, generator=g) * 0.02
    conf = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    q, s = quantize_int8_convrot(w)
    dense = dequantize_int8_convrot(q, s, conf, out_dtype=torch.float32)
    q2, s2 = quantize_int8_convrot(dense)
    assert torch.equal(q, q2) and torch.allclose(s, s2)
    assert (dense - w).norm() / w.norm() < 0.02
    f8 = (w / 0.001).to(torch.float8_e4m3fn)
    out = dequantize_comfy(f8, torch.tensor(0.001), {"format": "float8_e4m3fn"}, out_dtype=torch.float32)
    assert torch.allclose(out, f8.to(torch.float32) * 0.001)
    assert torch.equal(dequantize_comfy(q, s, conf, out_dtype=torch.float32), dense)
