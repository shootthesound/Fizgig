"""MiniMax H3's int8 keeps the pruned int8 file's own weights, so a 16-bit DiT (the pruned bf16 files: curve table, no
int8 codes) can't load at int8 (#357: picking int8, or Auto on a big card, stopped with "needs a pre-quantized
checkpoint"). Auto now plans 4-bit for such a file, Start refuses int8 picked by hand with what to pick instead, and the
int8 file plans as before. Fake files: the checks only read the safetensors header."""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fizgig.families import launch, quant  # noqa: E402
from fizgig.families.registry import get  # noqa: E402
from fizgig.minimax import common as T  # noqa: E402


def _file(path, keys):
    head = json.dumps({k: {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]} for k in keys}).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head)
    return str(path)


def _pair(tmp_path):
    base = ["adaln_t_table", "blocks.0.attn.qkv.weight"]
    return (_file(tmp_path / "h3_pruned_int8_convrot.safetensors", base + ["blocks.0.attn.qkv.comfy_quant"]),
            _file(tmp_path / "h3_pruned_bf16.safetensors", base))


def test_header_tells_int8_from_16bit(tmp_path):
    int8, bf16 = _pair(tmp_path)
    assert T.is_pruned_checkpoint(int8) and T.is_pruned_checkpoint(bf16)      # both pruned: the curve table
    assert T.has_int8_codes(int8) and not T.has_int8_codes(bf16)
    assert not T.has_int8_codes(str(tmp_path / "missing.safetensors"))


def test_h3_image_auto_plans_4bit_for_a_16bit_file(tmp_path):
    int8, bf16 = _pair(tmp_path)
    desc = get("minimax_image")
    drv = desc.load_driver()
    assert quant.plan(desc, drv, "auto", -1, free_gb=30.0, megapixels=0.25, dit_path=int8)[:2] == ("int8", 0)
    assert quant.plan(desc, drv, "auto", -1, free_gb=30.0, megapixels=0.25, dit_path=bf16)[:2] == ("nf4", 0)
    assert quant.plan(desc, drv, "auto", -1, free_gb=30.0, megapixels=0.25)[:2] == ("int8", 0)   # no file: as before
    why = drv.precision_problem("int8", bf16)
    assert "has none" in why and "minimax_h3_image_pruned_int8_convrot.safetensors" in why
    assert drv.precision_problem("int8", int8) == "" and drv.precision_problem("nf4", bf16) == ""
    assert drv.precision_problem("hqq", bf16) == ""


def test_h3_video_auto_plans_4bit_for_a_16bit_file(tmp_path):
    int8, bf16 = _pair(tmp_path)
    assert T.plan_base_quant(30.0, True)[0] == "int8"
    mode, n, _c, why = T.plan_base_quant(30.0, True, int8=False)
    assert (mode, n) == ("nf4", 0) and "no int8" in why
    drv = get("minimax").load_driver()
    assert "minimax_h3_fl2va_pruned_int8_convrot.safetensors" in drv.precision_problem("int8", bf16)
    assert drv.precision_problem("int8", int8) == ""


def test_start_refuses_int8_picked_for_a_16bit_file(tmp_path):
    int8, bf16 = _pair(tmp_path)
    for key, pref in (("minimax_image", "minimax_image_dit"), ("minimax", "minimax_dit")):
        desc = get(key)
        labels = {**launch.PRECISION_LABELS, **desc.precision_labels}
        pick = {"FAMILY_PRECISION": labels["int8"], "models": {pref: bf16}}
        out = launch.precision_problems(desc, pick)
        assert len(out) == 1 and "4-bit" in out[0], (key, out)
        assert launch.precision_problems(desc, {**pick, "models": {pref: int8}}) == []
        assert launch.precision_problems(desc, {**pick, "FAMILY_PRECISION": labels["auto"]}) == []
        assert launch.precision_problems(desc, {**pick, "FAMILY_PRECISION": labels["nf4"]}) == []
    klein = get("klein")                                   # a family without the hook: nothing to say
    assert launch.precision_problems(klein, {"FAMILY_PRECISION": "int8", "models": {}}) == []
