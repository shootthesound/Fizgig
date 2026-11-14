"""The Auto plan counts the trainable adapter beyond the rank-16 LoRA the description's figures were measured with:
Krea 2's LoKR factor 8 on AdEMAMix with EMA (#337, a 16 GB card told nf4 fits, then spilling into system RAM)."""
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fizgig.families import quant  # noqa: E402
from fizgig.families.registry import get  # noqa: E402


def _header_only(path, driver, prefix="model.diffusion_model."):
    """A safetensors header naming every Linear the driver's LoRA wraps (no weights: the plan reads the header)."""
    names = [m for g in driver.block_map(None) for b in g.blocks for m in b.modules]
    hdr = {prefix + n + ".weight": {"dtype": "BF16", "shape": [3072, 3072], "data_offsets": [0, 0]} for n in names}
    hdr[prefix + "blocks.0.attn_norm.weight"] = {"dtype": "BF16", "shape": [3072], "data_offsets": [0, 0]}
    raw = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw)
    return len(names)


def _run(path, **kw):
    return dict(dict(dit_path=str(path), network_type="lora", network_dim=16, lokr_factor=8,
                     optimizer_type="adamw8bit", ema_decay=0.98), **kw)


def test_adapter_extra_counts_lokr_optimizer_and_ema(tmp_path):
    drv = get("krea2").load_driver()
    p = tmp_path / "dit.safetensors"
    assert _header_only(p, drv) > 100
    assert quant.adapter_extra_gb(drv, _run(p)) == 0.0                         # the figures' own adapter
    assert quant.adapter_extra_gb(drv, _run(p, network_dim=8, ema_decay=0)) == 0.0
    big = quant.adapter_extra_gb(drv, _run(p, network_type="lokr", optimizer_type="ademamix8bit"))
    no_ema = quant.adapter_extra_gb(drv, _run(p, network_type="lokr", optimizer_type="ademamix8bit", ema_decay=0))
    fp32 = quant.adapter_extra_gb(drv, _run(p, network_type="lokr", optimizer_type="adamw"))
    assert 0 < no_ema < big < fp32
    assert quant.adapter_extra_gb(drv, _run(tmp_path / "missing.safetensors", network_type="lokr")) == 0.0


def test_lokr_with_ema_is_not_told_nf4_fits_a_16gb_card(tmp_path):
    desc, drv = get("krea2"), get("krea2").load_driver()
    p = tmp_path / "dit.safetensors"
    _header_only(p, drv)
    assert quant.plan(desc, drv, "auto", -1, free_gb=14.7, megapixels=0.5)[:2] == ("nf4", 0)   # a rank-16 LoRA
    p_, swap, why = quant.plan(desc, drv, "auto", -1, free_gb=14.7, megapixels=0.5, adapter_gb=3.0)
    assert (p_, swap) != ("nf4", 0) or "nothing fits" in why
