"""Extract's identity mode (#301): a model's identity blocks keep 4x the rank, in a fine-tune extraction and in a
LoRA's rank reduction; every other block keeps the rank asked."""
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fizgig.extraction.model_diff import extract_diff_loras, identity_layout
from fizgig.families.extract import IDENTITY_RANK, extract_weight_only
from fizgig.families.lorafile import family_keys
from fizgig.families.registry import get as get_family


def test_checkpoints_are_told_apart_by_their_block_layout():
    moss = {f"layers.{i}.attention.qkv.weight" for i in range(42)}
    qwen = {f"transformer_blocks.{i}.attn.to_q.weight" for i in range(32)}
    assert identity_layout(moss)[1] == set(range(24, 36))
    assert identity_layout(qwen)[1] == set(range(10, 15))
    assert identity_layout({f"layers.{i}.x.weight" for i in range(30)}) is None       # another model's layers
    assert identity_layout({"blocks.0.x.weight"}) is None


def test_finetune_extraction_gives_the_identity_blocks_4x_the_rank(tmp_path):
    g = torch.Generator().manual_seed(0)
    base = {f"layers.{i}.attention.qkv.weight": torch.randn(64, 64, generator=g) for i in range(42)}
    tuned = {k: v + 0.1 * torch.randn(64, 64, generator=g) for k, v in base.items()}
    save_file(base, str(tmp_path / "base.safetensors"))
    save_file(tuned, str(tmp_path / "tuned.safetensors"))
    for identity in (False, True):
        [out] = extract_diff_loras(str(tmp_path / "base.safetensors"), str(tmp_path / "tuned.safetensors"),
                                   str(tmp_path / ("id" if identity else "plain")), [4], identity=identity, device="cpu")
        assert out.endswith("_r4_id.safetensors" if identity else "_r4.safetensors")
        with safe_open(out, "pt") as f:
            ranks = {k: f.get_tensor(k).shape[0] for k in f.keys() if k.endswith("lora_down.weight")}
            meta = f.metadata()
        for k, r in ranks.items():
            i = int(k.split("layers_")[1].split("_")[0])
            assert r == (4 * IDENTITY_RANK if identity and 24 <= i <= 35 else 4), (k, r)
        assert ("fizgig_extraction_identity" in meta) == identity


def test_rank_reduction_keeps_4x_in_the_identity_blocks(tmp_path):
    moss = get_family("moss_image")
    drv = moss.load_driver()
    mods = {b.id: b.modules[0] for g in drv.block_map() for b in g.blocks}
    g = torch.Generator().manual_seed(1)
    sd = {}
    for bid in ("block_0", "block_24", "block_35", "block_36"):
        kd, ku, ka = family_keys(moss, mods[bid])
        sd[kd], sd[ku], sd[ka] = torch.randn(32, 48, generator=g), torch.randn(48, 32, generator=g), torch.tensor(32.0)
    save_file(sd, str(tmp_path / "src.safetensors"))
    extract_weight_only(moss, str(tmp_path / "src.safetensors"), str(tmp_path / "out.safetensors"), 4, identity=True)
    with safe_open(str(tmp_path / "out.safetensors"), "pt") as f:
        for bid, want in (("block_0", 4), ("block_24", 16), ("block_35", 16), ("block_36", 4)):
            assert f.get_tensor(family_keys(moss, mods[bid])[0]).shape[0] == want, bid
