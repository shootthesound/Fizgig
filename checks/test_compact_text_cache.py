"""Compact Krea 2 text caches (dataset/compact_text.py): only the real tokens on disk, rebuilt at load. Lossless where
it matters - the same mask, the same real tokens bit for bit, and the same forward and loss on a tiny Krea 2 as the
full cache - while an old full cache loads exactly as before."""
import os
import types

import torch
from safetensors.torch import save_file


def _cond(n, seq=64, seed=1):
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(seq, dtype=torch.bool)
    m[:n] = True
    m[-2:] = True                                     # the encoder's layout: prompt, padding, template suffix
    h = torch.randn(seq, 3, 24, generator=g)          # padding rows are NOT zero in a real encoder's output
    return {"hidden_states": h, "attention_mask": m}


def _item(tmp_path, name, cap):
    return types.SimpleNamespace(text_encoder_output_cache_path=str(tmp_path / f"{name}_krea2_te.safetensors"),
                                 caption=cap, item_key=name)


def test_compact_cache_is_lossless_for_a_tiny_krea2(tmp_path, monkeypatch):
    from fizgig.dataset import compact_text
    from fizgig.families import cache
    from fizgig.families.registry import get as get_family
    from fizgig.krea2 import driver as KD
    from fizgig.krea2.model import SingleMMDiTConfig, SingleStreamDiT
    from fizgig.krea2.sampling import gather_valid_text
    monkeypatch.setattr(KD, "DTYPE", torch.float32)
    desc = get_family("krea2")
    assert desc.text_cache_trim
    conds = [_cond(n, seed=i) for i, n in enumerate((5, 11, 3))]
    full, comp = [], []
    for i, c in enumerate(conds):
        old = _item(tmp_path, f"old{i}", "a, b")
        save_file({f"cond__{k}": v for k, v in c.items()}, old.text_encoder_output_cache_path,
                  metadata={"caption1": "a, b"})                       # a cache written before: full length
        new = _item(tmp_path, f"new{i}", "a, b")
        cache.save_cond(desc, new, c)
        assert compact_text.is_compact(new.text_encoder_output_cache_path)
        assert os.path.getsize(new.text_encoder_output_cache_path) < 0.6 * os.path.getsize(
            old.text_encoder_output_cache_path)
        f = compact_text.load(old.text_encoder_output_cache_path)
        assert torch.equal(f["cond__hidden_states"], c["hidden_states"])       # an old cache: exactly as written
        full.append(f)
        comp.append(compact_text.load(new.text_encoder_output_cache_path))
    for f, c in zip(full, comp):
        assert f.keys() == c.keys()
        assert f["cond__hidden_states"].shape == c["cond__hidden_states"].shape
        m = f["cond__attention_mask"]
        assert torch.equal(m, c["cond__attention_mask"])
        assert torch.equal(f["cond__hidden_states"][m], c["cond__hidden_states"][m])
        assert not c["cond__hidden_states"][~m].any()                      # the padding comes back as zeros
        a, am = gather_valid_text(f["cond__hidden_states"][None], m[None])
        b, bm = gather_valid_text(c["cond__hidden_states"][None], m[None])
        assert torch.equal(a, b) and torch.equal(am, bm)                    # what the DiT is given: identical
    torch.manual_seed(0)
    dit = SingleStreamDiT(SingleMMDiTConfig(features=64, tdim=32, txtdim=24, heads=4, multiplier=2, layers=2, patch=2,
                                            channels=16, txtlayers=3, txtheads=4, txtkvheads=4)).float().eval()
    drv = desc.load_driver()
    lat = torch.randn(3, 16, 8, 8, generator=torch.Generator().manual_seed(3))

    def run(cs):
        cs = [{k[len("cond__"):]: v[None] for k, v in c.items()} for c in cs]
        with torch.no_grad():
            out = [drv._forward(dit, lat[i:i + 1], torch.tensor([0.4]), cs[i]) for i in range(3)]
            loss = drv.training_loss_batch(dit, lat, drv.stack_conds(cs), torch.Generator().manual_seed(7))[0]
        return out, loss
    fo, fl = run(full)
    co, cl = run(comp)
    assert all(torch.equal(a, b) for a, b in zip(fo, co)) and torch.equal(fl, cl)


def test_dop_files_and_the_share_read_compact_files(tmp_path, monkeypatch):
    from fizgig.dataset import compact_text
    from fizgig.families import cache, train
    from fizgig.families.registry import get as get_family
    from fizgig.web import helpers
    desc = get_family("krea2")
    it = _item(tmp_path, "img", "girl, red hat, smile, outdoors")
    cache.save_cond(desc, it, _cond(6))
    assert compact_text.is_compact(it.text_encoder_output_cache_path)
    # a DOP / uncond file in the same format
    up = str(tmp_path / "uncond_krea2_te.safetensors")
    cache.save_cond_file(desc, up, "", _cond(2))
    assert train._load_dop_cond(up)["hidden_states"].shape == (1, 64, 3, 24)
    # a training share: a compact file goes as it is and comes back as it is
    dst = str(tmp_path / "share" / "img_krea2_te.safetensors")
    assert helpers._trim(it.text_encoder_output_cache_path, dst)
    back = str(tmp_path / "back.safetensors")
    helpers._untrim(dst, back)
    a, b = compact_text.load(back), compact_text.load(it.text_encoder_output_cache_path)
    assert all(torch.equal(a[k], b[k]) for k in b)
    # off: full files again; a family whose model reads its padding is never compacted
    monkeypatch.setenv("FIZGIG_COMPACT_TEXT", "0")
    cache.save_cond(desc, _item(tmp_path, "off", "x"), _cond(3))
    assert not compact_text.is_compact(str(tmp_path / "off_krea2_te.safetensors"))
    monkeypatch.delenv("FIZGIG_COMPACT_TEXT")
    other = types.SimpleNamespace(arch_id="qwen_image21", text_cache_trim=False)
    cache.save_cond(other, _item(tmp_path, "q", "x"), _cond(3))
    assert not compact_text.is_compact(str(tmp_path / "q_krea2_te.safetensors"))
