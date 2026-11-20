"""Vimoo's Glyph-ByT5 comes from Glyph-SDXL-v2 since Vimoo stopped shipping it (12 Nov 2026): the download is
converted to the file Vimoo used to ship, and the tokenizer is built from Glyph-SDXL-v2's colour and font lists."""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")


def test_glyph_downloads_from_its_own_home():
    from fizgig.families.registry import get
    f = next(m for m in get("vimoo").model_files if m.pref_key == "vimoo_glyph")
    assert (f.repo, f.path) == ("nlpcvcode/Glyph-SDXL-v2", "checkpoints/byt5_model.pt")
    assert f.local_name == "vimoo_v1_glyph_byt5_fp32.safetensors" and f.convert == "fizgig.vimoo.text:convert_glyph"
    assert len(f.revision) == 40


def test_added_tokens_match_vimoos_list():
    from fizgig.vimoo.text import glyph_added_tokens
    colors = {"white": 0, "black": 1}
    fonts = {"en-Montserrat-Regular": 0, "en-Poppins-Italic": 1, "kr-Nanum": 0}
    got = glyph_added_tokens(colors, fonts)
    assert got["<extra_id_0>"] == 259 and got["<extra_id_124>"] == 383
    assert got["<color-0>"] == 384 and got["<color-1>"] == 385
    assert [got["<en-font-0>"], got["<en-font-1>"], got["<kr-font-0>"]] == [386, 387, 388]


def test_downloader_converts(tmp_path, monkeypatch):
    """fetch_weight downloads the .pt, converts it into the models folder under the local name and cleans up."""
    from safetensors.torch import load_file

    import fizgig.scripts.fetch_models as fm
    from fizgig.utils import hf_public
    from fizgig.vimoo import text
    w = next(w for w in fm.FAMILIES["vimoo"] if w.pref_key == "vimoo_glyph")
    emb = torch.randn(1510, 1472)
    sd = {"embed_tokens.weight": emb, "block.0.layer.0.SelfAttention.q.weight": torch.randn(4, 4)}
    (tmp_path / "checkpoints").mkdir()
    torch.save(sd, tmp_path / "checkpoints" / "byt5_model.pt")
    seen = {}

    def fake_download(**kw):
        seen.update(kw)
        return str(tmp_path / kw["filename"])
    monkeypatch.setattr(hf_public, "download", fake_download)
    monkeypatch.setattr(fm, "_valid_safetensors", lambda p, n: os.path.isfile(p))
    prefs = {}
    assert fm.fetch_weight(w, str(tmp_path), prefs, log=lambda *a: None)
    assert seen["revision"] == w.revision and seen["repo_id"] == "nlpcvcode/Glyph-SDXL-v2"
    out = load_file(prefs["vimoo_glyph"])
    assert os.path.basename(prefs["vimoo_glyph"]) == "vimoo_v1_glyph_byt5_fp32.safetensors"
    assert sorted(os.listdir(tmp_path)) == ["vimoo_v1_glyph_byt5_fp32.safetensors"]     # the .pt and its folder gone
    assert torch.equal(out["shared.weight"], emb) and torch.equal(out["encoder.embed_tokens.weight"], emb)
    assert set(out) == {"shared.weight", "encoder.embed_tokens.weight", "encoder.block.0.layer.0.SelfAttention.q.weight"}
    torch.save(sd, tmp_path / "x.pt")                                                 # a hand-downloaded .pt loads
    pt = text._load_file(str(tmp_path / "x.pt"))
    assert set(pt) == set(out) and torch.equal(pt["shared.weight"], emb)
