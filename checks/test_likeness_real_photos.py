"""The Samples gallery's 'your photos' line (web/gallery_likeness.real_photos): the person's own training photos scored
against the same 3 reference photos - the realistic top for a preview. The 3 and photos with no face are left out, a
Look Filter scan with the same 3 and the same model is reused, and a changed photo changes the signature."""
import json
import os
import types

import numpy as np
from PIL import Image

from fizgig.web import gallery_likeness as G, prep_jobs

REFS = ["r1.png", "r2.png", "r3.png"]


def _folder(tmp_path, n=8):
    for name in REFS + [f"p{i}.png" for i in range(n)] + ["noface.png"]:
        Image.new("RGB", (8, 8)).save(tmp_path / name)
    Image.new("L", (8, 8)).save(tmp_path / "p0-masklabel.png")          # a mask beside a photo: never a photo
    return str(tmp_path)


def _emb(name="adaface"):
    return types.SimpleNamespace(cache_id=name, name=name)


def test_scores_the_photos_not_the_references(tmp_path, monkeypatch):
    d = _folder(tmp_path)
    seen = []

    def embed(path, emb=None):
        name = os.path.basename(path)
        seen.append(name)
        if name == "noface.png":
            return None
        i = int(name[1]) if name.startswith("p") else 0
        return np.array([0.5 + i * 0.05, 0.0])                           # p0 0.5 ... p7 0.85 against base [1, 0]
    monkeypatch.setattr(prep_jobs, "_embed", embed)
    r = G.real_photos(d, REFS, [np.array([1.0, 0.0])] * 3, _emb())
    assert not set(seen) & set(REFS) and "p0-masklabel.png" not in seen
    assert r["n"] == 8 and abs(r["median"] - 0.675) < 1e-6 and abs(r["best"] - 0.85) < 1e-6
    assert r["p25"] <= r["median"] <= r["p75"]


def test_reuses_a_look_filter_scan_with_the_same_photos_and_model(tmp_path, monkeypatch):
    d = _folder(tmp_path)
    look = {"baselines": list(reversed(REFS)), "embedder": "adaface",
            "scores": {**{f"p{i}": 0.6 for i in range(8)}, "r1": 0.99, "gone": 0.1, "noface": None}}
    (tmp_path / "fizgig_look_scores.json").write_text(json.dumps(look))
    monkeypatch.setattr(prep_jobs, "_embed", lambda *a, **k: 1 / 0)       # never embeds: the scan answers
    r = G.real_photos(d, REFS, [], _emb())
    assert r["n"] == 8 and r["median"] == 0.6                              # the 3, a gone photo and no-face left out
    monkeypatch.setattr(prep_jobs, "_embed", lambda *a, **k: np.array([1.0, 0.0]))
    assert G.real_photos(d, REFS, [np.array([0.7, 0.0])] * 3, _emb("dinov2@yolo:default"))["median"] == 0.7  # other model


def test_too_few_faces_is_none_and_the_signature_follows_the_photos(tmp_path, monkeypatch):
    d = _folder(tmp_path, n=2)
    monkeypatch.setattr(prep_jobs, "_embed", lambda *a, **k: np.array([1.0, 0.0]))
    assert G.real_photos(d, REFS, [np.array([1.0, 0.0])] * 3, _emb()) is None   # 2 photos (+ no face): too few
    sig = G._dataset_sig(d, REFS)
    assert "r1.png" not in sig and "p1.png" in sig
    Image.new("RGB", (9, 9)).save(tmp_path / "p1.png")
    os.utime(tmp_path / "p1.png", (1, 1))
    assert G._dataset_sig(d, REFS) != sig
