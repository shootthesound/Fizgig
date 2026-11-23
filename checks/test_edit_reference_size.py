"""An edit model's workbench reference (Studio, Royale) is used at the Reference MP in its own shape, each side a
multiple of the family's grid - every edit model packs a reference as tokens of its own size - not cropped to the
render's width and height. Studio's MP box stays live for edit models."""
import os
from types import SimpleNamespace

from PIL import Image

from fizgig.families.workbench import WorkbenchEngine

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "fizgig", "web", "static",
                   "app.js")


def _eng():
    eng = WorkbenchEngine.__new__(WorkbenchEngine)
    eng.desc = SimpleNamespace(reference_grid=32, reference_strength=False, reference_kind="edit")
    return eng


def test_the_reference_keeps_its_shape_at_the_mp(tmp_path):
    p = tmp_path / "r.png"
    Image.new("RGB", (1200, 800)).save(p)
    eng = _eng()
    w, h = eng._ref_size(str(p), 1.0, 768, 1344)
    assert (w, h) == (1216, 832) and w % 32 == 0 and h % 32 == 0          # 3:2, about 1 MP
    w, h = eng._ref_size(str(p), 0.25, 768, 1344)
    assert (w, h) == (608, 416)
    assert eng._ref_size(str(tmp_path / "gone.png"), 1.0, 768, 1344) == (768, 1344)


def test_the_render_uses_it(tmp_path):
    p = tmp_path / "r.png"
    Image.new("RGB", (800, 1200)).save(p)
    eng = _eng()
    got = []
    eng._reference_latents = lambda path, w, h: got.append((w, h)) or "z"
    state = SimpleNamespace(ref_image_path=str(p), ref_megapixels=0.5)
    assert eng._edit_refs(state, str(p), 1024, 1024) == ["z"]
    assert got == [eng._ref_size(str(p), 0.5, 1024, 1024)] and got[0][1] > got[0][0]


def test_studios_mp_box_is_live_for_edit_models():
    js = open(APP, encoding="utf-8").read()
    assert '$("stRefMp").disabled = L.reference_kind === "edit"' not in js
