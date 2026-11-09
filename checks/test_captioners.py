"""Captioners speak Fizgig's language, and never load in a training run.

Every registered captioner declares what Fizgig needs to run it safely (kind, files pinned, the memory it needs), its
class is importable and has the three methods, and a stub captioner goes through the worker's own path to a written
caption file. And the memory rule (fizgig/captioners/__init__.py): no training code imports the captioner package, so
no captioner can enter a run's memory plan."""
import ast
import os
import re
from pathlib import Path

import pytest

from fizgig import captioners
from fizgig.captioners import (KINDS, Captioner, CaptionerDescription, CaptionRequest, CaptionResult, HFRepo,
                               PrefFile, finish, with_trigger)

SRC = Path(__file__).resolve().parents[1] / "src" / "fizgig"


@pytest.mark.parametrize("d", captioners.CAPTIONERS, ids=lambda d: d.key)
def test_each_captioner_declares_what_fizgig_needs(d):
    assert re.fullmatch(r"[a-z0-9_]+", d.key)
    assert d.label and d.kind in KINDS and d.licence and d.vram_gb > 0
    assert re.fullmatch(r"fizgig\.captioners\.[a-z0-9_]+:[A-Za-z0-9_]+", d.module)
    if isinstance(d.files, HFRepo):                      # pinned: a change upstream can't swap the files
        assert d.files.repo and re.fullmatch(r"[0-9a-f]{40}", d.files.revision)
        assert not d.files.code_revision or re.fullmatch(r"[0-9a-f]{40}", d.files.code_revision)
    else:
        assert isinstance(d.files, PrefFile) and d.files.pref
    assert not d.tasks or d.default_task in d.tasks
    assert not (d.kind == "tags" and d.instruction), "a tagger doesn't follow instructions"


def test_keys_and_labels_are_unique():
    keys = [d.key for d in captioners.CAPTIONERS]
    labels = [d.label for d in captioners.CAPTIONERS]
    assert len(set(keys)) == len(keys) and len(set(labels)) == len(labels)


@pytest.mark.parametrize("d", captioners.CAPTIONERS, ids=lambda d: d.key)
def test_each_class_has_the_three_jobs(d):
    mod, _, cls = d.module.partition(":")
    path = SRC.parent / Path(*mod.split(".")).with_suffix(".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))          # read, not imported: no torch / model code here
    node = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls), None)
    assert node is not None, f"{cls} not in {path.name}"
    methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
    assert {"load", "caption"} <= methods
    assert any(isinstance(b, ast.Name) and b.id == "Captioner" for b in node.bases)


def test_training_code_never_imports_captioners():
    """The memory rule: a captioner loads only in the Captions tab's worker. Training (families/, training/, every
    model package's trainer path) must not import fizgig.captioners - auto-recaption calls Qwen3-VL directly."""
    bad = []
    for sub in ("families", "training", "networks", "modules", "dataset"):
        for f in (SRC / sub).rglob("*.py"):
            text = f.read_text(encoding="utf-8", errors="replace")
            if re.search(r"\bfizgig\.captioners\b|from fizgig import captioners", text):
                bad.append(str(f.relative_to(SRC)))
    assert not bad, f"training code imports the captioner registry: {bad}"


def test_finish_tags_and_sentences():
    tags = CaptionResult(tags=[("smile", 0.9), ("long_hair", 0.8), ("watermark", 0.95), ("hat", 0.2)])
    assert finish(tags, threshold=0.35, leave_out=["watermark"]) == "smile, long hair"
    assert finish(tags, threshold=0.35, underscores=True) == "watermark, smile, long_hair"
    assert finish(CaptionResult(text="  a  woman\non a beach ")) == "a woman on a beach"
    assert with_trigger("a woman", "zwxem") == "zwxem, a woman"
    assert with_trigger("a woman", "") == "a woman" and with_trigger("", "zwxem") == ""


class _Stub(Captioner):
    def load(self):
        self.loaded = True

    def caption(self, image, request):
        assert image.mode == "RGB"                         # Fizgig hands an upright RGB picture, see-through on white
        assert image.getpixel((0, 0)) == (255, 255, 255)
        return CaptionResult(tags=[("1girl", 0.9), ("solo", 0.6), ("blurry", 0.1)])


def test_a_stub_captioner_goes_through_the_worker(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from PIL import Image

    from fizgig.scripts import batch_caption as bc
    desc = CaptionerDescription(key="stub", label="Stub tagger", kind="tags", module="x:_Stub",
                                files=HFRepo("a/b", "0" * 40), licence="MIT", vram_gb=1.0)
    monkeypatch.setattr(captioners, "get", lambda k: desc if k == "stub" else None)
    monkeypatch.setattr(captioners, "make", lambda d, dev, files: _Stub(d, dev, files))
    img = tmp_path / "a.png"
    Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(img)      # see-through: must reach the captioner as white
    (tmp_path / "list.txt").write_text(str(img) + "\n", encoding="utf-8")
    state = bc._load_state_from_config({"captioner": "stub"}, "cpu")
    assert state.captioner.files["revision"] == "0" * 40 and state.captioner.loaded
    bc._run_job(state, {"list_file": str(tmp_path / "list.txt"), "trigger": "zwxem", "caption_ext": ".txt"})
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "zwxem, 1girl, solo"
    bc.free(state)
    assert state.captioner is None


def test_captions_tab_offers_the_registry():
    from fizgig.web import captions
    with_file = captions.options({}, "qwen.safetensors")
    without = captions.options({}, "")
    assert with_file["models"] == [d.label for d in captioners.CAPTIONERS]
    assert with_file["default_model"] == captioners.get("qwen3vl").label
    assert captioners.get("qwen3vl").label not in without["models"]       # its file isn't set
    assert captions.need_gb(captioners.get("florence2_base").label) == 2.5
