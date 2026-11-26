"""Auto-recaption "revise, don't rewrite" (#244, @bkrew): with the tick on, the first try is handed the picture's caption
(without its trigger word) and corrects it; the second try writes from scratch. Either way the caption you had before
auto-recaption is kept beside the pictures (fizgig_caption_history.json) and offered back by Edit caption while the file
still holds the AI's caption."""
import json
import os
import types

import pytest

from fizgig.krea2.caption_text import drop_trigger, revise_instruction


def _watch(tmp_path, revise, seen, answer="a woman seen from behind"):
    from fizgig.families import loss_watch
    from fizgig.krea2 import embedder, utils
    mp = pytest.MonkeyPatch()
    mp.setattr(utils, "load_krea2_text_encoder", lambda *a, **k: types.SimpleNamespace(to=lambda d: None))

    def fake(enc, img, detailed=False, instruction=None, max_new_tokens=0):
        seen.append(instruction)
        return answer
    mp.setattr(embedder, "generate_caption", fake)
    obj = types.SimpleNamespace(captioner_path="x", instr="Caption it.", instr_detailed="Every detail.",
                                trigger_word="zwx", triggers={}, trigger_position="start", caption_ext=".txt",
                                auto_recaption=True, revise=revise,
                                watch=types.SimpleNamespace(where=lambda k: ("", str(tmp_path), k)))
    return loss_watch, obj, mp


def test_first_try_corrects_your_caption_second_writes_anew(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "a.txt").write_text("zwx, a woman facing the camera, my_term", encoding="utf-8")
    seen = []
    lw, obj, mp = _watch(tmp_path, True, seen)
    try:
        lw.Watch._recaption(obj, [("a", None, 1)], {"a": "item"}, "cpu")
        assert seen[0].startswith("Caption it.") and '"a woman facing the camera, my_term"' in seen[0]
        assert "zwx" not in seen[0]                       # the trigger goes back on after, not into the revision
        assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "zwx, a woman seen from behind"
        lw.Watch._recaption(obj, [("a", None, 2)], {"a": "item"}, "cpu")
        assert seen[1] == "Every detail."                 # the second try starts over
    finally:
        mp.undo()


def test_tick_off_keeps_the_first_try_as_it_was(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "a.txt").write_text("zwx, a woman", encoding="utf-8")
    seen = []
    lw, obj, mp = _watch(tmp_path, False, seen)
    try:
        lw.Watch._recaption(obj, [("a", None, 1)], {"a": "item"}, "cpu")
        assert seen == ["Caption it."]
    finally:
        mp.undo()


def test_your_caption_is_kept_and_offered_back_until_you_change_it(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "a.txt").write_text("zwx, my own caption", encoding="utf-8")
    seen = []
    lw, obj, mp = _watch(tmp_path, False, seen, answer="first AI caption")
    try:
        lw.Watch._recaption(obj, [("a", None, 1)], {"a": "item"}, "cpu")
        now = (tmp_path / "a.txt").read_text(encoding="utf-8")
        assert lw.caption_before(str(tmp_path), "a", now) == "zwx, my own caption"
        mp.setattr(__import__("fizgig.krea2.embedder", fromlist=["x"]), "generate_caption",
                   lambda *a, **k: "second AI caption")
        lw.Watch._recaption(obj, [("a", None, 2)], {"a": "item"}, "cpu")
        now = (tmp_path / "a.txt").read_text(encoding="utf-8")
        assert now == "zwx, second AI caption"
        assert lw.caption_before(str(tmp_path), "a", now) == "zwx, my own caption"     # still yours, not the 1st AI's
        assert lw.caption_before(str(tmp_path), "a", "zwx, edited by hand") is None     # you changed it: not offered
        hist = json.loads((tmp_path / lw.BEFORE_FILE).read_text(encoding="utf-8"))
        assert hist == {"a": {"before": "zwx, my own caption", "after": "zwx, second AI caption"}}
    finally:
        mp.undo()


def test_the_editor_gets_it(tmp_path):
    pytest.importorskip("torch")
    from fizgig.families.loss_watch import keep_before
    from fizgig.web import image_loss
    keep_before(str(tmp_path), "a", "mine", "the AI's")
    run = {"image_folder": str(tmp_path)}
    assert image_loss.caption_before(run, "a", "the AI's") == "mine"
    assert image_loss.caption_before(run, "a", "something else") is None
    assert image_loss.caption_before(run, "b", "the AI's") is None


def test_trigger_dropped_either_end():
    assert drop_trigger("zwx, a woman", "zwx") == "a woman"
    assert drop_trigger("a woman, zwx", "zwx") == "a woman"
    assert drop_trigger("Zwx,a woman", "zwx") == "a woman"
    assert drop_trigger("a woman", "") == "a woman"
    assert drop_trigger("zwxman, a woman", "zwx") == "zwxman, a woman"
    assert "\"a woman\"" in revise_instruction("Do it.", "a woman")


def _cmd(tmp_path, **values):
    import sys
    from fizgig.families import form, launch
    from fizgig.families.registry import get
    desc, models = get("zimage"), {}
    for mf in desc.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    cap = tmp_path / "qwen3vl.safetensors"
    cap.write_bytes(b"0")
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
         "DATASET_CONFIG": str(tmp_path / "d.toml"), "KREA2_AUTO_RECAPTION": True, **values}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                      cache_root=str(tmp_path / "c"), samples_dir=str(tmp_path / "s"), captioner=str(cap))
    return desc, v, str(cap), next(s.cmd for s in launch.plan(desc, inp).stages if s.name == "Training")


def test_the_tick_reaches_the_trainer_and_shows_with_auto_recaption(tmp_path):
    from fizgig.families import form
    from fizgig.families.train import setup_parser
    assert form.DEFAULTS["RECAPTION_REVISE"] is False
    desc, v, cap, cmd = _cmd(tmp_path, RECAPTION_REVISE=True)
    assert "--auto_recaption" in cmd and setup_parser().parse_args(cmd[2:]).recaption_revise is True
    *_, cmd = _cmd(tmp_path)
    assert "--recaption_revise" not in cmd and setup_parser().parse_args(cmd[2:]).recaption_revise is False
    assert "RECAPTION_REVISE" not in form.state(desc, v, captioner=cap)["hidden"]
    assert "RECAPTION_REVISE" in form.state(desc, {**v, "KREA2_AUTO_RECAPTION": False}, captioner=cap)["hidden"]
