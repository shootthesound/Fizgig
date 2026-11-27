"""The loss watch on the Captions tab (Peter, 27 Nov 2026). Each caption card shows its picture's verdict from the run
training on the folder (or the folder's last run with a loss log); a caption saved on the Captions tab while a run
trains on that folder is queued for that run, so the trainer re-encodes it at the next epoch, as an edit from the
run's per-image loss view is; and every run of a model with the loss watch gets the text encoder, so a queued edit is
re-encoded even with no loss-watch tick on. The per-image loss view has Edit on the Captions tab."""
import json
import os
import sys
import time

from fizgig.families import form, launch
from fizgig.families.registry import get
from fizgig.web import image_loss


def _run(tmp_path, result="running", started=None, name="r1"):
    folder = tmp_path / "pics"
    folder.mkdir(exist_ok=True)
    out = tmp_path / f"out_{name}"
    (out / "loss_log").mkdir(parents=True, exist_ok=True)
    return {"id": name, "lora": name, "family": "klein", "result": result, "image_folder": str(folder),
            "output_dir": str(out), "started_at": started if started is not None else time.time() - 100}


def test_a_pictures_key_on_the_captions_tab(tmp_path):
    run = _run(tmp_path)
    folder = run["image_folder"]
    assert image_loss.folder_key(run, folder, "a.png") == "a"
    crops = os.path.join(folder, "fizgig_detail_crops", "1.0")
    assert image_loss.folder_key(run, crops, "a.png") == "fizgig_detail_crops/a"
    assert image_loss.folder_key(run, str(tmp_path / "elsewhere"), "a.png") is None


def test_a_save_during_a_run_is_queued_for_it(tmp_path):
    live, done = _run(tmp_path, name="live"), _run(tmp_path, result="finished", name="done")
    assert image_loss.live_run_on([done, live], live["image_folder"]) is live
    assert image_loss.live_run_on([done], live["image_folder"]) is None
    image_loss.queue_caption(live, "a", "  a woman, from behind  ")
    image_loss.queue_caption(live, "b", "a dog")
    q = json.load(open(os.path.join(live["output_dir"], "loss_log", "caption_updates.json"), encoding="utf-8"))
    assert q == {"a": "a woman, from behind", "b": "a dog"}          # the trainer's queue, as the loss view writes it


def test_the_cards_get_the_runs_verdicts(tmp_path):
    run = _run(tmp_path, result="finished")
    log = os.path.join(run["output_dir"], "loss_log")
    with open(os.path.join(log, "per_image_loss.jsonl"), "w", encoding="utf-8") as f:
        for ep in (1, 2):
            for k, loss in (("a", 0.3), ("b", 0.1), ("s2:c", 0.2), ("fizgig_detail_crops/a", 0.2)):
                f.write(json.dumps({"epoch": ep, "key": k, "t": 0.5, "loss": loss}) + "\n")
    json.dump({"epoch": 2, "images": {"a": {"verdict": "stuck", "multiplier": 0.5}, "b": {"verdict": "easy"}}},
              open(os.path.join(log, "problem_images.json"), "w", encoding="utf-8"))
    image_loss.queue_caption({**run, "output_dir": run["output_dir"]}, "a", "fixed")
    got = image_loss.folder_verdicts([run], run["image_folder"], lambda r: True)
    assert got["run"]["id"] == run["id"] and got["run"]["epoch"] == 2 and not got["run"]["live"]
    assert got["items"]["a"]["verdict"] == "stuck" and got["items"]["a"]["queued"]
    assert got["items"]["b"]["verdict"] == "easy" and "s2:c" not in got["items"]
    assert "fizgig_detail_crops/a" in got["items"]
    assert image_loss.folder_verdicts([run], str(tmp_path / "other"), lambda r: True) is None


def test_every_loss_watch_run_gets_the_text_encoder(tmp_path):
    desc, models = get("klein"), {}
    assert desc.loss_watch
    for mf in desc.model_files:
        f = tmp_path / f"{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
         "DATASET_CONFIG": str(tmp_path / "d.toml"), "SAMPLE_EVERY_N_EPOCHS": "0",
         "KREA2_LOSS_WATCH": False, "KREA2_PER_IMAGE_LR": False, "KREA2_AUTO_RECAPTION": False,
         "KREA2_WARMUP_LOOK": False}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder="",
                      cache_root=str(tmp_path / "c"), samples_dir=str(tmp_path / "s"))
    cmd = next(s.cmd for s in launch.plan(desc, inp).stages if s.name == "Training")
    assert "--text_encoder" in cmd


def test_the_page_shows_verdicts_and_queues_saves():
    js = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js"),
              encoding="utf-8").read()
    assert "/api/captions/loss?folder=" in js and "capLossTag(it), ta," in js
    assert "status.textContent = capSavedSay(r, it);" in js and "imageLossToCaptions(im)" in js
    assert '$("capSearch").value = name.replace' in js and 'showTab("captions")' in js
    # the pill's words show at once on hover (its own box, no help cursor); the Loss watch filter sits between
    # Sort and View, greyed out when no run here has a loss log
    css = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "style.css"),
               encoding="utf-8").read()
    pill = css[css.index(".cap-lw-pill {"):css.index("}", css.index(".cap-lw-pill {"))]
    assert ".cap-lw-pill[data-tip]:hover::after" in css and "cursor: help" not in pill
    html = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "index.html"),
                encoding="utf-8").read()
    assert html.index('id="capSort"') < html.index('id="capLossFilter"') < html.index('id="capView"')
    assert "sel.disabled = !Object.keys(counts).length;" in js
