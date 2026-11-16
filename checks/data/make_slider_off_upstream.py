"""Made once: a prompt slider as Fizgig had it before the by-epoch switch - upstream ea4f47d4, 9.7.4 - for
checks/test_slider_by_epoch.py: with the tick off, Fizgig must give the same. From the repo, with that commit's files in
a folder of their own:

    git archive --format=tar -o base.tar ea4f47d4
    tar -x -f base.tar -C <an empty folder>
    python -I checks/data/make_slider_off_upstream.py <that folder>/src checks/data/slider_off_upstream.json

For each model with slider training (every picture model's prompt slider, MiniMax H3 Image's pairs and ACE-Step's
songs) and case, only what proves the slider off is upstream's, so a change elsewhere (a new option, another default,
a hint reworded) leaves it as it is: the Training command's slider part (each --slider_* option with its values, and
--train_blocks when the slider's Ultra adds it; None when Start refuses), Start's slider problems (those that start
with "Slider") and the slider previews' prompt file, the temporary folder as <tmp>, the Python as <python> and paths
with "/"; and the Slider card's rows on show (form() builds them, form.state hides them) - each one's key, kind, label
and values (its default and choices), not its hint or other words on the page, and without the Train by epoch tick
and a row's by_epoch look, which that commit doesn't have.

Left out, as Fizgig differs there on purpose: a box of the picture or an end with several lines (Start refuses them
with the tick off, saying to turn it on) and a push strength by epoch such as 3:10:2 (refused in other words) -
test_slider_by_epoch.py checks those - and a prompt that starts with "-" (fixed apart: the trainer took it for an
option and stopped)."""
import json
import os
import sys
import tempfile

BASE = {"FAMILY_KIND": "slider", "FAMILY_SLIDER_SOURCE": "Prompts", "MAX_TRAIN_EPOCHS": "8", "LORA_NAME": "t",
        "FAMILY_SLIDER_BASE": "a photo of a man,", "FAMILY_SLIDER_POS": "smiling", "FAMILY_SLIDER_NEG": "frowning",
        "FAMILY_SLIDER_GUIDANCE": "3"}
CASES = {
    "one line each": {},
    "anchors": {"FAMILY_SLIDER_ANCHORS": "a photo of a dog\na photo of a cat", "FAMILY_SLIDER_ANCHOR_WEIGHT": "1.5"},
    "push number": {"FAMILY_SLIDER_GUIDANCE": " 2.5 "},
    "push 0": {"FAMILY_SLIDER_GUIDANCE": "0"},
    "push empty": {"FAMILY_SLIDER_GUIDANCE": ""},
    "epochs as words": {"FAMILY_SLIDER_POS": "smiling @5-8", "FAMILY_SLIDER_ANCHORS": "a photo of a man @-4\na dog @2"},
    "no -1 end": {"FAMILY_SLIDER_NEG": ""},
    "ultra": {"FAMILY_SLIDER_ULTRA": True},          # its blocks (--train_blocks), on a model that has them
    "photo pairs": {"FAMILY_SLIDER_SOURCE": "Photo pairs"},
    # with its -1 end folder (an empty one, <tmp>/minus): the pair slider's own command
    "photo pairs, -1 folder": {"FAMILY_SLIDER_SOURCE": "Photo pairs", "FAMILY_SLIDER_DIR": "<tmp>/minus"},
}
SKIP = ("FAMILY_SLIDER_BY_EPOCH",)          # the tick: not the plain slider's
ROW = ("key", "kind", "label", "default", "choices")  # what a row keeps: what it is, its label and values - no hint


def slider_args(argv, ultra):
    """The Training command's slider part, in order: each --slider_* option with its values, and --train_blocks when
    the slider's Ultra adds it (`ultra`) - [[option, value, ...], ...]; None with no command (Start refused)."""
    if argv is None:
        return None
    opts = []
    for x in argv:
        if x.startswith("--"):
            opts.append([x])
        elif opts:                          # its value ("-1" too)
            opts[-1].append(x)
    return [o for o in opts if o[0].startswith("--slider_") or (ultra and o[0] == "--train_blocks")]


def observe(root):
    """{model: {"rows": {key: row}, "cases": {case: {slider_args, problems, preview, shown}}}} from the fizgig imported
    (its files in `root`, a new empty folder)."""
    from fizgig.families import form, launch
    from fizgig.families.registry import get, training_families

    def norm(x):
        if isinstance(x, str):
            return x.replace(sys.executable, "<python>").replace(root, "<tmp>").replace("\\", "/")
        if isinstance(x, (list, tuple)):
            return [norm(y) for y in x]
        return {k: norm(y) for k, y in x.items()} if isinstance(x, dict) else x

    out = {}
    os.makedirs(os.path.join(root, "minus"), exist_ok=True)
    for fam in sorted(d.key for d in training_families() if d.slider_training):      # every model with sliders
        desc, models = get(fam), {}
        for mf in desc.model_files:
            path = os.path.join(root, f"{fam}_{mf.pref_key}.safetensors")
            with open(path, "wb") as f:
                f.write(b"0")
            models[mf.pref_key] = path
        card = next(s for s in form.form(desc)["sections"] if s["id"] == "slider")["rows"]
        rows, cases = {}, {}
        for name, case in CASES.items():
            case = {k: x.replace("<tmp>", root) if isinstance(x, str) else x for k, x in case.items()}
            v = {**form.DEFAULTS, **BASE, "LORA_OUTPUT_DIR": os.path.join(root, "out"),
                 "DATASET_CONFIG": os.path.join(root, "dataset.toml"), **case}
            inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=root, image_folder="",
                              cache_root=os.path.join(root, "cache"), samples_dir=os.path.join(root, "out", "sample"))
            p = launch.plan(desc, inp)
            hidden = set(form.state(desc, v, "")["hidden"])
            shown = [r for r in card if r["key"] not in hidden and r["key"] not in SKIP]
            rows.update({r["key"]: {k: r[k] for k in ROW if k in r} for r in shown})
            ultra = bool(desc.slider_ultra_blocks and v.get("FAMILY_SLIDER_ULTRA"))
            cases[name] = {"slider_args": slider_args(next((s.cmd for s in p.stages if s.name == "Training"), None),
                                                      ultra),
                           "problems": [x for x in p.problems if x.startswith("Slider")],
                           "shown": [r["key"] for r in shown],
                           "preview": next((t for f, t in p.files if f.endswith("_slider_prompt.txt")), None)}
        out[fam] = norm({"rows": rows, "cases": cases})
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(sys.argv[1]))
    import fizgig
    print("fizgig from", os.path.dirname(fizgig.__file__), file=sys.stderr)
    with tempfile.TemporaryDirectory() as tmp:
        got = observe(tmp)
    with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
        json.dump({"how": __doc__, "models": got}, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
