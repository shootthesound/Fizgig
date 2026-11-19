"""#376: the Samples tab's First / Last frame - MiniMax H3's clip previews start / end on them. The rows show only
for a family that declares preview_keyframes; the launch sends them for clip previews on the model's own base (not
the Reference base, a slider or an edit); a set path that isn't there stops Start; the driver encodes each picture
at the preview's size and anchors it at frame 0 / the last frame, a still uses neither."""
import sys

import torch
from PIL import Image

from fizgig.families import form, launch
from fizgig.families.registry import get as get_family
from fizgig.families.train import setup_parser


def _plan(tmp_path, fam, **values):
    desc, models = get_family(fam), {}
    for mf in desc.model_files:
        f = tmp_path / f"{fam}_{mf.pref_key}.safetensors"
        f.write_bytes(b"0")
        models[mf.pref_key] = str(f)
    pics = tmp_path / "pics"
    pics.mkdir(exist_ok=True)
    Image.new("RGB", (64, 64), "red").save(pics / "a.jpg")
    (pics / "a.txt").write_text("a person", encoding="utf-8")
    v = {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"), "MAX_TRAIN_EPOCHS": "4",
         "DATASET_CONFIG": str(tmp_path / "dataset.toml"), **values}
    inp = form.inputs(desc, v, models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(pics),
                      cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    return desc, inp


def _train_cmd(tmp_path, fam, **values):
    desc, inp = _plan(tmp_path, fam, **values)
    return next(s.cmd for s in launch.plan(desc, inp).stages if s.name == "Training")


def _frame(tmp_path, name="first.png", size=(300, 200)):
    p = tmp_path / name
    Image.new("RGB", size, "blue").save(p)
    return str(p)


def test_rows_only_where_declared():
    keys = lambda fam: {r["key"] for s in form.form(get_family(fam))["sections"] for r in s["rows"] if r}
    assert {"SAMPLE_FIRST_FRAME", "SAMPLE_LAST_FRAME"} <= keys("minimax")
    assert not {"SAMPLE_FIRST_FRAME", "SAMPLE_LAST_FRAME"} & keys("klein")


def test_launch_sends_the_frames(tmp_path):
    first, last = _frame(tmp_path), _frame(tmp_path, "last.png")
    cmd = _train_cmd(tmp_path, "minimax", SAMPLE_FIRST_FRAME=first, SAMPLE_LAST_FRAME=last)
    assert cmd[cmd.index("--sample_first_frame") + 1] == first
    assert cmd[cmd.index("--sample_last_frame") + 1] == last
    a = setup_parser().parse_args(["--family", "minimax", "--dit", "d", "--output_dir", "o", "--output_name", "n",
                                   "--sample_first_frame", first, "--sample_last_frame", last])
    assert (a.sample_first_frame, a.sample_last_frame) == (first, last)


def test_no_frames_no_flags(tmp_path):
    cmd = _train_cmd(tmp_path, "minimax")
    assert "--sample_first_frame" not in cmd and "--sample_last_frame" not in cmd


def test_reference_base_takes_them_as_references(tmp_path):
    first, last = _frame(tmp_path), _frame(tmp_path, "last.png")
    cmd = _train_cmd(tmp_path, "minimax", SAMPLE_FIRST_FRAME=first, SAMPLE_LAST_FRAME=last,
                     H3_TRAIN_BASE="Reference (ref2va)")
    i = cmd.index("--sample_ref_images")
    assert cmd[i + 1:i + 3] == [first, last]
    assert "--sample_first_frame" not in cmd and "--sample_last_frame" not in cmd
    a = setup_parser().parse_args(["--family", "minimax", "--dit", "d", "--output_dir", "o", "--output_name", "n",
                                   "--sample_ref_images", first, last])
    assert a.sample_ref_images == [first, last]


def test_reference_base_relabels_the_rows():
    desc = get_family("minimax")
    st = form.state(desc, {"H3_TRAIN_BASE": "Reference (ref2va)"})
    assert st["labels"]["SAMPLE_FIRST_FRAME"] == "Reference 1" and st["labels"]["SAMPLE_LAST_FRAME"] == "Reference 2"
    assert "<Picture 1>" in st["notes"]["SAMPLE_FIRST_FRAME"]
    assert "SAMPLE_FIRST_FRAME" not in form.state(desc, {}).get("labels", {})


def test_reference_prompts_carry_their_tags():
    drv = get_family("minimax").load_driver()

    class TE:
        def encode_with_reference(self, caption, images):
            return torch.zeros(1, 7, 8), torch.arange(7)

    out = drv.encode_text_with_ref_pictures(TE(), ["a", "b"], ["pic"])
    assert [tuple(c["hidden_states"].shape) for c in out] == [(7, 8), (7, 8)]
    assert all(c["token_tags"].tolist() == list(range(7)) for c in out)
    drv.encode_images = lambda vae, imgs: [torch.zeros(24, 4, 6)]
    drv.set_preview_references({"device": "cpu"}, [Image.new("RGB", (96, 64))])
    kw = drv._preview_ref_kw(out[0])
    assert kw["ref_latents"][0].shape == (1, 24, 1, 4, 6) and kw["text_token_tags"] is out[0]["token_tags"]
    assert drv._preview_ref_kw({"hidden_states": torch.zeros(3, 8)}) == {}          # a text-only prompt: none


def test_missing_picture_stops_start(tmp_path):
    desc, inp = _plan(tmp_path, "minimax", SAMPLE_FIRST_FRAME=str(tmp_path / "gone.png"))
    assert any("First frame" in p and "isn't there" in p for p in launch.start_problems(desc, inp))


def test_driver_anchors_first_and_last(tmp_path):
    drv = get_family("minimax").load_driver()
    seen = []

    def enc(vae, imgs):
        seen.append(imgs[0].shape)
        h, w = imgs[0].shape[:2]
        return [torch.zeros(24, h // 16, w // 16)]

    drv.encode_images = enc
    drv.set_preview_keyframes({"device": "cpu"}, {"first": _frame(tmp_path), "last": _frame(tmp_path, "l.png")})
    kfs = drv._preview_keyframes(768, 640, 56)
    assert [i for i, _ in kfs] == [0, 55]
    assert all(z.shape == (1, 24, 1, 40, 48) for _, z in kfs)
    assert seen == [(640, 768, 3), (640, 768, 3)]                       # cropped to the preview's shape
    drv._preview_keyframes(768, 640, 56)
    assert len(seen) == 2                                                # cached at that size
    drv._preview_keyframes(512, 512, 22)
    assert len(seen) == 4                                                # a smaller rung: encoded again
    assert drv._preview_keyframes(768, 640, 1) is None                   # a still uses neither


def test_a_moved_run_takes_its_frames(tmp_path):
    from fizgig.web import helpers
    first, last = _frame(tmp_path), _frame(tmp_path, "last.png")
    files = helpers._move_extras({"SAMPLE_FIRST_FRAME": first, "SAMPLE_LAST_FRAME": last})[1]
    assert files == {"SAMPLE_FIRST_FRAME": first, "SAMPLE_LAST_FRAME": last}
