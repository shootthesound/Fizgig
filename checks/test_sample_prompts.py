"""Each preview shows the prompt it was made with: a round's prompts are recorded (with its start time) before its
pictures appear, so a Sample now after a prompt edit, sharing its epoch with the epoch's own previews, shows its own."""
import ast, datetime, json, logging, os, re, shutil, subprocess, sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fizgig.web import server  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig")
TRAIN = os.path.join(ROOT, "families", "train.py")
APP = os.path.join(ROOT, "web", "static", "app.js")


def _trainer_fn(name, dt=datetime):
    """One function from train.py (the module needs the full training stack to import)."""
    src = open(TRAIN, encoding="utf-8").read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"os": os, "json": json, "datetime": dt, "logger": logging.getLogger("t")}
    exec(compile(ast.Module([node], []), TRAIN, "exec"), ns)
    return ns[name]


def _pic(run, epoch, idx, ts):
    return f"{run}_e{epoch:06d}_{idx:02d}_{ts}_{1234 + idx}.png"


def test_sample_now_after_an_edit_shows_its_own_prompt(tmp_path):
    stamps = iter(["20261025115734", "20261025120120"])

    class _DT:
        @staticmethod
        def now():
            return datetime.datetime.strptime(next(stamps), "%Y%m%d%H%M%S")
    record_round = _trainer_fn("_record_round", type("m", (), {"datetime": _DT}))
    record_prompts = _trainer_fn("_record_prompts")
    d, run = str(tmp_path), "R"
    record_prompts(d, run, 40, ["old a", "old b"])                 # epoch 40's previews (before rounds existed)
    record_round(d, run, 41, ["old a", "old b"])                   # Sample now in epoch 41, before the edit
    record_round(d, run, 41, ["new a", "old b"])                   # the epoch's own previews, after it
    record_prompts(d, run, 41, ["new a", "old b"])
    rounds = server.round_prompts(d, run)
    assert [r[:2] for r in rounds] == [[41, "20261025115734"], [41, "20261025120120"]]
    assert server.picture_prompt(rounds, _pic(run, 41, 0, "20261025115734"), 41, 0) == "old a"
    assert server.picture_prompt(rounds, _pic(run, 41, 0, "20261025120120"), 41, 0) == "new a"
    assert server.picture_prompt(rounds, _pic(run, 41, 0, "20261025120121"), 41, 0) == "new a"
    assert server.picture_prompt(rounds, _pic(run, 40, 0, "20261025112345"), 40, 0) is None   # older: as before
    assert server.picture_prompt(rounds, _pic(run, 41, 5, "20261025120120"), 41, 5) is None
    assert server.epoch_prompts(d, run) == {"40": ["old a", "old b"], "41": ["new a", "old b"]}  # "rounds" ignored


def test_round_recorded_before_the_pictures_render():
    src = open(TRAIN, encoding="utf-8").read()
    body = src[src.index("    def _previews(epoch):"):src.index("    def _make_ckpt_previews(path):")]
    rec = body.index("_record_round(sample_dir, output_name, epoch, prompts)")
    for render in ("ckpt_previews.render(", "round.json.tmp", "local = _render_previews("):
        assert rec < body.index(render)


def test_samples_api_gives_each_picture_its_prompt(tmp_path):
    d, run = str(tmp_path), "R"
    json.dump({"41": ["new a"], "rounds": [[41, "20261025115734", ["old a"]], [41, "20261025120120", ["new a"]]]},
              open(os.path.join(d, f".prompts_{run}.json"), "w"))
    for ts in ("20261025115734", "20261025120120"):
        open(os.path.join(d, _pic(run, 41, 0, ts)), "wb").write(b"x")
    got = {}

    class H:
        def json(self, data, code=200):
            got.update(data)
    server.Handler._samples(H(), d, run)
    by = {it["name"]: it.get("prompt") for it in got["items"]}
    assert by[_pic(run, 41, 0, "20261025115734")] == "old a" and by[_pic(run, 41, 0, "20261025120120")] == "new a"


def test_gallery_uses_the_pictures_prompt_and_redraws_when_records_land():
    src = open(APP, encoding="utf-8").read()
    fig = src[src.index("function previewFigure("):src.index("function withMode(")]
    assert "it.prompt != null ? it.prompt :" in fig and "own && own[it.idx]" in fig
    assert 'it.name + "@" + it.time + "@" + (it.prompt || "")' in src
    assert "JSON.stringify(g.epoch_prompts || {})" in src
    if not shutil.which("node"):
        pytest.skip("node not installed")
    js = ("const el=(t,a,...k)=>({t,a,k});const withMode=(f)=>f;const mediaFor=()=>null;const previewLabel=()=>'';\n"
          + fig + "\nconst cap=(f)=>f.k[1].k[1];"
          "console.log(JSON.stringify([cap(previewFigure({idx:0,epoch:41,prompt:'mine'},['run'],null,{'41':['ep']})).a.text,"
          "cap(previewFigure({idx:0,epoch:41},['run'],null,{'41':['ep']})).a.text,"
          "cap(previewFigure({idx:0,epoch:9},['run'],null,{'41':['ep']})).a.text]))")
    assert json.loads(subprocess.check_output(["node", "-e", js], text=True)) == ["mine", "ep", "run"]
