"""A folder's own name is never a glob pattern. glob.glob(os.path.join(folder, "*")) matches nothing when the folder's
path holds [ ] (a training folder "photos [v2]", an output folder "LoRAs [klein]", Fizgig installed under one): Move a
run and a helper share sent none of the training folder's files, MOSS-Image / Qwen Image found no shards of a sharded
model, and the share's rounds were never cleared. Every glob in src/fizgig now escapes the folder part (glob.escape,
helpers._glob); only utils/vram_monitor.py globs system patterns on purpose."""
import ast
import tarfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "fizgig"
PATTERNS_ON_PURPOSE = {"utils/vram_monitor.py"}


def _unescaped(tree):
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("glob", "iglob")
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "glob" and n.args):
            continue
        a = n.args[0]
        if isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) and a.func.attr == "join" and a.args:
            first = a.args[0]
            if isinstance(first, ast.Constant) or (isinstance(first, ast.Call) and isinstance(first.func, ast.Attribute)
                                                   and first.func.attr == "escape"):
                continue
        elif isinstance(a, ast.Call) and isinstance(a.func, ast.Attribute) and a.func.attr == "escape":
            continue
        yield n.lineno


def test_every_glob_escapes_its_folder():
    bad = []
    for p in sorted(SRC.rglob("*.py")):
        rel = p.relative_to(SRC).as_posix()
        if rel in PATTERNS_ON_PURPOSE:
            continue
        bad += [f"{rel}:{line}" for line in _unescaped(ast.parse(p.read_text(encoding="utf-8")))]
    assert not bad, f"glob with an unescaped folder: {bad}"


def test_the_rule_catches_the_old_shape():
    tree = ast.parse("glob.glob(os.path.join(folder, '*'))\nglob.glob(os.path.join(glob.escape(folder), '*'))\n"
                     "glob.glob(_dir('files', 'sh*'))\nglob.glob(os.path.join('/fixed', '*'))\n")
    assert list(_unescaped(tree)) == [1, 3]


def test_move_a_run_packs_a_bracketed_training_folder(tmp_path):
    from fizgig.web import helpers
    folder = tmp_path / "photos [v2]"
    folder.mkdir()
    for n in ("a.png", "a.txt", "notes.doc"):
        (folder / n).write_bytes(b"x")
    out = tmp_path / "LoRAs [klein]"
    state = out / "t-000001-state"
    state.mkdir(parents=True)
    (state / "training_state.json").write_text("{}", encoding="utf-8")
    path = helpers._move_pack(str(out), "t", str(state), str(folder), {})
    with tarfile.open(path) as tf:
        names = tf.getnames()
    assert "images/a.png" in names and "images/a.txt" in names and "images/notes.doc" not in names
