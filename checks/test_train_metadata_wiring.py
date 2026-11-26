"""The family trainer's LoRA save adds the training recipe (training/metadata.py::training_metadata), and every name
its metadata() closure reads is bound in train_family - a typo there would only show up as a silently missing recipe,
since the save never lets metadata stop it."""
import ast
import builtins
import inspect

from fizgig.families import train as T


def test_save_metadata_carries_the_recipe():
    src = inspect.getsource(T.train_family)
    assert "training_metadata(" in src and "training_spec=_spec" in src and "md.setdefault(k, v)" in src


def test_metadata_closure_reads_only_bound_names():
    tree = ast.parse(inspect.getsource(T))
    mod = set(vars(T)) | set(dir(builtins))
    tf = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_family")
    bound = {a.arg for a in tf.args.args + tf.args.kwonlyargs}
    for n in ast.walk(tf):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            bound.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(n.name)
            bound |= {a.arg for a in n.args.args + n.args.kwonlyargs}
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
    md = next(n for n in ast.walk(tf) if isinstance(n, ast.FunctionDef) and n.name == "metadata")
    used = {n.id for n in ast.walk(md) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert not (used - bound - mod)
