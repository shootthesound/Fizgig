"""A lambda or nested function made inside `except ... as exc:` that reads `exc` breaks when it runs
after the block: Python deletes the name when the block ends, so the call raises NameError instead of
showing the error (a failed Gizmo take showed nothing). Copy the message out first: err = str(exc)."""
import ast
import os

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _params(fn):
    a = fn.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    names.update(x.arg for x in (a.vararg, a.kwarg) if x)
    return names


def _late_reads(tree):
    for h in ast.walk(tree):
        if not isinstance(h, ast.ExceptHandler) or not h.name:
            continue
        for stmt in h.body:
            for fn in ast.walk(stmt):
                if not isinstance(fn, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef)) or h.name in _params(fn):
                    continue
                body = [fn.body] if isinstance(fn, ast.Lambda) else fn.body
                if any(isinstance(n, ast.Name) and n.id == h.name and isinstance(n.ctx, ast.Load)
                       for part in body for n in ast.walk(part)):
                    yield fn.lineno, h.name


def test_the_rule_catches_a_late_read():
    bad = "try:\n    pass\nexcept Exception as exc:\n    later(lambda: print(exc))\n"
    good = "try:\n    pass\nexcept Exception as exc:\n    err = str(exc)\n    later(lambda: print(err))\n"
    bound = "try:\n    pass\nexcept Exception as exc:\n    later(lambda exc=exc: print(exc))\n"
    assert list(_late_reads(ast.parse(bad))) == [(4, "exc")]
    assert not list(_late_reads(ast.parse(good))) and not list(_late_reads(ast.parse(bound)))


def test_no_deferred_call_reads_an_except_name():
    found = []
    for root, _dirs, files in os.walk(SRC):
        for f in files:
            if f.endswith(".py"):
                path = os.path.join(root, f)
                with open(path, encoding="utf-8") as fh:
                    tree = ast.parse(fh.read())
                found += [f"{os.path.relpath(path, SRC)}:{line} reads '{name}'" for line, name in _late_reads(tree)]
    assert not found, found
