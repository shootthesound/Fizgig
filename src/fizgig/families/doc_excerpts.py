"""Keep the driver guide's code excerpts identical to the drivers they quote.

An excerpt in a docs/drivers/*.md file is a marker naming a function, followed by a fenced python block:

    <!-- excerpt: qwen_image21/driver.py QwenImage21Driver.training_loss -->
    ```python
    ...the function's source, exactly...
    ```

    python src/fizgig/families/doc_excerpts.py --check    # exit 1 and name every excerpt that no longer matches
    python src/fizgig/families/doc_excerpts.py --update   # rewrite every excerpt from the current source

Paths are relative to src/fizgig. The name is `Class.method` or a module-level `function`. Run --check before a
release (and --update after changing a quoted driver), so the guide never shows code the drivers no longer run.
"""
import argparse
import ast
import glob
import os
import re
import sys
import textwrap

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))          # src/fizgig
DOCS = os.path.abspath(os.path.join(SRC, "..", "..", "docs", "drivers"))
MARK = re.compile(r"<!-- excerpt: (\S+) (\S+) -->\n```python\n(.*?)^```", re.S | re.M)   # empty blocks too


def source_of(path, name):
    """The exact source of `Class.method` / `function` in src/fizgig/<path>, decorators included, dedented."""
    text = open(os.path.join(SRC, path), encoding="utf-8").read()
    lines = text.splitlines()
    parts = name.split(".")
    nodes = ast.parse(text).body
    node = None
    for i, part in enumerate(parts):
        node = next((n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                     and n.name == part), None)
        if node is None:
            raise KeyError(f"{name} not found in {path}")
        nodes = node.body
    start = min([d.lineno for d in node.decorator_list] + [node.lineno])
    return textwrap.dedent("\n".join(lines[start - 1:node.end_lineno]))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--update", action="store_true")
    a = ap.parse_args()
    stale, total = [], 0
    for md in sorted(glob.glob(os.path.join(DOCS, "*.md"))):
        doc = open(md, encoding="utf-8").read()

        def fix(m):
            nonlocal total
            total += 1
            path, name, shown = m.group(1), m.group(2), m.group(3).rstrip("\n")
            try:
                live = source_of(path, name)
            except (KeyError, OSError) as e:
                stale.append(f"{os.path.basename(md)}: {path} {name} - {e}")
                return m.group(0)
            if live != shown:
                stale.append(f"{os.path.basename(md)}: {path} {name} - differs from the source")
            return f"<!-- excerpt: {path} {name} -->\n```python\n{live}\n```"

        new = MARK.sub(fix, doc)
        if a.update and new != doc:
            open(md, "w", encoding="utf-8", newline="\n").write(new)
    if a.update:
        print(f"{total} excerpts written from the source" + (f"; not found: {stale}" if any("not found" in s for s in stale) else ""))
        sys.exit(0)
    for s in stale:
        print("STALE", s)
    print(f"{total} excerpts, {len(stale)} stale")
    sys.exit(1 if stale else 0)


if __name__ == "__main__":
    main()
