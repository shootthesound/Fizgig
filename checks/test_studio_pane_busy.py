"""Studio: each preview pane says when a render (or the model load) is going, under its picture, so it shows while you
work the sliders below - the status line and bar sit above the output (Peter, 26 Nov 2026)."""
import os
import re

STATIC = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static")


def _read(name):
    with open(os.path.join(STATIC, name), encoding="utf-8") as f:
        return f.read()


def test_every_pane_has_its_line_and_the_poll_fills_it():
    html, js = _read("index.html"), _read("app.js")
    for k in ("Nol", "Base", "Tweak"):
        fig = re.search(rf'<figcaption id="st{k}Cap">.*?</figure>', html, re.S)
        assert fig and f'id="stBusy{k}"' in fig.group(0)            # inside the pane, under its caption
    poll = js[js.index("async function studioPollOnce()"):]
    assert poll.index("studioPaneBusy(st, going)") < poll.index("S.st.foreign && st.load_by")   # before the early return
