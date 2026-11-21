"""Studio with no LoRA: its one render is the Tweaked pane (the Baseline pane and the arrows are hidden), so the arrows'
remembered "Tweaked hidden" must not hide it too - every render vanished, and showed when a LoRA was added."""
import os
import re

STATIC = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static")


def test_tweaked_pane_shows_without_a_lora():
    css = open(os.path.join(STATIC, "style.css"), encoding="utf-8").read()
    assert re.search(r"#tab-studio\.no-lora #stBasePane", css)                       # the layout this is about
    assert re.search(r"#tab-studio\.no-lora \.st-preview\.only-base \.st-tweak-pane\s*\{\s*display:\s*block", css)
    # in Keep in view (the dock) the pane stays a flex column, or its picture stops scaling to the panel
    assert re.search(r"\.st-dock \.st-pane\s*\{\s*display:\s*flex", css)
    assert re.search(r"#tab-studio\.no-lora \.st-dock \.st-preview\.only-base \.st-tweak-pane\s*\{\s*display:\s*flex", css)


def test_panes_open_both_each_time():
    """A pane hidden last time isn't remembered: Studio opens with both (Peter, 21 Nov 2026)."""
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    assert 'localStorage.setItem("fizgig_st_split"' not in js and 'getItem("fizgig_st_split")' not in js
    assert 'split("both");' in js


def test_hidden_tweaked_rules_skip_no_lora():
    js = open(os.path.join(STATIC, "app.js"), encoding="utf-8").read()
    for line in js.splitlines():
        if 'S.st.split === "base"' in line and "split(" not in line and "S.st.split =" not in line \
                and "? " not in line.split('S.st.split === "base"')[1][:3]:
            assert "noLora" in line, line.strip()
