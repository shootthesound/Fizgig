"""Preferences > Speed-ups for your graphics card: the two per-card switches are read from prefs.json by the run itself
(every process: training, workers, helpers, pods), and an environment variable wins over the tick."""
import json
import os

import pytest

from fizgig import user_data
from fizgig.families import quant
from fizgig.modules import rocm_conv


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    path = tmp_path / "prefs.json"
    monkeypatch.setattr(user_data, "PREFS_FILE", str(path))
    monkeypatch.delenv("FIZGIG_PIN_SWAP", raising=False)
    monkeypatch.delenv("FIZGIG_ROCM_BANDS", raising=False)
    return lambda **kv: path.write_text(json.dumps(kv), encoding="utf-8")


def test_pinning_follows_the_tick(prefs, monkeypatch):
    assert quant.pin_swap_ok(1.0, 100) is False             # no prefs file: off
    prefs(pin_block_swap="1")
    assert quant.pin_swap_ok(1.0, 100) is True
    assert quant.pin_swap_ok(10.0, 13.9) is False           # still only with room to spare
    prefs(pin_block_swap="0")
    assert quant.pin_swap_ok(1.0, 100) is False
    monkeypatch.setenv("FIZGIG_PIN_SWAP", "1")               # the variable wins
    assert quant.pin_swap_ok(1.0, 100) is True
    prefs(pin_block_swap="1")
    monkeypatch.setenv("FIZGIG_PIN_SWAP", "0")
    assert quant.pin_swap_ok(1.0, 100) is False


def test_amd_bands_follow_the_tick(prefs, monkeypatch):
    assert rocm_conv.wanted() is True                        # on unless switched off
    prefs(rocm_conv_bands="0")
    assert rocm_conv.wanted() is False
    monkeypatch.setenv("FIZGIG_ROCM_BANDS", "1")
    assert rocm_conv.wanted() is True
    monkeypatch.setenv("FIZGIG_ROCM_BANDS", "0")
    prefs(rocm_conv_bands="1")
    assert rocm_conv.wanted() is False


def test_the_page_saves_and_resets_them():
    from fizgig.web import server
    assert server.PREF_DEFAULTS["pin_block_swap"] == "0" and server.PREF_DEFAULTS["rocm_conv_bands"] == "1"
    assert server.PREF_DEFAULTS["rocm_miopen"] == "1"
    assert {"pin_block_swap", "rocm_conv_bands", "rocm_miopen"} <= set(server.SETTING_PREFS)
    js = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "fizgig", "web",
                           "static", "app.js"), encoding="utf-8").read()
    assert 'tick("pin_block_swap", "0")' in js and 'tick("rocm_conv_bands", "1")' in js
    assert 'tick("rocm_miopen", "1")' in js
    assert "host.append(cardSpeedupsCard());" in js


def test_every_tick_on_the_card_saves_as_1_or_0():
    """A tick shows on when its saved value is "1" (String(S.set[key]) === "1"), so the save must store "1" / "0":
    Faster block swap was missed and saved as "True", showing unticked after a reload though it was on."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    js = open(os.path.join(root, "src", "fizgig", "web", "static", "app.js"), encoding="utf-8").read()
    card = js[js.index("function cardSpeedupsCard()"):]
    card = card[:card.index("\nfunction ", 1)]
    keys = set(re.findall(r'tick\("([a-z0-9_]+)"', card))
    assert {"pin_block_swap", "send_only_block_swap", "rocm_conv_bands", "rocm_miopen"} <= keys
    py = open(os.path.join(root, "src", "fizgig", "web", "server.py"), encoding="utf-8").read()
    start = py.index('for k in ("lan_discovery", "update_check"')
    block = py[start:py.index("*(k for k in ch if k in autosave_keys)", start)]
    missing = [k for k in sorted(keys) if f'"{k}"' not in block]
    assert not missing, f"saved as True/False, not 1/0: {missing}"


def test_show_how_popup_closes():
    """Preferences' Show how opens the Sysmem pop-up with its close buttons wired (they were only wired by the launch
    pop-up, so opened from Preferences it couldn't be dismissed), and without Don't show this again."""
    import shutil
    import subprocess
    from pathlib import Path
    if not shutil.which("node"):
        pytest.skip("node not installed")
    js = (Path(__file__).resolve().parents[1] / "src/fizgig/web/static/app.js").read_text(encoding="utf-8")
    assert "onclick: () => sysmemOpen(true)" in js
    start = js.index("function sysmemOpen(")
    fn = js[start:js.index("\n}\n", start) + 3]
    script = """
const assert = require('node:assert/strict');
const made = {};
const node = () => { const n = {style: {}, checked: false, classList: {s: new Set(), add(c) {this.s.add(c);},
  remove(c) {this.s.delete(c);}, contains(c) {return this.s.has(c);}}}; n.closest = () => n; return n; };
const $ = (id) => (made[id] = made[id] || node());
let saved = 0; const api = () => { saved++; return Promise.resolve(); };
""" + fn + """
for (const how of ["sysmemClose", "sysmemX", "backdrop"]) {
  sysmemOpen(true);
  const m = $("sysmemModal");
  assert(m.classList.contains("active"));
  assert.equal($("sysmemNever").style.display, "none");
  if (how === "backdrop") m.onclick({target: m}); else $(how).onclick();
  assert(!m.classList.contains("active"), how);
}
assert.equal(saved, 0);
"""
    subprocess.run(["node", "-e", script], check=True)
