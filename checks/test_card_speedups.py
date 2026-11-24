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
