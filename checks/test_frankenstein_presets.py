"""Frankenstein mode's presets: Studio serves the mode its own (Reset All + the family's), every named block a real
block at a blend weight in 0..1; the page applies them with every other block at 0 (the model as loaded) - the LoRA
rule, every other block at 1, made Reset All blend in 100% of the other model - and keeps your presets for the mode
apart from your LoRA presets (a LoRA strength read as a blend weight)."""
import os
import re

import pytest

pytest.importorskip("torch")
from fizgig.families.registry import get  # noqa: E402
from fizgig.web import studio  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "web", "static", "app.js")


def _layout(key):
    return studio.layout(get(key), {}, lambda prefs, k: "")


def test_h3_image_serves_its_frankenstein_presets():
    L = _layout("minimax_image")
    names = [p["name"] for p in L["frank_presets"]]
    assert names[0] == "✨Reset All" and L["frank_presets"][0]["keep"] is None
    assert "✨Motion" in names and "✨Motion, gentle" in names
    ids = {b["id"] for g in L["groups"] for b in g["blocks"]}
    for p in L["frank_presets"][1:]:
        assert p["blocks"] and set(p["blocks"]) <= ids, p["name"]
        assert all(0.0 < w <= 1.0 for w in p["blocks"].values()), p["name"]
    motion = dict(get("minimax_image").frankenstein_presets)["✨Motion"]
    assert dict(motion) == {f"h3blk_{i}": 0.75 for i in (26, 27, 28, 32, 33, 34)}


def test_a_family_without_the_mode_serves_none():
    assert _layout("zimage")["frank_presets"] == []


def test_start_needs_the_other_side_lora_too(tmp_path):
    # without H3's Turbo LoRA baked into H3 video's side its blocks aren't few-step blocks and every blend is noise:
    # Start said nothing and loaded without it
    from fizgig.web import server
    desc = get("minimax_image")
    key, _name, lora_key, _s = desc.frankenstein_with[0]
    dit = tmp_path / "h3video.safetensors"
    dit.write_bytes(b"x")
    frank, problems = server._frank_plan(desc, {key: str(dit), lora_key: str(tmp_path / "missing.safetensors")})
    assert len(problems) == 1 and "Turbo LoRA" in problems[0], problems
    lora = tmp_path / "turbo.safetensors"
    lora.write_bytes(b"x")
    frank, problems = server._frank_plan(desc, {key: str(dit), lora_key: str(lora)})
    assert not problems and frank["path"] == str(dit) and frank["lora"] == str(lora) and frank["strength"] == 0.75


def test_the_tick_hides_the_lora_rows():
    # the mode loads no LoRA: the Primary / Donor rows and their mixes showed until the mode had loaded (and could be
    # filled in for a load that ignores them)
    js = open(APP, encoding="utf-8").read()
    css = open(os.path.join(os.path.dirname(APP), "style.css"), encoding="utf-8").read()
    assert 'classList.toggle("frank-asked", studioFrankAsked())' in js
    assert '$("stFrank").addEventListener("change", studioFrankTick)' in js
    rule = css[css.index("#tab-studio.frank-asked .st-lora-row"):]
    rule = rule[:rule.index("}")]
    for sel in ("#tab-studio.frank-asked .st-lora-row", "#tab-studio.frank-asked #stMixRows",
                "#tab-studio.frank-asked #stDmixRows"):
        assert sel in rule, sel
    assert "display: none" in css[css.index(rule):css.index(rule) + len(rule) + 40]


def test_block_values_stay_within_a_blend_share():
    # in the mode a block is a share of the other model, 0..1: the value box, ± and − / + went to -3..3, lucky rolls
    # to 1.5, and the server quietly clamped - the screen said one thing, the render did another
    js = open(APP, encoding="utf-8").read()
    css = open(os.path.join(os.path.dirname(APP), "style.css"), encoding="utf-8").read()
    assert "const stLo = () => (S.st.frank ? 0 : -3), stHi = () => (S.st.frank ? 1 : 3);" in js
    sset = js[js.index("function studioSet(id, k, v, live)"):]
    assert 'if (k === "ps") v = stClampP(v);' in sset[:400]
    ssync = js[js.index("function studioSync(ids)"):]
    assert "if (S.st.frank) for (const b of Object.values(S.st.blocks)) b.ps = stClampP(b.ps);" in ssync[:200]
    assert 'side === "p" ? stClampP(r)' in js
    assert '$("stLuckyLo").value = "0"; $("stLuckyHi").value = "0.6";' in js     # lucky rolls 0 to 0.6 in the mode
    assert "#tab-studio.frankenstein .st-flip, #tab-studio.frankenstein .st-bal { display: none; }" in css


def test_page_applies_them_over_zero_and_keeps_your_presets_apart():
    js = open(APP, encoding="utf-8").read()
    apply = js[js.index("async function studioPresetApply"):js.index("// Royale's reference strength")]
    assert "const rest = S.st.frank ? 0 : 1;" in apply and "ps: rest" in apply
    assert "studioBuiltins().find" in apply
    assert re.search(r'studioPresetKey\(\) \{ return S\.st\.layout\.key \+ \(S\.st\.frank \? "__frankenstein" : ""\)', js)
    for call in ("/api/studio/presets?family=", "/api/studio/preset_load", "/api/studio/preset_save"):
        i = js.index(call, js.index("async function studioPresets"))
        assert "studioPresetKey()" in js[i:i + 120], call
