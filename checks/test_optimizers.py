"""Prodigy and Muon optimizer support: catalog availability, vendored fallbacks, d * lr reporting, owns_its_rate,
UI form state forcing/unlocking (Prodigy locks LEARNING_RATE='1.0' and disables Adaptive LR; Muon keeps standard LR
and Adaptive LR while auto-filling RMS-matched args; clean reset on optimizer transitions), and launch plan."""

import builtins
import logging
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from fizgig.families import form, launch, registry  # noqa: E402
from fizgig.training.muon import Muon as FallbackMuon  # noqa: E402
from fizgig.training.optimizers import (  # noqa: E402
    MUON_DEFAULT_ARGS,
    PRODIGY_DEFAULT_ARGS,
    available_optimizers,
    create_optimizer,
    optimizer_lr,
    owns_its_rate,
    parse_optimizer_args,
    strip_muon_args,
    strip_prodigy_args,
)
from fizgig.training.prodigy import Prodigy as VendoredProdigy  # noqa: E402
from fizgig.training.step_log import StepLog, read as read_steplog  # noqa: E402


def test_prodigy_and_muon_in_available_optimizers():
    opts = available_optimizers()
    assert "prodigy" in opts
    assert "muon" in opts
    assert "automagic3" in opts
    assert "adamw" in opts


def test_parse_optimizer_args_types_and_booleans():
    parsed = parse_optimizer_args(
        "weight_decay=0.01 betas=0.9,0.99 decouple=True safeguard_warmup=true "
        "use_bias_correction=false d_coef=1.5 adjust_lr_fn=match_rms_adamw"
    )
    assert parsed == {
        "weight_decay": 0.01,
        "betas": (0.9, 0.99),
        "decouple": True,
        "safeguard_warmup": True,
        "use_bias_correction": False,
        "d_coef": 1.5,
        "adjust_lr_fn": "match_rms_adamw",
    }


def test_strip_prodigy_and_muon_args():
    assert strip_prodigy_args(PRODIGY_DEFAULT_ARGS) == ""
    assert strip_prodigy_args("decouple=True weight_decay=0.01 d_coef=0.5 safeguard_warmup=True") == ""
    assert strip_prodigy_args("d_coef=2.0 slice_p=11 betas=0.9,0.99") == "betas=0.9,0.99"
    assert strip_prodigy_args("weight_decay=0.01") == "weight_decay=0.01"
    assert strip_prodigy_args("") == ""

    assert strip_muon_args(MUON_DEFAULT_ARGS) == ""
    assert strip_muon_args("adjust_lr_fn=match_rms_adamw weight_decay=0.02 momentum=0.9 ns_steps=5") == ""
    assert strip_muon_args("momentum=0.9 betas=0.9,0.99") == "betas=0.9,0.99"
    assert strip_muon_args("weight_decay=0.01") == "weight_decay=0.01"
    assert strip_muon_args("") == ""


def test_prodigy_creation_defaults_and_overrides(caplog):
    p = [nn.Parameter(torch.randn(8, 8))]
    opt, label = create_optimizer("prodigy", p, 1.0)
    assert owns_its_rate(opt)
    g = opt.param_groups[0]
    assert g["lr"] == 1.0
    assert g["decouple"] is True
    assert g["weight_decay"] == pytest.approx(0.01)
    assert g["d_coef"] == pytest.approx(1.0)
    assert g["use_bias_correction"] is True
    assert g["safeguard_warmup"] is True
    assert optimizer_lr(opt) == pytest.approx(1e-6)

    # User overrides in Optimizer Args win over defaults
    p2 = [nn.Parameter(torch.randn(8, 8))]
    opt2, label2 = create_optimizer("prodigy", p2, 1.0, "d_coef=0.5 weight_decay=0.05 slice_p=11 use_bias_correction=False")
    g2 = opt2.param_groups[0]
    assert g2["d_coef"] == pytest.approx(0.5)
    assert g2["weight_decay"] == pytest.approx(0.05)
    assert g2["slice_p"] == 11
    assert g2["use_bias_correction"] is False
    assert "d_coef=0.5" in label2

    # Tiny CLI LR (< 0.1) is promoted to 1.0 with an info log
    p3 = [nn.Parameter(torch.randn(8, 8))]
    with caplog.at_level(logging.INFO):
        opt3, _ = create_optimizer("prodigy", p3, 1e-4)
    assert opt3.param_groups[0]["lr"] == 1.0
    assert "using lr=1.0" in caplog.text


def test_prodigy_vendored_fallback_when_prodigyopt_missing(monkeypatch):
    real_import = builtins.__import__

    def _no_prodigyopt(name, *args, **kwargs):
        if name == "prodigyopt" or name.startswith("prodigyopt."):
            raise ImportError("simulated missing prodigyopt")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "prodigyopt", raising=False)
    monkeypatch.setattr(builtins, "__import__", _no_prodigyopt)

    p = [nn.Parameter(torch.randn(8, 8))]
    opt, _ = create_optimizer("prodigy", p, 1.0)
    assert isinstance(opt, VendoredProdigy)
    assert owns_its_rate(opt)
    assert optimizer_lr(opt) == pytest.approx(1e-6)


def test_muon_creation_step_and_fallback():
    torch.manual_seed(7)
    lora_down = nn.Parameter(torch.randn(16, 64))
    lora_up = nn.Parameter(torch.zeros(64, 16))
    opt, label = create_optimizer("muon", [lora_down, lora_up], 1e-4)
    assert not owns_its_rate(opt)
    g = opt.param_groups[0]
    assert g["lr"] == pytest.approx(1e-4)
    assert g["adjust_lr_fn"] == "match_rms_adamw"
    assert g["weight_decay"] == pytest.approx(0.01)
    assert g["momentum"] == pytest.approx(0.95)
    assert g["nesterov"] is True
    assert g["ns_steps"] == 5
    assert optimizer_lr(opt) == pytest.approx(1e-4)

    x = torch.randn(4, 64)
    target = torch.randn(4, 64)
    down_before = lora_down.detach().clone()
    opt.zero_grad()
    loss = (((x @ lora_down.T) @ lora_up.T - target) ** 2).mean()
    loss.backward()
    opt.step()
    assert not torch.equal(lora_down, down_before)
    assert not torch.equal(lora_up, torch.zeros_like(lora_up))

    # Custom Muon args override defaults, and Prodigy keys are stripped cleanly
    p2 = [nn.Parameter(torch.randn(8, 16))]
    opt2, _ = create_optimizer("muon", p2, 2e-4, f"{PRODIGY_DEFAULT_ARGS} momentum=0.9 ns_steps=4 weight_decay=0.02")
    g2 = opt2.param_groups[0]
    assert g2["momentum"] == pytest.approx(0.9)
    assert g2["ns_steps"] == 4
    assert g2["weight_decay"] == pytest.approx(0.02)

    # Non-2D parameter transparently uses FallbackMuon without raising ValueError
    p_conv = [nn.Parameter(torch.randn(8, 4, 3, 3)), nn.Parameter(torch.randn(8))]
    opt_fb, _ = create_optimizer("muon", p_conv, 1e-4)
    assert isinstance(opt_fb, FallbackMuon)
    for t in p_conv:
        t.grad = torch.randn_like(t)
    before_conv = p_conv[0].detach().clone()
    opt_fb.step()
    assert not torch.equal(p_conv[0], before_conv)


def test_prodigy_steps_adapt_d_and_steplog_records_effective_lr(tmp_path):
    torch.manual_seed(42)
    w = nn.Parameter(torch.randn(16, 16))
    target = torch.randn(16, 16)
    opt, _ = create_optimizer("prodigy", [w], 1.0, PRODIGY_DEFAULT_ARGS)
    slog = StepLog(str(tmp_path), "prodigy_run")

    lrs = []
    for step in range(1, 15):
        opt.zero_grad()
        loss = ((w - target) ** 2).mean()
        loss.backward()
        opt.step()
        cur_lr = optimizer_lr(opt)
        lrs.append(cur_lr)
        slog.add(step, 1, loss.item(), cur_lr)
    slog.flush()

    assert lrs[0] == pytest.approx(1e-6)
    assert max(lrs) > 1e-6
    assert max(lrs) < 0.1
    data = read_steplog(slog.path)
    assert len(data["points"]) == 14
    assert all(pt[3] is not None and pt[3] < 0.1 for pt in data["points"])
    assert data["points"][-1][3] == pytest.approx(lrs[-1], rel=1e-3)


def test_fallback_to_adamw_clamps_high_lr(caplog):
    p = [nn.Parameter(torch.randn(4, 4))]
    with caplog.at_level(logging.WARNING):
        opt, label = create_optimizer("nonexistent_optimizer", p, 1.0)
    assert label == "adamw (fallback)"
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert "clamped from 1" in caplog.text


def test_non_prodigy_and_non_muon_strip_foreign_kwargs():
    p = [nn.Parameter(torch.randn(4, 4))]
    opt, label = create_optimizer("adamw", p, 1e-4, f"{PRODIGY_DEFAULT_ARGS} {MUON_DEFAULT_ARGS}")
    assert label.startswith("adamw")
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert not owns_its_rate(opt)

    opt_p, _ = create_optimizer("prodigy", [nn.Parameter(torch.randn(4, 4))], 1.0, MUON_DEFAULT_ARGS)
    assert owns_its_rate(opt_p)


def test_form_state_prodigy_and_muon_transitions(tmp_path):
    krea = registry.get("krea2")
    sdxl = registry.get("sdxl")

    # 1. Selecting prodigy locks LEARNING_RATE="1.0", disables Adaptive LR / warmup (per-image LR stays: a loss weight),
    #    and auto-fills PRODIGY_DEFAULT_ARGS
    vals = {
        **form.DEFAULTS,
        "OPTIMIZER_TYPE": "prodigy",
        "LEARNING_RATE": "1e-4",
        "OPTIMIZER_ARGS": "",
        "ADAPTIVE_LR": True,
        "KREA2_PER_IMAGE_LR": True,
        "KREA2_WARMUP_LOOK": True,
    }
    st = form.state(krea, vals)
    assert "LEARNING_RATE" in st["disabled"]
    assert "ADAPTIVE_LR" in st["disabled"]
    assert "KREA2_PER_IMAGE_LR" not in st["disabled"]
    assert "KREA2_WARMUP_LOOK" in st["disabled"]
    assert "OPTIMIZER_ARGS" not in st["disabled"]
    assert {"ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX"} <= set(st["hidden"])
    assert st["force"]["LEARNING_RATE"] == "1.0"
    assert st["force"]["OPTIMIZER_ARGS"] == PRODIGY_DEFAULT_ARGS
    assert st["force"]["ADAPTIVE_LR"] is False
    assert "KREA2_PER_IMAGE_LR" not in st["force"]
    assert st["force"]["KREA2_WARMUP_LOOK"] is False
    assert st["notes"]["OPTIMIZER_TYPE"].endswith("Per-image LR still works, as a loss weight on each image.")
    st_off = form.state(krea, {**vals, "KREA2_PER_IMAGE_LR": False})   # its own state is left as it is, either way
    assert "KREA2_PER_IMAGE_LR" not in st_off["force"] and "KREA2_PER_IMAGE_LR" not in st_off["disabled"]

    # 2. Switching from prodigy to muon unlocks LEARNING_RATE (restores "1e-4"), strips Prodigy args,
    #    auto-fills MUON_DEFAULT_ARGS, and keeps Adaptive LR available
    vals_muon = {**vals, **st["force"], "OPTIMIZER_TYPE": "muon"}
    st_muon = form.state(krea, vals_muon)
    assert "LEARNING_RATE" not in st_muon["disabled"]
    assert "ADAPTIVE_LR" not in st_muon["disabled"]
    assert "OPTIMIZER_ARGS" not in st_muon["disabled"]
    assert st_muon["force"]["LEARNING_RATE"] == "1e-4"
    assert st_muon["force"]["OPTIMIZER_ARGS"] == MUON_DEFAULT_ARGS
    assert "Newton-Schulz" in st_muon["notes"]["OPTIMIZER_TYPE"]
    assert "match_rms_adamw" in st_muon["notes"]["OPTIMIZER_ARGS"]

    # 3. Switching from muon back to prodigy strips Muon args, locks LR="1.0", and fills PRODIGY_DEFAULT_ARGS
    vals_back_prodigy = {**vals_muon, **st_muon["force"], "OPTIMIZER_TYPE": "prodigy"}
    st_back_prodigy = form.state(krea, vals_back_prodigy)
    assert st_back_prodigy["force"]["LEARNING_RATE"] == "1.0"
    assert st_back_prodigy["force"]["OPTIMIZER_ARGS"] == PRODIGY_DEFAULT_ARGS

    # 4. Switching from muon to adamw8bit clears MUON_DEFAULT_ARGS back to ""
    vals_adam = {**vals_muon, **st_muon["force"], "OPTIMIZER_TYPE": "adamw8bit"}
    st_adam = form.state(krea, vals_adam)
    assert st_adam["force"]["OPTIMIZER_ARGS"] == ""

    # 5. Switching from prodigy to SDXL adamw restores SDXL's 5e-5 LR and clears args
    st_sdxl = form.state(sdxl, {**vals, **st["force"], "OPTIMIZER_TYPE": "adamw"})
    assert st_sdxl["force"]["LEARNING_RATE"] == "5e-5"
    assert st_sdxl["force"]["OPTIMIZER_ARGS"] == ""


def test_launch_plan_with_prodigy_and_muon(tmp_path):
    krea = registry.get("krea2")
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    (img_dir / "a.png").write_bytes(b"x")
    (img_dir / "a.txt").write_text("zwxem a photo", encoding="utf-8")
    models = {}
    for mf in krea.model_files:
        if mf.required or mf.pref_key == krea.training_adapter:
            f = tmp_path / f"{mf.pref_key}.safetensors"
            f.write_bytes(b"0")
            models[mf.pref_key] = str(f)

    # Prodigy plan
    inp_p = form.inputs(
        krea,
        {
            **form.DEFAULTS,
            "LORA_NAME": "test_prodigy",
            "LORA_OUTPUT_DIR": str(tmp_path / "out"),
            "DATASET_CONFIG": str(tmp_path / "dataset.toml"),
            "OPTIMIZER_TYPE": "prodigy",
            "LEARNING_RATE": "1.0",
            "OPTIMIZER_ARGS": "",
            "ADAPTIVE_LR": True,
        },
        models=models,
        python=sys.executable,
        repo_dir=str(tmp_path),
        image_folder=str(img_dir),
        cache_root=str(tmp_path / "cache"),
        samples_dir=str(tmp_path / "out" / "sample"),
    )
    pl_p = launch.plan(krea, inp_p)
    assert pl_p.problems == []
    cmd_p = pl_p.stages[-1].cmd
    assert cmd_p[cmd_p.index("--optimizer_type") + 1] == "prodigy"
    assert cmd_p[cmd_p.index("--learning_rate") + 1] == "1.0"
    assert "--adaptive_lr" not in cmd_p
    assert cmd_p[cmd_p.index("--optimizer_args") + 1] == PRODIGY_DEFAULT_ARGS

    # Muon plan (with stale Prodigy args in OPTIMIZER_ARGS: stripped and replaced with MUON_DEFAULT_ARGS)
    inp_m = form.inputs(
        krea,
        {
            **form.DEFAULTS,
            "LORA_NAME": "test_muon",
            "LORA_OUTPUT_DIR": str(tmp_path / "out2"),
            "DATASET_CONFIG": str(tmp_path / "dataset2.toml"),
            "OPTIMIZER_TYPE": "muon",
            "LEARNING_RATE": "1e-4",
            "OPTIMIZER_ARGS": PRODIGY_DEFAULT_ARGS,
            "ADAPTIVE_LR": False,
        },
        models=models,
        python=sys.executable,
        repo_dir=str(tmp_path),
        image_folder=str(img_dir),
        cache_root=str(tmp_path / "cache"),
        samples_dir=str(tmp_path / "out2" / "sample"),
    )
    pl_m = launch.plan(krea, inp_m)
    assert pl_m.problems == []
    cmd_m = pl_m.stages[-1].cmd
    assert cmd_m[cmd_m.index("--optimizer_type") + 1] == "muon"
    assert cmd_m[cmd_m.index("--learning_rate") + 1] == "0.0001"
    assert cmd_m[cmd_m.index("--optimizer_args") + 1] == MUON_DEFAULT_ARGS


def test_app_js_prodigy_and_muon_switch_and_reset():
    script = r"""
const fs = require("fs");
const code = fs.readFileSync("src/fizgig/web/static/app.js", "utf8");
const start = code.indexOf("const PRODIGY_DEFAULT_ARGS =");
const end = code.indexOf("// a combo row's dropdown:", start);
const snippet = code.slice(start, end);

let synced = [];
const S = {
  values: {OPTIMIZER_TYPE: "adamw8bit", LEARNING_RATE: "1e-4", OPTIMIZER_ARGS: "", ADAPTIVE_LR: true, KREA2_PER_IMAGE_LR: true, KREA2_WARMUP_LOOK: true},
  fam: {form: {presets: [{name: "std", values: {OPTIMIZER_TYPE: "adamw8bit", LEARNING_RATE: 0.0001}}]}}
};
const syncInputs = (keys) => { synced.push(...keys); };
const refreshStateSoon = () => {};
const saveStateSoon = () => {};
const sampleLiveSoon = () => {};
const saveTriggerSoon = () => {};
const $ = () => ({value: ""});

eval(snippet + "\nglobalThis.onValue = onValue; globalThis.PRODIGY_DEFAULT_ARGS = PRODIGY_DEFAULT_ARGS; globalThis.MUON_DEFAULT_ARGS = MUON_DEFAULT_ARGS;");

// 1. Switch to prodigy
onValue({key: "OPTIMIZER_TYPE"}, "prodigy");
if (S.values.LEARNING_RATE !== "1.0") throw new Error("Expected LR 1.0, got " + S.values.LEARNING_RATE);
if (S.values.OPTIMIZER_ARGS !== PRODIGY_DEFAULT_ARGS) throw new Error("Expected prodigy args, got " + S.values.OPTIMIZER_ARGS);
if (S.values.ADAPTIVE_LR !== false || S.values.KREA2_WARMUP_LOOK !== false) {
  throw new Error("Expected adaptive/warmup off");
}
if (S.values.KREA2_PER_IMAGE_LR !== true) throw new Error("Expected per-image LR left on with prodigy");
if (synced.includes("KREA2_PER_IMAGE_LR")) throw new Error("per-image LR should not be touched by prodigy");

// 2. Switch from prodigy to muon
onValue({key: "OPTIMIZER_TYPE"}, "muon");
if (S.values.LEARNING_RATE !== "1e-4") throw new Error("Expected LR reset to 1e-4 for muon, got " + S.values.LEARNING_RATE);
if (S.values.OPTIMIZER_ARGS !== MUON_DEFAULT_ARGS) throw new Error("Expected muon args, got " + S.values.OPTIMIZER_ARGS);

// 3. Switch from muon to prodigy
onValue({key: "OPTIMIZER_TYPE"}, "prodigy");
if (S.values.LEARNING_RATE !== "1.0") throw new Error("Expected LR 1.0 for prodigy, got " + S.values.LEARNING_RATE);
if (S.values.OPTIMIZER_ARGS !== PRODIGY_DEFAULT_ARGS) throw new Error("Expected prodigy args from muon, got " + S.values.OPTIMIZER_ARGS);

// 4. Switch from prodigy to muon, then to adamw8bit
onValue({key: "OPTIMIZER_TYPE"}, "muon");
onValue({key: "OPTIMIZER_TYPE"}, "adamw8bit");
if (S.values.LEARNING_RATE !== "1e-4") throw new Error("Expected LR 1e-4 for adamw8bit, got " + S.values.LEARNING_RATE);
if (S.values.OPTIMIZER_ARGS !== "") throw new Error("Expected args cleared from muon, got " + S.values.OPTIMIZER_ARGS);

// 5. Switch to prodigy, then to automagic3
onValue({key: "OPTIMIZER_TYPE"}, "prodigy");
onValue({key: "OPTIMIZER_TYPE"}, "automagic3");
if (S.values.LEARNING_RATE !== "1e-6") throw new Error("Expected LR 1e-6 for automagic3, got " + S.values.LEARNING_RATE);
if (S.values.OPTIMIZER_ARGS !== "") throw new Error("Expected args cleared for automagic3, got " + S.values.OPTIMIZER_ARGS);
console.log("JS_OK");
"""
    res = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    assert "JS_OK" in res.stdout


def test_automagic_still_turns_per_image_lr_off():
    krea = registry.get("krea2")
    st = form.state(krea, {**form.DEFAULTS, "OPTIMIZER_TYPE": "automagic3", "LEARNING_RATE": "1e-6",
                           "ADAPTIVE_LR": True, "KREA2_PER_IMAGE_LR": True, "KREA2_WARMUP_LOOK": True})
    assert {"ADAPTIVE_LR", "KREA2_PER_IMAGE_LR", "KREA2_WARMUP_LOOK"} <= set(st["disabled"])
    assert st["force"]["ADAPTIVE_LR"] is False and st["force"]["KREA2_PER_IMAGE_LR"] is False
    assert st["force"]["KREA2_WARMUP_LOOK"] is False
    assert st["notes"]["OPTIMIZER_TYPE"] == ("automagic3 sets its own learning rate: Adaptive LR, per-image LR and the "
                                             "look warm-up are off.")


@pytest.mark.parametrize("opt,lr,keeps", [("prodigy", "1.0", True), ("automagic3", "1e-6", False),
                                          ("adamw8bit", "1e-4", True)])
def test_launch_per_image_lr_with_prodigy(tmp_path, opt, lr, keeps):
    krea = registry.get("krea2")
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    (img_dir / "a.png").write_bytes(b"x")
    (img_dir / "a.txt").write_text("zwxem a photo", encoding="utf-8")
    models = {}
    for mf in krea.model_files:
        if mf.required or mf.pref_key == krea.training_adapter:
            f = tmp_path / f"{mf.pref_key}.safetensors"
            f.write_bytes(b"0")
            models[mf.pref_key] = str(f)
    inp = form.inputs(krea, {**form.DEFAULTS, "LORA_NAME": "t", "LORA_OUTPUT_DIR": str(tmp_path / "out"),
                             "DATASET_CONFIG": str(tmp_path / "dataset.toml"), "OPTIMIZER_TYPE": opt,
                             "LEARNING_RATE": lr, "OPTIMIZER_ARGS": "", "ADAPTIVE_LR": True,
                             "KREA2_LOSS_WATCH": True, "KREA2_PER_IMAGE_LR": True, "KREA2_WARMUP_LOOK": True},
                      models=models, python=sys.executable, repo_dir=str(tmp_path), image_folder=str(img_dir),
                      cache_root=str(tmp_path / "cache"), samples_dir=str(tmp_path / "out" / "sample"))
    pl = launch.plan(krea, inp)
    assert pl.problems == []
    cmd = pl.stages[-1].cmd
    assert ("--per_image_lr" in cmd) is keeps
    assert "--log_per_image_loss" in cmd
    if opt != "adamw8bit":
        assert "--warmup_look_outliers" not in cmd and "--adaptive_lr" not in cmd


def test_trainer_keeps_per_image_lr_only_for_prodigy():
    src = open(os.path.join(os.path.dirname(__file__), "..", "src", "fizgig", "families", "train.py"),
               encoding="utf-8").read()
    blk = src[src.index("    if owns_its_rate(optimizer):        # Automagic v3"):]
    blk = blk[:blk.index("box_learning_rate = learning_rate")]
    assert '_keeps_pilr = str(optimizer_type or "").strip().lower() == "prodigy"' in blk
    assert "adaptive_lr = warmup_look_outliers = False" in blk
    assert "if not _keeps_pilr:\n            per_image_lr = False" in blk
    assert "m = 1.0                        # Automagic v3 / Prodigy owns the rate" in src   # lr_acc path untouched
