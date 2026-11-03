"""Pause pressed a second time: Save & stop. A tiny CPU family trains for real (a toy DiT, toy encoders) through the
cache and the trainer: the restore point is saved mid-epoch and the run stops, Resume trains the rest of that epoch
and carries the step count, the LR schedule and the loss log on; a save that fails keeps training. Plus the server's
side (the request, its failure note, the epochs a restore point holds) and the button's wiring."""
import dataclasses
import json
import os
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from fizgig.families import registry
from fizgig.families.driver import FamilyDriver

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPE = 4                                    # 8 pictures in two buckets, Batch Size 2: 4 steps an epoch


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Module()
        for n in ("to_q", "to_k", "to_v"):
            setattr(self.attention, n, nn.Linear(4, 4))
        self.attention.to_out = nn.ModuleList([nn.Linear(4, 4)])
        self.feed_forward = nn.Module()
        self.feed_forward.w1, self.feed_forward.w2 = nn.Linear(4, 4), nn.Linear(4, 4)

    def forward(self, x, c):
        a = self.attention
        x = x + a.to_out[0](a.to_q(x) * torch.sigmoid(a.to_k(c)) + a.to_v(x))
        return x + self.feed_forward.w2(torch.tanh(self.feed_forward.w1(x)))


class _Dit(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.layers = nn.ModuleList([_Layer() for _ in range(3)])

    def forward(self, x, c):
        for layer in self.layers:
            x = layer(x, c)
        return x


class TinyDriver(FamilyDriver):
    supports_references = False

    def load_dit(self, path, device):
        return _Dit().to(device)

    def max_blocks_to_swap(self, dit=None):
        return 0

    def enable_gradient_checkpointing(self, dit, on=True):
        pass

    def load_vae(self, path, device):
        return "vae"

    def encode_images(self, vae, images):
        out = []
        for im in images:
            t = torch.from_numpy(np.asarray(im)).float().permute(2, 0, 1) / 255.0
            t = torch.nn.functional.avg_pool2d(t[None], 8)[0]
            out.append(torch.cat([t, t[:1]], 0))
        return out

    def load_text_encoder(self, path, device):
        return "te"

    def unload_text_encoder(self, te):
        pass

    def encode_text(self, te, captions):
        return [{"emb": torch.tensor([float(len(c) % 7), 1.0, 0.5, float(len(c) % 3)])} for c in captions]

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0, loss_mask=None, prior=None, **kw):
        b = latents.shape[0]
        t = torch.rand(b, generator=generator) * (max_t - min_t) + min_t
        noise = torch.randn(latents.shape, generator=generator)
        xt = (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise
        v = dit(xt.flatten(2).transpose(1, 2), cond["emb"].view(b, 1, 4)).transpose(1, 2).reshape(latents.shape)
        per = ((v - (noise - latents)) ** 2).flatten(1).mean(1)
        return per.mean(), {"t": float(t[0]), "ts": [float(x) for x in t], "per": per}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        t = torch.full((latents.shape[0],), (min_t + max_t) / 2)
        noise = torch.randn(latents.shape, generator=generator)
        return {"xt": (1 - t.view(-1, 1, 1, 1)) * latents + t.view(-1, 1, 1, 1) * noise, "target": noise - latents}

    def predict(self, dit, state, cond):
        b = state["xt"].shape[0]
        x = state["xt"].flatten(2).transpose(1, 2)
        return dit(x, cond["emb"].view(b, 1, 4)).transpose(1, 2).reshape(state["xt"].shape)


class _CpuTorch(types.ModuleType):
    """The trainer's torch, with its "cuda" device on the CPU."""
    def __getattr__(self, n):
        return getattr(torch, n)

    @staticmethod
    def device(x, *a):
        return torch.device("cpu") if x == "cuda" else torch.device(x, *a)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from PIL import Image
    d = tmp_path_factory.mktemp("savestop")
    img = d / "img"
    img.mkdir()
    rng = np.random.default_rng(0)
    for i in range(8):
        h = 64 if i % 2 == 0 else 96
        Image.fromarray(rng.integers(0, 255, (h, 64, 3), dtype=np.uint8)).save(img / f"p{i}.png")
        (img / f"p{i}.txt").write_text(f"a photo number {i}")
    cfg = d / "ds.toml"
    cfg.write_text(f'[general]\nresolution = [64, 64]\ncaption_extension = ".txt"\nbatch_size = 2\n'
                   f'enable_bucket = true\n\n[[datasets]]\nimage_directory = "{img.as_posix()}"\n'
                   f'cache_directory = "{(d / "cache").as_posix()}"\nnum_repeats = 1\n')
    return d, str(cfg)


@pytest.fixture
def tiny(monkeypatch, data):
    """The trainer on a tiny CPU family (zimage's description, this file's driver), the cache made."""
    import fizgig.families.cache as C
    import fizgig.families.train as T
    desc = dataclasses.replace(registry.FAMILIES["zimage"], driver=f"{__name__}:TinyDriver", supports_batch=True)
    monkeypatch.setitem(registry.FAMILIES, "zimage", desc)
    monkeypatch.setattr(T, "torch", _CpuTorch("torch"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a, **k: 0)
    d, cfg = data
    if not (d / "cache").exists():
        for stage in ("latents", "text"):
            monkeypatch.setattr(sys, "argv", ["cache", "--family", "zimage", "--stage", stage, "--dataset_config",
                                              cfg, "--model", "x", "--device", "cpu"])
            C.main()
    return T, cfg


def _run(T, cfg, out, press_at=0, resume=None, **kw):
    """One training run of 3 epochs; press_at: Pause, then Pause again, right after that step (as the UI does)."""
    if press_at:
        class _Presses(T.StepLog):
            def add(self, step, *a, **k):
                super().add(step, *a, **k)
                if step == press_at:
                    open(os.path.join(str(out), ".pause_requested"), "w").write("pause\n")
                    open(os.path.join(str(out), ".savestop_now"), "w").close()
        T_StepLog = T.StepLog
        T.StepLog = _Presses
    args = dict(network_dim=4, network_alpha=4, learning_rate=1e-3, lr_scheduler="cosine", max_train_epochs=3,
                save_state=True, compile_blocks="off", seed=7)
    args.update(kw)
    try:
        T.train_family("zimage", "x", cfg, str(out), "t", resume_state_dir=resume, **args)
        return "done"
    except SystemExit as e:
        assert e.code in (0, None)
        return "stopped"
    finally:
        if press_at:
            T.StepLog = T_StepLog


def _losses(out):
    with open(os.path.join(str(out), ".fizgig_loss_t.jsonl"), encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if '"l"' in ln]


def test_second_press_saves_a_restore_point_then_resume_finishes_the_epoch(tiny, tmp_path, caplog):
    T, cfg = tiny
    full = tmp_path / "full"
    assert _run(T, cfg, full) == "done"
    out = tmp_path / "run"
    caplog.set_level("INFO")
    assert _run(T, cfg, out, press_at=SPE + 2) == "stopped"          # 2 steps into epoch 2
    sd = out / "t-000002-state"
    assert {"lora.safetensors", "optimizer.pt", "rng.pt", "training_state.json"} <= set(os.listdir(sd))
    meta = json.loads((sd / "training_state.json").read_text())
    assert meta["epoch"] == 1 and meta["global_step"] == SPE + 2
    mid = meta["mid_epoch"]
    assert mid["epoch"] == 2 and mid["steps"] == 2 and len(mid["batches"]) == 2 and mid["scheduler_steps"] == 6
    assert all(isinstance(loss, float) for _k, loss in mid["batches"])
    assert "Restore point saved at step 6" in caplog.text
    assert not (out / "t-000002.safetensors").exists()             # it stopped inside epoch 2
    assert [r["s"] for r in _losses(out)] == list(range(1, 7))
    from fizgig.families import launch
    assert launch.latest_state(str(out), "t") == str(sd)          # what Resume picks

    caplog.clear()
    os.remove(out / ".pause_requested")
    assert _run(T, cfg, out, resume=str(sd)) == "done"
    assert "matched none" not in caplog.text                        # the epoch's own batches, the trained ones passed
    log = _losses(out)
    assert [r["s"] for r in log] == list(range(1, 3 * SPE + 1))   # 12 steps, as the run without a stop
    assert [r["e"] for r in log] == [1] * 4 + [2] * 4 + [3] * 4
    # the LR schedule carries on where it stood: the same rates as the run that never stopped
    assert [r["lr"] for r in log] == [r["lr"] for r in _losses(full)]
    meta = json.loads((sd / "training_state.json").read_text())   # epoch 2's end rewrote the folder: a whole epoch
    assert meta["epoch"] == 2 and "mid_epoch" not in meta
    assert (out / "t.safetensors").exists()


def test_with_accumulation_it_waits_for_the_optimizer_step(tiny, tmp_path):
    T, cfg = tiny
    out = tmp_path / "acc"
    assert _run(T, cfg, out, press_at=SPE + 1, gradient_accumulation_steps=2) == "stopped"
    meta = json.loads((out / "t-000002-state" / "training_state.json").read_text())
    assert meta["global_step"] == SPE + 2 and meta["mid_epoch"]["scheduler_steps"] == 3
    os.remove(out / ".pause_requested")
    assert _run(T, cfg, out, resume=str(out / "t-000002-state"), gradient_accumulation_steps=2) == "done"
    assert [r["s"] for r in _losses(out)] == list(range(1, 13))


def test_at_an_epochs_last_step_resume_does_its_epoch_end(tiny, tmp_path, caplog):
    T, cfg = tiny
    out = tmp_path / "end"
    assert _run(T, cfg, out, press_at=2 * SPE, ema_decay=0.9) == "stopped"
    mid = json.loads((out / "t-000002-state" / "training_state.json").read_text())["mid_epoch"]
    assert mid["steps"] == SPE
    os.remove(out / ".pause_requested")
    caplog.set_level("INFO")
    assert _run(T, cfg, out, resume=str(out / "t-000002-state"), ema_decay=0.9) == "done"
    assert (out / "t-000002.safetensors").exists()                 # epoch 2's save, after the resume
    avg = sum(loss for _k, loss in mid["batches"]) / SPE           # its average from the saved losses
    assert f"epoch 2/3  avr_loss={avg:.4f}" in caplog.text


def test_a_failed_save_keeps_training_and_pauses_at_the_epochs_end(tiny, tmp_path, monkeypatch, caplog):
    T, cfg = tiny
    real = T._save_state

    def full_disk(*a, folder_epoch=None, **k):
        if folder_epoch is not None:
            raise OSError(28, "No space left on device")
        return real(*a, folder_epoch=folder_epoch, **k)
    monkeypatch.setattr(T, "_save_state", full_disk)
    monkeypatch.setattr(T.time, "sleep", lambda s: None)
    out = tmp_path / "fail"
    caplog.set_level("INFO")
    assert _run(T, cfg, out, press_at=SPE + 2) == "stopped"
    assert "Save & stop: no restore point - saving it failed" in caplog.text
    assert "No space left" in (out / ".savestop_failed").read_text()
    meta = json.loads((out / "t-000002-state" / "training_state.json").read_text())
    assert meta["epoch"] == 2 and "mid_epoch" not in meta           # the first press's pause, at the epoch's end
    assert [r["s"] for r in _losses(out)] == list(range(1, 9))


# ---- the server and the page ------------------------------------------------------------------------------------
def test_server_second_press_and_its_failure_note(tmp_path, monkeypatch):
    from fizgig.web import server
    monkeypatch.setattr(server, "append_log", lambda s: None)
    monkeypatch.setitem(server._run, "save_stop", False)
    monkeypatch.setitem(server._run, "save_stop_note", "")
    out = str(tmp_path)
    assert "error" in server._save_stop(out, "Caching latents", "zimage", {}, False)
    assert not os.path.exists(os.path.join(out, ".savestop_now"))
    assert server._save_stop(out, "Training", "zimage", {}, False) == {"ok": True}
    assert os.path.exists(os.path.join(out, ".savestop_now")) and server._run["save_stop"]
    r = {"save_stop": True, "output_dir": out}
    server._save_stop_check(r)
    assert r["save_stop"]                                           # nothing failed yet
    with open(os.path.join(out, ".savestop_failed"), "w") as f:
        f.write("saving it failed: disk full")
    server._save_stop_check(r)
    assert not r["save_stop"] and r["save_stop_note"] == "saving it failed: disk full"
    assert not server._run["save_stop"] and not os.path.exists(os.path.join(out, ".savestop_failed"))


def test_a_restore_points_epochs_done(tmp_path):
    from fizgig.web import server
    d = tmp_path / "t-000003-state"
    d.mkdir()
    (d / "training_state.json").write_text(json.dumps({"epoch": 2, "mid_epoch": {"epoch": 3}}))
    assert server.state_epochs_done(str(d), 3) == 2
    (d / "training_state.json").write_text(json.dumps({"epoch": 3}))
    assert server.state_epochs_done(str(d), 3) == 3


def test_wiring():
    src = open(os.path.join(ROOT, "src/fizgig/web/server.py"), encoding="utf-8").read()
    assert '".savestop_now", ".savestop_failed"' in src                # a new run never takes an old request
    assert "return self.json(_save_stop(out, stage, fam, vals, saving))" in src
    js = open(os.path.join(ROOT, "src/fizgig/web/static/app.js"), encoding="utf-8").read()
    assert '"Save & stop"' in js and "Saving restore point, then stopping…" in js
    tr = open(os.path.join(ROOT, "src/fizgig/families/train.py"), encoding="utf-8").read()
    # no share round starts on a resumed part-epoch, and none is cut short mid-round
    assert "not (resume_left and epoch == start_epoch)" in tr
    assert "if share_now[0] is not None or only is not None or catching[0]:" in tr
