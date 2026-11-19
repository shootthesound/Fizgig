"""A helper's preview worker: another computer's training run, rendered here. It loads the run's model the way the
trainer does (same family, base precision, Turbo LoRA, SFT look, context LoRA, the LoRA's own shape), keeps it
loaded, and renders each round it is handed from that epoch's LoRA with the trainer's own preview code, so the
pictures are the ones the training GPU would have made.

    python -m fizgig.families.preview_worker --setup <setup.json> --jobs <dir>

setup.json: {family, options, precision, network: {dim, alpha, type, factor, blocks}, paths: {dit, vae, text_encoder,
speed_lora, look_lora, context}, context_strength}. A job is <dir>/<id>.job.json, the round.json the trainer wrote
plus "lora" (its file) and "out" (where the pictures go); the worker claims it by renaming it to .working, and
answers with <id>.done.json ({ok, files, seconds} or {ok: false, error}). <dir>/worker.json says what it is doing.
With no job for --idle seconds it exits and the GPU is free again.
"""
import argparse
import glob
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from fizgig.families import quant  # noqa: E402
from fizgig.families.lora import FamilyLoRA  # noqa: E402
from fizgig.families.registry import get as get_family  # noqa: E402
from fizgig.families.train import (CONTEXT, LOOK, SPEED, TRAINABLE_PREVIEW, _encode_override,  # noqa: E402
                                   _render_previews, _small_card_previews)

logger = logging.getLogger("fizgig.preview_worker")


def _write(path, data):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(path + ".tmp", path)


def plan(desc, driver, setup, precision):
    """(precision, blocks_to_swap, why) for the base this helper loads. A model that plans itself (description.own_plan:
    MiniMax H3) is asked as the Batch size note asks it (driver.plan_pictures, web/batch_plan.py), at the run's
    precision, then the trainer's preview caps apply (_plan_previews: 768x640 / 22 frames on a 16 GB card). The shared
    planner has no figures for it: an int8 run's helper loaded int8 with nothing streamed and ran out of memory on a
    16 GB card (#372, @Nessa2026691). The others plan from their measured figures, as before."""
    if not desc.own_plan:
        return quant.plan(desc, driver, precision, -1)
    paths, n = setup["paths"], setup.get("network") or {}
    run = {"dit_path": paths["dit"], "network_type": n.get("type") or "lora", "network_dim": int(n.get("dim") or 16),
           "lokr_factor": int(n.get("factor") or 8), "context_lora_path": paths.get("context") or ""}
    mode, swap = driver.plan_pictures(precision, -1, megapixels=1.0, pictures=1, run=run)
    if hasattr(driver, "_plan_previews"):
        driver._plan_previews(swap)
    return mode, swap, "the model's own plan for this card"


class Worker:
    def __init__(self, setup, jobs):
        self.jobs = jobs
        self.state = os.path.join(jobs, "worker.json")
        self.desc = get_family(setup["family"])
        if self.desc is None:
            raise RuntimeError(f"Fizgig here doesn't know the model family {setup['family']!r}")
        self.driver = self.desc.load_driver()
        self.driver.set_options(dict(setup.get("options") or {}))
        self.device = torch.device("cuda")
        paths = setup["paths"]
        self.te_path = paths["text_encoder"]
        self.ck = setup.get("checkpoint") or None
        if self.ck:
            self._load_checkpoint(setup, paths)
            return
        precision, swap, why = plan(self.desc, self.driver, setup, setup.get("precision") or "auto")
        self.say("loading", f"Loading {self.desc.display_name} ({precision}" + (f", {swap} blocks swapped" if swap else "")
                 + ")")
        logger.info(f"[helper] {self.desc.display_name} at {precision}, block swap {swap} ({why})")
        self.dit, self.swapped = quant.load_base(self.driver, paths["dit"], self.device, precision, swap)
        self.precision = precision
        self.net = FamilyLoRA(self.dit, self.driver, device=self.device)
        n = setup["network"]
        self.net.add_trainable(n["dim"], n["alpha"], blocks=set(n["blocks"]) if n.get("blocks") else None,
                               kind=n.get("type") or "lora", factor=n.get("factor") or 8)
        if paths.get("context"):
            self.net.add_file(paths["context"], CONTEXT, float(setup.get("context_strength") or 1.0))
            self.driver.frozen_file_added(self.dit, paths["context"], float(setup.get("context_strength") or 1.0),
                                          "context")
        self.speed = self.desc.preview_speed() if paths.get("speed_lora") else None
        if self.speed is not None:
            self.net.add_file(paths["speed_lora"], SPEED, self.speed.strength)
            self.driver.frozen_file_added(self.dit, paths["speed_lora"], self.speed.strength, "speed")
            self.net.set_enabled(SPEED, False)
            self.net.move_adapter(SPEED, "cpu")
        self.look = bool(paths.get("look_lora"))
        if self.look:
            st = float(self.desc.preview_look_strength or 1.0)
            self.net.add_file(paths["look_lora"], LOOK, st)
            self.driver.frozen_file_added(self.dit, paths["look_lora"], st, "speed")
            self.net.set_enabled(LOOK, False)
            self.net.move_adapter(LOOK, "cpu")
        self.vae = self.driver.load_vae(paths["vae"], self.device)
        self.lowmem = _small_card_previews()
        self.enc_key, self.enc, self.neg_key, self.neg = None, None, None, None
        self.say("ready", "Ready")

    def _load_checkpoint(self, setup, paths):
        """Rounds on the family's preview checkpoint (MOSS-Image's SFT, Klein's Distilled), as the trainer's
        _CheckpointPreviews renders them: the checkpoint frozen, each epoch's LoRA a frozen file on it (#339)."""
        path = self.ck.get("path")
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"The preview checkpoint isn't here ({path or 'no file set in Preferences'})")
        self.say("loading", f"Loading {os.path.basename(path)}")
        self.dit, self.swapped = self.driver.load_preview_checkpoint(path, self.device, int8=bool(self.ck.get("int8")))
        logger.info(f"[helper] {os.path.basename(path)} (preview checkpoint)"
                    + (f", {self.swapped} blocks streamed" if self.swapped else ""))
        self.precision = "checkpoint"
        self.net = FamilyLoRA(self.dit, self.driver, device=self.device)
        if paths.get("context"):
            self.net.add_file(paths["context"], CONTEXT, float(setup.get("context_strength") or 1.0))
        self.speed_path = paths.get("speed_lora")
        self.speed = self.look = None
        self.vae = self.driver.load_vae(paths["vae"], self.device)
        self.lowmem = False
        self.enc_key, self.enc, self.neg_key, self.neg = None, None, None, None
        self.say("ready", "Ready")

    def _run_checkpoint(self, job):
        import dataclasses
        t0 = time.time()
        if self.net.has(TRAINABLE_PREVIEW):
            self.net.swap_file(TRAINABLE_PREVIEW, job["lora"])
        else:
            self.net.add_file(job["lora"], TRAINABLE_PREVIEW)
        conds, neg = self._conds(job)
        turbo = float(job.get("turbo") or 0)
        sp = self.desc.preview_speed()
        if turbo > 0 and sp is not None and self.speed_path:    # a checkpoint that takes the speed LoRA (Kromoss)
            if not self.net.has(SPEED):
                self.net.add_file(self.speed_path, SPEED, turbo)
                self.driver.frozen_file_added(self.dit, self.speed_path, turbo, "speed")
            self.net.set_strength(SPEED, turbo)
            settings = sp.settings
        else:
            if self.net.has(SPEED):
                self.net.set_strength(SPEED, 0.0)
            settings = dataclasses.replace(self.desc.preview_checkpoint_sampling, cfg=float(job["cfg"]))
        with torch.no_grad():
            files = _render_previews(self.driver, self.dit, self.net, self.vae, conds, job["out"], int(job["epoch"]),
                                     output_name=job["output_name"], steps=int(job["steps"]), cfg=float(job["cfg"]),
                                     neg=neg, width=int(job["width"]), height=int(job["height"]),
                                     seed=int(job["seed"]), speed=settings, swapped=bool(self.swapped))
        return {"ok": True, "files": [os.path.basename(p) for p in files], "seconds": round(time.time() - t0, 1),
                "precision": self.precision}

    def say(self, state, text, **extra):
        _write(self.state, {"state": state, "text": text, "pid": os.getpid(), "at": time.time(), **extra})

    def _conds(self, job):
        if job.get("conds_file"):          # the run's own encodings, sent with the round (no text encoder here)
            d = torch.load(job["conds_file"], map_location="cpu", weights_only=True)
            return d["conds"], d.get("neg")
        prompts = list(job["prompts"])
        if self.enc_key != prompts:
            self.enc = _encode_override(self.driver, self.te_path, prompts, self.dit, self.device,
                                        parkable=not self.swapped)
            self.enc_key = prompts
        neg = None
        if job.get("negative") is not None:
            if self.neg_key != job["negative"]:
                self.neg = _encode_override(self.driver, self.te_path, [job["negative"]], self.dit, self.device,
                                            parkable=not self.swapped)[0]
                self.neg_key = job["negative"]
            neg = self.neg
        return self.enc, neg

    def run(self, job):
        if self.desc.preview_keyframes:      # First / Last frame (#376): this round's, or none (a worker outlives runs)
            self.driver.set_preview_keyframes(self.vae, {k: p for k, p in (job.get("keyframes") or {}).items()
                                                         if p and os.path.isfile(p)})
        if self.ck:
            return self._run_checkpoint(job)
        t0 = time.time()
        n = self.net.load_trainable(job["lora"])
        if n == 0:
            raise RuntimeError("The epoch's LoRA matched none of this model's layers")
        conds, neg = self._conds(job)
        speed = None
        if self.speed is not None and float(job.get("turbo") or 0) > 0:
            self.net.set_strength(SPEED, float(job["turbo"]))
            speed = self.speed.settings
        look = self.look and float(job.get("look") or 0) > 0
        if look:
            self.net.set_strength(LOOK, float(job["look"]))
        with torch.no_grad():
            files = _render_previews(self.driver, self.dit, self.net, self.vae, conds, job["out"], int(job["epoch"]),
                                     output_name=job["output_name"], steps=int(job["steps"]), cfg=float(job["cfg"]),
                                     neg=neg, width=int(job["width"]), height=int(job["height"]),
                                     seed=int(job["seed"]), speed=speed, lowmem=self.lowmem,
                                     swapped=bool(self.swapped), strengths=tuple(job.get("strengths") or ()),
                                     look=look)
        return {"ok": True, "files": [os.path.basename(p) for p in files], "seconds": round(time.time() - t0, 1),
                "precision": self.precision}

    def loop(self, idle):
        last = time.time()
        while True:
            todo = sorted(glob.glob(os.path.join(glob.escape(self.jobs), "*.job.json")))
            if not todo:
                if os.path.exists(os.path.join(self.jobs, "stop")) or time.time() - last > idle:
                    self.say("stopped", "Stopped")
                    return
                time.sleep(0.5)
                continue
            path = todo[0]
            work = path[:-len(".job.json")] + ".working"
            try:
                os.replace(path, work)
            except OSError:
                continue
            jid = os.path.basename(path)[:-len(".job.json")]
            try:
                with open(work, encoding="utf-8") as f:
                    job = json.load(f)
                self.say("rendering", f"Rendering epoch {job.get('epoch')}", job=jid, epoch=job.get("epoch"))
                res = self.run(job)
            except Exception as e:
                logger.exception(f"[helper] job {jid} failed")
                res = {"ok": False, "error": str(e) or type(e).__name__}
            _write(os.path.join(self.jobs, jid + ".done.json"), res)
            try:
                os.remove(work)
            except OSError:
                pass
            self.say("ready", "Ready")
            last = time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--setup", required=True)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--idle", type=float, default=600.0, help="seconds without a job before the worker exits")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    with open(a.setup, encoding="utf-8") as f:
        setup = json.load(f)
    os.makedirs(a.jobs, exist_ok=True)
    try:
        w = Worker(setup, a.jobs)
    except Exception as e:
        logger.exception("[helper] could not load the model")
        _write(os.path.join(a.jobs, "worker.json"), {"state": "failed", "text": str(e) or type(e).__name__,
                                                       "pid": os.getpid(), "at": time.time()})
        sys.exit(1)
    w.loop(a.idle)


if __name__ == "__main__":
    main()
