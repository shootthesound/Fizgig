"""Standard-layer LoRA trainer for any described family (fizgig.families), driven through the family's driver.

    python src/fizgig/families/train.py --family qwen_image21 --dit ... --dataset_config ... --output_dir ...

The loop is family-agnostic: Fizgig dataset + bucketing (batch 1), frozen adapters (the family's training adapter,
off for previews; a context LoRA, on for previews; neither in saves), Adaptive LR or a step scheduler, gradient
clipping, optional EMA, per-epoch checkpoints in the family's LoRA key format with SAI metadata, resumable state
dirs and the GUI's pause contract. Everything model-specific (loading, noise/target/timesteps, forward, sampling,
decoding, which Linears a LoRA wraps) comes from the driver.

Why training adapters exist: on Qwen Image 2.1 a plain LoRA collapsed at lr 5e-4 and wobbled at 1e-4; with the same
Adaptive LR a no-adapter run fell to 37 likeness at step 2000 while Fizgig's adapter held 74-77 (26 Sep 2026).
"""
import argparse
import dataclasses
import datetime
import gc
import glob
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
import uuid
from multiprocessing import Value

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.dataset.config import (BlueprintGenerator, ConfigSanitizer,  # noqa: E402
                                   generate_dataset_group_by_blueprint, load_user_config)
from fizgig.families import quant  # noqa: E402
from fizgig.families.lora import FamilyLoRA  # noqa: E402
from fizgig.families.registry import get as get_family  # noqa: E402
from fizgig.training.adaptive_lr import AdaptiveLR  # noqa: E402
from fizgig.training.resume_lr import resume_lr  # noqa: E402
from fizgig.training.metadata import (build_metadata, latest_sample_image, refresh_checkpoint_thumbnail,  # noqa: E402
                                      resolve_title, sample_for_epoch, thumbnail_data_uri)
from fizgig.training.train_utils import LossRecorder, prune_state_dirs, validate_output_name  # noqa: E402
from fizgig.training.step_log import StepLog, take_request  # noqa: E402
from fizgig.utils.upright import open_upright

logger = logging.getLogger(__name__)

ADAPTER = "training_adapter"
CONTEXT = "context"
SPEED = "speed_lora"
LOOK = "preview_look"         # a LoRA for previews only (MOSS-Image's SFT look LoRA), on CPU between previews
TRAINABLE_PREVIEW = "epoch_lora"         # the epoch's LoRA, frozen on a preview checkpoint


class _Collator:
    def __init__(self, shared_epoch, dataset):
        self.shared_epoch = shared_epoch
        self.dataset = dataset

    def __call__(self, examples):
        wi = torch.utils.data.get_worker_info()
        ds = wi.dataset if wi is not None else self.dataset
        ds.set_current_epoch(self.shared_epoch.value)
        return examples[0]


def _step_scheduler(optimizer, kind, warmup, total, cycles=1, power=1.0):
    def f(s):
        if warmup and s < warmup:
            return (s + 1) / warmup
        if kind in ("constant", "constant_with_warmup"):
            return 1.0
        prog = min(1.0, (s - warmup) / max(1, total - warmup))
        if kind == "cosine":
            return 0.5 * (1 + math.cos(math.pi * prog))
        if kind == "cosine_with_restarts":
            return 0.5 * (1 + math.cos(math.pi * ((prog * cycles) % 1.0)))
        if kind == "linear":
            return 1.0 - prog
        if kind == "polynomial":
            return (1.0 - prog) ** power
        return 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def _save_state(output_dir, output_name, net, optimizer, *, epoch, global_step, arch_id, extra=None, ema=None):
    state_dir = os.path.join(output_dir, f"{output_name}-{epoch:06d}-state")
    os.makedirs(state_dir, exist_ok=True)
    net.save(os.path.join(state_dir, "lora.safetensors"), dtype=torch.float32)
    torch.save(optimizer.state_dict(), os.path.join(state_dir, "optimizer.pt"))
    if ema is not None:
        torch.save(ema.state_dict(), os.path.join(state_dir, "ema.pt"))
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng, os.path.join(state_dir, "rng.pt"))
    with open(os.path.join(state_dir, "training_state.json"), "w", encoding="utf-8") as f:   # commit marker, last
        json.dump({"epoch": epoch, "global_step": global_step, "architecture": arch_id,
                   "optimizer": optimizer.__class__.__name__, **(extra or {})}, f)
    logger.info(f"[state] saved -> {state_dir}")
    return state_dir


def _optimizer_family_groups(desc, net, lr):
    """The LoRA's trainable parameters split by the description's optimizer_families (first match on the dotted
    module name; the rest are "other"), as optimizer param groups - or None when the family declares none or
    everything lands in one group."""
    if not desc.optimizer_families:
        return None
    from fizgig.families.lora import TRAINABLE
    buckets, counts = {}, {}
    for full, w in net.wrapped.items():
        if TRAINABLE not in w.adapters:
            continue
        fam = next((name for name, needles in desc.optimizer_families if any(n in full for n in needles)), "other")
        ps = [p for p in w.adapters[TRAINABLE].parameters() if p.requires_grad]
        if ps:
            buckets.setdefault(fam, []).extend(ps)
            counts[fam] = counts.get(fam, 0) + 1
    if len(buckets) < 2:
        return None
    order = [n for n, _ in desc.optimizer_families] + ["other"]
    return [{"params": buckets[f], "lr": float(lr), "family": f, "modules": counts[f]} for f in order if f in buckets]


def _legacy_perm(driver, net):
    """{old parameter index: family LoRA parameter index} for an older trainer's state (driver.legacy_state_order), or
    None when the orders already agree. Only modules that train here count, in the old order, down then up."""
    order = driver.legacy_state_order(net.dit)
    if order is None:
        return None
    from fizgig.families.lora import TRAINABLE
    here, i = {}, 0
    for full, w in net.wrapped.items():
        if TRAINABLE in w.adapters:
            for part, _p in enumerate(w.adapters[TRAINABLE].parameters()):
                here[(full, part)] = i
                i += 1
    old = [(m, part) for m in order if (m, 0) in here for part in range(2) if (m, part) in here]
    if len(old) != len(here):
        raise RuntimeError(f"[resume] the old state's layout ({len(old)} tensors) does not match this LoRA "
                           f"({len(here)}) - different rank, blocks or network type?")
    return {k: here[key] for k, key in enumerate(old)}


def _load_state(state_dir, net, optimizer, device, arch, untagged_own=False, driver=None):
    """(epoch, global step, training_state.json) of a saved state, its LoRA weights and RNG restored. The optimizer
    (and, by the caller, the EMA) is restored only from a state this family wrote: another trainer's (the original
    Krea 2's) orders its parameters differently, so its moments would land on the wrong tensors - that state goes on
    from its weights with a fresh optimizer.
    A pause saved by an accelerate trainer (the old Klein trainer: model.safetensors holding the LoRA, the epoch in the
    folder name, adaptive_lr_state.json beside it) goes on the same way; its global step is None (the caller counts
    it from the epoch)."""
    legacy = os.path.join(state_dir, "model.safetensors")
    if not os.path.isfile(os.path.join(state_dir, "training_state.json")) and os.path.isfile(legacy):
        m = re.search(r"-(\d{6})-state$", os.path.basename(os.path.normpath(state_dir)))
        if not m:
            raise RuntimeError(f"[resume] {state_dir}: no epoch in the folder name ('<lora name>-000012-state')")
        if net.load_trainable(legacy) == 0:
            raise RuntimeError(f"[resume] {state_dir} matched none of this LoRA's modules - different rank or "
                               f"target modules?")
        meta = {"epoch": int(m.group(1)), "own_state": False}
        side = os.path.join(state_dir, "adaptive_lr_state.json")
        if os.path.isfile(side):
            with open(side, encoding="utf-8") as f:
                meta["adaptive_lr_state"] = json.load(f)
        logger.info("[resume] this pause was saved by the old trainer: continuing from its LoRA weights and epoch"
                    + (" (and its adaptive LR state)" if "adaptive_lr_state" in meta else "")
                    + ", with a fresh optimizer")
        return meta["epoch"], None, meta
    for need in ("lora.safetensors", "optimizer.pt", "training_state.json"):
        if not os.path.isfile(os.path.join(state_dir, need)):
            raise RuntimeError(f"[resume] {state_dir} is not a saved training state (missing {need}). Pick the "
                               f"folder named like '<lora name>-000012-state'.")
    if net.load_trainable(os.path.join(state_dir, "lora.safetensors")) == 0:
        raise RuntimeError(f"[resume] {state_dir} matched none of this LoRA's modules - different rank or "
                           f"target modules?")
    with open(os.path.join(state_dir, "training_state.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["own_state"] = meta.get("architecture") == arch or (untagged_own and "architecture" not in meta)
    if meta["own_state"]:
        sd = torch.load(os.path.join(state_dir, "optimizer.pt"), map_location=device)
        perm = _legacy_perm(driver, net) if (driver is not None and "architecture" not in meta) else None
        if perm is not None:            # an older trainer's order: each moment back onto its own tensor
            sd["state"] = {perm[int(k)]: v for k, v in sd["state"].items()}
            meta["legacy_perm"] = perm
            logger.info("[resume] an older trainer's state: optimizer moments remapped into this LoRA's order")
        # another Optimizer Type than the run trained with (#246): its saved moments don't fit this one, so the LoRA's
        # weights carry on with a fresh optimizer (a state from before the type was saved: as its groups tell)
        was, now = meta.get("optimizer"), optimizer.__class__.__name__
        if was and was != now:
            why = f"it trained with {was}, and this resume uses {now}"
        elif len(sd.get("param_groups") or []) != len(optimizer.param_groups):
            why = "it trained with another Optimizer Type than this resume uses"
        else:
            why = ""
        if why:
            meta["fresh_optimizer"] = True
            logger.info(f"[resume] {why}: continuing from the state's LoRA weights and epoch, with a fresh "
                        f"optimizer (it settles within a few steps). To carry the optimizer on too, resume with "
                        f"{was or 'the run' + chr(39) + 's own'} Optimizer Type.")
        else:
            optimizer.load_state_dict(sd)
    else:
        logger.info("[resume] this state was saved by another trainer: continuing from its LoRA weights, with a "
                    "fresh optimizer and weight average (they settle within a few steps)")
    rng_path = os.path.join(state_dir, "rng.pt")
    if os.path.exists(rng_path):
        rng = torch.load(rng_path)
        torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    return int(meta.get("epoch", 0)), int(meta.get("global_step", 0)), meta


def _preview_vram(tag, reset_peak=False):
    """One line of VRAM state at a preview waypoint (the same lines Krea 2 logs, #123)."""
    try:
        if not torch.cuda.is_available():
            return
        if reset_peak:
            torch.cuda.reset_peak_memory_stats()
        a = torch.cuda.memory_allocated() / 1024 ** 3
        r = torch.cuda.memory_reserved() / 1024 ** 3
        pk = torch.cuda.max_memory_reserved() / 1024 ** 3
        f = torch.cuda.mem_get_info()[0] / 1024 ** 3
        logger.info(f"[preview-vram] {tag}: allocated {a:.2f} GB, reserved {r:.2f} GB (peak {pk:.2f} GB), "
                    f"free {f:.2f} GB")
    except Exception:
        pass


def _small_card_previews():
    """Cards under 20 GB get the low-memory preview treatment (Krea 2's rule): the training DiT parks on CPU for
    the VAE decode and the preview canvas caps at 768 px. FIZGIG_PREVIEW_LOWMEM=1/0 forces it; FIZGIG_SIM_VRAM_GB
    simulates a smaller card."""
    ov = os.environ.get("FIZGIG_PREVIEW_LOWMEM", "").strip()
    if ov in ("0", "1"):
        return ov == "1"
    try:
        if not torch.cuda.is_available():
            return False
        sim = os.environ.get("FIZGIG_SIM_VRAM_GB", "").strip()
        total = float(sim) if sim else torch.cuda.get_device_properties(0).total_memory / 1e9
        return total < 20.0
    except Exception:
        return False


def _record_prompts(sample_dir, output_name, epoch, prompts):
    """The prompts an epoch's previews used (<sample_dir>/.prompts_<name>.json, by epoch), so the gallery shows each
    preview's own prompt after a mid-run change, not the run's first ones."""
    path = os.path.join(sample_dir, f".prompts_{output_name}.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    data[str(epoch)] = [str(p) for p in prompts]
    try:
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(path + ".tmp", path)
    except OSError as e:
        logger.warning(f"[sample] could not record the preview prompts ({e})")


def _read_sample_override(output_dir):
    """The live sample override (<output_dir>/.sample_override.json, written by the desktop's status-bar panel or the
    web UI's Samples tab): {prompts, prompt (the first), seed, width, height, steps, cfg, turbo, every, until, after}
    while a prompt is set, else None (every / until / after: the preview cadence, see preview_due). A "prompts" list (the web's, every line of its box) or a single "prompt"; the reference image field
    is ignored. steps / cfg / turbo (the Turbo LoRA's strength) are None when the override doesn't set them."""
    path = os.path.join(output_dir, ".sample_override.json")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        prompts = d.get("prompts") if isinstance(d.get("prompts"), list) else [d.get("prompt", "")]
        prompts = [str(p).strip() for p in prompts if str(p or "").strip() and not str(p).strip().startswith("#")]
        if prompts:
            def num(k, kind):            # one bad value drops only itself, never the prompts
                try:
                    return kind(float(d[k])) if d.get(k) not in (None, "") else None
                except (TypeError, ValueError):
                    return None
            steps, cfg, turbo = num("steps", int), num("cfg", float), num("turbo", float)
            look = d.get("look") if isinstance(d.get("look"), str) else None   # the look LoRA's path, "" = off
            # the preview-model tick (Qwen's Turbo, Klein's Distilled): its file's path when ticked, "" = the training
            # model, None = not sent (the run keeps what it started with)
            ck = d.get("checkpoint") if isinstance(d.get("checkpoint"), str) else None
            return {"prompts": prompts, "prompt": prompts[0], "seed": int(d.get("seed", 1234)), "checkpoint": ck,
                    "width": int(d.get("width", 1024)), "height": int(d.get("height", 1024)),
                    "steps": steps if steps and steps > 0 else None, "cfg": cfg if cfg and cfg > 0 else None,
                    "turbo": turbo if turbo is not None and turbo >= 0 else None,
                    "look": look, "look_strength": num("look_strength", float),
                    **{k: (lambda v: v if v is not None and v >= 0 else None)(num(k, int))
                       for k in ("every", "until", "after")}}
    except (OSError, ValueError, TypeError):
        pass
    return None


HELPER_PREVIEWS = ".helper_previews.json"   # written by the web UI while a helper renders this run's previews
HELPER_PREVIEWS_OFF = ".helper_previews_off.json"   # the trainer's: what keeps this run's previews on this GPU
HELPER_ROUNDS = ".helper_rounds"             # <output_dir>/.helper_rounds/<epoch>/: a round handed to the helper


HELPER_SHARE = ".helper_share.json"   # written by the web UI while a helper trains a share of each epoch's images
SHARE_DIR = ".helper_share"           # <output_dir>/.helper_share/: setup.json, <epoch>/start, job, theirs, lead


def _helper_share(output_dir, output_name=""):
    """The helper that trains a share of each epoch ({name, frac}), while the web UI says it's there for this run
    (a note left by the run before, in the same folder, names that run)."""
    try:
        with open(os.path.join(output_dir, HELPER_SHARE), encoding="utf-8") as f:
            d = json.load(f)
        ok = (d.get("on") and time.time() - float(d.get("at", 0)) < 60 and 0 < float(d.get("frac", 0)) < 1
              and (not output_name or d.get("run") == output_name))
        return d if ok else None
    except (OSError, ValueError, TypeError):
        return None


HELPER_SHARE_WAIT = ".helper_share_wait.json"   # the web UI's: this launch's helper is getting ready


def _share_wait_note(output_dir, output_name, launch):
    """The web UI's hold on this launch's first epoch while its helper gets ready ({name, text}), while it keeps it
    fresh, else None."""
    try:
        with open(os.path.join(output_dir, HELPER_SHARE_WAIT), encoding="utf-8") as f:
            d = json.load(f)
        ok = (d.get("run") == output_name and d.get("launch") == launch and time.time() - float(d.get("at", 0)) < 20)
        return d if ok else None
    except (OSError, ValueError, TypeError):
        return None


def _share_split(seed, epoch, keys, frac):
    """The helper's images this epoch: ordered by a hash of the seed, the epoch and each file's name (the same on
    both sides, so nothing is listed or sent), and drawn separately from the face crops (any file with "face" in
    its name) and the rest - each GPU gets its share of each, and with two or more face crops neither gets them
    all. Over the run every image trains on both GPUs."""
    import hashlib
    name = lambda k: os.path.basename(str(k))
    out = set()
    for part in ([k for k in keys if "face" in name(k).lower()], [k for k in keys if "face" not in name(k).lower()]):
        order = sorted(part, key=lambda k: hashlib.md5(f"{seed}:{epoch}:{name(k)}".encode()).hexdigest())
        n = round(len(order) * float(frac))
        if len(order) >= 2:
            n = min(max(n, 1), len(order) - 1)
        out.update(order[:n])
    return out


def _write_json(path, data):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f)
    for n in range(20):          # Windows refuses the replace while the web UI has the file open for a moment
        try:
            os.replace(path + ".tmp", path)
            return
        except PermissionError:
            if n == 19:
                raise
            time.sleep(0.05)


class _Compare(Exception):
    """The round handed to the helper also renders here (Check picture)."""


def _helper_previews(output_dir):
    """The helper that renders this run's previews ({name, ...}), while the web UI says it's there (its note is
    refreshed every few seconds; an old one means the helper or the web UI went away and this GPU renders)."""
    try:
        with open(os.path.join(output_dir, HELPER_PREVIEWS), encoding="utf-8") as f:
            d = json.load(f)
        return d if d.get("on") and time.time() - float(d.get("at", 0)) < 60 else None
    except (OSError, ValueError, TypeError):
        return None


def preview_due(done, every, until=0, after=0):
    """Whether epoch `done` previews (or saves): every `every` epochs, or with `until` and `after` both set (How often
    to preview / save, switched on: "then from epoch `until`"), every `every` before epoch `until`, then from it on
    whole multiples of `after`."""
    if until > 0 and after > 0 and done >= until:
        return done % after == 0
    return every > 0 and done % every == 0


SLIDER_LIVE = ".slider_prompts.json"            # written by the web UI while a prompt slider trains
SLIDER_LIVE_APPLIED = ".slider_prompts_applied.json"   # what the trainer took, and at which epoch (the UI reads it)


def _read_slider_live(output_dir):
    """A prompt slider's new prompts, set while it trains: {base, pos, neg, guidance, anchors} or None - the form's own
    three (what the picture is, what each end adds), all needed, as at launch; anchors may be empty."""
    try:
        with open(os.path.join(output_dir, SLIDER_LIVE), encoding="utf-8") as f:
            d = json.load(f)
        out = {"base": str(d.get("base") or "").strip(), "pos": str(d.get("pos") or "").strip(),
               "neg": str(d.get("neg") or "").strip(), "guidance": float(d.get("guidance")),
               "anchors": _anchor_lines(d.get("anchors"))}
        return out if out["base"] and out["pos"] and out["neg"] and out["guidance"] > 0 else None
    except (OSError, ValueError, TypeError):
        return None


def _anchor_lines(raw):
    """A prompt slider's anchor prompts from a list or one prompt per line, blanks dropped."""
    if isinstance(raw, str):
        raw = raw.splitlines()
    return [str(x).strip() for x in (raw or []) if str(x).strip()]


def _encode_override(driver, te_path, prompts, dit, device, parkable=True, references=None):
    """Encode the override's prompts mid-run. The text encoder loads beside the training DiT when it fits; otherwise the
    DiT waits on CPU for the encode (a block-swapped DiT is small enough to stay, and the driver picks a smaller
    encoder when VRAM is short)."""
    from fizgig.families import quant
    te_gb = os.path.getsize(te_path) / 1024 ** 3 if te_path and os.path.exists(te_path) else 0.0
    tok = driver.park_for(dit, device, te_gb + 2.0, "the override prompt's text encoder")
    park = tok is None and parkable and quant.free_vram_gb() < te_gb + 2.0
    if park:
        quant.move(dit, "cpu")
        torch.cuda.empty_cache()
    try:
        if references:
            te = driver.load_reference_text_encoder(te_path, device)
        else:
            te = driver.load_text_encoder(te_path, device)
        try:
            if references:
                return driver.encode_text_with_references(te, list(prompts), [references] * len(prompts))
            return driver.encode_text(te, list(prompts))
        finally:
            driver.unload_text_encoder(te)
            del te
            torch.cuda.empty_cache()
    finally:
        if tok is not None:
            driver.unpark(dit, device, tok)
        if park:
            quant.move(dit, device)


def _cap_canvas(width, height, cap=768):
    long = max(width, height)
    if long <= cap:
        return width, height
    s = cap / long
    return max(64, int(width * s) // 32 * 32), max(64, int(height * s) // 32 * 32)


@torch.no_grad()
def _reference_size(w, h, area):
    """An edit preview's canvas: the reference's aspect at `area` pixels, in multiples of 64 (a training bucket's
    step, so the reference's vision tokens and latents line up as they do in training)."""
    r = w / h
    return max(64, round((area * r) ** 0.5 / 64) * 64), max(64, round((area / r) ** 0.5 / 64) * 64)


def _load_references(paths, width, height):
    """uint8 (H, W, 3) arrays of the preview's reference images and the preview's (width, height): the first (the
    photo the edit is made to) sets the preview's shape at its area; extra pictures (a multi-picture edit's photo_1,
    photo_2 ...) keep their own shape at that area, as they are sized in training."""
    import numpy as np
    from PIL import Image, ImageOps
    from fizgig.utils.upright import open_upright
    imgs = [open_upright(p).convert("RGB") for p in paths]
    size = _reference_size(*imgs[0].size, width * height)
    return [np.array(ImageOps.fit(im, size if i == 0 else _reference_size(*im.size, width * height), Image.LANCZOS))
            for i, im in enumerate(imgs)], size


class _SliderBank(torch.utils.data.Dataset):
    """Prompt-pair sliders train on no dataset: the base model renders a bank of practice latents from the neutral
    prompt, and each step noises one of them."""

    def __init__(self, latents):
        self.latents = latents
        self.datasets = []
        self.num_train_items = len(latents)

    def set_current_epoch(self, epoch):
        pass

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, i):
        return {"latents": self.latents[i]}


# The dial is the point, so a slider preview shows it moving: one prompt, one seed, these strengths side by side.
SLIDER_PREVIEW_MULTIPLIERS = (-1.0, 0.0, 1.0)


def _slider_strip(frames, multipliers):
    from PIL import Image, ImageDraw, ImageFont
    w, h = frames[0].size
    gap = 8
    try:
        font = ImageFont.load_default(size=max(14, w // 40))
    except Exception:
        font = ImageFont.load_default()
    labels = ["strength 0 (slider off)" if m == 0 else f"strength {m:+g}" for m in multipliers]
    # the band fits the text (the font grows with the picture): its full height, descenders too, padded evenly
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    top, bottom = min(probe.textbbox((0, 0), t, font=font)[1] for t in labels),         max(probe.textbbox((0, 0), t, font=font)[3] for t in labels)
    pad = max(6, (bottom - top) // 3)
    band = (bottom - top) + 2 * pad
    strip = Image.new("RGB", (w * len(frames) + gap * (len(frames) - 1), h + band), (16, 16, 16))
    draw = ImageDraw.Draw(strip)
    for k, (im, label) in enumerate(zip(frames, labels)):
        x = k * (w + gap)
        strip.paste(im, (x, band))
        draw.text((x + 8, pad - top), label, fill=(236, 236, 236), font=font)
    return strip


def _pair_slider_caption(user_config):
    """The caption an image-pair slider's photos share (the most common one if they were edited apart)."""
    from collections import Counter
    general = user_config.get("general", {})
    counts = Counter()
    for d in user_config.get("datasets", []):
        folder = d.get("image_directory")
        ext = d.get("caption_extension") or general.get("caption_extension") or ".txt"
        if not folder or not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            if f.endswith(ext):
                with open(os.path.join(folder, f), encoding="utf-8", errors="replace") as fh:
                    text = fh.read().strip()
                if text:
                    counts[text] += 1
    return counts.most_common(1)[0][0] if counts else None


def _prompt_slider_step(driver, dit, net, latents, enc, gen, *, guidance, min_t, max_t, anchor=None,
                        anchor_weight=1.0):
    """Concept Sliders, textual form, on flow matching. With the adapter at 0 the frozen model predicts the
    neutral, positive and negative prompts at one noised practice latent; the adapter then trains at +1 toward
    v_neutral + guidance * (v_pos - v_neg) and at -1 toward the mirror. Each pole is backpropagated before the
    flip (checkpointed blocks recompute at the strength of the moment).

    An anchor (#252, a prompt the slider must leave alone, e.g. "a photo of a man" for a hair-length dial meant for
    women): the frozen model's prediction for it at the same latent is a target at +1 and -1 too, weighted by
    anchor_weight, so the dial learns its change without carrying it onto the anchor. Returns (the detached mean
    slider loss, t, the anchor's mean loss or None)."""
    state = driver.noise_latents(latents, gen, min_t=min_t, max_t=max_t)
    n_c, p_c, g_c = enc
    a_total = None
    try:
        net.set_trainable_multiplier(0.0)
        with torch.no_grad():
            v_n = driver.predict(dit, state, n_c).float()
            delta = float(guidance) * (driver.predict(dit, state, p_c).float() - driver.predict(dit, state, g_c).float())
            v_a = driver.predict(dit, state, anchor).float() if anchor is not None else None
        total = 0.0
        for m in (1.0, -1.0):
            net.set_trainable_multiplier(m)
            loss = F.mse_loss(driver.predict(dit, state, n_c).float(), v_n + m * delta)
            (0.5 * loss).backward()
            total += 0.5 * loss.item()
            if v_a is not None:          # its own forward and backward, so the peak is one prediction's, as before
                a_loss = F.mse_loss(driver.predict(dit, state, anchor).float(), v_a)
                (0.5 * float(anchor_weight) * a_loss).backward()
                a_total = (a_total or 0.0) + 0.5 * a_loss.item()
    finally:
        net.set_trainable_multiplier(1.0)
    return total, state["t"], a_total


def _dop_pairs(triggers, classes):
    """--dop_trigger "zwxem, qrtpl" + --dop_class "woman, man" -> [(trigger, class), ...]: one class for every
    trigger when only one is given, nothing (the trigger dropped) when none."""
    trig = [t.strip() for t in str(triggers or "").split(",") if t.strip()]
    cls = [c.strip() for c in str(classes or "").split(",")]
    if len(cls) == 1:
        cls = cls * len(trig)
    if len(cls) != len(trig):
        raise RuntimeError(f"[dop] {len(trig)} trigger(s) ({', '.join(trig)}) but {len(cls)} replacement(s) - give "
                           "one replacement for each trigger, in the same order, or one for them all")
    return list(zip(trig, cls))


def _dop_caption(caption, pairs):
    """DOP's caption (multi_person.dop_caption, which the Subjects card's preview shows)."""
    from fizgig.families.multi_person import dop_caption
    return dop_caption(caption, pairs)


def _dop_captions(group):
    """item_key -> (the caption its text cache was written from - every family's text cache keeps it as caption1 -,
    that cache file's path)."""
    from safetensors import safe_open
    out = {}
    for ds in group.datasets:
        bm = getattr(ds, "batch_manager", None)
        for bucket in (bm.buckets.values() if bm is not None else ()):
            for it in bucket:
                path = getattr(it, "text_encoder_output_cache_path", None)
                k = str(path or "")            # an item's own (two folders' same-named files share an item_key)
                if k in out or not path:
                    continue
                try:
                    with safe_open(path, framework="pt") as f:
                        out[k] = ((f.metadata() or {}).get("caption1"), path)
                except Exception:
                    out[k] = (None, path)
    return out


def _dop_path(te_path):
    """Where an item's trigger-free text encoding is kept: beside its own text cache."""
    root, ext = os.path.splitext(te_path)
    return root + "_dop" + (ext or ".safetensors")


def _load_dop_cond(path):
    """A DOP text encoding from its file, batched as the loader batches the cache."""
    from safetensors.torch import load_file
    return {k[len("cond__"):]: v[None] for k, v in load_file(path).items() if k.startswith("cond__")}


def _dop_step(driver, dit, net, latents, cond, gen, *, weight, min_t, max_t, accum, swapped=False):
    """Differential Output Preservation (AI-Toolkit's DOP, by Ostris): the same picture, freshly noised, with the
    trigger swapped for the class word. The target is the model's own prediction there with the adapter at 0, so
    without the trigger the LoRA learns to change nothing. Backpropagated here; returns the detached loss."""
    state = driver.noise_latents(latents, gen, min_t=min_t, max_t=max_t)
    try:
        net.set_trainable_multiplier(0.0)
        if swapped:     # a forward with no backward: block swap must stream it as inference, as previews do, or a
            driver.block_swap_mode(dit, inference=True)     # block is left on the CPU for the next pass
        with torch.no_grad():
            target = driver.predict(dit, state, cond).float()
    finally:
        net.set_trainable_multiplier(1.0)
        if swapped:
            driver.block_swap_mode(dit, inference=False)
    loss = F.mse_loss(driver.predict(dit, state, cond).float(), target)
    (loss * weight / accum).backward()
    return loss.item()


class _MaskPrior:
    """Masked prior preservation (OneTrainer's masked_prior_preservation_weight, AI-Toolkit's inverted_mask_prior):
    outside a picture's mask the target is the model's own prediction with the LoRA at 0, so the LoRA learns to
    leave the rest of the picture as the base model draws it instead of only learning it less. The driver calls
    base() with its forward at the step's noised input, before the training forward (families/driver.py
    prior_base / with_prior)."""

    def __init__(self, driver, dit, net, weight, swapped):
        self.driver, self.dit, self.net, self.weight, self.swapped = driver, dit, net, float(weight), bool(swapped)

    def base(self, forward):
        try:
            self.net.set_trainable_multiplier(0.0)
            if self.swapped:     # a forward with no backward: streamed as inference, as DOP's is
                self.driver.block_swap_mode(self.dit, inference=True)
            with torch.no_grad():
                return forward().detach().float()
        finally:
            self.net.set_trainable_multiplier(1.0)
            if self.swapped:
                self.driver.block_swap_mode(self.dit, inference=False)


def _dop_ft_targets(driver, dit, loader, dop_cond, cond_for, device, *, samples, seed, min_t, max_t):
    """A fine-tune's DOP targets, made before its first step while the model is still the original: for each image,
    `samples` noised inputs (the family's own timestep rule) and the original model's prediction there for the
    trigger-free caption. A fine-tune has no adapter to switch off, so these stand in for the base model. Kept on
    the CPU in bf16: item key -> [(state, prediction), ...]."""
    gen = torch.Generator().manual_seed(int(seed) + 7919)
    out = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="[dop] the original model's predictions"):
            keys = [str(k) for k in batch.get("item_te_paths") or []]
            dc = cond_for(batch, driver.batch_cond(batch, device))
            if dc is None or keys[0] in out:
                continue
            latents = batch["latents"].to(device)
            rows = []
            for _ in range(int(samples)):
                state = driver.noise_latents(latents, gen, min_t=min_t, max_t=max_t)
                pred = driver.predict(dit, state, dc)
                rows.append(({k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in state.items()},
                             pred.detach().to("cpu", torch.bfloat16)))
            out[keys[0]] = rows
    return out


def _dop_ft_loss(driver, dit, rows, cond, device):
    """One saved input of this image: the model's prediction now against the original's there."""
    state, target = rows[random.randrange(len(rows))]
    state = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in state.items()}
    return F.mse_loss(driver.predict(dit, state, cond).float(), target.to(device).float())


def _render_previews(driver, dit, net, vae, encoded, out_dir, epoch, *, output_name, steps, cfg, neg, width,
                     height, seed, ema=None, speed=None, lowmem=False, swapped=False, refs=None, slider=False,
                     strengths=(), look=False):
    """Previews on the RESIDENT training model: training adapter OFF (the deployment setup), the family's speed
    LoRA ON if one is loaded (it lives on CPU between previews). On small cards the DiT parks on CPU for the
    decode. File names match the other trainers so the GUI gallery reads them:
    <name>_e<epoch>_<idx>_<timestamp>_<seed>.png. `speed` is the speed LoRA's SamplingSettings or None; `refs` the edit previews' reference latents; `slider`
    renders each prompt at -1 / 0 / +1 on one seed and saves them as one strip. `strengths`: each prompt again at
    these LoRA strengths, same seed, saved beside it as <...>_s<strength x100>.png (_s050 = 0.5)."""
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    device = next(iter(dit.parameters())).device
    _preview_vram("preview start", reset_peak=True)
    paths = []
    was_training = dit.training
    done = set()           # what this preview has changed, so the finally undoes exactly that, wherever it failed
    try:
        net.set_enabled(ADAPTER, False)
        done.add("adapter")
        if speed is not None:
            done.add("speed")              # before the move: a move that runs out of memory still gets put back
            net.move_adapter(SPEED, device)
            net.set_enabled(SPEED, True)
        if look:
            done.add("look")
            net.move_adapter(LOOK, device)
            net.set_enabled(LOOK, True)
        if ema is not None:
            ema.swap_in()
            done.add("ema")
        dit.eval()
        if swapped:
            done.add("swap")
            driver.block_swap_mode(dit, inference=True)
        ref_kw = {"refs": [r.to(device) for r in refs]} if refs else {}
        lats = []
        extra = () if slider else tuple(strengths or ())
        mults = SLIDER_PREVIEW_MULTIPLIERS if slider else (None,) + extra
        # previews run the blocks uncompiled (Compile Blocks): eval mode, the preview size and the speed LoRA are each a
        # new graph, and building them took ~5 minutes of Triton/MSVC work on the first preview after every start or
        # resume - a handful of eager steps costs seconds. Training keeps its compiled graphs
        _eager = torch.compiler.set_stance("force_eager")
        _eager.__enter__()
        done.add("eager")
        for i, cond in enumerate(encoded):
            for m in mults:
                if m is not None or extra:
                    net.set_trainable_multiplier(1.0 if m is None else m)
                if speed is not None:
                    # the speed LoRA's own CFG, unless the Samples tab asks for more (then the negative applies too)
                    _cfg = cfg if cfg and cfg > 1.0 else speed.cfg
                    lats.append(driver.generate(dit, cond, width, height, steps=steps, seed=seed + i, cfg=_cfg,
                                                neg_cond=neg if _cfg > 1.0 else None,
                                                sigmas=speed.sigmas, options=speed.options, **ref_kw))
                else:
                    lats.append(driver.generate(dit, cond, width, height, steps=steps, seed=seed + i, cfg=cfg,
                                                neg_cond=neg, **ref_kw))
        tok = driver.park_for(dit, device, None, "decode")   # None = the shared rule below
        if tok is not None:
            done.add("decode_park")
        park = lowmem and not swapped and tok is None   # a swapped DiT is already mostly on CPU
        if park:                            # #123: never hold the training DiT and the VAE decode together
            done.add("lowmem_park")
            _preview_vram("before decode")
            quant.move(dit, "cpu")
            torch.cuda.empty_cache()
            _preview_vram("DiT parked for the decode")
        per = len(mults)
        for i in range(len(lats) // per):
            p = os.path.join(out_dir, f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}.png")
            if slider:
                frames = [driver.decode(vae, lat, width, height) for lat in lats[i * per:(i + 1) * per]]
                strip = driver.slider_preview(frames, SLIDER_PREVIEW_MULTIPLIERS)
                paths += driver.save_preview(_slider_strip(frames, SLIDER_PREVIEW_MULTIPLIERS) if strip is None
                                             else strip, p)
                continue
            for m, lat in zip(mults, lats[i * per:(i + 1) * per]):      # decoded one at a time: a song is big
                q = p if m is None else f"{p[:-4]}_s{int(round(m * 100)):03d}.png"
                paths += driver.save_preview(driver.decode(vae, lat, width, height), q)
    finally:
        if "eager" in done:
            _eager.__exit__(None, None, None)
        if slider or strengths:
            net.set_trainable_multiplier(1.0)  # a preview that failed mid-dial must not leave the LoRA scaled
        if "decode_park" in done:
            driver.unpark(dit, device, tok)
        if "lowmem_park" in done:
            quant.move(dit, device)
            _preview_vram("after decode, DiT restored")
        if "swap" in done:
            driver.block_swap_mode(dit, inference=False)
        if "ema" in done:
            ema.swap_out()
        if "speed" in done:
            net.set_enabled(SPEED, False)
            net.move_adapter(SPEED, "cpu")
        if "look" in done:
            net.set_enabled(LOOK, False)
            net.move_adapter(LOOK, "cpu")
        if "adapter" in done:
            net.set_enabled(ADAPTER, True)
        dit.train(was_training)
        torch.cuda.empty_cache()
        _preview_vram("after preview cleanup")
    logger.info(f"[sample] epoch {epoch}: {len(paths)} preview(s) -> {out_dir}")
    return paths


class _CheckpointPreviews:
    """Training previews on the family's preview checkpoint (train_preview_checkpoint: Klein's Distilled), with the
    old Klein trainer's memory handoff: the driver parks the training model, loads the checkpoint (its own swap by
    card, optionally INT8), the epoch's LoRA - the live adapter, EMA weights if EMA is on - and the context LoRA ride
    on it as frozen adapters, and the training model is put back exactly. Between epochs the checkpoint stays in
    system RAM when it isn't block-swapped and the cache mode allows (auto: decided once per run, free RAM >= 18 GB,
    so caching never flip-flops); any failure to reuse it falls back to a fresh load."""

    def __init__(self, driver, path, device, cache_mode="auto", int8=False, context=None, speed=None):
        self.driver, self.path, self.device = driver, path, torch.device(device)
        self.cache_mode, self.int8, self.context = str(cache_mode or "auto").lower(), bool(int8), context
        self.speed = speed                 # (path, strength): the speed LoRA, on a checkpoint that takes it (Kromoss)
        self.cached = None                 # (model, FamilyLoRA, swapped) on the CPU between epochs
        self._auto = None
        self._lora_file = os.path.join(os.environ.get("TEMP") or os.environ.get("TMPDIR") or "/tmp",
                                       f"fizgig_preview_lora_{os.getpid()}.safetensors")

    def _cache_on(self, swapped):
        if self.cache_mode == "off" or swapped:
            return False
        if self.cache_mode == "on":
            return True
        if self._auto is None:
            try:
                import psutil
                self._auto = psutil.virtual_memory().available / 1e9 >= 18.0
            except Exception:
                self._auto = False
        return self._auto

    def render(self, dit, net, ema, render):
        """Park, put the checkpoint up with this epoch's LoRA, call render(model, adapters), restore."""
        if ema is not None:
            ema.swap_in()
        try:
            net.save(self._lora_file, dtype=torch.bfloat16)
        finally:
            if ema is not None:
                ema.swap_out()
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        token = self.driver.park_for_preview(dit, self.device)
        m = fl = None
        swapped = 0
        try:
            gc.collect()
            torch.cuda.empty_cache()
            if self.cached is not None:
                try:
                    m, fl, swapped = self.cached
                    self.cached = None
                    quant.move(m, self.device)
                    fl.swap_file(TRAINABLE_PREVIEW, self._lora_file)
                    logger.info("[sample] preview checkpoint reused from RAM (no disk reload)")
                except Exception:
                    logger.warning("[sample] preview checkpoint reuse failed - reloading from disk", exc_info=True)
                    m = fl = None
            if m is None:
                m, swapped = self.driver.load_preview_checkpoint(self.path, self.device, int8=self.int8)
                fl = FamilyLoRA(m, self.driver, device=self.device)
                fl.add_file(self._lora_file, TRAINABLE_PREVIEW)
                if self.context:
                    fl.add_file(self.context[0], CONTEXT, self.context[1])
                logger.info(f"[sample] previews on {os.path.basename(self.path)}"
                            + (f" ({swapped} blocks streamed)" if swapped else ""))
            return render(m, fl, swapped)
        finally:
            keep = m is not None and self._cache_on(swapped)
            if keep:
                try:
                    quant.move(m, "cpu")
                    self.cached = (m, fl, swapped)
                except Exception:
                    logger.warning("[sample] could not keep the preview checkpoint in RAM", exc_info=True)
                    self.cached = None
            del m, fl
            gc.collect()
            gc.collect()
            torch.cuda.empty_cache()
            self.driver.unpark_after_preview(dit, self.device, token)
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            try:
                os.remove(self._lora_file)
            except OSError:
                pass


def _largest_bucket_mp(group):
    """Megapixels of the run's largest actual bucket (what sets activation memory), or None."""
    try:
        return max(w * h / 1e6 for ds in group.datasets for (w, h) in ds.batch_manager.bucket_resos)
    except Exception:
        return None


class _FineTune:
    """The fine-tune's side of the training loop: the rotator and its master, the window plan for this card, the
    whole-rotation length and cadence, a fresh optimizer per window, and the full-checkpoint save."""

    def __init__(self, driver, desc, dit, net, dit_path, device, *, max_train_epochs, rotations, save_every_rotations,
                 rotate_every, start_window, epochs_done, fused, lr, master_dir, mp=None, group=None, max_parts=0,
                 held_grads=False):
        from fizgig.families import ft
        from fizgig.utils.device import plannable_free_vram
        self.ft, self.driver, self.desc, self.dit, self.src, self.lr = ft, driver, desc, dit, dit_path, lr
        self.be = driver.ft_backend(dit, device, dit_path, group) or ft.SharedBackend(driver, dit, device, dit_path,
                                                                                         group)
        where = os.environ.get("FIZGIG_FT_MASTER", "auto")
        self.scratch = os.path.join(master_dir, ".fizgig_ft_master")
        gb, where = self.be.build_master(where, self.scratch)
        n_always = self.be.start_always()
        logger.info(f"[finetune] bf16 master: {len(list(self.be.master.keys()))} weights, {gb:.1f} GB "
                    + ("in system RAM" if where == "ram" else f"ON DISK (system memory can't comfortably hold it; "
                       f"untouched weights read from the model file, trained ones spill there)")
                    + f"; {n_always} always-on Linears ({self.be.always_label}) train all run")
        if where == "ram":
            try:
                import psutil
                avail = psutil.virtual_memory().available / 1e9
                if avail < 10:
                    logger.warning(f"[finetune] system RAM is tight after the master ({avail:.0f} GB left) - "
                                   f"FIZGIG_FT_MASTER=disk keeps the master on disk instead")
            except Exception:
                pass
        n_blocks = self.be.cycle_len()
        logger.info(f"[finetune] planning with {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
                    f"{plannable_free_vram():.2f} GB free")
        windows, stream, why, usable = self.be.plan(
            plannable_free_vram(), allow_stream=os.environ.get("FIZGIG_NO_FT_STREAM") != "1", mp=mp,
            max_parts=max_parts, held_grads=held_grads)
        for line in why:
            logger.info(f"[finetune] {line}")
        if windows is None:
            raise RuntimeError(f"[finetune] ~{usable:.1f} GB of usable VRAM is below what the fine-tune needs even "
                               f"with depth-split windows and streamed blocks. Close other GPU apps, or train a LoRA.")
        self.sched = ft.schedule(windows, n_blocks, rotate_every=max(1, int(rotate_every)),
                                 start_window=int(start_window))
        self.cycle = self.sched.cycle_epochs
        self.epochs = max(1, int(rotations)) * self.cycle
        self.save_every = max(1, int(save_every_rotations)) * self.cycle
        self.epochs_done = int(epochs_done)
        self.streamer = getattr(self.be, "streamer", None) or (True if stream else None)
        driver.ft_cycle(self.cycle, self.epochs_done, self.epochs)
        self.fused = ft.FusedSteps(lr) if fused else None
        self.opt_label = ("per-parameter " if fused else "") + ft.make_optimizer([torch.zeros(1)], lr)[1] + " (fine-tune)"
        logger.info(f"[finetune] FULL FINE-TUNE - {self.sched.describe()}")
        logger.info(f"[finetune] {rotations} rotation(s) of {self.cycle} epoch(s) = {self.epochs} epochs; checkpoint"
                    f" + preview every {save_every_rotations} rotation(s) = every {self.save_every} epochs "
                    f"(epochs {', '.join(str(e + self.epochs_done) for e in range(self.save_every, self.epochs + 1, self.save_every)[:6])}"
                    f"{' ...' if self.epochs // self.save_every > 6 else ''}); never mid-rotation")
        self._pause_noted = False

    def rotation_done(self, done):
        return done % self.cycle == 0

    def note_pause(self, done):
        if not self._pause_noted:
            nxt = (done // self.cycle + 1) * self.cycle
            logger.info(f"[pause] requested - finishing this rotation first (a checkpoint is only ever a whole "
                        f"rotation): pausing after epoch {nxt + self.epochs_done}")
            self._pause_noted = True

    def needs_window(self, epoch):
        """A new window this epoch - or the same one after a save parked it."""
        return list(self.sched.active_at(epoch)) != list(self.be.active)

    def enter(self, epoch):
        """The epoch's window trainable, with a fresh optimizer (the old one's state belonged to the outgoing
        weights; the caller has dropped it). Returns (params, optimizer); the optimizer is None under
        free-each-gradient."""
        want = self.sched.active_at(epoch)
        if self.fused is not None:
            self.fused.detach()
        gc.collect()
        torch.cuda.empty_cache()
        n = self.be.rotate(want)
        torch.cuda.reset_peak_memory_stats()
        params = self.be.trainable_params()
        logger.info(f"[finetune] epoch {epoch + 1 + self.epochs_done}: window {self.sched.window_at(epoch) + 1}/"
                    f"{self.sched.n_windows} {want} - "
                    f"{n} Linears trainable, "
                    f"{sum(p.numel() for p in params) / 1e9:.2f}B parameters")
        if self.fused is not None:
            self.fused.attach(params)
            return params, None
        return params, self.ft.make_optimizer(params, self.lr)[0]

    def park(self):
        """Everything back to NF4 with the master complete - before a save, a preview, or the end."""
        if self.fused is not None:
            self.fused.detach()
        gc.collect()
        self.be.park()
        gc.collect()
        torch.cuda.empty_cache()

    def save(self, path, epoch):
        n_ep = epoch + self.epochs_done
        meta = {"fizgig_finetune": f"{self.desc.key}-rotation", "fizgig_source_base": os.path.basename(self.src),
                "fizgig_next_start_window": str(self.sched.window_at(epoch)), "fizgig_ft_n_windows":
                str(self.sched.n_windows), "fizgig_ft_epochs_done": str(n_ep), "modelspec.architecture":
                self.desc.modelspec_arch if hasattr(self.desc, "modelspec_arch") else self.desc.key}
        logger.info(f"[finetune] saving the full checkpoint at epoch {n_ep} -> {os.path.basename(path)} "
                    f"(~{os.path.getsize(self.src) / 1e9:.0f} GB; training waits for it)")
        replaced, total = self.be.save(path, meta)
        logger.info(f"[save] {path} ({replaced}{'/' + str(total) if total else ''} tensors trained)")


def train_family(family, dit_path, dataset_config, output_dir, output_name, *, network_dim=32, network_alpha=32,
                 learning_rate=1e-4, max_train_epochs=16, save_every_n_epochs=1, save_until_epoch=0,
                 save_every_n_epochs_after=0, save_state=False,
                 save_state_on_train_end=False, keep_last_n_states=2, seed=42, precision="bf16",
                 training_adapter=None, training_adapter_strength=1.0,
                 context_lora_path=None, context_lora_strength=1.0, min_timestep=0.0, max_timestep=1.0,
                 speed_lora=None, speed_lora_strength=None, preview_lora=None, preview_lora_strength=1.0,
                 vae_path=None, te_path=None, sample_prompts=None, sample_every_n_epochs=0, sample_until_epoch=0,
                 sample_every_n_epochs_after=0, sample_width=None,
                 sample_height=None, sample_steps=None, sample_cfg_scale=None, sample_negative=None,
                 sample_at_first=False, sample_seed=42, sample_reference=None, sample_image=None,
                 sample_strengths=(), slider_pairs=False, slider_diff_weight=1.0, slider_prompts=None, slider_guidance=3.0,
                 slider_anchors=(), slider_anchor_weight=1.0,
                 train_blocks=None,
                 slider_bank=16, slider_bank_res=768,
                 metadata_title=None, metadata_author=None, metadata_description=None, metadata_license=None,
                 metadata_tags=None, metadata_trigger_phrase=None, metadata_thumbnail=None,
                 resume_state_dir=None, adaptive_lr=False, adaptive_lr_min=1e-4, adaptive_lr_max=2e-4,
                 max_grad_norm=1.0, ema_decay=0.0, optimizer_type="adamw", optimizer_args="",
                 lr_scheduler="constant", lr_warmup_steps=0, lr_scheduler_num_cycles=1, lr_scheduler_power=1.0,
                 gradient_accumulation_steps=1, compile_blocks="auto", gradient_checkpointing="auto", blocks_to_swap=0, network_type="lora", lokr_factor=8,
                 log_per_image_loss=False, per_image_lr=False, auto_recaption=False, warmup_look_outliers=False,
                 trigger_word=None, trigger_position="start", recaption_triggers=None, recaption_instruction=None,
                 recaption_instruction_detailed=None, recaption_task=None, recaption_task_detailed=None,
                 captioner=None, finetune=False, ft_rotations=10, ft_save_every_rotations=1, ft_rotate_every=1, ft_max_parts=0,
                 ft_start_window=0,
                 ft_epochs_done=0, ft_fused_backward=False, reg_lr_multiplier=0.2, preview_checkpoint=None,
                 preview_checkpoint_cache="auto", preview_int8=False, family_options=None,
                 dop=False, dop_trigger=None, dop_class="", dop_weight=1.0, dop_every=1, dop_ft_samples=8,
                 lr_boost=None, share_worker=None, masked_training=False, mask_floor=0.1, mask_prior=0.0):
    desc = get_family(family)
    if desc is None or not desc.training_ready:
        raise RuntimeError(f"unknown or untrainable family {family!r}")
    validate_output_name(output_name)          # before any model loads, not at the first save an epoch in (#70)
    if context_lora_path and finetune:
        raise RuntimeError("A Context LoRA trains a LoRA on top of another one; a fine-tune trains the base model "
                           "itself, so the two don't combine. Drop --context_lora_path.")
    driver = desc.load_driver()
    driver.set_options(dict(family_options or {}))      # before the data: an option may shape the dataset
    if train_blocks:
        train_blocks = driver.expand_train_blocks(train_blocks)
    arch = desc.arch_id
    speed_desc = desc.preview_speed() if speed_lora else None
    if speed_lora and speed_desc is None:
        logger.warning(f"[sample] {desc.display_name} declares no preview speed LoRA - ignoring --speed_lora")
        speed_lora = None
    # strength 0: previews render without it, and it loads at the first preview a live change asks it for
    speed_idle = speed_desc is not None and speed_lora_strength is not None and speed_lora_strength <= 0
    if speed_idle:
        logger.info(f"[sample] {speed_desc.name} at strength 0 - previews render without it until a strength is "
                    "asked for")
    # --speed_lora on its own means the turbo's own recipe (strength, steps); the GUI passes its Samples-tab values
    asked_steps, asked_cfg = sample_steps, sample_cfg_scale   # a preview checkpoint takes these when given
    sample_steps = sample_steps or (speed_desc.settings.steps if speed_desc and not speed_idle else desc.preview_steps)
    sample_cfg_scale = desc.preview_cfg if sample_cfg_scale is None else sample_cfg_scale
    sample_width = sample_width or desc.preview_width
    sample_height = sample_height or desc.preview_height
    lowmem = _small_card_previews()
    if lowmem and max(sample_width, sample_height) > 768:
        sample_width, sample_height = _cap_canvas(sample_width, sample_height)
        logger.info(f"[preview] card under 20 GB: preview canvas capped to {sample_width}x{sample_height} and the "
                    f"DiT parks on CPU for the decode (#123). FIZGIG_PREVIEW_LOWMEM=0 turns this off.")
    device = torch.device("cuda")
    quant.apply_vram_cap()          # FIZGIG_SIM_VRAM_GB: behave like a smaller card
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ---- sliders: a signed dial, trained at +1 and -1 ------------------------------------------------
    slider = bool(slider_pairs or slider_prompts)
    if slider:
        if not desc.slider_training:
            raise RuntimeError(f"{desc.display_name} does not offer slider training")
        if slider_pairs and slider_prompts:
            raise RuntimeError("[slider] image pairs and prompt pairs are two different sliders - pick one")
        if slider_prompts and (len(slider_prompts) != 3 or not all(str(x).strip() for x in slider_prompts[1:])):
            raise RuntimeError("[slider] prompt pairs need NEUTRAL POSITIVE NEGATIVE (the two poles non-empty)")
        if network_type != "lora":
            logger.info("[slider] network type %s -> LoRA: a slider is a plain LoRA whose strength is the dial",
                        network_type)
            network_type = "lora"
        if adaptive_lr or ema_decay or log_per_image_loss or per_image_lr or auto_recaption or warmup_look_outliers:
            logger.info("[slider] adaptive LR, weight averaging and the per-image loss watch are off: they assume "
                        "one target per image, and a slider step has two")
        adaptive_lr, ema_decay = False, 0.0
        log_per_image_loss = per_image_lr = auto_recaption = warmup_look_outliers = False
        if slider_pairs:
            logger.info("[slider] IMAGE-PAIR SLIDER: the adapter trains at +1 toward each image and at -1 toward "
                        "its pair (difference weight %g)", float(slider_diff_weight))
        else:
            logger.info("[slider] PROMPT-PAIR SLIDER: base '%s' | +1 -> '%s' | -1 -> '%s' (guidance %g)",
                        *slider_prompts, float(slider_guidance))
            slider_anchors = _anchor_lines(slider_anchors)
            if slider_anchors:
                logger.info("[slider] %d anchor prompt(s), one a step in turn, weight %g - the dial leaves them "
                            "as the model has them: %s", len(slider_anchors), float(slider_anchor_weight),
                            " | ".join(slider_anchors))
            if not (te_path and vae_path):
                raise RuntimeError("[slider] prompt pairs need the text encoder and VAE paths")
            if str(slider_prompts[0]).strip():     # the dial is shown on the picture it was trained on
                # and on the first anchor, whose strip should barely move
                sample_prompts = [str(slider_prompts[0]).strip()] + slider_anchors[:1]
    else:
        slider_anchors = []

    # ---- full fine-tune (families/ft.py): the base model itself trains, a component window at a time -------------
    if finetune:
        if not desc.finetune:
            raise RuntimeError(f"{desc.display_name} does not offer a full fine-tune")
        if slider:
            raise RuntimeError("[finetune] a slider is a LoRA - untick Fine-tune or the slider")
        if resume_state_dir:
            raise RuntimeError("[finetune] a fine-tune continues from its saved checkpoint (the Model / --dit), not a "
                               "state folder")
        _why = driver.ft_source_unfit(dit_path)
        if _why:
            raise RuntimeError(f"[finetune] {os.path.basename(dit_path)} {_why}")
        if precision != "nf4" or blocks_to_swap:
            logger.info(f"[finetune] frozen base: NF4 (asked {precision}, swap {blocks_to_swap}) - the fine-tune "
                        f"trunk is always 4-bit, and blocks outside the window stream only if the plan needs it")
        precision, blocks_to_swap = "nf4", 0
        _off = [n for n, on in (("EMA", ema_decay), ("Adaptive LR", adaptive_lr), ("auto-recaption", auto_recaption),
                                ("torch.compile", str(compile_blocks).lower() in ("on", "outside")))
                if on]
        if _off:
            logger.info(f"[finetune] off for a fine-tune: {', '.join(_off)}")
        ema_decay, adaptive_lr, auto_recaption, compile_blocks = 0.0, False, False, "off"
        if ft_fused_backward and int(gradient_accumulation_steps or 1) > 1:
            logger.info("[finetune] free-each-gradient steps every parameter as its gradient lands - accumulation off")
            gradient_accumulation_steps = 1
        if lr_scheduler != "constant":
            logger.info(f"[finetune] LR scheduler {lr_scheduler} -> constant (the optimizer is rebuilt every window)")
            lr_scheduler = "constant"
        if ft_fused_backward and max_grad_norm:
            logger.info("[finetune] free-each-gradient: gradient clipping is off (it needs every gradient at once)")
            max_grad_norm = 0.0

    # ---- data ------------------------------------------------------------------------------------
    shared_epoch = Value("i", 0)
    if slider_prompts:
        user_config = {"general": {"resolution": [slider_bank_res, slider_bank_res]}}
        group = _SliderBank([None] * max(1, int(slider_bank)))    # filled once the DiT is loaded
    else:
        if not dataset_config:
            raise RuntimeError("--dataset_config is required (only a prompt-pair slider trains without a dataset)")
        user_config = load_user_config(dataset_config)
        blueprint = BlueprintGenerator(ConfigSanitizer()).generate(user_config, argparse.Namespace(),
                                                                   architecture=arch)
        if masked_training:
            if not desc.masked_training:
                raise RuntimeError(f"{desc.display_name} has no masked training - drop --masked_training")
            from fizgig.dataset.image_dataset import ImageDataset
            ImageDataset.loss_masks = True      # each still with a photo-masklabel.png gets its mask (dataset/masks.py)
        group = generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True,
                                                    num_timestep_buckets=None, shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items - run the cache stages (families/cache.py) first.")
    masked_images = 0
    if masked_training and slider_prompts:
        logger.info("[masks] a prompt-pair slider trains on pictures it makes itself: masked training is off")
        masked_training = False
    elif masked_training:
        masked_images = len({str(getattr(it, "mask_path", "")) for ds in group.datasets
                             for b in ds.batch_manager.buckets.values() for it in b if getattr(it, "mask_path", None)})
        logger.info(f"[masks] masked training: {masked_images} picture(s) with a mask, outside the mask counts "
                    f"x{float(mask_floor):g}")
    mask_prior = float(mask_prior or 0.0) if masked_training else 0.0
    if mask_prior and (slider or finetune):
        logger.info("[masks] keeping the model's own picture outside the mask needs a LoRA to switch off (not a "
                    "slider or a fine-tune): it's off for this run")
        mask_prior = 0.0
    elif mask_prior:
        logger.info(f"[masks] outside the mask, the LoRA is kept to the model's own picture at x{mask_prior:g} "
                    "(one extra forward a step, without the LoRA)")

    def _loss_mask(batch):
        """The step's per-cell loss weights (masked training), or None: a picture without a mask trains as normal."""
        m = batch.get("loss_mask") if masked_training else None
        if m is None:
            return None
        from fizgig.dataset.masks import mask_weight
        return mask_weight(m.to(device).float(), mask_floor)
    if slider_pairs:
        shared = _pair_slider_caption(user_config)
        if shared:                           # the dial is shown on what both ends of a pair have in common
            sample_prompts = [shared]
            logger.info("[slider] previews use the pairs' caption: '%s'", shared)
    if slider:
        driver.slider_setup(group)           # e.g. H3: still previews unless the pairs are clips
    for ds in group.datasets:
        if getattr(ds, "batch_size", 1) != 1:
            raise RuntimeError("Fizgig trains one image at a time: set batch_size = 1 in the dataset config. For a "
                               "larger effective batch use --gradient_accumulation_steps (the same averaged "
                               "gradient, at the memory of batch 1).")
    loader = DataLoader(group, batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=(lambda b: b[0]) if slider_prompts else _Collator(shared_epoch, group))
    steps_per_epoch = len(loader)
    logger.info(f"{desc.display_name} training: {group.num_train_items} items, {max_train_epochs} epochs, "
                f"{steps_per_epoch} steps/epoch")
    _launch = uuid.uuid4().hex[:12]      # this launch: a re-run of the same name is a new training share
    # what keeps this run on one GPU (the share's half would train without it), for the Helpers tab to say so
    _share_off = ("a fine-tune" if finetune else
                  "Your subject and other people" if dop else
                  "Track how each image trains and the per-image options" if (
                      log_per_image_loss or per_image_lr or auto_recaption or warmup_look_outliers) else "")
    if share_worker is None and not slider_prompts:
        try:                     # for a helper's training share (web/helpers.py): this run's command and its cache
            os.makedirs(os.path.join(output_dir, SHARE_DIR), exist_ok=True)
            for _old in glob.glob(os.path.join(output_dir, SHARE_DIR, "[0-9]" * 6)):
                shutil.rmtree(_old, ignore_errors=True)     # an earlier run's rounds: never this run's to combine
            with open(dataset_config, encoding="utf-8") as f:
                _ds_text = f.read()
            _write_json(os.path.join(output_dir, SHARE_DIR, "setup.json"), {
                "launch": _launch, "off": _share_off,
                "family": desc.key, "args": sys.argv[1:], "dataset_config": _ds_text, "output_name": output_name,
                "cache_dirs": sorted({ds.cache_directory for ds in getattr(group, "datasets", [])
                                      if getattr(ds, "cache_directory", None)}), "max_train_epochs": max_train_epochs})
        except Exception as e:
            logger.info(f"[share] no training share for this run ({e})")
    # A fine-tune's regularisation set: a dataset block marked is_reg, trained at a fixed reduced LR so it tethers
    # the model's prior rather than teaching a subject (a full fine-tune has no rank bound on its drift). A LoRA's
    # update is rank-bounded, so there the block is trained as ordinary images, with a warning.
    reg_keys = set()
    reg_ds = [ds for ds in getattr(group, "datasets", []) if getattr(ds, "is_reg", False)]
    if reg_ds and finetune:
        for ds in reg_ds:
            for bucket in ds.batch_manager.buckets.values():
                reg_keys.update(str(it.item_key) for it in bucket)
        logger.info(f"[reg] {len(reg_keys)} regularisation image(s) at x{reg_lr_multiplier:g} LR "
                    f"({group.num_train_items - len(reg_keys)} subject items)")
        if len(reg_keys) >= group.num_train_items - len(reg_keys):
            logger.warning("[reg] regularisation images are at least half the training set - the multiplier only "
                           "reads as an LR cut while they are the minority (Adafactor normalises by a second moment "
                           "the majority dominates)")
    elif reg_ds:
        logger.warning("[reg] the dataset config has a regularisation block, but this is a LoRA run - those images "
                       "train as ordinary images at full LR; remove the is_reg block if that is not what you want")

    # ---- Auto precision / swap: planned on an empty card (the description's figures include the preview VAE)
    auto_precision = precision == "auto"
    if precision == "auto" or blocks_to_swap < 0:
        req = (precision, blocks_to_swap)
        res = user_config.get("general", {}).get("resolution") or [1024, 1024]
        mp = (res[0] * res[1] if isinstance(res, (list, tuple)) else res * res) / 1e6
        own = driver.plan_run(precision, blocks_to_swap, group=group, run=dict(
            dit_path=dit_path, network_type=network_type, network_dim=network_dim, lokr_factor=lokr_factor,
            optimizer_type=optimizer_type, training_adapter=training_adapter, context_lora_path=context_lora_path,
            ema_decay=0.98 if str(ema_decay).lower().startswith("short") else ema_decay))
        if own is not None:
            precision, blocks_to_swap, why = own
        else:
            precision, blocks_to_swap, why = quant.plan(desc, driver, precision, blocks_to_swap, megapixels=mp)
            why += f" at {mp:.2f} MP"
            if share_worker is not None:
                # a helper's share trains at the lead's precision (the same base numbers): say so plainly, before the
                # model loads, when this card can't hold it even with the most blocks swapped (the lead trains on)
                short = quant.share_shortfall(desc, driver, precision, blocks_to_swap, mp)
                if short:
                    raise SystemExit(f"[share] This GPU can't train the lead's {desc.precision_labels.get(precision, precision)} "
                                     f"base: {short}. The lead trains every image itself. To share with this GPU, "
                                     "train on the lead with a smaller base (Auto, int8 or 4-bit).")
        logger.info(f"[precision] Auto plan: {precision}, block swap {blocks_to_swap} ({why}); asked {req}")

    # ---- torch.compile (Krea 2's rule), decided here on the empty card as the original does - the free VRAM its
    # checks read would be the loaded DiT's leftovers later - and applied last, after every LoRA has patched the
    # forwards and the slider bank has rendered. Auto weighs the warm-up against this run's length; On places the
    # checkpoint where it fits at the largest bucket.
    do_compile = False
    cb = str(compile_blocks or "off").lower()
    if desc.compiles and cb != "off":
        try:
            mp_max = max(w * h / 1e6 for ds in group.datasets for (w, h) in ds.batch_manager.bucket_resos)
        except Exception:
            mp_max = 0.25
        do_compile, why = driver.compile_plan(cb, group.num_train_items * max_train_epochs, precision,
                                              max(0, blocks_to_swap) if precision != "nf4" else 0, mp=mp_max)
        if cb == "auto":
            logger.info("[compile] auto: %s - %s", "ENABLED (checkpoint outside)" if do_compile == "outside"
                        else ("ENABLED" if do_compile else "off"), why)
        elif why:
            logger.info("[compile] %s", why)
    if auto_precision and not do_compile:
        alt = driver.auto_uncompiled_precision(dit_path, precision)
        if alt and alt != precision:
            logger.info(f"[precision] Auto: {desc.precision_labels.get(alt, alt)} instead of {precision} - this run "
                        "is not compiled, and uncompiled that is the faster base for this file")
            precision = alt

    # ---- gradient checkpointing: "auto" goes without it when the measured figure fits the empty card (LoRA only -
    # LoKR without checkpointing measured 3-16 GB over a LoRA at 1 MP, Qwen edit pairs ~26 GB; fine-tunes, sliders, edit pairs and
    # the mid-run captioner carry memory the figures don't cover)
    gc_off_blocks = 0
    if gradient_checkpointing == "auto":
        paired = any(d.get("control_directory") for d in user_config.get("datasets", []))
        if (quant.card_gb() < quant.NOGC_MIN_CARD_GB or finetune or slider or paired or auto_recaption
                or str(network_type).lower() != "lora"
                or gradient_accumulation_steps > 1):
            gradient_checkpointing = True
        else:
            kind, mp_ck = driver.nogc_load(group)
            gradient_checkpointing, why = quant.plan_checkpointing(
                desc, precision, max(0, blocks_to_swap) if precision != "nf4" else 0, mp_ck, compiled=bool(do_compile),
                kind=kind, rank=int(network_dim))
            logger.info(f"[checkpointing] gradient checkpointing {why}")
            if gradient_checkpointing:
                gc_off_blocks, why = quant.plan_partial_checkpointing(
                    desc, precision, max(0, blocks_to_swap) if precision != "nf4" else 0, mp_ck,
                    compiled=bool(do_compile), kind=kind, rank=int(network_dim))
                if gc_off_blocks:
                    logger.info(f"[checkpointing] gradient checkpointing {why}")
                    if do_compile:          # the partial figures are compiled with the checkpoint outside the graph
                        do_compile = "outside"
    gradient_checkpointing = bool(gradient_checkpointing)

    # ---- previews: encode prompts once, keep the VAE ---------------------------------------------
    encoded = neg = vae = ref_imgs = ref_latents = None
    slider_enc = slider_neg = None
    anchor_enc = []
    if slider_prompts:
        te = driver.load_text_encoder(te_path, device)
        _all = [{k: v[None] for k, v in c.items()}
                for c in driver.encode_text(te, [str(x) for x in slider_prompts] + list(slider_anchors))]
        slider_enc, anchor_enc = _all[:3], _all[3:]
        if sample_cfg_scale > 1.0 and desc.preview_negative is not None:
            # the practice pictures render as previews do: the Samples tab's negative (else the family's default)
            text = sample_negative if sample_negative is not None else desc.preview_negative
            slider_neg = {k: v[None] for k, v in driver.encode_text(te, [text])[0].items()}
        driver.unload_text_encoder(te)
        del te
        torch.cuda.empty_cache()
    # ---- DOP: each image's caption with the trigger swapped for the class word, encoded once ------------------
    dop_cond = {}
    if dop:
        trig = str(dop_trigger or trigger_word or metadata_trigger_phrase or "").strip()
        pairs = _dop_pairs(trig, dop_class) if trig else []
        why = ("a slider trains a dial, not a subject" if slider else
               "it needs the trigger word (--dop_trigger)" if not trig else
               "it needs the text encoder (--text_encoder)" if not te_path else "")
        if why:
            logger.warning(f"[dop] off: {why}")
        else:
            caps = _dop_captions(group)
            todo = {k: c2 for k, (c, _p) in caps.items() if (c2 := _dop_caption(c, pairs)) is not None}
            from fizgig.families import cache as _cache
            new = [k for k in sorted(todo) if not _cache.cond_file_matches(_dop_path(caps[k][1]), todo[k])]
            shown = ", ".join(f"'{t}' -> '{c}'" if c else f"'{t}' dropped" for t, c in pairs)
            if not todo:
                logger.warning(f"[dop] off: no caption contains {'any of ' if len(pairs) > 1 else ''}{shown}")
            else:
                keys = sorted(todo)
                if new:          # written beside each item's own text cache, so a re-run reads them back
                    te = driver.load_text_encoder(te_path, device)
                    try:
                        for i in tqdm(range(0, len(new), 4), desc="[dop] trigger-free captions"):
                            chunk = new[i:i + 4]
                            for k, c in zip(chunk, driver.encode_text(te, [todo[k] for k in chunk])):
                                _cache.save_cond_file(desc, _dop_path(caps[k][1]), todo[k], c, k)
                    finally:
                        driver.unload_text_encoder(te)
                        del te
                        torch.cuda.empty_cache()
                dop_cond.update({k: _dop_path(caps[k][1]) for k in keys})
                logger.info(f"[dop] trigger-free captions: {len(new)} encoded, {len(keys) - len(new)} already in "
                            "the cache")
                logger.info(f"[dop] Differential Output Preservation on {len(dop_cond)} of {len(caps)} images "
                            f"(x{dop_weight:g}, every {max(1, int(dop_every))} step(s)); {shown}: without the trigger the "
                            f"LoRA is held to the base model. e.g. '{todo[keys[0]][:120]}'")
                if len(pairs) > 1:
                    for t, _c in pairs:
                        n = sum(1 for k in todo if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", caps[k][0], re.I))
                        logger.info(f"[dop]   '{t}': {n} image(s)")
                if len(dop_cond) < len(caps):
                    logger.info(f"[dop] {len(caps) - len(dop_cond)} image(s) have no trigger in the caption - "
                                "they train as usual, with no preservation step")
    dop_every = max(1, int(dop_every or 1))
    if dop_cond:
        # each new shape compiles its own graph. The DOP pass's trigger-free captions bring text lengths the run's
        # own captions don't; with automatic dynamic shapes, the dynamic graph that follows failed to compile on
        # Krea 2 (inductor: KeyError 'unbacked_bindings', the step after the first DOP pass); off, the same run went
        # 30 steps clean. A run without DOP is left as it was
        import torch._dynamo as _dyn
        _dyn.config.automatic_dynamic_shapes = False
    dop_sum = [0.0, 0]
    sample_dir = os.path.join(output_dir, "sample")

    def dop_cond_for(batch, cond):
        """This item's conditioning with its trigger-free text, or None when the item has no DOP pass (no trigger,
        an edit pair, sound only)."""
        keys = [str(k) for k in batch.get("item_te_paths") or []]
        if (not dop_cond or len(keys) != 1 or keys[0] not in dop_cond or any(k.startswith("latents_control_")
                                                                             for k in batch)
                or bool(torch.as_tensor(batch.get("audio_only", False)).any())):
            return None
        dc = dict(cond)
        for n, v in _load_dop_cond(dop_cond[keys[0]]).items():
            if n not in dc:      # the cache keeps the text under other names: DOP can't swap it
                raise RuntimeError(f"[dop] {desc.display_name}'s encoded text has '{n}', which its cached "
                                   f"conditioning doesn't ({', '.join(sorted(cond))})")
            dc[n] = v.to(device) if torch.is_tensor(v) else v
        return dc
    dop_ft = None                  # a fine-tune's saved targets (made at the first window)
    dop_ft_file = os.path.join(output_dir, f"{output_name}_dop_targets.pt")
    if sample_reference and not driver.supports_references:
        logger.warning(f"[sample] {desc.display_name} has no edit previews - ignoring --sample_reference")
        sample_reference = None
    if sample_prompts and sample_every_n_epochs and te_path and vae_path:
        logger.info("[sample] encoding %d preview prompt(s) with %s", len(sample_prompts), desc.text_encoder_label)
        if sample_reference:
            ref_imgs, (sample_width, sample_height) = _load_references(sample_reference, sample_width, sample_height)
            logger.info(f"[sample] edit previews from {len(ref_imgs)} reference image(s) at "
                        f"{sample_width}x{sample_height}")
            te = driver.load_reference_text_encoder(te_path, device)
            encoded = driver.encode_text_with_references(te, sample_prompts, [ref_imgs] * len(sample_prompts))
            if sample_cfg_scale > 1.0 or not desc.samples_cfg_free:
                neg = driver.encode_text_with_references(te, [sample_negative or ""], [ref_imgs])[0]
        elif sample_image and desc.preview_image:
            from PIL import Image
            logger.info(f"[sample] previews see {os.path.basename(sample_image)} through the text encoder's vision "
                        f"path")
            te = driver.load_text_encoder(te_path, device)
            encoded = driver.encode_text_with_image(te, sample_prompts, open_upright(sample_image))
            if sample_cfg_scale > 1.0 or not desc.samples_cfg_free:
                neg = driver.encode_text(te, [sample_negative or ""])[0]     # negatives stay text-only
        else:
            te = driver.load_text_encoder(te_path, device)
            encoded = driver.encode_text(te, sample_prompts)
            if sample_cfg_scale > 1.0 or not desc.samples_cfg_free:
                neg = driver.encode_text(te, [sample_negative or ""])[0]
        driver.unload_text_encoder(te)
        del te
        torch.cuda.empty_cache()
        vae = driver.load_vae(vae_path, device)
        if ref_imgs:
            # one at a time: extra pictures have their own shapes, and a driver may stack its batch (Klein)
            ref_latents = [driver.encode_images(vae, [im])[0][None].cpu() for im in ref_imgs]
    elif sample_prompts and sample_every_n_epochs:
        logger.warning("[sample] previews need the text encoder and VAE paths - previews are off for this run")

    # ---- model ----------------------------------------------------------------------------------
    logger.info(f"Loading {desc.display_name} DiT ({precision}) from {dit_path}")
    dit, swapped = quant.load_base(driver, dit_path, device, precision, blocks_to_swap)
    if gradient_checkpointing:
        driver.enable_gradient_checkpointing(dit, True)
        dit.gc_off_blocks = gc_off_blocks       # the compiled blocks / H3's loop leave the last ones un-checkpointed
    net = FamilyLoRA(dit, driver, device=device)
    driver.prepare_training(dit, group, net)
    if training_adapter:
        n = net.add_file(training_adapter, ADAPTER, training_adapter_strength)
        if n == 0:
            raise RuntimeError(f"Training adapter {training_adapter} matched no {desc.display_name} modules.")
        driver.frozen_file_added(dit, training_adapter, training_adapter_strength, "adapter")
        logger.info(f"[adapter] training adapter ON ({n} Linears, strength {training_adapter_strength:g}) - frozen, "
                    f"off in previews, not saved into the LoRA")
    elif desc.training_adapter:                 # the family has one and this run goes without it
        logger.warning("[adapter] no training adapter for this run")
    if context_lora_path:
        n = net.add_file(context_lora_path, CONTEXT, context_lora_strength)
        driver.frozen_file_added(dit, context_lora_path, context_lora_strength, "context")
        logger.info(f"[context] {os.path.basename(context_lora_path)} frozen + active at {context_lora_strength:g} "
                    f"({n} Linears)")
    def _add_speed(strength):
        """The speed LoRA onto the training model, off and on CPU between previews; False when it matched nothing."""
        n = net.add_file(speed_lora, SPEED, strength)
        driver.frozen_file_added(dit, speed_lora, strength, "speed")
        if n == 0:
            net.remove(SPEED)
            logger.warning(f"[sample] {os.path.basename(speed_lora)} matched no {desc.display_name} layers "
                           f"(a turbo LoRA for another model?) - previews render without it")
            return False
        net.set_enabled(SPEED, False)
        net.move_adapter(SPEED, "cpu")
        logger.info(f"[sample] {speed_desc.name}: {n} Linears, on CPU between previews, on only while they render "
                    f"(strength {strength:g})")
        return True

    look_said = [False]

    def _look_ready(path=None, strength=None):
        """The preview look LoRA on the training model (added the first time its file is there: it may still be
        downloading), off and on CPU between previews. False = previews without it. path / strength: the live
        override's, when the Samples tab's tick turned it on after the run started."""
        path, strength = path or preview_lora, strength or preview_lora_strength
        if not path or look_said[0] == "bad":
            return False
        if net.has(LOOK):
            return True
        if not os.path.isfile(path):
            if not look_said[0]:
                logger.info(f"[sample] {os.path.basename(path)} isn't downloaded yet - previews render without "
                            "it until it is")
                look_said[0] = True
            return False
        n = net.add_file(path, LOOK, strength)
        driver.frozen_file_added(dit, path, strength, "speed")
        if n == 0:
            net.remove(LOOK)
            logger.warning(f"[sample] {os.path.basename(path)} matched no {desc.display_name} layers - "
                           "previews render without it")
            look_said[0] = "bad"
            return False
        net.set_enabled(LOOK, False)
        net.move_adapter(LOOK, "cpu")
        logger.info(f"[sample] {os.path.basename(path)}: {n} Linears at {strength:g}, on previews "
                    "only (on CPU between them)")
        return True

    if speed_lora and encoded is not None and not speed_idle:
        if not _add_speed(speed_desc.strength if speed_lora_strength is None else speed_lora_strength):
            speed_lora = None
            sample_steps = desc.preview_steps
    if network_type == "lokr" and "lokr" not in desc.network_types:
        raise RuntimeError(f"{desc.display_name} does not offer LoKR")
    ftr = None
    if finetune:
        ftr = _FineTune(driver, desc, dit, net, dit_path, device, max_train_epochs=None, rotations=ft_rotations,
                        save_every_rotations=ft_save_every_rotations, rotate_every=ft_rotate_every,
                        start_window=ft_start_window, epochs_done=ft_epochs_done, fused=ft_fused_backward,
                        lr=learning_rate, master_dir=output_dir, mp=_largest_bucket_mp(group), group=group,
                        max_parts=ft_max_parts, held_grads=bool(dop_cond))
        max_train_epochs, save_every_n_epochs = ftr.epochs, ftr.save_every
        if ftr.streamer is not None:
            swapped = 1          # blocks stream: previews and caption re-encodes must not park + restore the whole DiT
    else:
        blocks = None
        if train_blocks:
            known = {b.id for g in driver.block_map(dit) for b in g.blocks}
            unknown = sorted(set(train_blocks) - known)
            if unknown:
                raise RuntimeError(f"--train_blocks: {', '.join(unknown)} are not {desc.display_name} blocks")
            blocks = set(train_blocks)
            logger.info(f"[blocks] the LoRA trains {len(blocks)} of {len(known)} blocks only: {', '.join(train_blocks)}")
        net.add_trainable(network_dim, network_alpha, blocks=blocks, kind=network_type, factor=lokr_factor)
    def render_bank():
        """The practice bank: the base model's own renders of the neutral prompt (adapter at 0, training adapter
        off, the speed LoRA on when it is loaded), decoded and re-encoded into training latents."""
        nonlocal vae
        if vae is None:
            vae = driver.load_vae(vae_path, device)
        n_c = {k: v[0].to(device) for k, v in slider_enc[0].items()}
        was_training = dit.training
        net.set_trainable_multiplier(0.0)
        net.set_enabled(ADAPTER, False)
        use_speed = speed_lora is not None and speed_desc is not None and net.has(SPEED)
        if use_speed:
            net.move_adapter(SPEED, device)
            net.set_enabled(SPEED, True)
        dit.eval()
        bank = []
        with torch.no_grad(), driver.still_renders():     # practice pictures are stills, whatever the previews are
            for i in tqdm(range(len(group)), desc="[slider] practice images"):
                if use_speed:
                    lat = driver.generate(dit, n_c, slider_bank_res, slider_bank_res, steps=speed_desc.settings.steps,
                                          seed=seed + 1000 + i, cfg=speed_desc.settings.cfg,
                                          sigmas=speed_desc.settings.sigmas, options=speed_desc.settings.options)
                else:
                    lat = driver.generate(dit, n_c, slider_bank_res, slider_bank_res, steps=desc.preview_steps,
                                          seed=seed + 1000 + i, cfg=sample_cfg_scale,
                                          # on the GPU with the prompt: SDXL joins the two before moving them
                                          **({"neg_cond": {k: v[0].to(device) for k, v in slider_neg.items()}}
                                             if slider_neg is not None else {}))
                bank.append(driver.practice_latent(vae, lat, slider_bank_res))
        if use_speed:
            net.set_enabled(SPEED, False)
            net.move_adapter(SPEED, "cpu")
        net.set_enabled(ADAPTER, True)
        net.set_trainable_multiplier(1.0)
        if was_training:
            dit.train()
        group.latents = bank
        torch.cuda.empty_cache()
        logger.info("[slider] %d practice images rendered at %dx%d", len(bank), slider_bank_res, slider_bank_res)

    if slider_prompts:
        render_bank()
    if do_compile:                       # decided before the load (below the Auto plan), applied last
        driver.compile_blocks(dit, "outside" if do_compile == "outside" else "inside",
                              blocks_to_swap if swapped else 0)
    from fizgig.training.optimizers import create_optimizer, group_rates, optimizer_lr, owns_its_rate
    if ftr is not None:
        params, optimizer, opt_label = [], None, ftr.opt_label
    else:
        params = net.parameters()
        logger.info((f"LoKR factor {lokr_factor}" if network_type == "lokr" else
                     f"LoRA rank {network_dim} alpha {network_alpha:g}") +
                    f": {len(net.trainable_modules())} modules, {sum(p.numel() for p in params) / 1e6:.1f}M "
                    f"trainable params")
        if share_worker is None:         # a round's LoRA in bf16, for a training share's estimate before its first round
            try:
                with open(os.path.join(output_dir, SHARE_DIR, "setup.json"), encoding="utf-8") as f:
                    _s = json.load(f)
                _s["lora_bytes"] = 2 * sum(p.numel() for p in params)
                _s["precision"] = precision      # Auto's pick here: the helper trains on the same base numbers
                _write_json(os.path.join(output_dir, SHARE_DIR, "setup.json"), _s)
            except (OSError, ValueError):
                pass
        opt_params = params
        if str(optimizer_type or "").lower() == "automagic3":
            groups = _optimizer_family_groups(desc, net, learning_rate)
            if groups:                   # one rate per group: each family of modules finds its own
                opt_params = groups
                logger.info("[optimizer] per-family rates: " + ", ".join(
                    f"{g['family']} ({g['modules']} modules)" for g in groups) + " - each votes its own rate")
            if desc.automagic_sign_window and "polarity_history" not in (optimizer_args or ""):
                optimizer_args = f"{optimizer_args or ''} polarity_history={desc.automagic_sign_window}".strip()
                logger.info(f"[optimizer] Automagic v3: sign window {desc.automagic_sign_window} (this family's "
                            f"default; set polarity_history in Optimizer Args to override)")
        _oargs = optimizer_args or ""
        if (desc.optimizer_weight_decay is not None and "weight_decay" not in _oargs
                and "adam" in str(optimizer_type or "").lower()):
            _oargs = (_oargs + f" weight_decay={desc.optimizer_weight_decay:g}").strip()
        optimizer, opt_label = create_optimizer(optimizer_type, opt_params, learning_rate, _oargs,
                                                eps_floor_8bit=desc.optimizer_eps_floor_8bit)
    if owns_its_rate(optimizer):        # Automagic v3 sets its own rate: the watcher and schedulers stand down
        if adaptive_lr:
            logger.info("[adaptive_lr] ignored - the optimizer sets its own learning rate")
        if per_image_lr or warmup_look_outliers:
            logger.info("[per-image LR] per-image LR and the look warm-up are off - the optimizer sets its own rate")
        adaptive_lr = per_image_lr = warmup_look_outliers = False
        logger.info(f"[optimizer] {opt_label} owns the learning rate from here ({learning_rate:.2e} is its start); "
                    f"the LR scheduler stands down")
    box_learning_rate = learning_rate   # the Learning Rate box, before adaptive LR's midpoint (saved for a resume)
    if adaptive_lr:                     # the watcher owns the rate: start at the geometric midpoint of Min/Max
        learning_rate = math.sqrt(adaptive_lr_min * adaptive_lr_max)
        for g in optimizer.param_groups:
            g["lr"] = learning_rate
        logger.info(f"[adaptive_lr] ENABLED - start_lr={learning_rate:.3e} min_lr={adaptive_lr_min:.3e} "
                    f"max_lr={adaptive_lr_max:.3e} (the Learning Rate box is ignored)")
    adaptive = (AdaptiveLR(adaptive_lr_min, adaptive_lr_max, clip_signal=desc.adaptive_lr_clip_signal)
                if adaptive_lr else None)
    ema = None
    if str(ema_decay).lower().startswith("short"):
        # short-run mode (from H3): the normal ramp never reaches 0.98 on a short run, so the window is sized to the
        # run instead - decay 1 - 4/steps (about the last quarter), fast ramp
        from fizgig.training.ema import EMAWeights
        _total = max(1, int(group.num_train_items) * max(1, int(max_train_epochs)))
        ema_decay = min(0.995, max(0.5, 1.0 - 4.0 / _total))
        ema = EMAWeights(net, ema_decay, ramp=2)
        logger.info(f"[ema] SHORT-RUN mode: {_total} steps -> decay {ema_decay:.3f} (window ~ a quarter of the run), "
                    f"fast ramp - checkpoints and previews use the average")
    elif ema_decay and float(ema_decay) > 0:
        from fizgig.training.ema import EMAWeights
        ema = EMAWeights(net, float(ema_decay))
        logger.info(f"[ema] ON at decay {float(ema_decay):g} - checkpoints and previews use the running average")

    start_epoch = global_step = 0
    if resume_state_dir:
        _fresh_lrs = [g["lr"] for g in optimizer.param_groups] if optimizer is not None else []
        start_epoch, global_step, meta = _load_state(resume_state_dir, net, optimizer, device, arch,
                                                     untagged_own=desc.resumes_untagged_states, driver=driver)
        if global_step is None:
            global_step = start_epoch * steps_per_epoch
        if adaptive:
            adaptive.load_state_dict(meta.get("adaptive_lr_state"))
        if (optimizer is not None and meta.get("own_state") and not meta.get("fresh_optimizer")
                and not owns_its_rate(optimizer)):
            resume_lr(optimizer, _fresh_lrs, meta, box_learning_rate, adaptive)
        if ema is not None and meta["own_state"] and os.path.exists(os.path.join(resume_state_dir, "ema.pt")):
            _esd = torch.load(os.path.join(resume_state_dir, "ema.pt"), map_location="cpu")
            if meta.get("legacy_perm"):     # an older trainer's order, as the optimizer's
                _sh = [None] * len(_esd["shadow"])
                for _i, _t in enumerate(_esd["shadow"]):
                    _sh[meta["legacy_perm"][_i]] = _t
                _esd["shadow"] = _sh
            ema.load_state_dict(_esd)
        logger.info(f"[resume] from {resume_state_dir}: continuing at epoch {start_epoch + 1}/{max_train_epochs}")
    from fizgig.families.loss_watch import Watch
    watch = Watch(output_dir, group, user_config, driver, log=log_per_image_loss, per_image_lr=per_image_lr,
                  auto_recaption=auto_recaption, warmup_look=warmup_look_outliers,
                  resume=bool(resume_state_dir), start_epoch=start_epoch, te_path=te_path,
                  trigger_word=trigger_word, trigger_position=trigger_position,
                  recaption_triggers=json.loads(recaption_triggers) if recaption_triggers else None,
                  recaption_instruction=recaption_instruction,
                  recaption_instruction_detailed=recaption_instruction_detailed, captioner_path=captioner,
                  recaption_task=recaption_task, recaption_task_detailed=recaption_task_detailed)
    # Gradient accumulation (Krea 2's rule): N micro-batches average into one update - each loss is divided by N, the
    # optimizer steps every N and at every epoch end (a partial group is flushed), so updates per epoch are
    # ceil(steps / N) and the schedule runs in those updates. Sliders already backpropagate two poles per step: 1.
    accum = 1 if slider else max(1, int(gradient_accumulation_steps or 1))
    updates_per_epoch = math.ceil(steps_per_epoch / accum)
    if accum > 1:
        logger.info(f"[grad_accum] {accum} micro-batches per optimizer step (effective batch {accum}); "
                    f"{updates_per_epoch} updates/epoch")
    elif int(gradient_accumulation_steps or 1) > 1:
        logger.info("[grad_accum] off for sliders: each step already backpropagates both ends")
    scheduler = None
    if ftr is None and not adaptive and not owns_its_rate(optimizer):
        scheduler = _step_scheduler(optimizer, lr_scheduler, lr_warmup_steps, updates_per_epoch * max_train_epochs,
                                    lr_scheduler_num_cycles, lr_scheduler_power)
        import warnings
        with warnings.catch_warnings():
            # a resume moves the schedule to where the run paused before the first optimizer step, on purpose;
            # PyTorch's "lr_scheduler.step() before optimizer.step()" warning is for training loops, not this
            warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before")
            for _ in range(start_epoch * updates_per_epoch if accum > 1 else global_step):
                scheduler.step()

    last_prompt = [None]
    slider_history = []                     # [{epoch, prompts, guidance}] - each live change, for the saved metadata
    anchor_losses = []                      # this epoch's anchor losses (a prompt slider with anchors)

    def slider_live(done):
        """A prompt slider's new prompts (the web UI's Slider prompts panel), taken at an epoch boundary: the poles
        encoded again, and a new base renders a new practice bank and becomes the previews' prompt. Training goes on
        from where it is, toward the new ends."""
        nonlocal slider_prompts, slider_guidance, slider_enc, encoded, sample_prompts, slider_anchors, anchor_enc
        new = _read_slider_live(output_dir) if slider_prompts else None
        if new is None:
            return
        # each end is the base and its own words, as the launch builds them (families/launch.py)
        prompts = [new["base"], f"{new['base']} {new['pos']}", f"{new['base']} {new['neg']}"]
        if (prompts == [str(x) for x in slider_prompts] and new["guidance"] == float(slider_guidance)
                and new["anchors"] == list(slider_anchors)):
            return
        base_changed = prompts[0] != str(slider_prompts[0])
        logger.info("[slider] new prompts from epoch %d: base '%s' | +1 -> '%s' | -1 -> '%s' (guidance %g)%s",
                    done + 1, *prompts, new["guidance"],
                    f" | anchors: {' | '.join(new['anchors'])}" if new["anchors"] else " | no anchors")
        parked = []
        try:
            enc = _encode_override(driver, te_path, prompts + new["anchors"], dit, device, parkable=not swapped)
            _all = [{k: v[None] for k, v in c.items()} for c in enc]
            slider_enc, anchor_enc = _all[:3], _all[3:]
            if base_changed:
                if desc.preview_park_optimizer and optimizer is not None:     # room for the renders, as previews do
                    for st in optimizer.state.values():
                        for k, v in st.items():
                            if torch.is_tensor(v) and v.is_cuda:
                                st[k] = v.to("cpu")
                                parked.append((st, k))
                    torch.cuda.empty_cache()
                render_bank()
            # the previews followed the old base (and its first anchor): they follow the new ones
            if prompts[0] and sample_prompts == [str(slider_prompts[0])] + list(slider_anchors[:1]):
                sample_prompts = [prompts[0]] + new["anchors"][:1]
                if encoded is not None and not sample_reference and not (sample_image and desc.preview_image):
                    encoded = [enc[0]] + enc[3:4]           # text-only previews: the new prompts' own encodings
        except Exception as exc:
            logger.warning(f"[slider] could not take the new prompts ({type(exc).__name__}: {str(exc)[:200]}) - "
                           "training goes on with the ones it had", exc_info=True)
            return
        finally:
            for st, k in parked:
                st[k] = st[k].to(device)
        slider_prompts, slider_guidance, slider_anchors = prompts, new["guidance"], new["anchors"]
        slider_history.append({"epoch": done, "prompts": prompts, "guidance": new["guidance"],
                               "anchors": new["anchors"]})
        try:
            with open(os.path.join(output_dir, SLIDER_LIVE_APPLIED), "w", encoding="utf-8") as f:
                json.dump({"epoch": done, "base": new["base"], "pos": new["pos"], "neg": new["neg"],
                           "guidance": new["guidance"], "anchors": new["anchors"],
                           "practice_rerendered": base_changed, "at": time.time()}, f)
        except OSError:
            pass

    def metadata(epoch):
        thumb = None if (metadata_thumbnail or "").lower() in ("off", "none") else (
            metadata_thumbnail or latest_sample_image(output_dir))
        md = build_metadata(None, arch, time.time(),
                            title=metadata_title if metadata_title is not None else resolve_title(
                                output_name, metadata_trigger_phrase),
                            author=metadata_author,
                            description=metadata_description if metadata_description is not None else last_prompt[0],
                            license=metadata_license, tags=metadata_tags, trigger_phrase=metadata_trigger_phrase,
                            thumbnail=thumbnail_data_uri(thumb))
        md.update({"ss_network_module": f"fizgig.families ({desc.key}, {network_type})",
                   "ss_network_dim": str(network_dim if network_type == "lora" else lokr_factor),
                   "ss_network_alpha": str(network_alpha if network_type == "lora" else 1.0),
                   **({"ss_lokr_factor": str(lokr_factor)} if network_type == "lokr" else {}),
                   "ss_architecture": arch, "ss_epoch": str(epoch),
                   "ss_optimizer": opt_label, "ss_learning_rate": f"{learning_rate:g}",
                   "ss_training_adapter": os.path.basename(training_adapter) if training_adapter else "none"})
        if context_lora_path:
            md.update({"ss_context_lora": os.path.basename(context_lora_path),
                       "ss_context_lora_strength": str(context_lora_strength)})
        if slider:
            # the deploy contract: strength is the dial (-1 one pole, +1 the other); tools read this
            md.update({"ss_slider": "prompt_pairs" if slider_prompts else "image_pairs"})
            if slider_prompts:
                md.update({"ss_slider_prompts": json.dumps([str(x) for x in slider_prompts]),
                           "ss_slider_guidance": f"{float(slider_guidance):g}"})
                if slider_anchors:
                    md.update({"ss_slider_anchors": json.dumps(list(slider_anchors)),
                               "ss_slider_anchor_weight": f"{float(slider_anchor_weight):g}"})
                if slider_history:          # the prompts it was steered through, from which epoch
                    md.update({"ss_slider_prompt_history": json.dumps(slider_history)})
            else:
                md.update({"ss_slider_diff_weight": f"{float(slider_diff_weight):g}"})
        if train_blocks:
            md.update({"ss_train_blocks": ",".join(train_blocks)})
        if masked_training:
            md.update({"ss_masked_training": "True", "ss_mask_floor": f"{float(mask_floor):g}",
                       "ss_masked_images": str(masked_images)})
            if mask_prior:
                md["ss_mask_prior"] = f"{mask_prior:g}"
        md.update(driver.run_metadata())
        return md

    def save_lora(path, epoch):
        if ftr is not None:
            ftr.save(path, epoch)
            return
        if ema is not None:
            ema.swap_in()
        try:
            # the family's own format: split by default, the fused checkpoint layout only when the description says so
            net.save(path, metadata(epoch), fuse=desc.lora.save_fused)
            # a ComfyUI-fused export is a separate transform (driver.export_lora_state_dict), offered explicitly
        finally:
            if ema is not None:
                ema.swap_out()
        logger.info(f"[save] {path}")

    previews_off = [False]

    def previews(epoch):
        """The epoch's previews. A preview that fails (out of memory, say) switches previews off for the rest of the
        run and training carries on, as the original Krea 2 trainer did: a picture is never worth the run."""
        if encoded is None or previews_off[0]:
            return
        parked = []
        try:
            if desc.preview_park_optimizer and optimizer is not None:
                for st in optimizer.state.values():
                    for k, v in st.items():
                        if torch.is_tensor(v) and v.is_cuda:
                            st[k] = v.to("cpu")
                            parked.append((st, k))
                if parked:
                    torch.cuda.empty_cache()
                    logger.info(f"[preview] {len(parked)} optimizer tensors parked on CPU for the preview")
            _previews(epoch)
        except Exception as exc:
            previews_off[0] = True
            logger.warning(f"[previews] the epoch {epoch} preview failed ({type(exc).__name__}: {str(exc)[:200]}) - "
                           f"previews are off for the rest of this run; training continues", exc_info=True)
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            for st, k in parked:            # the next training step needs it back, whatever happened
                st[k] = st[k].to(device)

    ov_cache = [None, None]                  # the override's prompts and their encoding, kept while they don't change

    def _preview_due(done):
        """This epoch previews, by the launch cadence or the live override's (the Samples tab, changed mid-run)."""
        every, until, after = sample_every_n_epochs, sample_until_epoch, sample_every_n_epochs_after
        ov = _read_sample_override(output_dir)
        if ov:
            every = ov["every"] if ov.get("every") else every
            until = ov["until"] if ov.get("until") is not None else until
            after = ov["after"] if ov.get("after") is not None else after
        return preview_due(done, every, until, after)
    speed_bad = [False]                      # the speed LoRA matched no layers: not tried again
    ck_tried = [None]                        # the preview checkpoint a live change last asked for
    compared = [0]                           # the helper's last Check picture request this run answered

    def _previews(epoch):
        conds, w, h, sd, prompts = encoded, sample_width, sample_height, sample_seed, sample_prompts
        steps_now, cfg_now, ask_steps, ask_cfg = sample_steps, sample_cfg_scale, asked_steps, asked_cfg
        speed_now = speed_desc.settings if (speed_lora and speed_desc) else None
        base_strength = (speed_desc.strength if speed_lora_strength is None else speed_lora_strength) \
            if speed_now is not None else 0
        strength_now = base_strength
        ov = _read_sample_override(output_dir)
        if ov and slider:       # a slider preview is its own prompt across strengths: an override never applies
            logger.info("[sample override] ignored - slider previews always show the slider's own prompt")
            ov = None
        if ov:
            logger.info(f"[sample override] active - {len(ov['prompts'])} prompt(s), '{ov['prompt'][:60]}' "
                        f"seed={ov['seed']} {ov['width']}x{ov['height']}")
            try:
                if ov_cache[0] != ov["prompts"]:     # encoded again only when the prompts changed
                    ov_cache[:] = [list(ov["prompts"]), _encode_override(driver, te_path, ov["prompts"], dit, device,
                                                                         parkable=not swapped, references=ref_imgs)]
                conds = ov_cache[1]
                sd, prompts = ov["seed"], list(ov["prompts"])
                if ov.get("steps"):
                    steps_now = ask_steps = ov["steps"]
                if ov.get("cfg") is not None:
                    cfg_now = ask_cfg = ov["cfg"]
                if ov.get("turbo") is not None:
                    if speed_now is None and ov["turbo"] > 0:
                        logger.info("[sample override] Turbo strength ignored - no Turbo LoRA file is set in "
                                    "Preferences for this run")
                    elif speed_now is not None:
                        strength_now = ov["turbo"]
                if not ref_imgs:        # an edit keeps the reference's canvas: its latents are already encoded
                    w, h = ov["width"], ov["height"]
                    if lowmem and max(w, h) > 768:
                        w, h = _cap_canvas(w, h)
            except Exception:
                logger.exception("[sample override] could not encode the override prompt - using the configured ones")
                conds = encoded
        if not sd:          # seed 0 = a fresh random seed every preview round, as on the other trainers
            sd = random.randint(1, 2 ** 31 - 1)
        if cfg_now and cfg_now > 1.0 and neg is None:
            # CFG asked for mid-run on a run that has no negative encoded (a CFG-free family): render without it
            logger.warning(f"[sample] CFG {cfg_now:g} needs a negative prompt this run did not encode - this preview "
                           "renders at CFG 1")
            cfg_now = 1.0
        nonlocal ckpt_previews
        on_ck = ckpt_previews is not None
        if ov and ov.get("checkpoint") is not None and not slider and encoded is not None:
            # the Samples tab's preview-model tick, changed while the run goes: unticked = this round on the training
            # model (with the tab's own steps and CFG), ticked = on the checkpoint, loaded the first time it's asked for
            want = ov["checkpoint"]
            if not want:
                on_ck = False
            elif ckpt_previews is None or os.path.normcase(ckpt_previews.path) != os.path.normcase(want):
                if ck_tried[0] != want:         # asked once per file: a run that can't take it says so once
                    ck_tried[0] = want
                    ckpt_previews = _make_ckpt_previews(want) or ckpt_previews
                on_ck = ckpt_previews is not None and os.path.normcase(ckpt_previews.path) == os.path.normcase(want)
        if speed_now is not None:       # the Turbo LoRA at this round's strength; 0 = this round without it
            if strength_now <= 0:
                speed_now = None
            elif not on_ck or slider:     # on the training model: loaded the first time it is asked for
                if net.has(SPEED) or (not speed_bad[0] and _add_speed(strength_now)):
                    net.set_strength(SPEED, strength_now)
                else:
                    speed_bad[0], speed_now = True, None
            if on_ck and speed_now is not None and ckpt_previews.speed is None:
                speed_now = None                # the Turbo LoRA was added for the training model's previews
        if on_ck and not slider:
            ck = desc.preview_checkpoint_sampling          # the checkpoint's own recipe (Distilled: 4 steps, no CFG)
            # a checkpoint that takes the speed LoRA (Kromoss): this round with it while its strength is above 0
            turbo_ck = ckpt_previews.speed is not None and speed_now is not None
            ck_steps = (steps_now if turbo_ck else (ask_steps or ck.steps))
            ck_cfg = (cfg_now if turbo_ck else (ck.cfg if ask_cfg is None else ask_cfg))
            if not turbo_ck and ck.cfg <= 1.0 < ck_cfg:
                # a guidance-distilled checkpoint (Qwen's Turbo, Klein's Distilled) renders burnt above CFG 1
                logger.info(f"[sample] {os.path.basename(ckpt_previews.path)} renders at CFG 1 (it's distilled); the "
                            f"CFG {ck_cfg:g} asked for is for the training model's previews")
                ck_cfg = 1.0

            def _ck_render(m, fl, swp):
                if turbo_ck and not fl.has(SPEED):     # loaded onto the checkpoint the first round that asks for it
                    fl.add_file(ckpt_previews.speed[0], SPEED, strength_now)
                    driver.frozen_file_added(m, ckpt_previews.speed[0], strength_now, "speed")
                if ckpt_previews.speed is not None and fl.has(SPEED):
                    fl.set_strength(SPEED, strength_now if turbo_ck else 0.0)
                return _render_previews(
                    driver, m, fl, vae, conds, sample_dir, epoch, output_name=output_name, steps=ck_steps, cfg=ck_cfg,
                    neg=neg, width=w, height=h, seed=sd,
                    # the asked CFG, not the checkpoint's own: _render_previews reads CFG 1 as "the recipe's"
                    speed=speed_desc.settings if turbo_ck else dataclasses.replace(ck, cfg=ck_cfg),
                    swapped=bool(swp), refs=ref_latents)
            ckpt_previews.render(dit, net, ema, _ck_render)
            write_mode({"model": os.path.basename(ckpt_previews.path), "steps": ck_steps, "cfg": ck_cfg,
                        "turbo": speed_desc.name if turbo_ck else "", "strength": strength_now if turbo_ck else 0},
                       epoch)
            last_prompt[0] = prompts[-1] if prompts else None
            _record_prompts(sample_dir, output_name, epoch, prompts)
            return
        hp = _helper_previews(output_dir) if not (slider or ref_latents or finetune) else None
        compare = None  # Check picture: this round renders here too, and the helper's is compared with it
        if hp and hp.get("compare") and hp["compare"] == compared[0]:
            hp = {**hp, "compare": 0}                   # that request was this run's already
        if hp:          # a helper renders this round from the epoch's LoRA, with the same setup: training goes on
            lp, ls = ((ov["look"], ov.get("look_strength")) if ov and ov.get("look") is not None
                      else (preview_lora, preview_lora_strength))
            look_on = bool(lp) and look_said[0] != "bad"   # ticked: the helper fetches its own copy if need be
            if hp.get("compare"):                           # Check picture: exactly as this GPU renders it now
                look_on = bool(_look_ready(lp, ls)) if lp else False
            rd = os.path.join(output_dir, HELPER_ROUNDS, f"{epoch:06d}")
            try:
                os.makedirs(rd, exist_ok=True)
                if ema is not None:
                    ema.swap_in()
                try:
                    # exactly what trains; half the size (bf16) for a helper on a slow link (the lead measures it)
                    net.save(os.path.join(rd, "lora.safetensors"),
                             dtype=torch.bfloat16 if hp.get("dtype") == "bf16" else torch.float32)
                finally:
                    if ema is not None:
                        ema.swap_out()
                # the encoded prompts too: the helper needs no text encoder, and conditions exactly as here
                cpu = lambda x: ({k: cpu(v) for k, v in x.items()} if isinstance(x, dict) else
                                 [cpu(v) for v in x] if isinstance(x, (list, tuple)) else
                                 x.detach().cpu() if torch.is_tensor(x) else x)
                torch.save({"conds": cpu(conds), "neg": cpu(neg)}, os.path.join(rd, "conds.pt"))
                job = {"family": desc.key, "epoch": epoch, "output_name": output_name, "prompts": list(prompts),
                       "conds": "conds.pt",
                       "negative": (sample_negative or "") if neg is not None else None, "seed": sd,
                       "width": w, "height": h, "steps": steps_now, "cfg": cfg_now,
                       "turbo": strength_now if speed_now is not None else 0.0,
                       "look": (ls or preview_lora_strength) if look_on else 0.0,
                       "strengths": list(sample_strengths) if epoch > 0 else [],
                       "network": {"dim": network_dim, "alpha": network_alpha, "type": network_type,
                                   "factor": lokr_factor, "blocks": sorted(train_blocks) if train_blocks else None},
                       "options": dict(family_options or {}),
                       "precision": precision, "context": [context_lora_path, context_lora_strength]
                       if context_lora_path else None, "lowmem": bool(lowmem)}
                if hp.get("compare"):
                    compared[0] = hp["compare"]
                    compare = (rd, job)
                    raise _Compare
                with open(os.path.join(rd, "round.json.tmp"), "w", encoding="utf-8") as f:
                    json.dump(job, f)
                os.replace(os.path.join(rd, "round.json.tmp"), os.path.join(rd, "round.json"))
                logger.info(f"[sample] epoch {epoch}: previews handed to {hp.get('name') or 'the helper'} "
                            "- training carries on")
                last_prompt[0] = prompts[-1] if prompts else None
                _record_prompts(sample_dir, output_name, epoch, prompts)
                write_mode({"model": "", "steps": steps_now, "cfg": cfg_now,
                            "turbo": speed_desc.name if speed_now else "",
                            "strength": strength_now if speed_now else 0}, epoch, look_on, hp.get("name") or "helper")
                return
            except _Compare:
                logger.info(f"[sample] epoch {epoch}: Check picture - rendered here and on "
                            f"{hp.get('name') or 'the helper'}, to compare")
            except Exception:
                logger.warning("[sample] could not hand the previews to the helper - rendering them here",
                               exc_info=True)
                shutil.rmtree(rd, ignore_errors=True)
        local = _render_previews(driver, dit, net, vae, conds, sample_dir, epoch, output_name=output_name,
                         steps=steps_now, cfg=cfg_now, neg=neg, width=w, height=h,
                         seed=sd, ema=ema,
                         speed=speed_now, lowmem=lowmem,
                         swapped=bool(swapped), refs=ref_latents, slider=slider,
                         strengths=sample_strengths if epoch > 0 else (),   # before training the LoRA is all zero
                         look=(look_now := (_look_ready() if not ov or ov.get("look") is None   # as launched
                                            else ov["look"] != "" and _look_ready(ov["look"], ov.get("look_strength")))))
        if compare is not None:     # the helper's copy of this round, for Check picture (not for the gallery)
            rd, job = compare
            job["compare"] = [os.path.abspath(p) for p in local]
            with open(os.path.join(rd, "round.json.tmp"), "w", encoding="utf-8") as f:
                json.dump(job, f)
            os.replace(os.path.join(rd, "round.json.tmp"), os.path.join(rd, "round.json"))
        last_prompt[0] = prompts[-1] if prompts else None
        _record_prompts(sample_dir, output_name, epoch, prompts)
        write_mode({"model": "", "steps": steps_now, "cfg": cfg_now, "turbo": speed_desc.name if speed_now else "",
                    "strength": strength_now if speed_now else 0}, epoch, look_now)

    def _make_ckpt_previews(path):
        """Previews on this preview checkpoint, or None (with the reason logged): at the start, or when the Samples
        tab's tick turns it on mid-run."""
        if not desc.train_preview_checkpoint:
            logger.warning(f"[sample] {desc.display_name} has no checkpoint previews - ignoring --preview_checkpoint")
        elif finetune:      # the checkpoint carries a LoRA; a fine-tune changes the model itself, so it previews on that
            logger.info("[sample] a fine-tune previews on the model it trains, not the preview checkpoint")
        elif not os.path.isfile(path):
            logger.warning(f"[sample] preview checkpoint {path} not found - previews use the training model")
        else:
            from fizgig.families.base_check import checkpoint_takes_speed
            ck_speed = speed_lora if (speed_lora and speed_desc and checkpoint_takes_speed(desc, path)) else None
            if ck_speed:
                logger.info(f"[sample] {os.path.basename(path)} takes the {speed_desc.name}: previews on it use it "
                            f"while its strength is above 0")
            return _CheckpointPreviews(
                driver, path, device, preview_checkpoint_cache, preview_int8,
                context=(context_lora_path, context_lora_strength) if context_lora_path else None,
                speed=(ck_speed, speed_desc.strength if speed_lora_strength is None else speed_lora_strength)
                if ck_speed else None)
        return None

    ckpt_previews = _make_ckpt_previews(preview_checkpoint) if (preview_checkpoint and encoded is not None) else None
    def write_mode(mode, epoch=None, look=False, helper=""):
        """What this run's previews render with now, for the gallery (<sample_dir>/.previews_<name>.json): written at
        the start and after each preview round, so a live change of steps, CFG or Turbo strength shows. After a round
        (`epoch`) its setup joins `rounds` (epoch -> checkpoint / look / turbo), so each picture keeps its own."""
        if mode.get("turbo") and speed_desc is not None and speed_desc.gallery_note:
            mode = {**mode, "turbo_note": speed_desc.gallery_note}     # the family's word on its Turbo previews
        path = os.path.join(sample_dir, f".previews_{output_name}.json")
        try:
            with open(path, encoding="utf-8") as f:
                rounds = json.load(f).get("rounds") or {}               # this run's earlier rounds (a resume's too)
        except (OSError, ValueError, AttributeError):
            rounds = {}
        if epoch is not None:
            rounds[str(epoch)] = {"checkpoint": bool(mode.get("model")), "look": bool(look),
                                  "turbo": bool(mode.get("turbo")), **({"helper": helper} if helper else {})}
        mode = {**mode, "family": desc.key, "rounds": rounds}
        try:
            os.makedirs(sample_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(mode, f)
        except OSError as e:
            logger.warning(f"[sample] could not record the preview settings ({e})")

    if encoded is not None:         # what this run's previews render with, at the start
        if ckpt_previews is not None and not slider:
            ck = desc.preview_checkpoint_sampling
            mode = {"model": os.path.basename(preview_checkpoint), "turbo": "",
                    "steps": asked_steps or (ck.steps if ck else sample_steps),
                    "cfg": (ck.cfg if ck else sample_cfg_scale) if asked_cfg is None
                    else (1.0 if ck and ck.cfg <= 1.0 else asked_cfg)}
            if ckpt_previews.speed is not None and not speed_idle:
                mode.update(turbo=speed_desc.name, strength=ckpt_previews.speed[1], steps=sample_steps,
                            cfg=sample_cfg_scale)
        else:
            mode = {"model": "", "steps": sample_steps, "cfg": sample_cfg_scale,
                    "turbo": speed_desc.name if (speed_lora and speed_desc and not speed_idle) else "",
                    "strength": (speed_desc.strength if speed_lora_strength is None else speed_lora_strength)
                    if (speed_lora and speed_desc and not speed_idle) else 0}
        write_mode(mode)

    def state(epoch, required=False):
        """Save the resume state. A failed write (a full disk, or a sync / antivirus program holding the file) is
        retried once, then only warned about: the epoch's LoRA is already saved, and the state only serves a resume.
        A pause needs it (required=True), so there the second failure stops the run."""
        for attempt in (1, 2):
            try:
                _save_state(output_dir, output_name, net, optimizer, epoch=epoch, global_step=global_step,
                            arch_id=arch, ema=ema,
                            extra={"learning_rate": box_learning_rate,
                                   **({"adaptive_lr_state": adaptive.state_dict()} if adaptive else {})})
                return True
            except (OSError, RuntimeError) as e:
                if attempt == 2 and required:
                    raise
                if attempt == 1:
                    logger.warning(f"[state] saving the epoch {epoch} resume state failed ({e}) - trying again")
                    time.sleep(5)
                else:
                    # a half-written state must not count as the newest when states are pruned
                    shutil.rmtree(os.path.join(output_dir, f"{output_name}-{epoch:06d}-state"), ignore_errors=True)
                    logger.warning(f"[state] the epoch {epoch} resume state could not be saved ({e}). Training "
                                   f"carries on and the epoch's LoRA is saved; only resuming from this epoch is "
                                   f"lost. Check free disk space, and keep the output folder out of OneDrive / "
                                   f"Dropbox sync: {output_dir}")
        return False

    if sample_at_first and start_epoch == 0:
        previews(ftr.epochs_done if ftr is not None else 0)

    # ---- train ----------------------------------------------------------------------------------
    gen = torch.Generator().manual_seed(seed + start_epoch)
    pause_flag = os.path.join(output_dir, ".pause_requested")
    recorder = LossRecorder()
    # the web UI's loss graph: each step's loss, numbered across a resume / a fine-tune's continuation
    step_off = ftr.epochs_done * steps_per_epoch if ftr is not None else 0
    steplog = StepLog(output_dir, output_name, start_step=step_off + global_step)
    for _w in ("sample", "save"):          # a request left from an earlier run is not this one's
        take_request(output_dir, _w)
    progress = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step, desc="steps", smoothing=0)
    dit.train()
    pending = 0                            # micro-batches backpropagated since the last optimizer step
    last_grad_norm = None                  # pre-clip total norm of the most recent optimizer step (None = clipping off)
    lr_acc = []                            # this window's per-step LR multipliers (driver.step_policy)

    _block_params = {}

    def _step_freeze(blocks):
        """requires_grad off for the trainable adapter's parameters in `blocks` (block ids) - under a fine-tune, the
        active window's weights in those blocks; returns them."""
        if not blocks:
            return []
        if ftr is not None:
            out = [p for p in ftr.be.params_in_blocks(blocks) if p.requires_grad]
            for p in out:
                p.requires_grad_(False)
            return out
        if not _block_params:
            from fizgig.families.lora import TRAINABLE as _TR
            for _full, _w in net.wrapped.items():
                if _TR in _w.adapters:
                    _block_params.setdefault(driver.block_of(_full), []).extend(_w.adapters[_TR].parameters())
        out = [p for b in blocks for p in _block_params.get(b, ())]
        for p in out:
            p.requires_grad_(False)
        return out

    def _share_combine(sn):
        """The helper's share back: start + this GPU's change + the helper's (what one GPU would have moved over the
        whole epoch, give or take both halves starting from the same point). If it doesn't come back in time, the
        round's epochs it held are returned: this GPU then trains those images itself, so every image still trains
        every epoch."""
        from safetensors.torch import load_file, save_file
        took = time.time() - sn["t0"]
        _write_json(os.path.join(sn["dir"], "lead.json"), {"steps": sn["steps"], "seconds": round(took, 1)})
        theirs = os.path.join(sn["dir"], "theirs.safetensors")
        deadline, cap = time.time() + max(120.0, 3 * took), time.time() + 900
        if not os.path.exists(theirs):
            logger.info(f"[share] epoch {sn['epoch']}: waiting for {sn['name']}'s share")
        while not os.path.exists(theirs) and time.time() < min(deadline, cap):
            if os.path.exists(os.path.join(sn["dir"], "failed.json")):
                break
            _f = _helper_share(output_dir)              # still sending, training or handing it back: wait on
            if _f and int(_f.get("busy") or 0) == sn["epoch"]:
                deadline = max(deadline, time.time() + 60)
            time.sleep(0.5)
        # closed: whatever arrives for this round from now on is too late (the helper's side drops it)
        _write_json(os.path.join(sn["dir"], "closed.json"), {"at": time.time()})
        try:
            if not os.path.exists(theirs):
                logger.warning(f"[share] epoch {sn['epoch']}: {sn['name']}'s share didn't come back - this GPU "
                               "trains its images too")
                return list(range(sn["epoch"], sn["last"] + 1))
            mine = net.state_dict(torch.float32)
            try:
                th = load_file(theirs)                  # the helper's change over the round
                bad = next((k for k, v in th.items() if k not in mine or v.shape != mine[k].shape
                            or not torch.isfinite(v).all()), None)
            except Exception as e:
                th, bad = None, f"unreadable ({e})"
            if th is None or bad is not None:
                logger.warning(f"[share] epoch {sn['epoch']}: {sn['name']}'s share came back wrong ({bad}) - this "
                               "GPU trains its images too")
                return list(range(sn["epoch"], sn["last"] + 1))
            comb = {k: (v + th[k].float()) if k in th else v for k, v in mine.items()}
            save_file(comb, os.path.join(sn["dir"], "combined.safetensors"))
            before = [p.detach().float().clone() for p in ema.params] if ema is not None else None
            net.load_trainable(os.path.join(sn["dir"], "combined.safetensors"))
            if ema is not None:          # the average moves by the helper's change too, or saves and previews (made
                with torch.no_grad():    # from it) would hold only this GPU's half of the round
                    for sh, p, b in zip(ema.shadow, ema.params, before):
                        sh.add_(p.detach().float().to(sh.device) - b.to(sh.device))
            logger.info(f"[share] epoch {sn['epoch']}: combined - {sn['steps']} images here, the rest on {sn['name']} "
                        f"(waited {max(0.0, time.time() - sn['t0'] - took):.0f} s for it)")
            return []
        finally:
            for n in ("start.safetensors", "combined.safetensors", "theirs.safetensors"):
                try:
                    os.remove(os.path.join(sn["dir"], n))
                except OSError:
                    pass

    def _update():
        nonlocal pending, last_grad_norm
        if ftr is not None and ftr.fused is not None:
            pending = 0                    # every parameter already stepped from its own gradient hook
            lr_acc.clear()
            return
        if max_grad_norm:
            pre = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            last_grad_norm = pre.item()          # surface the pre-clip norm for logging
            if adaptive is not None and adaptive.clip_signal:
                adaptive.record_clip(pre, max_grad_norm)
        m = sum(lr_acc) / len(lr_acc) if lr_acc else 1.0
        lr_acc.clear()
        if optimizer.__class__.__name__ == "Automagic3":
            m = 1.0                        # Automagic v3 owns the rate (as on the old H3 trainer)
        if m != 1.0:                       # the window's mean, composed into the LR for this step only (never the loss:
            held = [g["lr"] for g in optimizer.param_groups]        # Adam is invariant to a constant on the gradient)
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] * m
        optimizer.step()
        if m != 1.0:
            for g, lr in zip(optimizer.param_groups, held):
                g["lr"] = lr
        driver.after_optimizer_step()
        if scheduler is not None and not catching[0]:
            scheduler.step()
        if ema is not None:
            ema.update()                   # after the clipped step, so the average tracks what was applied
        pending = 0

    # Warm-up reassurance: the first two epochs of a run start slowly (first-sight kernel planning, cuBLAS picks,
    # allocator and cache warm-up, and on a compiled run the blocks compiling for each new shape) - a crawling bar
    # looks like a hang, so a gentle note repeats every ~30 s while it lasts.
    warmup_note = [0.0]
    share_now = [None]       # this epoch's training share: {frac, epoch, ...} or None (this GPU trains every image)
    _share_keys = sorted({str(it.item_key) for ds in getattr(group, "datasets", [])
                          for b in getattr(getattr(ds, "batch_manager", None), "buckets", {}).values() for it in b})
    share_skip = [0]
    share_off_said = [False]
    catching = [False]       # training a round's images the helper didn't bring back (counted once already)

    def _epoch_batches(epoch):
        """This epoch's batches as (i, batch, only): its own pass (only None), then - when it closes a round whose
        share didn't come back - the helper's images of each of the round's epochs again (only: those keys), so
        both sides stay in step: every image trains every epoch whatever happens to the helper."""
        nonlocal global_step
        for i, batch in enumerate(loader):
            yield i, batch, None
        sn = share_now[0]
        if share_worker is not None or sn is None or epoch + 1 != sn["last"]:
            return
        if pending:                        # a partial group: settle the optimizer before the combine
            _update()
        _wn = adaptive._weight_norm(net.trainable_modules()) if adaptive and adaptive.prev_weight_norm else None
        missed = _share_combine(sn)
        _k = sn["last"] - sn["epoch"] + 1
        if _wn is not None and _k > 1:
            # the helper's change covers the round's k epochs: Adaptive LR's weight-growth check counts one
            # epoch's worth of it here, not k (or a multi-epoch round would read as a jump and roll back)
            adaptive.prev_weight_norm += (adaptive._weight_norm(net.trainable_modules()) - _wn) * (1 - 1 / _k)
        if not missed:
            return
        gs0, n0 = global_step, progress.n
        catching[0] = True
        try:
            for ep in missed:
                only = _share_split(seed, ep, _share_keys, sn["frac"])
                logger.info(f"[share] epoch {ep}: training {sn['name']}'s {len(only)} images here")
                for i, batch in enumerate(loader):
                    if str((batch.get("item_keys") or [""])[0]) in only:
                        yield i, batch, only
                        global_step, progress.n = gs0, n0      # the bar and the epoch shown stay where they are
            if pending:
                _update()
        finally:
            catching[0] = False
            global_step = gs0              # those steps were counted when they were skipped
            progress.n = n0
            progress.refresh()

    def _share_ok():
        # gradient_checkpointing off (or partly): a big card planned this run without it, and a helper trains with it - the halves
        # would differ, so no share (set a helper to share before the run starts and it keeps checkpointing)
        return (not (slider or ftr is not None or dop_cond) and watch.watch is None and optimizer is not None
                and gradient_checkpointing and not gc_off_blocks)

    def _worker_epochs():
        """--share_worker: each epoch the lead hands over (its job file: the start LoRA, the share, the LR), in turn;
        a stop file, or no job for 30 minutes, ends it."""
        _write_json(os.path.join(share_worker, "ready.json"), {"at": time.time(), "steps_per_epoch": steps_per_epoch})
        idle = time.time()
        while True:
            jobs = sorted(glob.glob(os.path.join(share_worker, "*.job.json")))
            if os.path.exists(os.path.join(share_worker, "stop")) or (not jobs and time.time() - idle > 1800):
                return
            if not jobs:
                time.sleep(0.5)
                continue
            work = jobs[0][:-len(".job.json")] + ".working"
            os.replace(jobs[0], work)
            with open(work, encoding="utf-8") as f:
                job = json.load(f)
            job["work"] = work
            share_now[0] = job
            for ep in job.get("epochs") or [job["epoch"]]:
                yield int(ep) - 1
            idle = time.time()

    if share_worker is None:
        # what keeps this run's previews on this GPU (a helper renders a normal LoRA run's only), for the web UI to say
        _pv_off = ("a slider's previews" if slider else "an edit LoRA's previews" if ref_latents else
                   "a fine-tune's previews" if finetune else
                   "previews on a separate preview model (such as Klein's Distilled)" if ckpt_previews is not None else "")
        _write_json(os.path.join(output_dir, HELPER_PREVIEWS_OFF), {"run": output_name, "launch": _launch, "why": _pv_off})
        if _pv_off and _helper_previews(output_dir):
            logger.info(f"[helpers] this run's previews render on this GPU: helpers don't render {_pv_off} yet")
    if share_worker is None and _share_ok():
        # Start together (the Helpers tab's tick): while the web UI is getting this run's helper ready, the first
        # epoch waits for it, so both sides train from the start. Unticked, the helper leaving or its start giving
        # up lets go of the wait (as does a web UI that stopped refreshing it)
        _noted = 0.0
        while _helper_share(output_dir, output_name) is None and not os.path.exists(pause_flag):
            _w = _share_wait_note(output_dir, output_name, _launch)
            if _w is None:
                break
            if time.time() - _noted > 30:
                logger.info(f"[share] waiting for {_w.get('name') or 'the helper'} to get ready before the first "
                            f"epoch, so both train from the start" + (f": {_w['text']}" if _w.get("text") else "")
                            + " (untick Start together on the Helpers tab to start now)")
                _noted = time.time()
            time.sleep(2)
    prior = _MaskPrior(driver, dit, net, mask_prior, swapped) if mask_prior else None
    for epoch in (_worker_epochs() if share_worker else range(start_epoch, max_train_epochs)):
        shared_epoch.value = epoch + 1
        driver.begin_epoch(epoch + 1 + (ftr.epochs_done if ftr is not None else 0))   # a continued fine-tune counts on
        if share_worker and epoch + 1 == int(share_now[0]["epoch"]):   # a round's first epoch: the lead's LoRA
            _sj = share_now[0]
            if net.load_trainable(_sj["start"]) == 0:
                raise RuntimeError("[share] the lead's LoRA matched none of this model's layers")
            # the lead's rate: its scheduler's place (this run's own schedule, stepped to it), or none - Adaptive LR
            # or a constant rate, held for the round as the lead holds it through an epoch
            scheduler = None
            if _sj.get("sched") is not None:
                scheduler = _step_scheduler(optimizer, lr_scheduler, lr_warmup_steps,
                                            updates_per_epoch * max_train_epochs, lr_scheduler_num_cycles,
                                            lr_scheduler_power)
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=r"Detected call of `lr_scheduler\.step\(\)` before")
                    for _ in range(int(_sj["sched"])):
                        scheduler.step()
            else:
                for g, lr in zip(optimizer.param_groups, _sj.get("lrs") or []):
                    g["lr"] = lr
            logger.info(f"[share] epoch {epoch + 1}: from the lead's LoRA, LR {optimizer.param_groups[0]['lr']:.3g}"
                        + (" (its schedule, at the lead's step)" if scheduler is not None else ""))
            _sj.update(steps=0, t0=time.time())
        elif share_worker is None:
            if share_now[0] is not None and epoch + 1 > share_now[0]["last"]:
                share_now[0] = None
            _hs = _helper_share(output_dir, output_name) if share_now[0] is None and _share_ok() else None
            if ((not gradient_checkpointing or gc_off_blocks) and not share_off_said[0] and share_now[0] is None
                    and _helper_share(output_dir, output_name)):
                share_off_said[0] = True
                logger.info("[share] a helper is ready to share, but this run trains without (all of) gradient checkpointing "
                            "(a big card's plan) and a helper trains with it - the halves would differ, so this GPU "
                            "trains every image. Set the helper to share before starting a run and both keep it on.")
            if _hs:                  # a helper trains its share of this round's epochs, from the same start
                # the round: up to `every` epochs, ending early at any epoch that previews or saves (so the pictures
                # and files hold both halves), at a pause, and at the run's end
                _last = epoch + 1
                while (_last < epoch + max(1, int(_hs.get("every") or 1)) and _last < max_train_epochs
                       and not (sample_every_n_epochs and _preview_due(_last))
                       and not preview_due(_last, save_every_n_epochs, save_until_epoch, save_every_n_epochs_after)
                       and not os.path.exists(pause_flag)):
                    _last += 1
                _sd = os.path.join(output_dir, SHARE_DIR, f"{epoch + 1:06d}")
                shutil.rmtree(_sd, ignore_errors=True)    # a resumed run's earlier go at this epoch: not this round
                os.makedirs(_sd, exist_ok=True)
                net.save(os.path.join(_sd, "start.safetensors"), dtype=torch.bfloat16)
                _write_json(os.path.join(_sd, "job.json"), {
                    "epoch": epoch + 1, "epochs": list(range(epoch + 1, _last + 1)), "frac": float(_hs["frac"]),
                    "seed": seed, "lrs": [g["lr"] for g in optimizer.param_groups],
                    "sched": scheduler.last_epoch if scheduler is not None else None, "launch": _launch})
                share_now[0] = {"epoch": epoch + 1, "last": _last, "frac": float(_hs["frac"]), "dir": _sd,
                                "steps": 0, "t0": time.time(), "name": _hs.get("name") or "the helper"}
                logger.info(f"[share] epoch{'s' if _last > epoch + 1 else ''} {epoch + 1}"
                            + (f"-{_last}" if _last > epoch + 1 else "") + f": {share_now[0]['name']} trains about "
                            f"{float(_hs['frac']) * 100:.0f}% of the images, combined at the end")
        if ftr is not None and ftr.needs_window(epoch):
            optimizer = params = None       # let go of the outgoing window before the swap, or both sit in VRAM
            params, optimizer = ftr.enter(epoch)                 # the epoch's window, and a fresh optimizer for it
        if ftr is not None and dop_cond and dop_ft is None:
            if ftr.epochs_done > 0 or start_epoch > 0:      # continued: the model is no longer the original
                if os.path.isfile(dop_ft_file):
                    dop_ft = torch.load(dop_ft_file, weights_only=False)
                    logger.info(f"[dop] the original model's predictions from {os.path.basename(dop_ft_file)}")
                else:
                    logger.warning(f"[dop] off: this fine-tune continues from a checkpoint and "
                                   f"{os.path.basename(dop_ft_file)} isn't in the output folder - DOP's targets "
                                   "are made from the original model at a fine-tune's start")
                    dop_ft, dop_cond = {}, {}
            else:
                _tp = time.time()
                dop_ft = _dop_ft_targets(driver, dit, loader, dop_cond, dop_cond_for, device,
                                         samples=dop_ft_samples, seed=seed, min_t=min_timestep, max_t=max_timestep)
                torch.save(dop_ft, dop_ft_file)
                mb = sum(t.numel() * t.element_size() for rows in dop_ft.values() for s, p in rows
                         for t in [p, *(v for v in s.values() if torch.is_tensor(v))]) / 1e6
                logger.info(f"[dop] fine-tune: {dop_ft_samples} of the original model's predictions for each of "
                            f"{len(dop_ft)} images ({mb:.0f} MB, {time.time() - _tp:.0f} s) - each DOP pass holds "
                            f"the model to one of them. Saved for a continued run: {os.path.basename(dop_ft_file)}")
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        _theirs = (_share_split(seed, epoch + 1, _share_keys, share_now[0]["frac"]) if share_now[0] is not None
                   else None)                       # the helper's images this epoch (both sides draw the same)
        for i, batch, _only in _epoch_batches(epoch):
            if epoch - start_epoch < 2 and time.time() - warmup_note[0] > 30.0:
                warmup_note[0] = time.time()
                logger.info("[warm-up] Warm-up phase — the first two epochs start slowly while the GPU plans kernels"
                            + (", compiles the blocks" if do_compile else "")
                            + " and fills its caches. Nothing is stuck; full speed arrives from epoch 3.")
            if _only is not None:              # a catch-up: the helper's images this GPU trains after all
                pass
            elif share_now[0] is not None:
                _mine = (str((batch.get("item_keys") or [""])[0]) in _theirs) == bool(share_worker)
                if _mine and share_worker and os.path.exists(share_now[0]["work"][:-len(".working")] + ".cancel"):
                    _mine = False              # the lead went on without this round: the rest of it is skipped
                if not _mine:                  # the other GPU trains this one this epoch
                    recorder.drop(step=i)
                    global_step += 1
                    progress.update(1)
                    share_skip[0] += 1
                    if scheduler is not None and share_skip[0] % accum == 0:
                        scheduler.step()       # the schedule stays where one GPU's would be
                    continue
                share_now[0]["steps"] += 1
            if watch.excluded(batch):          # two failed AI recaptions and still stuck: no forward, no loss
                recorder.drop(step=i)
                global_step += 1
                progress.update(1)
                continue
            _skip, _lrm = driver.step_policy(batch, epoch + 1 + (ftr.epochs_done if ftr is not None else 0))
            if _skip:                          # a retired category ("stop"): no forward, no loss, no record
                recorder.drop(step=i)
                global_step += 1
                progress.update(1)
                continue
            if lr_boost:             # experimental: items whose name holds a pattern train at a multiple of the LR
                _ik = " ".join(str(k) for k in batch.get("item_keys") or [])
                _lrm *= next((m for pat, m in lr_boost if pat in _ik), 1.0)
            lr_acc.append(_lrm)
            latents = batch["latents"].to(device)
            if slider:
                # both poles backpropagate before the strength flips: checkpointed blocks recompute their forward
                # in backward, at whatever strength is set then. The family's per-step freeze applies as on any step
                # (H3: the Training mode's block window, never the token refiner)
                optimizer.zero_grad(set_to_none=True)
                _frozen_sl = _step_freeze(driver.step_frozen_blocks(batch))
                if slider_prompts:
                    _l, _t, _al = _prompt_slider_step(
                        driver, dit, net, latents, slider_enc, gen, guidance=slider_guidance, min_t=min_timestep,
                        max_t=max_timestep, anchor=anchor_enc[global_step % len(anchor_enc)] if anchor_enc else None,
                        anchor_weight=slider_anchor_weight)
                    if _al is not None:
                        anchor_losses.append(_al)
                else:
                    cond = driver.batch_cond(batch, device)
                    _neg = batch.get("latents_control_0")
                    if _neg is None:
                        raise RuntimeError("[slider] this item has no pair image - an image-pair slider needs one in "
                                           "the second folder for every training image, matched by file name")
                    _neg = _neg.to(device)
                    _lm = _loss_mask(batch)
                    _l = 0.0
                    try:
                        for m, a, b in ((1.0, latents, _neg), (-1.0, _neg, latents)):
                            net.set_trainable_multiplier(m)
                            _pl, _info = driver.training_loss(dit, a, cond, gen, min_t=min_timestep,
                                                              max_t=max_timestep, diff_ref=b,
                                                              diff_weight=slider_diff_weight,
                                                              **({"loss_mask": _lm} if _lm is not None else {}))
                            (0.5 * _pl).backward()
                            _l += 0.5 * _pl.item()
                    finally:
                        net.set_trainable_multiplier(1.0)
                    _t = _info.get("t", 0.5)
                for _p in _frozen_sl:
                    _p.requires_grad_(True)
                loss, _info = torch.tensor(_l), {"t": _t}
            else:
                cond = driver.batch_cond(batch, device)
                refs = [batch[k].to(device) for k in sorted((k for k in batch if k.startswith("latents_control_")),
                                                            key=lambda k: int(k.rsplit("_", 1)[1]))]
                # per-step routing (a family's step_frozen_blocks): those blocks' trainable weights sit out this
                # step's forward and backward, so the backward stops at the first trained block and they get no grad
                _frozen_now = _step_freeze(driver.step_frozen_blocks(batch))
                _lm = _loss_mask(batch)
                loss, _info = driver.training_loss(dit, latents, cond, gen, min_t=min_timestep, max_t=max_timestep,
                                                   **({"refs": refs} if refs else {}),
                                                   **({"loss_mask": _lm} if _lm is not None else {}),
                                                   **({"prior": prior} if prior is not None and _lm is not None
                                                      else {}))
                if "lr_mult" in _info and lr_acc:
                    lr_acc[-1] *= _info["lr_mult"]     # where the step landed (the H3 noise-band LR)
                if pending == 0 and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                if reg_keys and all(str(k) in reg_keys for k in batch.get("item_keys") or [None]):
                    mult = reg_lr_multiplier       # a regularisation image: a fixed nudge, never the watch's
                else:
                    mult = watch.multiplier(batch)  # per-image LR (batch size 1): the raw loss is still what's recorded
                _scaled = loss * mult if mult != 1.0 else loss
                _dc = dop_cond_for(batch, cond) if dop_cond and global_step % dop_every == 0 else None
                _dk = str((batch.get("item_te_paths") or [None])[0])
                if _dc is not None and ftr is not None and dop_ft and _dk in dop_ft:
                    # one backward for both: the free-each-gradient fine-tune steps each weight as its gradient
                    # lands, so a second backward would be a second optimizer step
                    _dl = _dop_ft_loss(driver, dit, dop_ft[_dk], _dc, device)
                    _scaled = _scaled + dop_weight * _dl
                    dop_sum[0] += _dl.item()
                    dop_sum[1] += 1
                    _dc = None
                if ftr is not None and ftr.fused is not None:
                    ftr.fused.scale = lr_acc[-1] if lr_acc else 1.0   # each weight steps in this backward's hooks
                (_scaled / accum if accum > 1 else _scaled).backward()
                if _dc is not None and ftr is None:
                    dop_sum[0] += _dop_step(driver, dit, net, latents, _dc, gen, weight=dop_weight,
                                            min_t=min_timestep, max_t=max_timestep, accum=accum,
                                            swapped=bool(swapped))
                    dop_sum[1] += 1
                for _p in _frozen_now:
                    _p.requires_grad_(True)
            pending += 1
            stepped = False
            if pending >= accum:
                _update()
                stepped = True
            global_step += 1
            recorder.add(epoch=epoch, step=i, loss=loss.item())
            steplog.add(step_off + global_step, epoch + 1 + (ftr.epochs_done if ftr is not None else 0), loss.item(),
                        None if global_step % 20 else      # the rate now and then (Automagic's averages it)
                        learning_rate if optimizer is None else optimizer_lr(optimizer),
                        last_grad_norm if stepped else None)
            if pending == 0 and ftr is None:   # the web UI's Sample now / Save now, between optimizer steps
                if take_request(output_dir, "save"):
                    save_lora(os.path.join(output_dir, f"{output_name}-step{global_step:07d}.safetensors"), epoch + 1)
                    logger.info(f"[save now] step {global_step} saved as {output_name}-step{global_step:07d}.safetensors")
                if take_request(output_dir, "sample"):
                    logger.info(f"[sample now] previews at step {global_step} (epoch {epoch + 1} in progress)")
                    _tp = time.time()
                    previews(epoch + 1)
                    progress.start_t += time.time() - _tp
            watch.observe(epoch + 1, global_step, batch, _info.get("t", 0.5), loss.item())
            progress.set_postfix(avr_loss=f"{recorder.moving_average:.4f}", refresh=False)
            progress.update(1)

        if pending:                        # a partial group: settle the optimizer before the epoch-end work
            _update()
        _off = ftr.epochs_done if ftr is not None else 0
        logger.info(f"epoch {epoch + 1 + _off}/{max_train_epochs + _off}  avr_loss={recorder.moving_average:.4f}  "
                    f"step={global_step}  {(time.time() - t0) / max(1, steps_per_epoch):.2f}s/step  "
                    f"lr={learning_rate if optimizer is None else optimizer_lr(optimizer):.3e}  "
                    + (f"({group_rates(optimizer)})  " if optimizer is not None and group_rates(optimizer) else "") +
                    (f"  grad_norm {last_grad_norm:.4g}" if last_grad_norm is not None else "") +
                    f"peak VRAM {torch.cuda.max_memory_reserved() / 1024 ** 3:.1f} GB")
        steplog.flush()
        if share_worker and epoch + 1 < int((share_now[0].get("epochs") or [share_now[0]["epoch"]])[-1]):
            continue                         # a round of several epochs: the LoRA goes back after its last
        if share_now[0] is not None and share_worker:
            _sj = share_now[0]
            # its change over the round, against the start it was handed (small numbers: bf16 keeps them)
            from safetensors.torch import load_file, save_file
            _st = load_file(_sj["start"])
            _now = net.state_dict(torch.float32)
            save_file({k: (v - _st[k].float()).to(torch.bfloat16).contiguous() for k, v in _now.items() if k in _st},
                      _sj["out"] + ".part")
            os.replace(_sj["out"] + ".part", _sj["out"])
            _write_json(_sj["work"][:-len(".working")] + ".done.json",
                        {"ok": True, "steps": _sj["steps"], "seconds": round(time.time() - _sj["t0"], 1)})
            os.remove(_sj["work"])
            logger.info(f"[share] epoch {epoch + 1}: {_sj['steps']} images trained here, handed back")
            continue
        if dop_sum[1]:
            logger.info(f"[dop] epoch {epoch + 1}: preservation loss {dop_sum[0] / dop_sum[1]:.5f} over {dop_sum[1]} "
                        "step(s) - how far the LoRA moves the class prompt from the base model")
            dop_sum[0], dop_sum[1] = 0.0, 0
        if adaptive:
            adaptive.epoch_boundary(epoch, recorder.moving_average, net.trainable_modules(), optimizer)
        # problem-image verdicts + queued caption fixes / auto-recaptions, re-encoded before the next epoch
        watch.boundary(epoch + 1, dit, device, parkable=not swapped)

        done = epoch + 1
        # the save schedule (the Training tab's How often to save; a fine-tune saves at whole rotations)
        cadence = done < max_train_epochs and preview_due(
            done, save_every_n_epochs, *((0, 0) if ftr is not None else (save_until_epoch, save_every_n_epochs_after)))
        if ftr is not None:
            # a fine-tune saves and previews at whole rotations only, and the preview renders the checkpoint just
            # saved: the window goes back to NF4 (the master holds every trained weight) and its optimizer is
            # dropped first, which is also what frees the VRAM the render needs
            if os.path.exists(pause_flag) and done < max_train_epochs and not ftr.rotation_done(done):
                ftr.note_pause(done)
            if cadence or (os.path.exists(pause_flag) and done < max_train_epochs and ftr.rotation_done(done)):
                optimizer = params = None
                ftr.park()
                n_ep = done + ftr.epochs_done
                save_lora(os.path.join(output_dir, f"{output_name}-{n_ep:06d}.safetensors"), done)
                if cadence and sample_every_n_epochs:
                    _tp = time.time()
                    previews(n_ep)
                    progress.start_t += time.time() - _tp
                if os.path.exists(pause_flag):
                    logger.info(f"[pause] requested - rotation complete at epoch {n_ep}, checkpoint saved; exiting "
                                f"cleanly (continue from {output_name}-{n_ep:06d}.safetensors)")
                    progress.close()
                    sys.exit(0)
            continue
        if cadence:
            save_lora(os.path.join(output_dir, f"{output_name}-{done:06d}.safetensors"), done)
        state_saved = False
        if save_state and cadence:
            state_saved = state(done)
            if state_saved:
                prune_state_dirs(output_dir, output_name, keep_last_n_states)
        if sample_every_n_epochs and _preview_due(done):
            _tp = time.time()
            previews(done)
            progress.start_t += time.time() - _tp       # the bar's s/it is training speed, not previews
            # this epoch's checkpoint was saved before its preview existed, with the previous epoch's as its
            # thumbnail (#122): re-embed its own. An explicit --metadata_thumbnail (or "off") stays.
            if cadence and not (metadata_thumbnail or "").strip():
                own = sample_for_epoch(output_dir, output_name, done)
                if own:
                    refresh_checkpoint_thumbnail(os.path.join(output_dir, f"{output_name}-{done:06d}.safetensors"), own)
        if (os.path.exists(pause_flag) and done < max_train_epochs
                and not (share_now[0] is not None and done < share_now[0]["last"])):   # a share round ends first
            if state_saved:
                logger.info(f"[pause] requested - state for epoch {done} already saved; exiting cleanly")
            else:
                logger.info(f"[pause] requested - saving state at epoch {done} and exiting cleanly")
                state(done, required=True)
            progress.close()
            sys.exit(0)
        if anchor_losses:                # how far the dial moved its anchors this epoch (0 = not at all)
            logger.info(f"[slider] anchor loss, epoch {done}: {sum(anchor_losses) / len(anchor_losses):.5f} "
                        f"over {len(anchor_losses)} steps")
            anchor_losses.clear()
        if slider_prompts and done < max_train_epochs:
            _tp = time.time()
            slider_live(done)
            progress.start_t += time.time() - _tp       # the bar's s/it is training speed

    if share_worker:                     # a helper's share: the lead keeps the LoRA
        progress.close()
        logger.info("[share] the lead's run is done here")
        return None
    progress.close()
    steplog.flush()
    for _old in glob.glob(os.path.join(output_dir, SHARE_DIR, "*")):   # the share's rounds and bundles: done with
        if os.path.basename(_old) != "setup.json":
            try:                         # a file the web UI has open stays (Windows): never the run's final save
                shutil.rmtree(_old, ignore_errors=True) if os.path.isdir(_old) else os.remove(_old)
            except OSError:
                pass
    final = os.path.join(output_dir, f"{output_name}.safetensors")
    if ftr is not None:
        optimizer = params = None
        ftr.park()
        save_lora(final, max_train_epochs)
        ftr.be.cleanup()                  # the final checkpoint holds everything the scratch did
        if sample_every_n_epochs:
            previews(max_train_epochs + ftr.epochs_done)
        logger.info(f"Fine-tune complete -> {final}")
        return final
    save_lora(final, max_train_epochs)
    # the last epoch under its number too (#176): a resumed run that extends this one ends on the same plain name, and
    # without a numbered copy that epoch would be overwritten and lost
    shutil.copyfile(final, os.path.join(output_dir, f"{output_name}-{max_train_epochs:06d}.safetensors"))
    if save_state_on_train_end:
        state(max_train_epochs)
    logger.info(f"Training complete -> {final}")
    return final


def setup_parser():
    p = argparse.ArgumentParser(description="LoRA training for a described model family (standard layer)")
    p.add_argument("--family", required=True, help="family key, e.g. qwen_image21")
    p.add_argument("--dit", required=True)
    p.add_argument("--dataset_config", default=None, help="The dataset TOML (not used by a prompt-pair slider)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", required=True)
    p.add_argument("--precision", default="bf16", choices=("auto",) + quant.PRECISIONS,
                   help="base precision: bf16, int8 (8-bit, int8 matmuls), nf4 (4-bit) or auto (fits free VRAM)")
    p.add_argument("--blocks_to_swap", type=int, default=0,
                   help="blocks streamed between CPU and GPU (not with nf4); -1 = as few as fit free VRAM")
    p.add_argument("--network_dim", type=int, default=32)
    p.add_argument("--network_type", default="lora", choices=("lora", "lokr"))
    p.add_argument("--preview_lora", default=None,
                   help="a LoRA on previews of the training model only, never in training (MOSS-Image's SFT look "
                        "LoRA); a file not there yet is picked up at a later preview")
    p.add_argument("--preview_lora_strength", type=float, default=1.0)
    p.add_argument("--speed_lora_strength", type=float, default=None,
                   help="the preview speed LoRA's strength (default: the family's recommended value)")
    p.add_argument("--log_per_image_loss", action="store_true", help="detect problem images (Problem Images window)")
    p.add_argument("--per_image_lr", action="store_true", help="scale each step by the image's loss-watch multiplier")
    p.add_argument("--auto_recaption", action="store_true", help="recaption stuck images (needs --captioner)")
    p.add_argument("--warmup_look_outliers", action="store_true", help="LR warm-up for Look Filter outliers")
    p.add_argument("--trigger_position", default="start", choices=("start", "end"))
    p.add_argument("--recaption_triggers", default=None,
                   help="auto-recaption's trigger word per other folder, as JSON {key prefix: word}")
    p.add_argument("--recaption_instruction", default=None)
    p.add_argument("--recaption_instruction_detailed", default=None)
    p.add_argument("--recaption_task", default=None, help="the first try's Captions tab task, for the log")
    p.add_argument("--recaption_task_detailed", default=None, help="the second try's Captions tab task, for the log")
    p.add_argument("--captioner", default=None,
                   help="auto-recaption's captioner: the Krea 2 Qwen3-VL-4B text encoder file (as the Captions tab)")
    p.add_argument("--lokr_factor", type=int, default=8, help="LoKR only: w1 is about factor x factor")
    p.add_argument("--network_alpha", type=float, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_epochs", type=int, default=16)
    p.add_argument("--save_every_n_epochs", type=int, default=1)
    p.add_argument("--save_until_epoch", type=int, default=0,
                   help="With --save_every_n_epochs_after: saves every --save_every_n_epochs up to this epoch")
    p.add_argument("--save_every_n_epochs_after", type=int, default=0, help="...then on whole multiples of this")
    p.add_argument("--save_state", action="store_true")
    p.add_argument("--save_state_on_train_end", action="store_true")
    p.add_argument("--keep_last_n_states", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--training_adapter", default=None, help="Frozen training adapter (off in previews and saves)")
    p.add_argument("--training_adapter_strength", type=float, default=1.0)
    p.add_argument("--context_lora_path", default=None)
    p.add_argument("--context_lora_strength", type=float, default=1.0)
    p.add_argument("--speed_lora", default=None,
                   help="The family's preview speed LoRA (description.preview_speed_lora): previews only")
    p.add_argument("--min_timestep", type=float, default=0.0, help="Noise band floor (0-1)")
    p.add_argument("--max_timestep", type=float, default=1.0, help="Noise band ceiling (0-1)")
    p.add_argument("--vae", default=None)
    p.add_argument("--text_encoder", default=None)
    p.add_argument("--sample_prompts", default=None, help="One prompt per line")
    p.add_argument("--sample_every_n_epochs", type=int, default=0)
    p.add_argument("--sample_until_epoch", type=int, default=0,
                   help="With --sample_every_n_epochs_after: previews every --sample_every_n_epochs up to this epoch")
    p.add_argument("--sample_every_n_epochs_after", type=int, default=0,
                   help="...then on whole multiples of this")
    p.add_argument("--sample_width", type=int, default=None)
    p.add_argument("--sample_height", type=int, default=None)
    p.add_argument("--sample_steps", type=int, default=None)
    p.add_argument("--sample_cfg_scale", type=float, default=None)
    p.add_argument("--sample_negative", default=None)
    p.add_argument("--sample_at_first", action="store_true")
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--sample_strengths", default="",
                   help="Also render each preview at these LoRA strengths, comma-separated (e.g. 0.5)")
    p.add_argument("--family_option", action="append", default=[], metavar="KEY=VALUE",
                   help="a family's own training option (the driver's set_options), e.g. photo_blocks=20-49")
    p.add_argument("--preview_checkpoint", default=None,
                   help="Render training previews on this checkpoint (families with train_preview_checkpoint)")
    p.add_argument("--preview_checkpoint_cache", default="auto", choices=("auto", "on", "off"),
                   help="Keep the preview checkpoint in system RAM between epochs")
    p.add_argument("--preview_int8", action="store_true", help="INT8 matmuls on the preview checkpoint")
    p.add_argument("--sample_reference", nargs="+", default=None,
                   help="Edit previews: the photo every preview prompt edits, then any extra pictures it brings in")
    p.add_argument("--sample_image", default=None,
                   help="A picture previews see through the text encoder's vision path (families with preview_image)")
    p.add_argument("--slider_pairs", action="store_true",
                   help="Slider from image pairs: each training image is the +1 pole, its control_directory pair "
                        "(same file name) the -1 pole")
    p.add_argument("--slider_diff_weight", type=float, default=1.0,
                   help="Image-pair slider: 0 = plain loss, 1 = concentrate where the two images differ")
    p.add_argument("--slider_prompts", nargs=3, default=None, metavar=("NEUTRAL", "POSITIVE", "NEGATIVE"),
                   help="Slider from three prompts, no images (needs --text_encoder and --vae)")
    p.add_argument("--slider_guidance", type=float, default=3.0, help="Prompt-pair slider: how hard to push")
    p.add_argument("--slider_anchors", nargs="+", default=None, metavar="PROMPT",
                   help="Prompt-pair slider: prompts the dial must leave as they are (one a step, in turn)")
    p.add_argument("--slider_anchor_weight", type=float, default=1.0,
                   help="Prompt-pair slider: how much the anchors count against the dial's own loss")
    p.add_argument("--slider_bank", type=int, default=16, help="Prompt-pair slider: practice images to render")
    p.add_argument("--train_blocks", default="",
                   help="Comma-separated block ids (the driver's block map) to train; empty = every block")
    p.add_argument("--slider_bank_res", type=int, default=768, help="Prompt-pair slider: practice image size")
    for k in ("title", "author", "description", "license", "tags", "trigger_phrase", "thumbnail"):
        p.add_argument(f"--metadata_{k}", default=None)
    p.add_argument("--trigger_word", default=None, help="Recorded as the trigger phrase when none is given")
    p.add_argument("--resume", default=None)
    p.add_argument("--share_worker", default=None, help="a helper's training share: take each epoch's start LoRA "
                   "and share from this folder's job files, train this computer's images, write the result back")
    p.add_argument("--adaptive_lr", action="store_true")
    p.add_argument("--adaptive_lr_min", type=float, default=1e-4)
    p.add_argument("--adaptive_lr_max", type=float, default=2e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="micro-batches averaged into one optimizer step (sliders: always 1)")
    p.add_argument("--finetune", action="store_true",
                   help="full fine-tune of the base model (families whose driver offers one), a component window at a "
                        "time; saves full checkpoints at whole rotations")
    p.add_argument("--ft_rotations", type=int, default=10, help="fine-tune length in full rotations")
    p.add_argument("--ft_save_every_rotations", type=int, default=1,
                   help="checkpoint (and preview) every N rotations; never mid-rotation")
    p.add_argument("--ft_rotate_every", type=int, default=1, help="epochs each window trains before the next")
    p.add_argument("--ft_max_parts", type=int, default=0,
                   help="most parts per fine-tune window (0 = as many as fit; 1 = one part per window, most headroom)")
    p.add_argument("--ft_start_window", type=int, default=0, help="continuing: the window to start at")
    p.add_argument("--ft_epochs_done", type=int, default=0, help="continuing: epochs already trained (numbering)")
    p.add_argument("--ft_fused_backward", action="store_true",
                   help="step each weight as its gradient lands and free it (less VRAM; no clipping or accumulation)")
    p.add_argument("--reg_lr_multiplier", type=float, default=0.2,
                   help="fine-tune: LR multiplier for images in a dataset block marked is_reg = true")
    p.add_argument("--gradient_checkpointing", default="auto", choices=("auto", "on"),
                   help="auto: off for a compiled LoRA run with no block swap when the measured peak fits free VRAM")
    p.add_argument("--compile_blocks", default="auto", choices=("auto", "on", "outside", "off"),
                   help="torch.compile the DiT blocks (families with compiles=True); auto weighs warm-up vs run length")
    p.add_argument("--ema_decay", type=lambda v: v if str(v).lower().startswith("short") else float(v), default=0.0,
                   help="EMA decay (0.98), 0 = off, or 'short' (sized to the run)")
    p.add_argument("--optimizer_type", default="adamw")
    p.add_argument("--optimizer_args", default="")
    p.add_argument("--lr_scheduler", default="constant",
                   choices=["constant", "constant_with_warmup", "cosine", "cosine_with_restarts", "linear",
                            "polynomial"])
    p.add_argument("--lr_warmup_steps", type=int, default=0)
    p.add_argument("--lr_scheduler_num_cycles", type=int, default=1)
    p.add_argument("--lr_scheduler_power", type=float, default=1.0)
    p.add_argument("--dop", action="store_true",
                   help="Differential Output Preservation: without the trigger the LoRA is held to the base model")
    p.add_argument("--dop_trigger", default=None, help="DOP: the trigger(s) in the captions, comma-separated for several subjects "
                   "(default --trigger_word)")
    p.add_argument("--dop_class", default="",
                   help="DOP: what replaces each trigger, e.g. 'woman, man' (one for all, or one per "
                   "trigger in order); empty = the triggers are dropped")
    p.add_argument("--dop_weight", type=float, default=1.0, help="DOP: the preservation loss's weight")
    p.add_argument("--dop_every", type=int, default=1, help="DOP: a preservation pass every N steps")
    p.add_argument("--lr_boost", action="append", default=[], metavar="PATTERN=MULT",
                   help="experimental: items whose name contains PATTERN train at MULT x the LR on their step")
    p.add_argument("--dop_ft_samples", type=int, default=8,
                   help="DOP on a fine-tune: the original model's predictions made per image before the first step")
    p.add_argument("--masked_training", action="store_true",
                   help="weight each picture's loss by its mask (photo-masklabel.png beside photo.png): white counts "
                        "fully, grey partly, black at --mask_floor; pictures without a mask train as normal")
    p.add_argument("--mask_floor", type=float, default=0.1,
                   help="masked training: how much outside the mask counts (0 = nothing, 1 = as much as inside)")
    p.add_argument("--mask_prior", type=float, default=0.0,
                   help="masked training: outside the mask, keep the LoRA to the model's own prediction (LoRA at 0) "
                        "at this weight (OneTrainer's masked prior preservation; 0 = off; one extra forward a step)")
    return p


def main():
    if (sys.platform != "win32" and not os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
            and os.environ.get("FIZGIG_NO_EXPANDABLE") != "1"):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    logging.basicConfig(level=logging.INFO)
    a = setup_parser().parse_args()
    prompts = None
    if a.sample_prompts and os.path.exists(a.sample_prompts):
        with open(a.sample_prompts, encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    if not prompts and a.sample_image and a.sample_prompts:
        prompts = [""]          # a reference alone: 'generate from this picture'
    train_family(
        a.family, a.dit, a.dataset_config, a.output_dir, a.output_name, precision=a.precision,
        blocks_to_swap=a.blocks_to_swap, speed_lora_strength=a.speed_lora_strength, network_type=a.network_type, lokr_factor=a.lokr_factor,
        log_per_image_loss=a.log_per_image_loss, per_image_lr=a.per_image_lr, auto_recaption=a.auto_recaption,
        warmup_look_outliers=a.warmup_look_outliers, trigger_word=a.trigger_word, trigger_position=a.trigger_position,
        recaption_triggers=a.recaption_triggers,
        recaption_instruction=a.recaption_instruction,
        recaption_instruction_detailed=a.recaption_instruction_detailed, captioner=a.captioner,
        recaption_task=a.recaption_task, recaption_task_detailed=a.recaption_task_detailed,
        network_dim=a.network_dim, network_alpha=a.network_alpha, learning_rate=a.learning_rate,
        max_train_epochs=a.max_train_epochs, save_every_n_epochs=a.save_every_n_epochs,
        save_until_epoch=a.save_until_epoch, save_every_n_epochs_after=a.save_every_n_epochs_after, save_state=a.save_state,
        save_state_on_train_end=a.save_state_on_train_end, keep_last_n_states=a.keep_last_n_states, seed=a.seed,
        training_adapter=a.training_adapter, training_adapter_strength=a.training_adapter_strength,
        context_lora_path=a.context_lora_path, context_lora_strength=a.context_lora_strength,
        min_timestep=a.min_timestep, max_timestep=a.max_timestep, speed_lora=a.speed_lora, preview_lora=a.preview_lora,
        preview_lora_strength=a.preview_lora_strength,
        vae_path=a.vae, te_path=a.text_encoder,
        sample_prompts=prompts, sample_every_n_epochs=a.sample_every_n_epochs,
        sample_until_epoch=a.sample_until_epoch, sample_every_n_epochs_after=a.sample_every_n_epochs_after,
        sample_width=a.sample_width,
        sample_height=a.sample_height, sample_steps=a.sample_steps, sample_cfg_scale=a.sample_cfg_scale,
        sample_negative=a.sample_negative, sample_at_first=a.sample_at_first, sample_seed=a.sample_seed,
        sample_strengths=tuple(float(x) for x in a.sample_strengths.split(",") if x.strip()),
        sample_reference=list(a.sample_reference) if a.sample_reference else None, sample_image=a.sample_image,
        slider_pairs=a.slider_pairs, slider_diff_weight=a.slider_diff_weight, slider_prompts=a.slider_prompts,
        slider_guidance=a.slider_guidance, slider_anchors=a.slider_anchors or (),
        slider_anchor_weight=a.slider_anchor_weight, slider_bank=a.slider_bank, slider_bank_res=a.slider_bank_res,
        train_blocks=[b.strip() for b in a.train_blocks.split(",") if b.strip()] or None,
        metadata_title=a.metadata_title, metadata_author=a.metadata_author,
        metadata_description=a.metadata_description, metadata_license=a.metadata_license,
        metadata_tags=a.metadata_tags, metadata_trigger_phrase=a.metadata_trigger_phrase or a.trigger_word,
        metadata_thumbnail=a.metadata_thumbnail, resume_state_dir=a.resume, adaptive_lr=a.adaptive_lr,
        adaptive_lr_min=a.adaptive_lr_min, adaptive_lr_max=a.adaptive_lr_max, max_grad_norm=a.max_grad_norm,
        ema_decay=a.ema_decay, optimizer_type=a.optimizer_type, optimizer_args=a.optimizer_args,
        lr_scheduler=a.lr_scheduler, lr_warmup_steps=a.lr_warmup_steps,
        lr_scheduler_num_cycles=a.lr_scheduler_num_cycles, lr_scheduler_power=a.lr_scheduler_power,
        gradient_accumulation_steps=a.gradient_accumulation_steps, compile_blocks=a.compile_blocks,
        gradient_checkpointing=True if a.gradient_checkpointing == "on" else "auto",
        finetune=a.finetune, ft_rotations=a.ft_rotations, ft_save_every_rotations=a.ft_save_every_rotations,
        ft_rotate_every=a.ft_rotate_every, ft_max_parts=a.ft_max_parts, ft_start_window=a.ft_start_window, ft_epochs_done=a.ft_epochs_done,
        ft_fused_backward=a.ft_fused_backward, reg_lr_multiplier=a.reg_lr_multiplier,
        preview_checkpoint=a.preview_checkpoint, preview_checkpoint_cache=a.preview_checkpoint_cache,
        preview_int8=a.preview_int8,
        family_options=dict(o.split("=", 1) for o in a.family_option if "=" in o),
        dop=a.dop, dop_trigger=a.dop_trigger, dop_class=a.dop_class, dop_weight=a.dop_weight, dop_every=a.dop_every,
        dop_ft_samples=a.dop_ft_samples,
        lr_boost=[(x.split("=", 1)[0], float(x.split("=", 1)[1])) for x in a.lr_boost if "=" in x] or None,
        share_worker=a.share_worker, masked_training=a.masked_training, mask_floor=a.mask_floor,
        mask_prior=a.mask_prior)


if __name__ == "__main__":
    main()
