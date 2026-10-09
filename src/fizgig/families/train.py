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
import datetime
import gc
import json
import logging
import math
import os
import random
import re
import sys
import time
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
from fizgig.training.metadata import (build_metadata, latest_sample_image, refresh_checkpoint_thumbnail,  # noqa: E402
                                      resolve_title, sample_for_epoch, thumbnail_data_uri)
from fizgig.training.train_utils import LossRecorder, prune_state_dirs, validate_output_name  # noqa: E402

logger = logging.getLogger(__name__)

ADAPTER = "training_adapter"
CONTEXT = "context"
SPEED = "speed_lora"
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
        json.dump({"epoch": epoch, "global_step": global_step, "architecture": arch_id, **(extra or {})}, f)
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


def _read_sample_override(output_dir):
    """The GUI's live sample override (<output_dir>/.sample_override.json, written by the status-bar panel):
    {prompt, seed, width, height} while a prompt is set, else None. The reference image field is Klein's and ignored."""
    path = os.path.join(output_dir, ".sample_override.json")
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        prompt = str(d.get("prompt", "")).strip()
        if prompt:
            return {"prompt": prompt, "seed": int(d.get("seed", 1234)), "width": int(d.get("width", 1024)),
                    "height": int(d.get("height", 1024))}
    except (OSError, ValueError, TypeError):
        pass
    return None


def _encode_override(driver, te_path, prompt, dit, device, parkable=True, references=None):
    """Encode one override prompt mid-run. The text encoder loads beside the training DiT when it fits; otherwise the
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
                return driver.encode_text_with_references(te, [prompt], [references])
            return driver.encode_text(te, [prompt])
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
    """uint8 (H, W, 3) arrays of the preview's reference images at the first one's aspect and the preview's area,
    and that (width, height)."""
    import numpy as np
    from PIL import Image, ImageOps
    imgs = [Image.open(p).convert("RGB") for p in paths]
    size = _reference_size(*imgs[0].size, width * height)
    return [np.array(ImageOps.fit(im, size, Image.LANCZOS)) for im in imgs], size


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
    gap, band = 8, 30
    strip = Image.new("RGB", (w * len(frames) + gap * (len(frames) - 1), h + band), (16, 16, 16))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.load_default(size=max(14, w // 40))
    except Exception:
        font = ImageFont.load_default()
    for k, (im, m) in enumerate(zip(frames, multipliers)):
        x = k * (w + gap)
        strip.paste(im, (x, band))
        draw.text((x + 8, 7), "strength 0 (slider off)" if m == 0 else f"strength {m:+g}", fill=(236, 236, 236),
                  font=font)
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


def _prompt_slider_step(driver, dit, net, latents, enc, gen, *, guidance, min_t, max_t):
    """Concept Sliders, textual form, on flow matching. With the adapter at 0 the frozen model predicts the
    neutral, positive and negative prompts at one noised practice latent; the adapter then trains at +1 toward
    v_neutral + guidance * (v_pos - v_neg) and at -1 toward the mirror. Each pole is backpropagated before the
    flip (checkpointed blocks recompute at the strength of the moment). Returns the detached mean loss."""
    state = driver.noise_latents(latents, gen, min_t=min_t, max_t=max_t)
    n_c, p_c, g_c = enc
    try:
        net.set_trainable_multiplier(0.0)
        with torch.no_grad():
            v_n = driver.predict(dit, state, n_c).float()
            delta = float(guidance) * (driver.predict(dit, state, p_c).float() - driver.predict(dit, state, g_c).float())
        total = 0.0
        for m in (1.0, -1.0):
            net.set_trainable_multiplier(m)
            loss = F.mse_loss(driver.predict(dit, state, n_c).float(), v_n + m * delta)
            (0.5 * loss).backward()
            total += 0.5 * loss.item()
    finally:
        net.set_trainable_multiplier(1.0)
    return total, state["t"]


def _render_previews(driver, dit, net, vae, encoded, out_dir, epoch, *, output_name, steps, cfg, neg, width,
                     height, seed, ema=None, speed=None, lowmem=False, swapped=False, refs=None, slider=False):
    """Previews on the RESIDENT training model: training adapter OFF (the deployment setup), the family's speed
    LoRA ON if one is loaded (it lives on CPU between previews). On small cards the DiT parks on CPU for the
    decode. File names match the other trainers so the GUI gallery reads them:
    <name>_e<epoch>_<idx>_<timestamp>_<seed>.png. `speed` is the speed LoRA's SamplingSettings or None; `refs` the edit previews' reference latents; `slider`
    renders each prompt at -1 / 0 / +1 on one seed and saves them as one strip."""
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
        if ema is not None:
            ema.swap_in()
            done.add("ema")
        dit.eval()
        if swapped:
            done.add("swap")
            driver.block_swap_mode(dit, inference=True)
        ref_kw = {"refs": [r.to(device) for r in refs]} if refs else {}
        lats = []
        mults = SLIDER_PREVIEW_MULTIPLIERS if slider else (None,)
        for i, cond in enumerate(encoded):
            for m in mults:
                if m is not None:
                    net.set_trainable_multiplier(m)
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
        per = len(SLIDER_PREVIEW_MULTIPLIERS) if slider else 1
        for i in range(len(lats) // per):
            p = os.path.join(out_dir, f"{output_name}_e{epoch:06d}_{i:02d}_{ts}_{seed + i}.png")
            frames = [driver.decode(vae, lat, width, height) for lat in lats[i * per:(i + 1) * per]]
            if slider:
                strip = driver.slider_preview(frames, SLIDER_PREVIEW_MULTIPLIERS)
                paths += driver.save_preview(_slider_strip(frames, SLIDER_PREVIEW_MULTIPLIERS) if strip is None
                                             else strip, p)
            else:
                paths += driver.save_preview(frames[0], p)
    finally:
        if slider:
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

    def __init__(self, driver, path, device, cache_mode="auto", int8=False, context=None):
        self.driver, self.path, self.device = driver, path, torch.device(device)
        self.cache_mode, self.int8, self.context = str(cache_mode or "auto").lower(), bool(int8), context
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
                 rotate_every, start_window, epochs_done, fused, lr, master_dir, mp=None, group=None, max_parts=0):
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
            max_parts=max_parts)
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
                 learning_rate=1e-4, max_train_epochs=16, save_every_n_epochs=1, save_state=False,
                 save_state_on_train_end=False, keep_last_n_states=2, seed=42, precision="bf16",
                 training_adapter=None, training_adapter_strength=1.0,
                 context_lora_path=None, context_lora_strength=1.0, min_timestep=0.0, max_timestep=1.0,
                 speed_lora=None, speed_lora_strength=None,
                 vae_path=None, te_path=None, sample_prompts=None, sample_every_n_epochs=0, sample_width=None,
                 sample_height=None, sample_steps=None, sample_cfg_scale=None, sample_negative=None,
                 sample_at_first=False, sample_seed=42, sample_reference=None, sample_image=None,
                 slider_pairs=False, slider_diff_weight=1.0, slider_prompts=None, slider_guidance=3.0,
                 train_blocks=None,
                 slider_bank=16, slider_bank_res=768,
                 metadata_title=None, metadata_author=None, metadata_description=None, metadata_license=None,
                 metadata_tags=None, metadata_trigger_phrase=None, metadata_thumbnail=None,
                 resume_state_dir=None, adaptive_lr=False, adaptive_lr_min=1e-4, adaptive_lr_max=2e-4,
                 max_grad_norm=1.0, ema_decay=0.0, optimizer_type="adamw", optimizer_args="",
                 lr_scheduler="constant", lr_warmup_steps=0, lr_scheduler_num_cycles=1, lr_scheduler_power=1.0,
                 gradient_accumulation_steps=1, compile_blocks="auto", gradient_checkpointing=True, blocks_to_swap=0, network_type="lora", lokr_factor=8,
                 log_per_image_loss=False, per_image_lr=False, auto_recaption=False, warmup_look_outliers=False,
                 trigger_word=None, trigger_position="start", recaption_instruction=None,
                 recaption_instruction_detailed=None, captioner=None,
                 finetune=False, ft_rotations=10, ft_save_every_rotations=1, ft_rotate_every=1, ft_max_parts=0,
                 ft_start_window=0,
                 ft_epochs_done=0, ft_fused_backward=False, reg_lr_multiplier=0.2, preview_checkpoint=None,
                 preview_checkpoint_cache="auto", preview_int8=False, family_options=None):
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
    if speed_desc is not None and speed_lora_strength is not None and speed_lora_strength <= 0:
        logger.info(f"[sample] {speed_desc.name} at strength 0 - previews render without it")
        speed_lora = speed_desc = None
    # --speed_lora on its own means the turbo's own recipe (strength, steps); the GUI passes its Samples-tab values
    sample_steps = sample_steps or (speed_desc.settings.steps if speed_desc else desc.preview_steps)
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
            if not (te_path and vae_path):
                raise RuntimeError("[slider] prompt pairs need the text encoder and VAE paths")
            if str(slider_prompts[0]).strip():     # the dial is shown on the picture it was trained on
                sample_prompts = [str(slider_prompts[0]).strip()]

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
        group = generate_dataset_group_by_blueprint(blueprint.dataset_group, training=True,
                                                    num_timestep_buckets=None, shared_epoch=shared_epoch)
    if group.num_train_items == 0:
        raise RuntimeError("No training items - run the cache stages (families/cache.py) first.")
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

    # ---- previews: encode prompts once, keep the VAE ---------------------------------------------
    encoded = neg = vae = ref_imgs = ref_latents = None
    slider_enc = slider_neg = None
    if slider_prompts:
        te = driver.load_text_encoder(te_path, device)
        slider_enc = [{k: v[None] for k, v in c.items()}
                      for c in driver.encode_text(te, [str(x) for x in slider_prompts])]
        if sample_cfg_scale > 1.0 and desc.preview_negative is not None:
            # the practice pictures render as previews do: the Samples tab's negative (else the family's default)
            text = sample_negative if sample_negative is not None else desc.preview_negative
            slider_neg = {k: v[None] for k, v in driver.encode_text(te, [text])[0].items()}
        driver.unload_text_encoder(te)
        del te
        torch.cuda.empty_cache()
    sample_dir = os.path.join(output_dir, "sample")
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
            if sample_cfg_scale > 1.0:
                neg = driver.encode_text_with_references(te, [sample_negative or ""], [ref_imgs])[0]
        elif sample_image and desc.preview_image:
            from PIL import Image
            logger.info(f"[sample] previews see {os.path.basename(sample_image)} through the text encoder's vision "
                        f"path")
            te = driver.load_text_encoder(te_path, device)
            encoded = driver.encode_text_with_image(te, sample_prompts, Image.open(sample_image))
            if sample_cfg_scale > 1.0:
                neg = driver.encode_text(te, [sample_negative or ""])[0]     # negatives stay text-only
        else:
            te = driver.load_text_encoder(te_path, device)
            encoded = driver.encode_text(te, sample_prompts)
            if sample_cfg_scale > 1.0:
                neg = driver.encode_text(te, [sample_negative or ""])[0]
        driver.unload_text_encoder(te)
        del te
        torch.cuda.empty_cache()
        vae = driver.load_vae(vae_path, device)
        if ref_imgs:
            ref_latents = [z[None].cpu() for z in driver.encode_images(vae, ref_imgs)]
    elif sample_prompts and sample_every_n_epochs:
        logger.warning("[sample] previews need the text encoder and VAE paths - previews are off for this run")

    # ---- model ----------------------------------------------------------------------------------
    logger.info(f"Loading {desc.display_name} DiT ({precision}) from {dit_path}")
    dit, swapped = quant.load_base(driver, dit_path, device, precision, blocks_to_swap)
    if gradient_checkpointing:
        driver.enable_gradient_checkpointing(dit, True)
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
    if speed_lora and encoded is not None:
        _ss = speed_desc.strength if speed_lora_strength is None else speed_lora_strength
        n = net.add_file(speed_lora, SPEED, _ss)
        driver.frozen_file_added(dit, speed_lora, _ss, "speed")
    if speed_lora and encoded is not None and n == 0:
        net.remove(SPEED)
        logger.warning(f"[sample] {os.path.basename(speed_lora)} matched no {desc.display_name} layers "
                       f"(a turbo LoRA for another model?) - previews render without it, at "
                       f"{desc.preview_steps} steps")
        speed_lora = None
        sample_steps = desc.preview_steps
    elif speed_lora and encoded is not None:
        net.set_enabled(SPEED, False)
        net.move_adapter(SPEED, "cpu")
        logger.info(f"[sample] {speed_desc.name}: {n} Linears, on CPU between previews, on only while they render "
                    f"({sample_steps} steps, strength {speed_desc.strength if speed_lora_strength is None else speed_lora_strength:g})")
    if network_type == "lokr" and "lokr" not in desc.network_types:
        raise RuntimeError(f"{desc.display_name} does not offer LoKR")
    ftr = None
    if finetune:
        ftr = _FineTune(driver, desc, dit, net, dit_path, device, max_train_epochs=None, rotations=ft_rotations,
                        save_every_rotations=ft_save_every_rotations, rotate_every=ft_rotate_every,
                        start_window=ft_start_window, epochs_done=ft_epochs_done, fused=ft_fused_backward,
                        lr=learning_rate, master_dir=output_dir, mp=_largest_bucket_mp(group), group=group,
                        max_parts=ft_max_parts)
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
    if slider_prompts:
        # the practice bank: the base model's own renders of the neutral prompt (adapter at 0, training adapter
        # off, the speed LoRA on when it is loaded), decoded and re-encoded into training latents
        import numpy as np
        if vae is None:
            vae = driver.load_vae(vae_path, device)
        n_c = {k: v[0].to(device) for k, v in slider_enc[0].items()}
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
                                          **({"neg_cond": slider_neg} if slider_neg is not None else {}))
                img = driver.decode(vae, lat, slider_bank_res, slider_bank_res)
                bank.append(driver.encode_images(vae, [np.array(img)])[0][None].cpu())
        if use_speed:
            net.set_enabled(SPEED, False)
            net.move_adapter(SPEED, "cpu")
        net.set_enabled(ADAPTER, True)
        net.set_trainable_multiplier(1.0)
        group.latents = bank
        torch.cuda.empty_cache()
        logger.info("[slider] %d practice images rendered at %dx%d", len(bank), slider_bank_res, slider_bank_res)
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
        start_epoch, global_step, meta = _load_state(resume_state_dir, net, optimizer, device, arch,
                                                     untagged_own=desc.resumes_untagged_states, driver=driver)
        if global_step is None:
            global_step = start_epoch * steps_per_epoch
        if adaptive:
            adaptive.load_state_dict(meta.get("adaptive_lr_state"))
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
                  recaption_instruction=recaption_instruction,
                  recaption_instruction_detailed=recaption_instruction_detailed, captioner_path=captioner)
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
            else:
                md.update({"ss_slider_diff_weight": f"{float(slider_diff_weight):g}"})
        if train_blocks:
            md.update({"ss_train_blocks": ",".join(train_blocks)})
        md.update(driver.run_metadata())
        return md

    def save_lora(path, epoch):
        if ftr is not None:
            ftr.save(path, epoch)
            return
        if ema is not None:
            ema.swap_in()
        try:
            net.save(path, metadata(epoch))
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

    def _previews(epoch):
        conds, w, h, sd, prompts = encoded, sample_width, sample_height, sample_seed, sample_prompts
        ov = _read_sample_override(output_dir)
        if ov and slider:       # a slider preview is its own prompt across strengths: an override never applies
            logger.info("[sample override] ignored - slider previews always show the slider's own prompt")
            ov = None
        if ov:
            logger.info(f"[sample override] active - '{ov['prompt'][:60]}' seed={ov['seed']} {ov['width']}x{ov['height']}")
            try:
                conds = _encode_override(driver, te_path, ov["prompt"], dit, device, parkable=not swapped,
                                         references=ref_imgs)
                sd, prompts = ov["seed"], [ov["prompt"]]
                if not ref_imgs:        # an edit keeps the reference's canvas: its latents are already encoded
                    w, h = ov["width"], ov["height"]
                    if lowmem and max(w, h) > 768:
                        w, h = _cap_canvas(w, h)
            except Exception:
                logger.exception("[sample override] could not encode the override prompt - using the configured ones")
                conds = encoded
        if not sd:          # seed 0 = a fresh random seed every preview round, as on the other trainers
            sd = random.randint(1, 2 ** 31 - 1)
        if ckpt_previews is not None and not slider:
            ck = desc.preview_checkpoint_sampling          # the checkpoint's own recipe (Distilled: 4 steps, no CFG)
            ckpt_previews.render(dit, net, ema, lambda m, fl, swp: _render_previews(
                driver, m, fl, vae, conds, sample_dir, epoch, output_name=output_name, steps=ck.steps, cfg=ck.cfg,
                neg=None, width=w, height=h, seed=sd, speed=ck, swapped=bool(swp), refs=ref_latents))
            last_prompt[0] = prompts[-1] if prompts else None
            return
        _render_previews(driver, dit, net, vae, conds, sample_dir, epoch, output_name=output_name,
                         steps=sample_steps, cfg=sample_cfg_scale, neg=neg, width=w, height=h,
                         seed=sd, ema=ema,
                         speed=speed_desc.settings if (speed_lora and speed_desc) else None, lowmem=lowmem,
                         swapped=bool(swapped), refs=ref_latents, slider=slider)
        last_prompt[0] = prompts[-1] if prompts else None

    ckpt_previews = None
    if preview_checkpoint and encoded is not None:
        if not desc.train_preview_checkpoint:
            logger.warning(f"[sample] {desc.display_name} has no checkpoint previews - ignoring --preview_checkpoint")
        elif not os.path.isfile(preview_checkpoint):
            logger.warning(f"[sample] preview checkpoint {preview_checkpoint} not found - previews use the training "
                           f"model")
        else:
            ckpt_previews = _CheckpointPreviews(
                driver, preview_checkpoint, device, preview_checkpoint_cache, preview_int8,
                context=(context_lora_path, context_lora_strength) if context_lora_path else None)

    def state(epoch):
        _save_state(output_dir, output_name, net, optimizer, epoch=epoch, global_step=global_step, arch_id=arch,
                    ema=ema, extra={"adaptive_lr_state": adaptive.state_dict()} if adaptive else None)

    if sample_at_first and start_epoch == 0:
        previews(ftr.epochs_done if ftr is not None else 0)

    # ---- train ----------------------------------------------------------------------------------
    gen = torch.Generator().manual_seed(seed + start_epoch)
    pause_flag = os.path.join(output_dir, ".pause_requested")
    recorder = LossRecorder()
    progress = tqdm(total=steps_per_epoch * max_train_epochs, initial=global_step, desc="steps", smoothing=0)
    dit.train()
    pending = 0                            # micro-batches backpropagated since the last optimizer step
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

    def _update():
        nonlocal pending
        if ftr is not None and ftr.fused is not None:
            pending = 0                    # every parameter already stepped from its own gradient hook
            lr_acc.clear()
            return
        if max_grad_norm:
            pre = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
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
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update()                   # after the clipped step, so the average tracks what was applied
        pending = 0

    # Warm-up reassurance: the first two epochs of a run start slowly (first-sight kernel planning, cuBLAS picks,
    # allocator and cache warm-up, and on a compiled run the blocks compiling for each new shape) - a crawling bar
    # looks like a hang, so a gentle note repeats every ~30 s while it lasts.
    warmup_note = [0.0]
    for epoch in range(start_epoch, max_train_epochs):
        shared_epoch.value = epoch + 1
        if ftr is not None and ftr.needs_window(epoch):
            optimizer = params = None       # let go of the outgoing window before the swap, or both sit in VRAM
            params, optimizer = ftr.enter(epoch)                 # the epoch's window, and a fresh optimizer for it
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for i, batch in enumerate(loader):
            if epoch - start_epoch < 2 and time.time() - warmup_note[0] > 30.0:
                warmup_note[0] = time.time()
                logger.info("[warm-up] Warm-up phase — the first two epochs start slowly while the GPU plans kernels"
                            + (", compiles the blocks" if do_compile else "")
                            + " and fills its caches. Nothing is stuck; full speed arrives from epoch 3.")
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
            lr_acc.append(_lrm)
            latents = batch["latents"].to(device)
            if slider:
                # both poles backpropagate before the strength flips: checkpointed blocks recompute their forward
                # in backward, at whatever strength is set then. The family's per-step freeze applies as on any step
                # (H3: the Training mode's block window, never the token refiner)
                optimizer.zero_grad(set_to_none=True)
                _frozen_sl = _step_freeze(driver.step_frozen_blocks(batch))
                if slider_prompts:
                    _l, _t = _prompt_slider_step(driver, dit, net, latents, slider_enc, gen,
                                                 guidance=slider_guidance, min_t=min_timestep, max_t=max_timestep)
                else:
                    cond = driver.batch_cond(batch, device)
                    _neg = batch.get("latents_control_0")
                    if _neg is None:
                        raise RuntimeError("[slider] this item has no pair image - an image-pair slider needs one in "
                                           "the second folder for every training image, matched by file name")
                    _neg = _neg.to(device)
                    _l = 0.0
                    try:
                        for m, a, b in ((1.0, latents, _neg), (-1.0, _neg, latents)):
                            net.set_trainable_multiplier(m)
                            _pl, _info = driver.training_loss(dit, a, cond, gen, min_t=min_timestep,
                                                              max_t=max_timestep, diff_ref=b,
                                                              diff_weight=slider_diff_weight)
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
                loss, _info = driver.training_loss(dit, latents, cond, gen, min_t=min_timestep, max_t=max_timestep,
                                                   **({"refs": refs} if refs else {}))
                if "lr_mult" in _info and lr_acc:
                    lr_acc[-1] *= _info["lr_mult"]     # where the step landed (the H3 noise-band LR)
                if pending == 0 and optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                if reg_keys and all(str(k) in reg_keys for k in batch.get("item_keys") or [None]):
                    mult = reg_lr_multiplier       # a regularisation image: a fixed nudge, never the watch's
                else:
                    mult = watch.multiplier(batch)  # per-image LR (batch size 1): the raw loss is still what's recorded
                _scaled = loss * mult if mult != 1.0 else loss
                (_scaled / accum if accum > 1 else _scaled).backward()
                for _p in _frozen_now:
                    _p.requires_grad_(True)
            pending += 1
            if pending >= accum:
                _update()
            global_step += 1
            recorder.add(epoch=epoch, step=i, loss=loss.item())
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
                    f"peak VRAM {torch.cuda.max_memory_reserved() / 1024 ** 3:.1f} GB")
        if adaptive:
            adaptive.epoch_boundary(epoch, recorder.moving_average, net.trainable_modules(), optimizer)
        # problem-image verdicts + queued caption fixes / auto-recaptions, re-encoded before the next epoch
        watch.boundary(epoch + 1, dit, device, parkable=not swapped)

        done = epoch + 1
        cadence = bool(save_every_n_epochs) and done % save_every_n_epochs == 0 and done < max_train_epochs
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
            state(done)
            prune_state_dirs(output_dir, output_name, keep_last_n_states)
            state_saved = True
        if sample_every_n_epochs and done % sample_every_n_epochs == 0:
            _tp = time.time()
            previews(done)
            progress.start_t += time.time() - _tp       # the bar's s/it is training speed, not previews
            # this epoch's checkpoint was saved before its preview existed, with the previous epoch's as its
            # thumbnail (#122): re-embed its own. An explicit --metadata_thumbnail (or "off") stays.
            if cadence and not (metadata_thumbnail or "").strip():
                own = sample_for_epoch(output_dir, output_name, done)
                if own:
                    refresh_checkpoint_thumbnail(os.path.join(output_dir, f"{output_name}-{done:06d}.safetensors"), own)
        if os.path.exists(pause_flag) and done < max_train_epochs:
            if state_saved:
                logger.info(f"[pause] requested - state for epoch {done} already saved; exiting cleanly")
            else:
                logger.info(f"[pause] requested - saving state at epoch {done} and exiting cleanly")
                state(done)
            progress.close()
            sys.exit(0)

    progress.close()
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
    import shutil
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
    p.add_argument("--speed_lora_strength", type=float, default=None,
                   help="the preview speed LoRA's strength (default: the family's recommended value)")
    p.add_argument("--log_per_image_loss", action="store_true", help="detect problem images (Problem Images window)")
    p.add_argument("--per_image_lr", action="store_true", help="scale each step by the image's loss-watch multiplier")
    p.add_argument("--auto_recaption", action="store_true", help="recaption stuck images (needs --captioner)")
    p.add_argument("--warmup_look_outliers", action="store_true", help="LR warm-up for Look Filter outliers")
    p.add_argument("--trigger_position", default="start", choices=("start", "end"))
    p.add_argument("--recaption_instruction", default=None)
    p.add_argument("--recaption_instruction_detailed", default=None)
    p.add_argument("--captioner", default=None,
                   help="auto-recaption's captioner: the Krea 2 Qwen3-VL-4B text encoder file (as the Captions tab)")
    p.add_argument("--lokr_factor", type=int, default=8, help="LoKR only: w1 is about factor x factor")
    p.add_argument("--network_alpha", type=float, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--max_train_epochs", type=int, default=16)
    p.add_argument("--save_every_n_epochs", type=int, default=1)
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
    p.add_argument("--sample_width", type=int, default=None)
    p.add_argument("--sample_height", type=int, default=None)
    p.add_argument("--sample_steps", type=int, default=None)
    p.add_argument("--sample_cfg_scale", type=float, default=None)
    p.add_argument("--sample_negative", default=None)
    p.add_argument("--sample_at_first", action="store_true")
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--family_option", action="append", default=[], metavar="KEY=VALUE",
                   help="a family's own training option (the driver's set_options), e.g. photo_blocks=20-49")
    p.add_argument("--preview_checkpoint", default=None,
                   help="Render training previews on this checkpoint (families with train_preview_checkpoint)")
    p.add_argument("--preview_checkpoint_cache", default="auto", choices=("auto", "on", "off"),
                   help="Keep the preview checkpoint in system RAM between epochs")
    p.add_argument("--preview_int8", action="store_true", help="INT8 matmuls on the preview checkpoint")
    p.add_argument("--sample_reference", default=None, help="Edit previews: the photo every preview prompt edits")
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
    p.add_argument("--slider_bank", type=int, default=16, help="Prompt-pair slider: practice images to render")
    p.add_argument("--train_blocks", default="",
                   help="Comma-separated block ids (the driver's block map) to train; empty = every block")
    p.add_argument("--slider_bank_res", type=int, default=768, help="Prompt-pair slider: practice image size")
    for k in ("title", "author", "description", "license", "tags", "trigger_phrase", "thumbnail"):
        p.add_argument(f"--metadata_{k}", default=None)
    p.add_argument("--trigger_word", default=None, help="Recorded as the trigger phrase when none is given")
    p.add_argument("--resume", default=None)
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
        recaption_instruction=a.recaption_instruction,
        recaption_instruction_detailed=a.recaption_instruction_detailed, captioner=a.captioner,
        network_dim=a.network_dim, network_alpha=a.network_alpha, learning_rate=a.learning_rate,
        max_train_epochs=a.max_train_epochs, save_every_n_epochs=a.save_every_n_epochs, save_state=a.save_state,
        save_state_on_train_end=a.save_state_on_train_end, keep_last_n_states=a.keep_last_n_states, seed=a.seed,
        training_adapter=a.training_adapter, training_adapter_strength=a.training_adapter_strength,
        context_lora_path=a.context_lora_path, context_lora_strength=a.context_lora_strength,
        min_timestep=a.min_timestep, max_timestep=a.max_timestep, speed_lora=a.speed_lora,
        vae_path=a.vae, te_path=a.text_encoder,
        sample_prompts=prompts, sample_every_n_epochs=a.sample_every_n_epochs, sample_width=a.sample_width,
        sample_height=a.sample_height, sample_steps=a.sample_steps, sample_cfg_scale=a.sample_cfg_scale,
        sample_negative=a.sample_negative, sample_at_first=a.sample_at_first, sample_seed=a.sample_seed,
        sample_reference=[a.sample_reference] if a.sample_reference else None, sample_image=a.sample_image,
        slider_pairs=a.slider_pairs, slider_diff_weight=a.slider_diff_weight, slider_prompts=a.slider_prompts,
        slider_guidance=a.slider_guidance, slider_bank=a.slider_bank, slider_bank_res=a.slider_bank_res,
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
        finetune=a.finetune, ft_rotations=a.ft_rotations, ft_save_every_rotations=a.ft_save_every_rotations,
        ft_rotate_every=a.ft_rotate_every, ft_max_parts=a.ft_max_parts, ft_start_window=a.ft_start_window, ft_epochs_done=a.ft_epochs_done,
        ft_fused_backward=a.ft_fused_backward, reg_lr_multiplier=a.reg_lr_multiplier,
        preview_checkpoint=a.preview_checkpoint, preview_checkpoint_cache=a.preview_checkpoint_cache,
        preview_int8=a.preview_int8,
        family_options=dict(o.split("=", 1) for o in a.family_option if "=" in o))


if __name__ == "__main__":
    main()
