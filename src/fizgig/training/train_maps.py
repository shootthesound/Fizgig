"""Training maps: where on a few fixed pictures the model is still wrong, and where the LoRA is changing its prediction,
captured before training and after every epoch (the Samples tab draws them over the pictures, epoch by epoch).

Two maps, both on the family's own noised input (driver.noise_latents) and prediction (driver.predict), the hooks the
textual sliders use, so they work for every picture family:
  * "loss"   - each latent position's squared error against the training target (driver.map_target), channel mean.
  * "effect" - |prediction with the LoRA at 1 - prediction with it at 0| at the same input, channel mean.
A third where the family has the trigger_attention ability and the run a trigger word in the picture's caption:
  * "attn"   - the trigger word's share of each picture token's attention, mean over heads and blocks
               (training/attn_probe.py), from the LoRA-on forward, run eagerly.

The probes are a few training pictures (spread evenly through the sorted keys, so the same dataset picks the same ones),
each at a few fixed noise levels with fixed noise (a generator seeded per picture and level): between epochs only the
LoRA changes. No gradients; the model and LoRA in eval mode; the global CPU/CUDA RNG put back, so training carries on
exactly as it would have; a block-swapped DiT streams these forwards as inference (as previews do); the run's EMA (what
previews and saves use) swapped in. A failure is never fatal: one console line, and the maps stop for the run.

Output, in <output>/training_maps/<LoRA name>/ (runs often share an output folder): probes.json ({keys, timesteps}) and maps.jsonl, one line per epoch:
{"epoch", "maps": {key: {t: {"loss": grid, "effect": grid}}}} with each grid {"w", "h", "v": [row-major floats]}
pooled to at most MAX_SIDE on its long side (a latent grid is small already; the page scales it over the picture).
"""
import contextlib
import json
import logging
import os
import time

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DIRNAME = "training_maps"
TIMESTEPS = (0.25, 0.5, 0.75)    # noise levels: composition (high) to detail (low); not tiny (the target divides by t)
MAX_SIDE = 48


def maps_dir(output_dir, output_name):
    """Where a run's maps live: <output>/training_maps/<LoRA name>."""
    return os.path.join(output_dir or "", DIRNAME, str(output_name or "lora"))


def read(output_dir, output_name, key=None, t=None):
    """A run's maps for the page: {"keys", "timesteps", "epochs": [{"epoch", "maps"}]}, or None. With key and t, only
    that picture at that noise level ({"maps": {key: {t: ...}}}) - what the viewer shows, a ninth of the whole. A resumed
    run's epoch seen twice keeps the later one."""
    d = maps_dir(output_dir, output_name)
    try:
        probes = json.load(open(os.path.join(d, "probes.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return None
    by = {}
    try:
        with open(os.path.join(d, "maps.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue                         # a line still being written
                m = row.get("maps") or {}
                if key is not None:
                    one = (m.get(key) or {}).get(f"{float(t):g}") if t is not None else None
                    m = {key: {f"{float(t):g}": one}} if one is not None else {}
                by[int(row.get("epoch", 0))] = m
    except OSError:
        pass
    return {"keys": probes.get("keys") or [], "timesteps": probes.get("timesteps") or [],
            "crops": probes.get("crops") or {}, "attn": probes.get("attn") or [],
            "attn_words": probes.get("attn_words") or {}, "epochs": [{"epoch": e, "maps": by[e]} for e in sorted(by)]}


def crop_box(original_size, bucket_size):
    """The part of the photo a picture trained on, as [left, top, width, height] fractions of the photo: the bucket
    resize scales it to cover the bucket and crops the centre (dataset/image_dataset.resize_image_to_bucket, the same
    arithmetic). None when either size is unknown."""
    try:
        w, h = (int(x) for x in original_size)
        bw, bh = (int(x) for x in bucket_size)
    except (TypeError, ValueError):
        return None
    if min(w, h, bw, bh) <= 0:
        return None
    s = max(bw / w, bh / h)
    nw, nh = int(w * s + 0.5), int(h * s + 0.5)
    left, top = (nw - bw) // 2, (nh - bh) // 2
    return [round(left / nw, 5), round(top / nh, 5), round(bw / nw, 5), round(bh / nh, 5)]


def has_maps(output_dir, output_name):
    return os.path.isfile(os.path.join(maps_dir(output_dir, output_name), "maps.jsonl"))


def pick_keys(keys, n):
    """n keys spread evenly through the sorted list (the same dataset picks the same pictures)."""
    keys = sorted({str(k) for k in keys})
    n = max(0, min(int(n), len(keys)))
    if n <= 1:
        return keys[len(keys) // 2:len(keys) // 2 + 1] if n == 1 else []
    return [keys[round(i * (len(keys) - 1) / (n - 1))] for i in range(n)]


def pick_probes(keys, n):
    """The map pictures: one from each subject first (Multi Concept / Fizgig LoRA Isolation subjects are keyed "s<n>:";
    the training folder is the first), and one of Isolation's side-by-side pictures when the run has them (the subject
    beside a stranger: does the trigger land on the subject, not the stranger?), then the rest spread evenly through
    the subjects' own pictures. At least one per subject, so more than n when there are more subjects. Stills only;
    never a regularisation picture."""
    import re
    keys = sorted({str(k) for k in keys if "#" not in str(k) and not str(k).startswith("reg:")})
    groups = {}
    for k in keys:
        m = re.match(r"s\d+:", k)
        c = re.match(r"mp\d+:", k)              # one of a subject's composites (Isolation), one per subject
        g = ("composite " + (c.group(0) if c else "")) if "fizgig_mp_" in k else (m.group(0) if m else "")
        groups.setdefault(g, []).append(k)
    if len(groups) <= 1:
        return pick_keys(keys, n)
    out = [pick_keys(groups[g], 1)[0] for g in sorted(groups, key=lambda g: (g.startswith("composite"), g))]
    own = [k for k in keys if "fizgig_mp_" not in k]        # the rest from the subjects' own pictures
    for k in pick_keys(own, max(n, len(out)) + len(out)):
        if len(out) >= max(n, len(groups)):
            break
        if k not in out:
            out.append(k)
    return out


def _grid(m):
    """A (h, w) map -> {"w", "h", "v"}, pooled to <= MAX_SIDE on its long side, 4 significant figures."""
    m = m.float()
    h, w = m.shape
    k = max(1, -(-max(h, w) // MAX_SIDE))
    if k > 1:
        m = F.avg_pool2d(m[None, None], k, ceil_mode=True)[0, 0]
    h, w = m.shape
    return {"w": int(w), "h": int(h), "v": [float(f"{x:.4g}") for x in m.flatten().tolist()]}


def to_map(err, latents):
    """A per-element error in predict()'s layout -> a (h, w) map: a latent grid (B, C, h, w) or with a frame axis
    (B, C, T, h, w) averages its channels (first frame); tokens (B, N, D) or (N, D) average each token, laid out
    row by row on the patch grid the latents make (N = (h / p) * (w / p))."""
    if err.dim() == 4:
        return err.mean(dim=1)[0]
    if err.dim() == 5:
        return err.mean(dim=1)[0, 0]
    tok = err.mean(dim=-1).reshape(-1)
    h, w = latents.shape[-2:]
    n = tok.numel()
    p = max(1, round((h * w / max(1, n)) ** 0.5))
    gh, gw = -(-h // p), -(-w // p)
    if gh * gw != n:
        gh, gw = h // p, w // p
    if gh * gw != n:
        raise RuntimeError(f"can't lay {n} tokens on a {h}x{w} latent grid")
    return tok.view(gh, gw)


def _cpu(v):
    if torch.is_tensor(v):
        return v.detach().to("cpu", copy=True)
    if isinstance(v, (list, tuple)):
        return type(v)(_cpu(x) for x in v)
    if isinstance(v, dict):
        return {k: _cpu(x) for k, x in v.items()}
    return v


class TrainingMaps:
    def __init__(self, output_dir, output_name, datasets, *, images=3, timesteps=TIMESTEPS, seed=1234, driver=None,
                 desc=None, trigger=None):
        self.dir = maps_dir(output_dir, output_name)
        self.ts = tuple(float(t) for t in timesteps)
        self.seed = int(seed)
        self.batches = {}
        items = {}
        for ds in datasets or []:
            for reso, its in getattr(getattr(ds, "batch_manager", None), "buckets", {}).items():
                for it in its:
                    items.setdefault(str(it.item_key), (reso, it))
        self.keys = pick_probes(items, images)              # stills only, one per subject at least
        from fizgig.dataset.image_dataset import BucketBatchManager
        for k in self.keys:
            reso, it = items[k]
            try:      # its own one-picture batch from the cache, with its own caption
                self.batches[k] = {n: _cpu(v) for n, v in BucketBatchManager({reso: [it]}, 1)[0].items()}
            except Exception:
                logger.debug(f"[maps] could not read {k} from the cache", exc_info=True)
        self.keys = [k for k in self.keys if k in self.batches]
        # the part of each photo it trained on, so the viewer lines the maps up with it exactly
        self.crops = {k: c for k in self.keys
                      if (c := crop_box(getattr(items[k][1], "original_size", None),
                                        getattr(items[k][1], "bucket_size", None) or items[k][0]))}
        self.trig = self._trigger_positions(driver, desc, trigger, {k: items[k][1] for k in self.keys})
        self.failed = False

    @staticmethod
    def _trigger_positions(driver, desc, trigger, items):
        """key -> the trigger's positions in the text the model sees, for the pictures whose caption has it (the
        caption its text cache was written from). {} when the family or the run has no trigger map."""
        if not (desc is not None and getattr(desc, "trigger_attention", False) and driver is not None
                and str(trigger or "").strip()):
            return {}
        out = {}
        try:
            from safetensors import safe_open
            for k, it in items.items():
                path = getattr(it, "text_encoder_output_cache_path", None)
                if not path:
                    continue
                with safe_open(path, framework="pt") as f:
                    cap = (f.metadata() or {}).get("caption1") or ""
                per = {}
                for word in [w.strip() for w in str(trigger).split(",") if w.strip()]:
                    pos = driver.trigger_key_positions(cap, word)
                    if pos is None:
                        return {}
                    if pos:
                        per[word] = pos
                if per:
                    out[k] = per
        except Exception as e:
            logger.warning(f"[maps] the trigger-word map is off for this run ({type(e).__name__}: {str(e)[:160]})")
            return {}
        if not out:
            logger.info(f"[maps] no trigger-word map: '{trigger}' isn't in these pictures' captions")
        return out

    def start(self, resume=False):
        """A fresh run clears the old maps; a resume keeps adding to them."""
        os.makedirs(self.dir, exist_ok=True)
        if not resume:
            try:
                os.remove(os.path.join(self.dir, "maps.jsonl"))
            except OSError:
                pass
        with open(os.path.join(self.dir, "probes.json"), "w", encoding="utf-8") as f:
            json.dump({"keys": self.keys, "timesteps": list(self.ts), "crops": self.crops,
                       "attn": sorted(self.trig), "attn_words": {k: list(v) for k, v in self.trig.items()}}, f)
        logger.info(f"[maps] training maps on: {len(self.keys)} picture(s) x {len(self.ts)} noise level(s), "
                    "each epoch (the run card's Training maps button)"
                    + (f"; the trigger word's attention on {len(self.trig)} of them" if self.trig else ""))

    def capture(self, epoch, driver, dit, net, device, *, swapped=False, ema=None):
        """The maps for this epoch (0 = before training), appended to maps.jsonl. Never fatal."""
        if self.failed or not self.keys:
            return
        t0 = time.time()
        try:
            out = self._run(driver, dit, net, device, swapped=swapped, ema=ema, epoch=epoch)
        except Exception as e:
            self.failed = True
            logger.warning("[maps] the training maps failed this epoch, so they're off for the rest of this run. "
                           f"Training carries on. ({type(e).__name__}: {str(e)[:200]})")
            logger.debug("[maps] error:", exc_info=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return
        with open(os.path.join(self.dir, "maps.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": int(epoch), "maps": out}) + "\n")
        logger.info(f"[maps] epoch {epoch}: {len(self.keys)} picture(s), {time.time() - t0:.1f}s")

    def _run(self, driver, dit, net, device, *, swapped, ema, epoch):
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        modes = [(m, m.training) for m in (dit, net) if isinstance(m, torch.nn.Module)]
        streaming = swapped_ema = False
        out = {}
        try:
            if swapped:
                streaming = True
                driver.block_swap_mode(dit, inference=True)
            if ema is not None:
                ema.swap_in()
                swapped_ema = True
            for m, _ in modes:
                m.eval()
            with torch.no_grad():
                for j, k in enumerate(self.keys):
                    b = self.batches[k]
                    latents = b["latents"].to(device)
                    cond = driver.batch_cond(b, device)
                    per = {}
                    for i, t in enumerate(self.ts):
                        g = torch.Generator().manual_seed(self.seed * 1000003 + j * 1009 + i)
                        state = driver.noise_latents(latents, g, min_t=t, max_t=t)
                        target = driver.map_target(latents, state).float()
                        attn = None
                        if k in self.trig:          # the trigger's attention, on this same forward, run eagerly
                            from fizgig.training import attn_probe
                            stance = (torch.compiler.set_stance("force_eager")
                                      if hasattr(torch.compiler, "set_stance") else contextlib.nullcontext())
                            with stance, driver.trigger_attention_hooks(dit), \
                                    attn_probe.capturing(self.trig[k]) as probe:
                                pred = driver.predict(dit, state, cond).float()
                            attn = probe.result()
                        else:
                            pred = driver.predict(dit, state, cond).float()
                        loss = to_map((pred - target).pow(2), latents)
                        maps = {"loss": _grid(loss)}
                        if attn is not None:
                            # every trigger in the caption together; and each on its own when there are several
                            maps["attn"] = _grid(to_map(attn[""][None, :, None], latents))
                            if isinstance(self.trig[k], dict) and len(self.trig[k]) > 1:   # (a cut caption: maybe
                                # fewer than its words - the viewer then says there's no map for the missing one)
                                maps["attn_by"] = {w: _grid(to_map(a[None, :, None], latents))
                                                   for w, a in attn.items() if w}
                        if epoch > 0 and hasattr(net, "set_trainable_multiplier"):
                            net.set_trainable_multiplier(0.0)
                            try:
                                base = driver.predict(dit, state, cond).float()
                            finally:
                                net.set_trainable_multiplier(1.0)
                            maps["effect"] = _grid(to_map((pred - base).abs(), latents))
                        per[f"{t:g}"] = maps
                        del state, target, pred
                    out[k] = per
                    del latents, cond
        finally:
            for m, was in modes:
                m.train(was)
            if swapped_ema:
                ema.swap_out()
            if streaming:
                driver.block_swap_mode(dit, inference=False)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        return out
