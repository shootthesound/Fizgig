"""Training maps: where on a few fixed pictures the model is still wrong, and where the LoRA is changing its prediction,
captured before training and after every epoch (the Samples tab draws them over the pictures, epoch by epoch).

Two maps, both on the family's own noised input (driver.noise_latents) and prediction (driver.predict), the hooks the
textual sliders use, so they work for every picture family:
  * "loss"   - each latent position's squared error against the training target (driver.map_target), channel mean.
  * "effect" - |prediction with the LoRA at 1 - prediction with it at 0| at the same input, channel mean.

The probes are a few training pictures (spread evenly through the sorted keys, so the same dataset picks the same ones),
each at a few fixed noise levels with fixed noise (a generator seeded per picture and level): between epochs only the
LoRA changes. No gradients; the model and LoRA in eval mode; the global CPU/CUDA RNG put back, so training carries on
exactly as it would have; a block-swapped DiT streams these forwards as inference (as previews do); the run's EMA (what
previews and saves use) swapped in. A failure is never fatal: one console line, and the maps stop for the run.

Output, in <output>/training_maps/<LoRA name>/ (runs often share an output folder): probes.json ({keys, timesteps}) and maps.jsonl, one line per epoch:
{"epoch", "maps": {key: {t: {"loss": grid, "effect": grid}}}} with each grid {"w", "h", "v": [row-major floats]}
pooled to at most MAX_SIDE on its long side (a latent grid is small already; the page scales it over the picture).
"""
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
            "epochs": [{"epoch": e, "maps": by[e]} for e in sorted(by)]}


def has_maps(output_dir, output_name):
    return os.path.isfile(os.path.join(maps_dir(output_dir, output_name), "maps.jsonl"))


def pick_keys(keys, n):
    """n keys spread evenly through the sorted list (the same dataset picks the same pictures)."""
    keys = sorted({str(k) for k in keys})
    n = max(0, min(int(n), len(keys)))
    if n <= 1:
        return keys[len(keys) // 2:len(keys) // 2 + 1] if n == 1 else []
    return [keys[round(i * (len(keys) - 1) / (n - 1))] for i in range(n)]


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
    def __init__(self, output_dir, output_name, datasets, *, images=3, timesteps=TIMESTEPS, seed=1234):
        self.dir = maps_dir(output_dir, output_name)
        self.ts = tuple(float(t) for t in timesteps)
        self.seed = int(seed)
        self.batches = {}
        items = {}
        for ds in datasets or []:
            for reso, its in getattr(getattr(ds, "batch_manager", None), "buckets", {}).items():
                for it in its:
                    items.setdefault(str(it.item_key), (reso, it))
        self.keys = [k for k in pick_keys(items, images) if "#" not in k]     # stills only (a clip's still: no)
        from fizgig.dataset.image_dataset import BucketBatchManager
        for k in self.keys:
            reso, it = items[k]
            try:      # its own one-picture batch from the cache, with its own caption
                self.batches[k] = {n: _cpu(v) for n, v in BucketBatchManager({reso: [it]}, 1)[0].items()}
            except Exception:
                logger.debug(f"[maps] could not read {k} from the cache", exc_info=True)
        self.keys = [k for k in self.keys if k in self.batches]
        self.failed = False

    def start(self, resume=False):
        """A fresh run clears the old maps; a resume keeps adding to them."""
        os.makedirs(self.dir, exist_ok=True)
        if not resume:
            try:
                os.remove(os.path.join(self.dir, "maps.jsonl"))
            except OSError:
                pass
        with open(os.path.join(self.dir, "probes.json"), "w", encoding="utf-8") as f:
            json.dump({"keys": self.keys, "timesteps": list(self.ts)}, f)
        logger.info(f"[maps] training maps on: {len(self.keys)} picture(s) x {len(self.ts)} noise level(s), "
                    "each epoch (Samples tab > Training maps)")

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
                        pred = driver.predict(dit, state, cond).float()
                        loss = to_map((pred - target).pow(2), latents)
                        maps = {"loss": _grid(loss)}
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
