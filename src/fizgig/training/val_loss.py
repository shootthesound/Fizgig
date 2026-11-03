"""Fixed validation loss: a few training images scored at the same noise levels with the same noise every check, so
the number only moves when the LoRA does (the step loss jumps around with each step's random timestep and noise).

The images are the ones named with --validation_pick (file names in the training folder, matched to the cache's
item keys by file stem), or - none picked, or none of them in the dataset - picked once from the run's
own item keys (evenly through their sorted list, so the same run picks the same images); each one's cached batch (latents + text embedding, on the CPU) is kept the first time training draws
it. A check runs every family's own `training_loss` with the timestep window pinned to one point and a generator
seeded per image and timestep: no gradients, the model and LoRA in eval mode, and the global CPU/CUDA RNG put back
afterwards, so training carries on exactly as it would have.

A block-swapped DiT (swapped=True) streams these forwards as inference, as previews and DOP's target pass do: in its
training layout the offloader moves each early block to the CPU during the forward and counts on the backward's hooks
to bring it back, so a forward with no backward leaves those blocks on the CPU and the next forward fails ("mat2 is on
cpu"). The training layout is put back afterwards, whatever happens.

The kept batches are the run's snapshot: a caption changed later (the Per-image loss viewer's fixes, auto-recaption -
both rewrite the cache's text embedding) trains from the next epoch but never reaches validation, so its line keeps
measuring the same thing. The snapshot is saved beside the loss log (.fizgig_val_<name>.pt) once every image is kept,
and a resumed run checks against it again; resumed with other validation settings (other images, noise levels or
seed), the check starts over and its line starts a new section (`seg`) from there.
"""
import logging
import os
import time

import torch

from fizgig.training.val_pick import picked_list, resolve_picked  # noqa: F401

logger = logging.getLogger(__name__)


def fixed_timesteps(n):
    """n evenly spaced noise levels inside (0, 1): cell centres, e.g. 4 -> 0.125, 0.375, 0.625, 0.875."""
    n = max(1, int(n))
    return [(k + 0.5) / n for k in range(n)]


def pick_keys(keys, n):
    """n keys spread evenly through the sorted list (deterministic for the same dataset)."""
    keys = sorted({str(k) for k in keys})
    n = max(0, min(int(n), len(keys)))
    if n == 0:
        return []
    if n == 1:
        return [keys[len(keys) // 2]]
    return [keys[round(i * (len(keys) - 1) / (n - 1))] for i in range(n)]


def _cpu(v):
    if torch.is_tensor(v):
        return v.detach().to("cpu", copy=True)
    if isinstance(v, (list, tuple)):
        return type(v)(_cpu(x) for x in v)
    if isinstance(v, dict):
        return {k: _cpu(x) for k, x in v.items()}
    return v


class FixedValidation:
    def __init__(self, keys, *, images=4, timesteps=4, seed=42, every_steps=0, picked=None):
        self.picked, self.missing = resolve_picked(picked, keys) if picked_list(picked) else ([], [])
        for m in self.missing:
            logger.warning(f"[validation] picked image {m!r} is not in this run's dataset (or its cache) - skipped")
        if picked_list(picked) and not self.picked:
            logger.warning("[validation] none of the picked images are in the dataset - using the automatic pick")
        self.wanted = self.picked or pick_keys(keys, images)
        self.ts = fixed_timesteps(timesteps)
        self.seed = int(seed)
        self.every_steps = max(0, int(every_steps or 0))
        self.batches = {}
        self.seg = 0                 # the validation line's section: +1 when a resume changes what's checked
        self.snapshot_path = None    # where the kept batches are saved once all are kept (the trainer sets it)
        self._saved = False

    def offer(self, batch):
        """Keep a copy of this training batch when it is one of the picked images and not kept yet."""
        for k in (str(x) for x in (batch.get("item_keys") or [])):    # Batch Size > 1: the batch it trains in
            if k in self.wanted and k not in self.batches:
                self.batches[k] = {n: _cpu(v) for n, v in batch.items()}
                if self.ready:
                    self.save()
                return

    def prefill(self, datasets):
        """Keep the picked images' batches straight from the cache, before training (the step-0 baseline): each its
        own one-image batch with its own caption. -> True when all are kept."""
        from fizgig.dataset.image_dataset import BucketBatchManager
        for ds in datasets or []:
            for reso, items in getattr(getattr(ds, "batch_manager", None), "buckets", {}).items():
                for it in items:
                    k = str(it.item_key)
                    if k in self.wanted and k not in self.batches:
                        try:
                            self.offer(BucketBatchManager({reso: [it]}, 1)[0])
                        except Exception:
                            logger.debug(f"[validation] could not read {k} for the baseline", exc_info=True)
        return self.ready

    def _setup(self):
        return {"wanted": list(self.wanted), "ts": [round(t, 6) for t in self.ts], "seed": self.seed}

    def save(self):
        """The snapshot to snapshot_path, once (never fatal)."""
        if self._saved or not self.snapshot_path or not self.ready:
            return
        self._saved = True
        try:
            tmp = self.snapshot_path + ".tmp"
            torch.save({"version": 1, **self._setup(), "seg": self.seg, "batches": self.batches}, tmp)
            os.replace(tmp, self.snapshot_path)
        except Exception:
            logger.debug("[validation] could not save the validation snapshot", exc_info=True)

    def restore(self, path):
        """A resumed run: its snapshot back -> "kept" (same settings: the same images, captions and noise as before),
        "changed" (other settings: kept afresh from now, as a new section of the line) or "none" (no snapshot -
        a run from before they were saved - kept afresh, the line as it was)."""
        self.snapshot_path = path
        try:
            snap = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(snap, dict) or snap.get("version") != 1:
                return "none"
        except Exception:
            return "none"
        prev_seg = int(snap.get("seg") or 0)
        if {k: snap.get(k) for k in ("wanted", "ts", "seed")} != self._setup() or \
                not isinstance(snap.get("batches"), dict) or set(snap["batches"]) != set(self.wanted):
            self.seg = prev_seg + 1
            return "changed"
        self.batches, self.seg, self._saved = dict(snap["batches"]), prev_seg, True
        return "kept"

    @property
    def ready(self):
        return bool(self.wanted) and len(self.batches) == len(self.wanted)

    def due(self, global_step):
        return self.every_steps > 0 and self.ready and global_step % self.every_steps == 0

    def run(self, driver, dit, device, net=None, swapped=False):
        """-> (mean loss, [mean at each timestep], seconds), or None when no image is kept yet. swapped: the DiT
        streams its blocks (block swap) - its forwards here run in the offloader's forward-only mode."""
        if not self.batches:
            return None
        t0 = time.time()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        modes = [(m, m.training) for m in (dit, net) if isinstance(m, torch.nn.Module)]
        per_t = [[] for _ in self.ts]
        streaming = False
        try:
            if swapped:
                streaming = True              # set first: a failure inside still puts the training layout back
                driver.block_swap_mode(dit, inference=True)
            for m, _ in modes:
                m.eval()
            with torch.no_grad():
                for j, k in enumerate(self.wanted):
                    batch = self.batches.get(k)
                    if batch is None:
                        continue
                    latents = batch["latents"].to(device)
                    cond = driver.batch_cond(batch, device)
                    refs = [batch[n].to(device) for n in sorted((n for n in batch if n.startswith("latents_control_")),
                                                                key=lambda n: int(n.rsplit("_", 1)[1]))]
                    for i, t in enumerate(self.ts):
                        g = torch.Generator().manual_seed(self.seed * 1000003 + j * 1009 + i)
                        loss, _ = driver.training_loss(dit, latents, cond, g, min_t=t, max_t=t,
                                                       **({"refs": refs} if refs else {}))
                        per_t[i].append(float(loss))
                    del latents, cond, refs
        finally:
            for m, was in modes:
                m.train(was)
            if streaming:                     # back to the training layout (every block where training expects it)
                driver.block_swap_mode(dit, inference=False)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        means = [sum(v) / len(v) for v in per_t if v]
        if not means:
            return None
        return sum(means) / len(means), means, time.time() - t0


FAILED_NOTE = ("[validation] The validation check failed this epoch, so it's been skipped and turned off for the rest "
               "of this run. Training carries on.")


def run_or_skip(validator, driver, dit, device, net=None, swapped=False):
    """validator.run, never fatal to the run: -> (result, failed). On an error, one short line on the console, the
    details in the log at debug level, and failed=True (the trainer then turns the check off for the run); run's
    own clean-up has put the model back in its training layout by then."""
    try:
        return validator.run(driver, dit, device, net, swapped=swapped), False
    except Exception:
        logger.warning(FAILED_NOTE)
        logger.debug("[validation] the check's error:", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, True
