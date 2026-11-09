"""Passive per-image loss logger (experiment/per-image-loss-logger).

Records, per training step, the image trained on, the timestep it drew, and the loss — so we can
study whether per-image loss *trajectories* (timestep-normalized) reveal image difficulty, outliers,
or a better training order/weighting. This is PURELY OBSERVATIONAL: it never touches gradients, the
learning rate, sampling, or ordering. It only writes a log.

Why timestep matters: a diffusion step's loss is dominated by the *random timestep* drawn that step
(loss at t≈0.95 is structurally huge vs t≈0.2), so raw per-image loss mostly ranks the dice roll,
not the image. We log the raw loss AND the timestep, plus a convenience residual against a running
per-timestep-bucket mean, so the intrinsic "harder/easier than average at this noise level" signal
can be recovered. The raw JSONL is the source of truth; do the real normalization offline.

Enable by setting the env var FIZGIG_PERIMAGE_LOSS_LOG=1 before launching training (the trainer
subprocess inherits it). Off by default → zero cost. Writes:
  <output_dir>/loss_log/per_image_loss.jsonl   — one JSON line per image-step

True per-image granularity needs batch size 1 (each step = one image), which is the Krea 2 /
typical LoRA default. For B>1 the step loss is the batch mean; we log it against the joined keys
and record the batch size so analysis can skip those.
"""
import json
import re
import logging
import os
import time

logger = logging.getLogger(__name__)


def _atomic_replace(tmp: str, dst: str, retries: int = 3, delay: float = 0.05) -> bool:
    """os.replace with brief retries for Windows sharing violations.

    os.replace needs DELETE access to dst, and Python's open() doesn't request
    FILE_SHARE_DELETE — so every atomic sidecar write fails with PermissionError
    while a reader holds the file open (the Problem Images window's own 4-second
    poll, OneDrive, an AV scanner). The reader's window is milliseconds; a couple
    of short retries clears it. Returns False (tmp cleaned up) if it never does."""
    for _ in range(retries):
        try:
            os.replace(tmp, dst)
            return True
        except PermissionError:
            time.sleep(delay)
        except OSError:
            break
    try:
        os.remove(tmp)
    except OSError:
        pass
    logger.warning("[loss-watch] could not update %s — a reader holds it open; "
                   "it will refresh next boundary", os.path.basename(dst))
    return False

_ENV_FLAG = "FIZGIG_PERIMAGE_LOSS_LOG"
_N_BUCKETS = 40  # timestep buckets over [0, 1] for the running-mean normalization. Finer buckets
                 # shrink the within-bucket loss spread that dominates per-sample residual noise —
                 # at 20 buckets that spread produced autocorrelated spurious improve-passes that
                 # inflated finish stamps and stalled plateau detection. krea2's logit-normal
                 # timestep sampling concentrates draws mid-range, so central buckets fill fast.


def is_enabled() -> bool:
    """True when FIZGIG_PERIMAGE_LOSS_LOG is set to a truthy value."""
    return os.environ.get(_ENV_FLAG, "").strip().lower() in ("1", "true", "on", "yes")


class PerImageLossLogger:
    """Append-only per-image loss recorder. No-op unless the env flag is set (or force=True)."""

    def __init__(self, output_dir: str, ema_beta: float = 0.9, force: bool = False):
        self.enabled = force or is_enabled()
        self.ema_beta = ema_beta
        self._f = None
        self._ema: dict[str, float] = {}          # item_key -> EMA of raw loss
        self._bucket_sum = [0.0] * _N_BUCKETS     # running loss sum per timestep bucket
        self._bucket_cnt = [0] * _N_BUCKETS
        self.n_records = 0
        if not self.enabled:
            return
        try:
            d = os.path.join(output_dir, "loss_log")
            os.makedirs(d, exist_ok=True)
            self.path = os.path.join(d, "per_image_loss.jsonl")
            self._f = open(self.path, "a", encoding="utf-8")
            logger.info(f"[loss-log] per-image loss logging ON -> {self.path}")
        except Exception as e:
            logger.warning(f"[loss-log] could not open log ({e}); disabling")
            self.enabled = False
            self._f = None

    def _bucket(self, t: float) -> int:
        b = int(min(max(t, 0.0), 0.999999) * _N_BUCKETS)
        return min(b, _N_BUCKETS - 1)

    def record(self, *, epoch: int, step: int, item_keys, timestep: float, loss: float) -> None:
        """Log one image-step. Silently no-ops when disabled; never raises into the training loop."""
        if not self.enabled or self._f is None:
            return
        try:
            loss = float(loss)
            t = float(timestep)
            keys = item_keys if isinstance(item_keys, (list, tuple)) else [item_keys]
            keys = [str(k) for k in keys] if keys else ["<unknown>"]
            key = keys[0] if len(keys) == 1 else "|".join(keys)

            b = self._bucket(t)
            self._bucket_sum[b] += loss
            self._bucket_cnt[b] += 1
            bmean = self._bucket_sum[b] / max(self._bucket_cnt[b], 1)

            prev = self._ema.get(key)
            ema = loss if prev is None else self.ema_beta * prev + (1.0 - self.ema_beta) * loss
            self._ema[key] = ema

            rec = {
                "epoch": int(epoch), "step": int(step), "key": key,
                "t": round(t, 5), "loss": round(loss, 6),
                "t_bucket": b, "t_bucket_mean": round(bmean, 6),
                "residual": round(loss - bmean, 6),   # loss minus running mean at this noise level
                "ema": round(ema, 6),                 # per-image EMA of raw loss
                "batch": len(keys),                   # >1 => batch-mean, not true per-image
            }
            self._f.write(json.dumps(rec) + "\n")
            self._f.flush()
            self.n_records += 1
        except Exception as e:
            logger.warning(f"[loss-log] record failed ({e})")

    def close(self) -> None:
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
            self._f = None


def where(key, folders):
    """(key prefix, folder, name in that folder) of a training key over {prefix: folder}, or (None, None, None).
    A key is its folder's prefix ("" for the training folder, "outfits/red/" for a sub-folder, "s2:" for another
    subject) + the picture's name without its extension; the longest matching prefix wins."""
    key = str(key)
    best = None
    for p in folders or {}:
        if key.startswith(p) and (best is None or len(p) > len(best)):
            best = p
    name = key[len(best):] if best is not None else ""
    # a sub-folder's or another folder's picture whose folder isn't in the map is no picture of a folder that is
    if best is None or "/" in name or re.match(r"(s\d+|reg):", name):
        return None, None, None
    return best, folders[best], name


class PerImageLossWatch:
    """Online per-image difficulty watcher — the actionable layer on top of the passive logger.

    Two GUI-toggleable behaviours share this one class:
      * problem-image detection (`write_jsonl` / always-on analytics): at each epoch boundary,
        classifies every image as STUCK (high residual, not descending — likely outlier/mislabel),
        LEARNING (high but descending — hard-but-good), or easy, logs stuck images to the console,
        and writes <output_dir>/loss_log/problem_images.json.
      * per-image adaptive LR (`apply_lr=True`): additionally emits a per-image loss multiplier —
        throttle STUCK images (default x0.5) so one bad image can't keep yanking the weights, give
        healthy/learned ones a gentle boost (default x1.1), leave hard-but-learning alone. At batch
        size 1, scaling the step's loss IS a per-image LR. Guardrails: no action during the warmup
        epochs (the trend needs data), the healthy x1.1 is the ONLY boost (every problem verdict
        reduces), and batch size > 1 disables scaling entirely (a batch mean isn't a per-image
        signal).

    Normalization matches the validated offline analyzer: residuals are recomputed at every epoch
    boundary against the CURRENT per-timestep-bucket means over the whole run so far — never the
    order-dependent live running mean (first sample per bucket would always read as residual 0).
    Raw records are kept in memory; a few thousand (key, epoch, bucket, loss) tuples is trivial.
    """

    def __init__(self, output_dir: str, *, apply_lr: bool = False, write_jsonl: bool = False,
                 warmup_epochs: int = 2, window: int = 5,
                 throttle_mult: float = 0.5, easy_mult: float = 1.1,
                 hi_q: float = 0.66, lo_q: float = 0.33,
                 persist_on: int = 2, persist_off: int = 3,
                 improve_frac: float = 0.12, improve_floor: float = 0.02,
                 suspect_mult: float = 0.7, easy_from_epoch: int = 3,
                 exhausted_mult: float = 0.6, exhaust_drop_frac: float = 0.3, exhaust_on: int = 2,
                 stuck_floor: float = 0.25, escalate_every: int = 2,
                 dataset_dir: str = None, caption_ext: str = ".txt", healthy_min: int = 3,
                 plateau_patience: int = 2, plateau_min_epochs: int = 8, family: str = "", folders=None):
        self.apply_lr = apply_lr
        self.warmup_epochs = warmup_epochs
        self.window = window
        self.throttle_mult = throttle_mult
        self.easy_mult = easy_mult
        self.hi_q = hi_q
        self.lo_q = lo_q
        self.persist_on = persist_on      # consecutive stuck votes to CONFIRM stuck
        self.persist_off = persist_off    # consecutive clear votes to RELEASE stuck
        # "Improving" test: split the last trend_window epochs into two halves and require the
        # recent half's mean residual to sit below the older half's by BOTH a fraction of the
        # excess AND ~1 standard error (estimated from that image's own residual scatter). With
        # one loss sample per image per epoch, slope signs and point-to-point drops both flap
        # inside the noise (validated on synthetic runs) — half-window means with a data-driven
        # noise bar is what actually separates a learner from a stuck outlier.
        self.improve_frac = improve_frac
        self.improve_floor = improve_floor
        self.trend_window = 8             # epochs; halves of 4 vs 4
        # Early-suspicion tier: caption-contradiction images are separable by MAGNITUDE from the
        # first epochs (extreme residual, not merely high), long before a trend exists. A robust
        # outlier (median + 1.5*IQR) for 2 consecutive boundaries — or a wild one (3*IQR) once —
        # gets a provisional, mild suspect_mult while the trend machinery gathers evidence.
        # Rationale: modern runs form identity in epochs 1-6 (real likeness by epoch 3 on clean
        # data), so waiting for trend confirmation acts after the damage window has closed.
        self.suspect_mult = suspect_mult
        self.easy_from_epoch = easy_from_epoch  # healthy boost needs a slightly steadier baseline
        # "Exhausted" tier: an image that PROVED a good run (residual dropped >= exhaust_drop_frac
        # of its early baseline) and then plateaued while still above-average difficulty. Its
        # caption is fine — the model has mined what it can — so further full-LR passes are mostly
        # overbake pressure. Gets exhausted_mult, and its improvement history SUPPRESSES the stuck/
        # suspect paths (a plateaued good-runner is not an outlier, whatever its current level).
        self.exhausted_mult = exhausted_mult
        self.exhaust_drop_frac = exhaust_drop_frac
        self.exhaust_on = exhaust_on
        # Escalating stuck throttle: staying confirmed IS accumulating evidence, so the penalty
        # deepens — throttle_mult on confirmation, halved every escalate_every further confirmed
        # epochs, floored at stuck_floor. A flat x0.5 forever still leaks half-strength poison all
        # run; near-zero from day one would deny a false conviction the gradient it needs to prove
        # itself and win release. Caption fixes reset history -> straight back to x1.0.
        # Floor x0.25 (was x0.1 until 9 Nov 2026): replaying 39 real runs, images convicted stuck still got easier
        # at full LR in runs without the throttle (by ~0.26 sd even among the fence's outliers) - x0.1 starved
        # images that were learnable, and with auto-recaption off nothing ever resets them.
        self.stuck_floor = stuck_floor
        self.escalate_every = escalate_every
        self.output_dir = output_dir

        self._records: list[tuple[str, int, int, float]] = []   # (key, epoch, bucket, loss)
        self._bsum = [0.0] * _N_BUCKETS
        self._bcnt = [0] * _N_BUCKETS
        self._mult: dict[str, float] = {}
        self.verdicts: dict[str, str] = {}
        self._epochs_seen: set[int] = set()
        self._batched_warned = False
        self._batched = False
        self._replaying = False   # True while resume_from_jsonl rebuilds state (mutes side effects)
        # Look-filter warm-up (curriculum): genuine-but-unusual images (tight angles, profiles)
        # ramp from a muted LR to full over the first epochs — see set_warmup_keys().
        self._warmup: set[str] = set()
        self._warmup_released: dict[str, int] = {}   # key -> epoch its ramp ended (proof or timeout)
        self._warmup_start = 0.4
        self._warmup_ramp_epochs = 4
        self._epoch_now = 1
        # Persistence state: a single noisy epoch must not flip a verdict. Per-key counters of
        # consecutive stuck / clear votes, plus the confirmed set (drives warnings + throttling).
        self._stuck_votes: dict[str, int] = {}
        self._clear_votes: dict[str, int] = {}
        self._suspect_votes: dict[str, int] = {}
        self._exhaust_votes: dict[str, int] = {}
        self._stuck_epochs: dict[str, int] = {}   # consecutive epochs confirmed (drives escalation)
        self._confirmed_stuck: set[str] = set()
        self._last_reported_stuck: set[str] = set()
        # Keys whose benefit of the doubt is spent (e.g. two failed AI recaptions): re-confirming
        # stuck EXCLUDES them from training entirely for the rest of the run — the trainer skips
        # their steps (no gradient, no loss recorded, so avr_loss stops carrying their permanent
        # error term). One-way, except reset_key (a manual caption edit re-admits the image).
        self._incorrigible: set[str] = set()
        self._excluded: set[str] = set()
        self.family = str(family or "")
        # excluded by an earlier run of this family: trains normally, but is already past its two recaptions, so a
        # confirmed stuck verdict excludes it at once (see _load_persistent_exclusions)
        self._known_hard: set[str] = set()
        # Health record: epochs a key spent comfortably OUT of the hard zone (or in a proven good
        # run). An ever-healthy image (>= healthy_min epochs) has PROVEN learnability — its later
        # souring is the mined-out/drift pattern, not caption poison — so at the exclusion
        # decision it's RETIRED as permanent exhausted (x exhausted_mult) instead of excluded.
        # Health survives reset_key on purpose: it's evidence about the image, not the caption.
        self.healthy_min = healthy_min
        self._healthy_epochs: dict[str, int] = {}
        self._retired: set[str] = set()

        # Best-epoch estimate: every image has a knowable FINISH epoch (the last boundary it
        # passed the improve test). When no image is improving for plateau_patience consecutive
        # boundaries, training is done extracting signal — the recommended checkpoint is the
        # ~75th percentile of per-image finish epochs (excluded images don't vote). This detects
        # "learning finished" (the honest stop anchor), NOT a certified quality peak — the GUI
        # frames it as a window to scrub in LoRA Royale. Note: late adaptive-LR reductions shrink
        # per-epoch drops below the improve floor, which can trip this slightly early.
        self.plateau_patience = plateau_patience
        self.plateau_min_epochs = plateau_min_epochs
        self._last_improving_epoch: dict[str, int] = {}
        self._improving_streak: dict[str, int] = {}   # consecutive improve-passes per key
        self._no_improve_streak = 0
        self._plateau_reported = False
        self._plateau_was_provisional = False
        self.plateaued = False
        self.plateau_pending = 0
        self.best_epoch_estimate = None

        # Persistent exclusions: fizgig_excluded.json lives IN the dataset folder, one entry per image with the
        # families that excluded it. A later run of the same family does NOT skip the image (Peter, 5 Oct 2026: it
        # taught the run until it got stuck, and "stuck" was that run's verdict): it trains normally, but is marked
        # as past its two AI recaptions, so if it is confirmed stuck again it is excluded at once. Each entry
        # snapshots the caption; a changed .txt (fixed offline) re-admits it fully. Mid-run caption edits clear
        # it via reset_key, which also removes the entry. Entries from before per-family records count for every
        # family. folders: {key prefix: folder} when the run trains several folders (sub-folders, other subjects):
        # each folder keeps its own file, its entries named as in that folder, so the record travels with it.
        self.folders = {p: d for p, d in (folders if folders is not None else {"": dataset_dir}).items()
                        if d and os.path.isdir(d)}
        self.dataset_dir = self.folders.get("")
        self.caption_ext = caption_ext
        self._excl_file = bool(self.folders) or None
        self._excl_data: dict[str, dict] = {}
        self._load_persistent_exclusions()

        # The JSONL logger stays the offline source of truth; force it on when the detection
        # toggle asks for it (env var still works on its own).
        self._jsonl = PerImageLossLogger(output_dir, force=write_jsonl)

    # ---- persistent exclusions -------------------------------------------------

    def where(self, key: str):
        """(key prefix, folder, name in that folder) of a training key, or (None, None, None) - the longest prefix
        wins, so a sub-folder's pictures are never read as the training folder's own."""
        return where(key, self.folders)

    def _current_caption(self, key: str):
        _p, folder, name = self.where(key)
        if not folder:
            return None
        p = os.path.join(folder, name + self.caption_ext)
        try:
            # utf-8-sig + replace: a BOM or stray legacy byte must not make the caption-changed
            # comparison (the exclusion pardon) read as "unreadable" and keep an image excluded.
            with open(p, encoding="utf-8-sig", errors="replace") as f:
                return f.read().strip()
        except Exception:
            return None

    def _load_persistent_exclusions(self) -> None:
        data = {}
        for prefix, folder in self.folders.items():
            path = os.path.join(folder, "fizgig_excluded.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data.update({prefix + str(k): v for k, v in json.load(f).items()})
            except Exception:
                logger.warning(f"[loss-watch] could not read {path} — ignoring")
        if not data:
            return
        pruned = []
        for key, entry in dict(data).items():
            cur = self._current_caption(key)
            if cur is not None and entry.get("caption") is not None and cur != entry["caption"]:
                # Caption changed since exclusion (fixed offline) — re-admit.
                pruned.append(key)
                del data[key]
            elif "families" not in entry or self.family in entry["families"]:
                self._known_hard.add(str(key))
                self._incorrigible.add(str(key))
        self._excl_data = data
        if pruned:
            self._write_persistent_exclusions()
            logger.info(f"[loss-watch] re-admitted {len(pruned)} previously-excluded image(s) "
                        f"whose captions changed: " + ", ".join(os.path.basename(k) for k in pruned))
        if self._known_hard:
            logger.info(f"[loss-watch] {len(self._known_hard)} image(s) were excluded by an earlier run "
                        f"(fizgig_excluded.json): they train normally, and are excluded at once if they get stuck "
                        f"again (their two AI recaptions are spent). Edit a caption to clear its record: "
                        + ", ".join(sorted(os.path.basename(k) for k in self._known_hard)))

    def _write_persistent_exclusions(self) -> None:
        if not self._excl_file:
            return
        per = {p: {} for p in self.folders}
        for key, entry in self._excl_data.items():
            prefix, _folder, name = self.where(key)
            if prefix is not None:
                per[prefix][name] = entry
        for prefix, entries in per.items():
            path = os.path.join(self.folders[prefix], "fizgig_excluded.json")
            if not entries and not os.path.exists(path):
                continue
            try:
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(entries, f, indent=2)
                _atomic_replace(tmp, path)
            except Exception:
                logger.warning(f"[loss-watch] could not write {path}", exc_info=True)

    def _record_exclusion(self, key: str, epoch: int) -> None:
        if not self._excl_file:
            return
        import time
        entry = self._excl_data.get(key) or {}
        fams = dict(entry.get("families") or {})
        fams[self.family or "?"] = {"epoch": int(epoch), "date": time.strftime("%Y-%m-%d %H:%M"),
                                    "reason": "still stuck after two AI recaptions"}
        self._excl_data[key] = {"families": fams, "caption": self._current_caption(key)}
        self._write_persistent_exclusions()

    def preflight(self, dataset_keys) -> None:
        """Reconcile persisted exclusion state with the actual training set (trainer calls this
        once, after the dataloader is built): prunes entries whose images have left the dataset
        (they'd pad the file forever)."""
        try:
            keys = {str(k) for k in dataset_keys}
            if not keys:
                return
            ghosts = {k for k in self._excl_data if k not in keys}
            if ghosts:
                for g in ghosts:
                    del self._excl_data[g]
                self._known_hard -= ghosts
                self._incorrigible -= ghosts
                self._write_persistent_exclusions()
                logger.info(f"[loss-watch] pruned {len(ghosts)} stale exclusion entries for "
                            f"images no longer in the dataset")
        except Exception:
            logger.warning("[loss-watch] preflight failed", exc_info=True)

    # ---- per-step ------------------------------------------------------------

    @staticmethod
    def _key_of(item_keys) -> str:
        keys = item_keys if isinstance(item_keys, (list, tuple)) else [item_keys]
        keys = [str(k) for k in keys] if keys else ["<unknown>"]
        return keys[0] if len(keys) == 1 else "|".join(keys)

    def set_warmup_keys(self, keys, *, start: float = 0.4, ramp_epochs: int = 4) -> None:
        """Look Consistency Filter outliers: genuine but UNUSUAL images (tight angles, profiles,
        occlusion) whose gradient fights the forming identity core in the first epochs — before
        the watch has any trend to act on. Prior-then-evidence: the embedding score covers the
        watch's warmup blindness (epochs 1-4), then normal verdicts take over. Each key starts
        at `start` and ramps linearly to x1.0 over `ramp_epochs` epochs, released EARLY the
        moment it proves it's improving. Stuck/suspect votes are held until 2 epochs after the
        ramp ends — a muted image learns slowly BY DESIGN and must not be convicted for it."""
        self._warmup = {str(k) for k in keys}
        self._warmup_start = float(start)
        self._warmup_ramp_epochs = max(1, int(ramp_epochs))

    def _warmup_mult(self, key: str):
        """Current ramp multiplier for an actively warming key, else None."""
        if key not in self._warmup or key in self._warmup_released:
            return None
        frac = (self._epoch_now - 1) / self._warmup_ramp_epochs
        if frac >= 1.0:
            return None
        return self._warmup_start + (1.0 - self._warmup_start) * frac

    def multiplier(self, item_keys) -> float:
        """Loss multiplier for the CURRENT step (looked up before observe; updates at epoch ends).
        Warm-up ramps apply even when per-image LR is off — they're their own toggle."""
        if self._batched:
            return 1.0
        key = self._key_of(item_keys)
        wm = self._warmup_mult(key)
        base = self._mult.get(key, 1.0) if self.apply_lr else 1.0
        return min(wm, base) if wm is not None else base

    def is_excluded(self, item_keys) -> bool:
        """True when this step's image is excluded from training (skip the step entirely)."""
        if self._batched:
            return False
        return self._key_of(item_keys) in self._excluded

    def observe(self, *, epoch: int, step: int, item_keys, timestep: float, loss: float) -> None:
        """Record one step's RAW (unscaled) loss. Never raises into the training loop."""
        try:
            keys = item_keys if isinstance(item_keys, (list, tuple)) else [item_keys]
            # Batch detection must also survive RESUME REPLAY, where a batched step's keys
            # arrive as one composite string ("a|b|c" — "|" can't appear in a filename), not
            # a list. Without this the latch came back clear after resume.
            if keys is not None and (len(keys) > 1
                                     or (len(keys) == 1 and "|" in str(keys[0]))):
                self._batched = True
                if self.apply_lr and not self._batched_warned:
                    self._batched_warned = True
                    logger.warning("[loss-watch] batch size > 1 — per-image LR disabled "
                                   "(a batch-mean loss isn't a per-image signal)")
            t = float(timestep)
            b = min(int(min(max(t, 0.0), 0.999999) * _N_BUCKETS), _N_BUCKETS - 1)
            loss = float(loss)
            key = self._key_of(item_keys)
            self._epoch_now = max(self._epoch_now, int(epoch))   # drives the warm-up ramp
            self._records.append((key, int(epoch), b, loss))
            self._bsum[b] += loss
            self._bcnt[b] += 1
            self._epochs_seen.add(int(epoch))
            if not self._replaying:   # replayed steps came FROM the jsonl — don't duplicate them
                self._jsonl.record(epoch=epoch, step=step, item_keys=item_keys, timestep=t, loss=loss)
        except Exception as e:
            logger.warning(f"[loss-watch] observe failed ({e})")

    # ---- per-epoch -----------------------------------------------------------

    @staticmethod
    def _slope(xs, ys) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        mx = sum(xs) / n
        my = sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom == 0:
            return 0.0
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom

    def epoch_boundary(self, epoch: int) -> dict:
        """Reclassify every image and refresh multipliers. Returns {key: verdict} (empty pre-warmup)."""
        try:
            if not self._records or len(self._epochs_seen) < self.warmup_epochs:
                return {}
            bucket_mean = [(self._bsum[i] / self._bcnt[i]) if self._bcnt[i] else 0.0
                           for i in range(_N_BUCKETS)]
            per_epoch: dict[str, dict[int, list[float]]] = {}
            raw_sum: dict[str, float] = {}
            raw_cnt: dict[str, int] = {}
            for key, ep, b, loss in self._records:
                per_epoch.setdefault(key, {}).setdefault(ep, []).append(loss - bucket_mean[b])
                raw_sum[key] = raw_sum.get(key, 0.0) + loss
                raw_cnt[key] = raw_cnt.get(key, 0) + 1

            stats = {}
            for key, ep_map in per_epoch.items():
                all_eps = sorted(ep_map)
                series = [sum(ep_map[e]) / len(ep_map[e]) for e in all_eps]  # raw per-epoch residual

                # Trend: compare the two halves of the last trend_window epochs. Half-means average
                # away the single-sample-per-epoch noise; the standard error of their difference
                # (from this image's own scatter) gives the noise bar the improve test must beat.
                tw = min(self.trend_window, len(series) - len(series) % 2)
                old_mean = new_mean = se = 0.0
                if tw >= 4:
                    win = series[-tw:]
                    half = tw // 2
                    old_half, new_half = win[:half], win[half:]
                    old_mean = sum(old_half) / half
                    new_mean = sum(new_half) / half
                    var = (sum((v - old_mean) ** 2 for v in old_half)
                           + sum((v - new_mean) ** 2 for v in new_half)) / max(tw - 2, 1)
                    se = (2.0 * var / half) ** 0.5

                res = series[-self.window:]
                eps = all_eps[-self.window:]
                # Early baseline: this image's residual level over its first few epochs — the
                # reference for "had a good run" (exhausted detection).
                nb = min(3, len(series))
                baseline = sum(series[:nb]) / nb
                stats[key] = {"mean_residual": sum(res) / len(res),
                              "slope": self._slope([float(e) for e in eps], res),
                              "first": old_mean, "last": new_mean, "se": se,
                              "trend_epochs": tw,
                              "baseline": baseline,
                              "total_drop": baseline - new_mean if tw >= 4 else 0.0,
                              "mean_loss": raw_sum[key] / raw_cnt[key],
                              "epochs": len(all_eps)}

            ordered = sorted(s["mean_residual"] for s in stats.values())
            hi = ordered[int(self.hi_q * (len(ordered) - 1))]
            lo = ordered[int(self.lo_q * (len(ordered) - 1))]
            # Robust outlier thresholds for the early-suspicion tier (floored at hi so the tier
            # never fires on merely-top-third images even when the IQR is degenerate/tiny).
            n = len(ordered)
            med = ordered[n // 2]
            iqr = ordered[int(0.75 * (n - 1))] - ordered[int(0.25 * (n - 1))]
            ext_hi = max(med + 1.5 * iqr, hi)
            ext_vhi = max(med + 3.0 * iqr, hi)
            n_ep = len(self._epochs_seen)
            # Exclusion cap: never exclude past half the dataset. Beyond that the flags are
            # relative rankings within a shrinking pool, not evidence of poison — over-cap
            # candidates retire as exhausted (still train, gently) instead.
            total_imgs = len(stats) + sum(1 for k in self._excluded if k not in stats)
            excl_cap = max(1, total_imgs // 2)

            # Per-epoch RAW votes -> persistence state machine. One noisy epoch can't flip a
            # verdict: stuck is CONFIRMED only after persist_on consecutive stuck votes, and
            # RELEASED only after persist_off consecutive clear votes. The suspect tier acts
            # earlier on MAGNITUDE alone (mild suspect_mult) while the trend evidence accrues.
            new_mult = {}
            improving_count = 0
            for key, s in stats.items():
                # Already excluded: frozen state — no votes, no releases, skipped by the trainer.
                if key in self._excluded:
                    s["verdict"] = "excluded"
                    s["multiplier"] = 0.0
                    new_mult[key] = 0.0
                    continue
                # Retired (ever-healthy image that would otherwise have been excluded): permanent
                # exhausted — still trains, gently. Only a manual caption edit un-retires it.
                if key in self._retired:
                    s["verdict"] = "exhausted"
                    s["multiplier"] = self.exhausted_mult if self.apply_lr else 1.0
                    new_mult[key] = self.exhausted_mult
                    continue
                # Improving = recent half-mean below older half-mean by a real margin (fraction of
                # the excess AND ~1 SE). Too little history -> not improving is unknowable; the
                # persistence gate + warmup keep that from mattering.
                drop = s["first"] - s["last"]
                improving = (s["trend_epochs"] >= 4
                             and drop >= max(self.improve_frac * max(s["first"], 0.0),
                                             self.improve_floor, s["se"]))
                s["improving"] = improving
                # A flat image passes the improve test spuriously ~10% of boundaries (noise beats
                # the floor), which inflates finish stamps and delays plateau detection — so an
                # image only COUNTS as improving after two consecutive passes (same persistence
                # principle as every other verdict here). The finish is stamped at the CENTER of
                # the detected drop window: a pass at epoch e reflects improvement around
                # e - trend_window/2; stamping e itself biases the estimate late by half a window.
                self._improving_streak[key] = self._improving_streak.get(key, 0) + 1 if improving else 0
                if self._improving_streak[key] >= 2:
                    improving_count += 1
                    self._last_improving_epoch[key] = max(1, epoch - self.trend_window // 2)
                # "Good run": this image's residual has dropped substantially from its early
                # baseline — proof the caption works and the model CAN learn it. A plateaued
                # good-runner is mined out, not stuck: its history suppresses the outlier paths.
                good_run = (s["trend_epochs"] >= 4 and s["baseline"] > 0.0
                            and s["total_drop"] >= max(self.exhaust_drop_frac * s["baseline"],
                                                       2.0 * self.improve_floor))
                # Health record: comfortably out of the hard zone, or a proven good run — either
                # is evidence the image IS learnable (protects it from exclusion later).
                if s["mean_residual"] < hi or good_run:
                    self._healthy_epochs[key] = self._healthy_epochs.get(key, 0) + 1
                # An image whose recent residual is <= 0 is no harder than average at the same
                # noise level — by definition not an outlier, whatever its trend. And with under
                # 4 epochs of history (fresh run, or history reset after a caption fix) there is
                # no trend to judge — it can't vote stuck yet.
                # Wobble bar (mean_residual > se): "hard" must be a real magnitude, not a rank.
                # The top third of a SPOTLESS dataset is still occupied by someone — without this,
                # the hardest clean images get convicted by percentile alone. An image only votes
                # stuck when it sits above average by more than its own epoch-to-epoch noise
                # (real caption poison clears this bar by an order of magnitude; validated by
                # replaying the 2026-07 real-run logs through both detector versions).
                # Outlier fence (9 Nov 2026): "hard" is also above median + 1.5 IQR (the suspect tier's ext_hi), not
                # just the top third. Late in a run almost nothing passes the improve test, so "top third and not
                # improving" had become "the hardest third": replaying 39 real runs convicted ~1/3 of every dataset,
                # each one at the floor by epoch 11. With the fence, 1026 convictions became 249, and the ones kept
                # are the least-improving of equally hard images.
                # Look-filter warm-up: release on proof of improvement or ramp completion, and
                # hold stuck/suspect votes until 2 epochs after the ramp ends — a deliberately
                # muted image reads as high-flat (the stuck signature) precisely BECAUSE it is
                # being protected; convicting it would defeat the warm-up.
                if key in self._warmup and key not in self._warmup_released:
                    if improving or (epoch - 1) >= self._warmup_ramp_epochs:
                        self._warmup_released[key] = epoch
                warm_hold = key in self._warmup and (
                    key not in self._warmup_released or epoch <= self._warmup_released[key] + 2)
                votes_stuck = (s["trend_epochs"] >= 4 and s["mean_residual"] >= ext_hi
                               and s["mean_residual"] > s["se"]
                               and s["last"] > 0.0 and not improving and not good_run
                               and not warm_hold)
                if key in self._confirmed_stuck:
                    self._stuck_epochs[key] = self._stuck_epochs.get(key, 0) + 1
                    self._clear_votes[key] = 0 if votes_stuck else self._clear_votes.get(key, 0) + 1
                    if self._clear_votes[key] >= self.persist_off:
                        self._confirmed_stuck.discard(key)
                        self._clear_votes[key] = 0
                        self._stuck_votes[key] = 0
                        # Tenure survives release ON PURPOSE: a noisy 1-2 epoch release must not
                        # reset the escalation ladder — re-confirmation resumes at depth. A real
                        # rehabilitation never re-confirms, and a caption fix wipes it (reset_key).
                else:
                    self._stuck_votes[key] = self._stuck_votes.get(key, 0) + 1 if votes_stuck else 0
                    if self._stuck_votes[key] >= self.persist_on:
                        self._confirmed_stuck.add(key)
                        self._clear_votes[key] = 0
                        self._stuck_epochs[key] = self._stuck_epochs.get(key, 0) + 1

                # Early-suspicion votes: extreme magnitude, not yet improving. (With <4 trend
                # epochs `improving` is always False, which is exactly right here — magnitude is
                # the only early signal, and release comes via the improve test once a trend forms.)
                # Same wobble bar once a trend exists; pre-trend (se unknowable) stays
                # magnitude-only — early suspicion is deliberately fast and mild.
                extreme = (s["mean_residual"] >= ext_hi and not improving and not good_run
                           and (s["trend_epochs"] < 4 or s["mean_residual"] > s["se"])
                           and not warm_hold)
                self._suspect_votes[key] = self._suspect_votes.get(key, 0) + 1 if extreme else 0
                suspect = (key not in self._confirmed_stuck
                           and (self._suspect_votes[key] >= 2
                                or (self._suspect_votes[key] >= 1 and s["mean_residual"] >= ext_vhi)))

                # Exhausted votes: proved a good run, now plateaued while still above-average
                # difficulty. Improving again (e.g. rest of the dataset caught up, or a caption
                # tweak) releases it immediately.
                exhaust_now = good_run and not improving and s["last"] > 0.0
                self._exhaust_votes[key] = self._exhaust_votes.get(key, 0) + 1 if exhaust_now else 0
                exhausted = self._exhaust_votes[key] >= self.exhaust_on

                # Release progress for the popup (stuck badge + improving trend = counting down).
                s["release_votes"] = self._clear_votes.get(key, 0) if key in self._confirmed_stuck else 0

                if key in self._excluded:
                    # Already excluded (this run, a previous run's persistent record, or replayed
                    # resume history) — report it as such; never re-run the ladder on frozen stats.
                    verdict, mult = "excluded", 0.0
                elif key in self._confirmed_stuck and key in self._incorrigible:
                    self._confirmed_stuck.discard(key)
                    self._last_reported_stuck.discard(key)
                    if len(self._excluded) >= excl_cap:
                        self._retired.add(key)
                        logger.warning(f"[loss-watch] epoch {epoch}: {os.path.basename(key)} "
                                       f"qualifies for exclusion but the cap "
                                       f"({excl_cap}/{total_imgs} images) is reached — retiring "
                                       f"as exhausted (x{self.exhausted_mult}) instead.")
                        verdict, mult = "exhausted", self.exhausted_mult
                    elif self._healthy_epochs.get(key, 0) >= self.healthy_min:
                        # Ever-healthy images have PROVEN learnability — a later souring is the
                        # mined-out/drift pattern, not caption poison. Retire as permanent
                        # exhausted instead of excluding; it keeps training, gently.
                        self._retired.add(key)
                        logger.warning(f"[loss-watch] epoch {epoch}: {os.path.basename(key)} was "
                                       f"healthy for {self._healthy_epochs[key]} epoch(s) earlier — "
                                       f"retiring as exhausted (x{self.exhausted_mult}) instead of "
                                       f"excluding.")
                        verdict, mult = "exhausted", self.exhausted_mult
                    else:
                        # Benefit of the doubt spent (two failed AI recaptions), stuck again, and
                        # NEVER healthy: excluded for the rest of the run. The trainer skips its
                        # steps — no gradient, no loss recorded — so avr_loss stops carrying its
                        # permanent error term. Only a manual caption edit (reset_key) re-admits it.
                        self._excluded.add(key)
                        self._record_exclusion(key, epoch)   # persists to <dataset>/fizgig_excluded.json
                        logger.warning(f"[loss-watch] epoch {epoch}: {os.path.basename(key)} EXCLUDED "
                                       f"from training — two AI captions couldn't fix it. Edit its "
                                       f"caption to re-admit it, or remove it from the dataset.")
                        verdict, mult = "excluded", 0.0
                elif key in self._confirmed_stuck:
                    # Escalate with tenure: x0.5 -> x0.25 (the floor). Staying confirmed is
                    # accumulating evidence; the leak shrinks as certainty grows.
                    tenure = self._stuck_epochs.get(key, 1)
                    mult = max(self.stuck_floor,
                               self.throttle_mult * (0.5 ** ((tenure - 1) // self.escalate_every)))
                    verdict = "stuck"
                    s["stuck_epochs"] = tenure
                elif suspect:
                    verdict, mult = "suspect", self.suspect_mult   # provisional early throttle
                elif exhausted:
                    verdict, mult = "exhausted", self.exhausted_mult
                elif votes_stuck:
                    verdict, mult = "watch", 1.0          # suspicious this epoch, not yet confirmed
                elif s["mean_residual"] >= hi:
                    verdict, mult = "learning", 1.0
                elif s["mean_residual"] <= lo and n_ep >= self.easy_from_epoch:
                    verdict, mult = "easy", self.easy_mult
                else:
                    verdict, mult = "mid", 1.0
                # Active warm-up ramp overrides the display + multiplier (exclusion still wins).
                wm = self._warmup_mult(key)
                if wm is not None and verdict != "excluded":
                    verdict, mult = "warmup", round(wm, 3)
                s["verdict"] = verdict
                s["earlier_exclusion"] = key in self._known_hard and verdict != "excluded"
                s["multiplier"] = mult if (self.apply_lr or verdict == "warmup") else 1.0
                new_mult[key] = mult
            self._mult = new_mult
            self.verdicts = {k: s["verdict"] for k, s in stats.items()}

            # An image excluded without records of its own (e.g. replayed resume history) still gets a stub report
            # row, or it would be invisible in the popup.
            for k in self._excluded:
                if k not in stats:
                    stats[k] = {"verdict": "excluded", "multiplier": 0.0, "mean_residual": 0.0,
                                "slope": 0.0, "first": 0.0, "last": 0.0, "se": 0.0,
                                "trend_epochs": 0, "baseline": 0.0, "total_drop": 0.0,
                                "mean_loss": 0.0, "epochs": 0, "improving": False}
                    self.verdicts[k] = "excluded"

            # Plateau detection + best-epoch estimate (see __init__ docs). The streak resets if
            # anything starts improving again (e.g. a mid-run caption fix revives an image), and
            # the report flag resets with it so a later re-plateau announces its updated estimate.
            # Tolerance: up to ~5% of images may read as "improving" from autocorrelated residual
            # noise at any boundary — demanding absolute zero never declares. Real revivals (e.g.
            # a caption fix) blow well past the tolerance.
            n_tracked = sum(1 for k in stats if k not in self._excluded)
            tol = max(1, round(0.05 * n_tracked))
            if n_ep >= self.plateau_min_epochs and improving_count <= tol:
                self._no_improve_streak += 1
            else:
                self._no_improve_streak = 0
            self.plateaued = self._no_improve_streak >= self.plateau_patience
            # Pending adjudication: images whose fate is still open — throttled awaiting release/
            # exclusion, or freshly reset (a recaption wiped their history, so their trend is
            # INVISIBLE to improving_count for ~4 epochs). A plateau declared while these exist is
            # PROVISIONAL: resolving them (exclusion cleans the gradient; a caption fix revives
            # learning) routinely gives the run a second wind, so the estimate can move later.
            # Real-run validated 2026-07-05: plateau fired at epoch 13 with 8 images mid-ladder.
            self.plateau_pending = sum(
                1 for k, s in stats.items()
                if k not in self._excluded
                and (s.get("verdict") in ("stuck", "suspect", "watch")
                     or s.get("trend_epochs", 0) < 4))
            finishes = sorted(e for k, e in self._last_improving_epoch.items()
                              if k not in self._excluded)
            if finishes:
                self.best_epoch_estimate = int(finishes[int(0.75 * (len(finishes) - 1))])
            # A provisional plateau upgrades to a confirmed one the moment the pending set
            # resolves — re-announce then, even if the plateau flag never broke in between.
            if self._plateau_reported and self._plateau_was_provisional and not self.plateau_pending:
                self._plateau_reported = False
            if self.plateaued and not self._plateau_reported and self.best_epoch_estimate \
                    and not self._replaying:
                be = self.best_epoch_estimate
                self._plateau_was_provisional = bool(self.plateau_pending)
                if self.plateau_pending:
                    logger.info(f"[loss-watch] epoch {epoch}: training looks plateaued for the "
                                f"settled images (best so far ≈ epoch {be}), but "
                                f"{self.plateau_pending} image(s) are still being adjudicated "
                                f"(throttled or freshly recaptioned). If they resolve, training "
                                f"may get a second wind and a LATER epoch may become the better "
                                f"checkpoint — trust the plateau once it re-declares with none "
                                f"pending.")
                else:
                    logger.info(f"[loss-watch] epoch {epoch}: training has PLATEAUED — no image "
                                f"has improved for {self.plateau_patience} epochs and nothing is "
                                f"pending adjudication. Estimated best checkpoint ≈ epoch {be} "
                                f"(75th percentile of per-image finish epochs). Scrub epochs "
                                f"{max(1, be - 2)}–{be + 2} in LoRA Royale to pick by eye — "
                                f"later epochs mainly add overbake risk.")
                self._plateau_reported = True
            elif not self.plateaued:
                self._plateau_reported = False

            # Warn only when the CONFIRMED set changes — not every epoch. During a resume replay
            # the diff is left unsynced, so the FINAL replayed boundary announces the accumulated
            # state once instead of narrating every historical flip.
            if self._confirmed_stuck != self._last_reported_stuck and not self._replaying:
                added = self._confirmed_stuck - self._last_reported_stuck
                removed = self._last_reported_stuck - self._confirmed_stuck
                action = f" — throttling LR x{self.throttle_mult}" if (self.apply_lr and not self._batched) else ""
                if added:
                    names = ", ".join(os.path.basename(k) for k in sorted(added))
                    logger.warning(f"[loss-watch] epoch {epoch}: image(s) confirmed STUCK "
                                   f"(persistently hard, not improving — check for bad/mislabeled "
                                   f"data): {names}{action}")
                if removed:
                    names = ", ".join(os.path.basename(k) for k in sorted(removed))
                    logger.info(f"[loss-watch] epoch {epoch}: no longer stuck: {names}")
                self._last_reported_stuck = set(self._confirmed_stuck)

            if self._replaying:
                return self.verdicts   # one report write at the end of the replay, not per epoch
            try:
                d = os.path.join(self.output_dir, "loss_log")
                os.makedirs(d, exist_ok=True)
                report = os.path.join(d, "problem_images.json")
                # Atomic write — the GUI polls this file and must never read a half-written dump.
                with open(report + ".tmp", "w", encoding="utf-8") as f:
                    # Composite batch keys ("a|b|c") are excluded from the report: a
                    # batch-mean isn't a per-image verdict, and the window filled with
                    # rows naming three images whose thumbnails can't load.
                    json.dump({"epoch": epoch, "apply_lr": self.apply_lr,
                               "batched": self._batched,
                               "improving_count": improving_count,
                               "plateaued": self.plateaued,
                               "pending_count": self.plateau_pending,
                               "best_epoch_estimate": self.best_epoch_estimate,
                               "images": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                              for kk, vv in s.items()}
                                          for k, s in stats.items() if "|" not in k}},
                              f, indent=2)
                _atomic_replace(report + ".tmp", report)
            except Exception:
                pass
            return self.verdicts
        except Exception as e:
            logger.warning(f"[loss-watch] epoch_boundary failed ({e})")
            return {}

    def resume_from_jsonl(self, up_to_epoch: int = None, resets: dict = None) -> int:
        """Rebuild the watcher's in-memory history after --resume by replaying this run's own
        per_image_loss.jsonl (resumed runs append to it, so it holds the full pre-pause record).

        Without this a resumed run restarts the watch blind: trends, verdicts, healthy-epoch
        credit and stuck tenure all re-warm from zero — only persistent exclusions survive —
        and the Problem Images window loses everything except the excluded rows.

        Replays observe() + epoch_boundary() exactly as the original run drove them (same
        records, same order, same votes — the state machines are deterministic). `resets` is
        {key: [(epoch, attempt, is_auto), ...]} (a bare tuple is accepted for one fix) from
        caption_updates_applied.json: the original run reset those images' histories at those
        boundaries (recaption / manual edit), so the replay must too, or an image would be
        judged on records its caption fix invalidated. Incorrigibility (attempt 2 spent) is
        applied AT its boundary, exactly as the live run did — applying it after the replay
        made the replay take different branches than the original run and invent verdicts.
        Console chatter and the report write are muted for all but the FINAL replayed epoch,
        which announces the restored state once. Returns the number of replayed epochs."""
        path = os.path.join(self.output_dir, "loss_log", "per_image_loss.jsonl")
        resets = resets or {}
        # Normalize: one fix may arrive as a bare tuple; sort each key's fixes by epoch.
        resets = {str(k): sorted([tuple(e) for e in (v if isinstance(v, list) else [v])],
                                 key=lambda e: e[0])
                  for k, v in resets.items()}

        def _apply_spent_attempts():
            # Benefit-of-the-doubt state that must survive even when there is nothing to
            # replay: an image whose 2nd (detailed) AI recaption was already spent goes back
            # on the exclusion track rather than getting a free third life.
            for k, entries in resets.items():
                for (_r_ep, att, is_auto) in entries:
                    if is_auto and att >= 2:
                        self.mark_incorrigible(k)

        if not os.path.exists(path):
            if resets or self._excluded:
                logger.warning("[loss-watch] resume: no per_image_loss.jsonl to replay — the "
                               "watch restarts blind (trends, verdicts, healthy credit and "
                               "stuck tenure re-warm from zero; persistent exclusions survive).")
            _apply_spent_attempts()
            return 0
        by_epoch: dict[int, list] = {}
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                        ep = int(r["epoch"])
                        if up_to_epoch is not None and ep > up_to_epoch:
                            continue   # beyond the restored state snapshot — don't replay ahead
                        by_epoch.setdefault(ep, []).append(
                            (int(r.get("step", 0)), str(r["key"]), float(r["t"]), float(r["loss"])))
                    except Exception:
                        continue   # one mangled line must not kill the resume
        except Exception as e:
            logger.warning(f"[loss-watch] resume replay: could not read {path} ({e})")
            _apply_spent_attempts()
            return 0
        if not by_epoch:
            _apply_spent_attempts()
            return 0
        epochs = sorted(by_epoch)
        n_steps = sum(len(v) for v in by_epoch.values())
        logger.info(f"[loss-watch] resume: replaying {n_steps} logged steps "
                    f"(epochs {epochs[0]}–{epochs[-1]}) to restore verdict history…")
        # reset_key also pardons exclusions (that's right for LIVE caption edits) — but these
        # are HISTORICAL resets, and some of those images were excluded later in the original
        # timeline. Their exclusions are the more recent fact; never let a replayed reset undo
        # one (or rewrite fizgig_excluded.json). Their record PURGE still happened in the
        # original run though — _purge_records_only reproduces it without the pardon, so the
        # old caption's records can't skew every other image's thresholds.
        preserved = set(self._excluded)
        self._replaying = True
        try:
            for ep in epochs:
                for _, key, t, loss in sorted(by_epoch[ep]):
                    self.observe(epoch=ep, step=0, item_keys=key, timestep=t, loss=loss)
                if ep == epochs[-1]:
                    self._replaying = False   # final BOUNDARY logs + writes the report once —
                    #                            flipped after the observes, which must stay muted
                    #                            or the last epoch's steps duplicate into the jsonl
                self.epoch_boundary(ep)
                # Reproduce the caption fixes the original run applied at this boundary — the
                # records before a fix describe the OLD caption and must not convict the new
                # one. Incorrigibility (2nd AI attempt spent) is applied HERE, at the same
                # boundary the live run applied it (right after the reset), so later replayed
                # boundaries take the same branches the original run took.
                for k, entries in resets.items():
                    for (r_ep, att, is_auto) in entries:
                        if r_ep != ep:
                            continue
                        if k in preserved:
                            self._purge_records_only(k)
                        else:
                            self.reset_key(k)
                        if is_auto and att >= 2:
                            self.mark_incorrigible(k)
        finally:
            self._replaying = False
        logger.info(f"[loss-watch] resume: verdict history restored through epoch {epochs[-1]}.")
        return len(epochs)

    def mark_incorrigible(self, key: str) -> None:
        """Spend a key's benefit of the doubt: if it re-confirms stuck, it goes straight to the
        stuck_floor (no escalation ladder). Called after the 2nd failed AI recaption; a manual
        caption edit (reset_key) clears it."""
        self._incorrigible.add(str(key))

    def _purge_records_only(self, key: str) -> None:
        """Replay-only sibling of reset_key for keys whose exclusion must be preserved.

        A historical reset must never pardon a LATER exclusion — but its record purge
        still happened in the original run, and letting the old caption's records survive
        skews the residual thresholds every OTHER image is judged against. So: purge the
        records + per-key stats, leave _incorrigible/_excluded/_retired/_excl_data alone."""
        key = str(key)
        self._records = [r for r in self._records if r[0] != key]
        for d in (self._stuck_votes, self._clear_votes, self._suspect_votes,
                  self._exhaust_votes, self._stuck_epochs, self._mult):
            d.pop(key, None)
        self._confirmed_stuck.discard(key)
        self._last_reported_stuck.discard(key)
        self._last_improving_epoch.pop(key, None)
        self._improving_streak.pop(key, None)
        self.verdicts.pop(key, None)

    def reset_key(self, key: str) -> None:
        """Forget one image's history — used after a live caption fix, since its stuck record
        reflects the OLD caption. It re-enters fresh (needs 4 epochs of new trend before it can
        vote stuck again) and its multiplier returns to 1.0 immediately."""
        key = str(key)
        self._records = [r for r in self._records if r[0] != key]
        for d in (self._stuck_votes, self._clear_votes, self._suspect_votes,
                  self._exhaust_votes, self._stuck_epochs, self._mult):
            d.pop(key, None)
        self._confirmed_stuck.discard(key)
        self._last_reported_stuck.discard(key)
        self._incorrigible.discard(key)
        self._known_hard.discard(key)  # ...and clears an earlier run's exclusion record
        self._excluded.discard(key)   # a manual caption edit re-admits an excluded image
        self._retired.discard(key)    # ...and un-retires a retired one
        if key in self._excl_data:    # ...including from the persistent dataset-folder record
            del self._excl_data[key]
            self._write_persistent_exclusions()
        self._last_improving_epoch.pop(key, None)
        self._improving_streak.pop(key, None)
        # NOTE: _healthy_epochs deliberately survives — it's evidence about the IMAGE's
        # learnability, independent of which caption it carried.
        self.verdicts.pop(key, None)

    def close(self) -> None:
        self._jsonl.close()
