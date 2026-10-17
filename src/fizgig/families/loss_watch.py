"""The per-image loss watch for described families: detect problem images, per-image LR, look-outlier warm-up and
auto-recaption, on the shared PerImageLossWatch (training/loss_logger.py).

This mirrors Krea 2's trainer integration (krea2/trainer.py) step for step - fresh-run cleanup, exclusion preflight,
look-outlier warm-up, resume replay from the run's own JSONL and the applied-captions ledger, and the between-epoch
caption repair. Stuck images are recaptioned by the shared captioner - Krea 2's Qwen3-VL-4B, the same model and
instructions as the Captions tab, for every family - and captions are re-encoded with the family's own driver.encode_text
into the family cache (cache.save_cond). The GUI's Problem Images window reads the same loss_log/ files as for Krea 2.
"""
import gc
import json
import logging
import os
import time

import torch

logger = logging.getLogger(__name__)

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _find_key(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(d, list):
        for v in d:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def _items(group):
    """item_key -> ItemInfo over every training bucket."""
    out = {}
    for ds in group.datasets:
        bm = getattr(ds, "batch_manager", None)
        if bm is not None:
            for bucket in bm.buckets.values():
                for it in bucket:
                    out[str(it.item_key)] = it
    return out


class Watch:
    """The run's loss watch and the state around it. `watch` is None when every toggle is off."""

    def __init__(self, output_dir, group, user_config, driver, *, log=False, per_image_lr=False, auto_recaption=False,
                 warmup_look=False, resume=False, start_epoch=0, te_path=None, trigger_word=None,
                 trigger_position="start", recaption_instruction=None, recaption_instruction_detailed=None,
                 captioner_path=None):
        from fizgig.training.loss_logger import PerImageLossWatch, is_enabled as env_on
        self.output_dir, self.group, self.driver, self.te_path = output_dir, group, driver, te_path
        self.trigger_word, self.trigger_position = trigger_word, trigger_position
        self.instr, self.instr_detailed = recaption_instruction, recaption_instruction_detailed
        self.recaptioned = {}
        self.captioner_path = captioner_path
        self.watch = None
        if not resume:
            _cleanup(output_dir)
        if auto_recaption and not (captioner_path and os.path.isfile(captioner_path)):
            logger.warning("[auto-recaption] needs the captioner (--captioner: the Krea 2 Qwen3-VL-4B text encoder "
                           "file) - off")
            auto_recaption = False
        if not (log or per_image_lr or auto_recaption or warmup_look or env_on()):
            self.auto_recaption = False
            return
        self.image_dir = _find_key(user_config, "image_directory")
        self.caption_ext = _find_key(user_config, "caption_extension") or ".txt"
        if not (self.image_dir and os.path.isdir(self.image_dir)):
            self.image_dir = None
        if auto_recaption and not self.image_dir:
            logger.warning("[auto-recaption] image_directory not found in the dataset config - auto-recaption off")
            auto_recaption = False
        self.auto_recaption = auto_recaption
        if auto_recaption:
            logger.info(f"[auto-recaption] ON - stuck images re-captioned from {self.image_dir}"
                        + (f" (trigger: '{trigger_word}')" if trigger_word else ""))
        self.watch = PerImageLossWatch(output_dir, apply_lr=per_image_lr, write_jsonl=True,
                                       dataset_dir=self.image_dir, caption_ext=self.caption_ext,
                                       family=self.driver.description.key)
        keys = set(_items(group))
        self.watch.preflight(keys)
        logger.info(f"[loss-watch] per-image loss watch ON (per_image_lr={per_image_lr})")
        if warmup_look:
            self._warmup(keys)
        if resume and start_epoch > 0:
            self._resume(start_epoch)

    # ---- setup helpers -------------------------------------------------------------------------------
    def _warmup(self, keys):
        path = os.path.join(self.image_dir or "", "fizgig_look_scores.json")
        try:
            with open(path, encoding="utf-8") as f:
                look = json.load(f)
        except FileNotFoundError:
            logger.warning("[look-warmup] fizgig_look_scores.json not found in the dataset folder - run the Look "
                           "Consistency Filter (Image Prep tab) first; warm-up off this run")
            return
        except Exception as e:
            logger.warning(f"[look-warmup] could not load look scores ({e}) - warm-up off this run")
            return
        cut = look.get("cutoff")
        # Scores carry the embedder that made them (ArcFace / DINOv2 / CLIP - different scales); the cutoff was
        # computed on that same scale when the scan ran, so it is used as is. Files from before the face backends
        # have no "embedder" and are ArcFace.
        emb = look.get("embedder_label") or look.get("embedder") or "ArcFace"
        if cut is None:
            logger.warning("[look-warmup] no cutoff in fizgig_look_scores.json (too few scored faces) - warm-up off")
            return
        outliers = {k for k, v in (look.get("scores") or {}).items() if isinstance(v, (int, float)) and v < float(cut)}
        gone, use = sorted(outliers - keys), outliers & keys
        if gone:
            logger.info(f"[look-warmup] {len(gone)} scored outlier(s) not in the dataset (moved via the Look "
                        f"Filter) - skipped")
        if use:
            self.watch.set_warmup_keys(use)
            logger.info(f"[look-warmup] {len(use)} look-outlier image(s) on LR warm-up x0.4 -> x1.0 "
                        f"({emb} scores, cutoff {float(cut):.2f})")

    def _resume(self, start_epoch):
        resets = {}
        try:
            with open(os.path.join(self.output_dir, "loss_log", "caption_updates_applied.json"),
                      encoding="utf-8") as f:
                for k, info in json.load(f).items():
                    for e in (info if isinstance(info, list) else [info]):
                        att, auto = int(e.get("attempt", 0) or 0), bool(e.get("auto"))
                        if auto:
                            self.recaptioned[k] = max(self.recaptioned.get(k, 0), att)
                        resets.setdefault(k, []).append((int(e.get("epoch", 0) or 0), att, auto))
        except Exception:
            pass
        self.watch.resume_from_jsonl(up_to_epoch=start_epoch, resets=resets)

    # ---- per step --------------------------------------------------------------------------------------
    def excluded(self, batch):
        return self.watch is not None and self.watch.is_excluded(batch.get("item_keys"))

    def multiplier(self, batch):
        return self.watch.multiplier(batch.get("item_keys")) if self.watch is not None else 1.0

    def observe(self, epoch, step, batch, t, loss):
        if self.watch is not None:
            self.watch.observe(epoch=epoch, step=step, item_keys=batch.get("item_keys"), timestep=t, loss=loss)

    # ---- epoch boundary --------------------------------------------------------------------------------
    def boundary(self, epoch, dit, device, parkable=True):
        """Reclassify, write problem_images.json, then apply queued caption edits and auto-recaptions."""
        if self.watch is not None:
            self.watch.epoch_boundary(epoch)
        try:
            self._captions(epoch, dit, device, parkable)
        except Exception:
            logger.warning("[caption-fix] caption repair failed this boundary - training continues", exc_info=True)

    def _captions(self, epoch, dit, device, parkable):
        path = os.path.join(self.output_dir, "loss_log", "caption_updates.json")
        processing = path + ".processing"
        updates = {}
        if os.path.exists(path):
            try:
                os.replace(path, processing)             # atomic claim: GUI edits made meanwhile land in a fresh file
                with open(processing, encoding="utf-8") as f:
                    updates = {str(k): str(v).strip() for k, v in json.load(f).items() if str(v).strip()}
            except Exception:
                logger.warning("[caption-fix] could not read caption_updates.json - skipping", exc_info=True)
                return
        auto = []
        if self.auto_recaption and self.watch is not None and self.image_dir:
            for k in sorted(k for k, v in self.watch.verdicts.items() if v == "stuck"):
                if k in updates or self.recaptioned.get(k, 0) >= 2:
                    continue
                for ext in _IMG_EXTS:
                    p = os.path.join(self.image_dir, os.path.basename(k) + ext)
                    if os.path.exists(p):
                        auto.append((k, p, self.recaptioned.get(k, 0) + 1))
                        break
        if not updates and not auto:
            _remove(processing)
            return
        if not self.te_path:
            logger.warning("[caption-fix] caption work is pending but no --text_encoder was given; left queued")
            _requeue(path, processing, updates)
            return
        items = _items(self.group)
        todo = [(k, items[k], c, 0) for k, c in updates.items() if k in items]
        for k in updates:
            if k not in items:
                logger.warning(f"[caption-fix] '{k}' not in the training set - skipped")
        auto = [a for a in auto if a[0] in items]
        if not todo and not auto:
            _remove(processing)
            return
        logger.info(f"[caption-fix] epoch boundary {epoch}: {len(todo)} manual edit(s), {len(auto)} stuck image(s) "
                    f"to auto-recaption...")
        from fizgig.families import cache, quant
        if parkable:
            quant.move(dit, "cpu")
            gc.collect()
            torch.cuda.empty_cache()
        ok = False
        try:
            if auto:
                todo += self._recaption(auto, items, device)
            if not todo:        # every stuck image failed to caption: nothing to re-encode, they retry next time
                _remove(processing)
                return
            te = self.driver.load_text_encoder(self.te_path, device)
            try:
                desc = self.driver.description
                for i in range(0, len(todo), 4):
                    chunk = todo[i:i + 4]
                    for (_, item, cap, _), cond in zip(chunk, self.driver.encode_text(te, [c for _, _, c, _ in chunk])):
                        item.caption = cap
                        cache.save_cond(desc, item, cond)     # the loader reads this file every step
                ok = True
            finally:
                self.driver.unload_text_encoder(te)
                del te
                gc.collect()
                torch.cuda.empty_cache()
        finally:
            if parkable:
                quant.move(dit, device)
        if not ok:     # already-written AI captions re-queue as if manual: no need to regenerate them
            _requeue(path, processing, {**updates, **{k: c for k, _, c, a in todo if a > 0}})
            return
        for k, _, _, attempt in todo:
            if attempt > 0:
                self.recaptioned[k] = max(self.recaptioned.get(k, 0), attempt)
        if self.watch is not None:
            for k, _, _, attempt in todo:
                self.watch.reset_key(k)
            for k, _, _, attempt in todo:
                if attempt >= 2:
                    self.watch.mark_incorrigible(k)
        _ack(self.output_dir, todo, epoch)
        _remove(processing)
        logger.info(f"[caption-fix] {len(todo)} caption(s) re-encoded - next epoch trains on the fixed text. "
                    f"Loss-watch history reset for: " + ", ".join(os.path.basename(k) for k, _, _, _ in todo))

    def _recaption(self, auto, items, device):
        """Caption the stuck images with the shared captioner (Krea 2's Qwen3-VL-4B, loaded for the call and freed
        before the family's encoder loads). Writes each .txt and returns (key, item, caption, attempt) rows to
        re-encode. A captioner that does not fit the card turns auto-recaption off for the run."""
        from fizgig.krea2.embedder import generate_caption
        from fizgig.krea2.utils import load_krea2_text_encoder
        try:
            enc = load_krea2_text_encoder(self.captioner_path, dtype=torch.bfloat16, device=device)
        except torch.OutOfMemoryError:
            logger.warning("[auto-recaption] off for the rest of this run: the captioner does not fit this card's "
                           "VRAM. Training continues; manual caption edits still apply.")
            self.auto_recaption = False
            gc.collect()
            torch.cuda.empty_cache()
            return []
        rows = []
        try:
            for k, img, attempt in auto:
                try:
                    instr = (self.instr_detailed if attempt >= 2 else self.instr) or None
                    cap = generate_caption(enc, img, detailed=attempt >= 2, instruction=instr)
                    if self.trigger_word:
                        cap = (f"{cap}, {self.trigger_word}" if str(self.trigger_position) == "end"
                               else f"{self.trigger_word}, {cap}")
                    try:
                        with open(os.path.join(self.image_dir, os.path.basename(k) + self.caption_ext), "w",
                                  encoding="utf-8") as f:
                            f.write(cap)
                    except OSError:
                        logger.warning(f"[auto-recaption] could not write the .txt for {k} - this run is fixed, "
                                       "a future re-cache will use the old caption")
                    rows.append((k, items[k], cap, attempt))
                    logger.info(f"[auto-recaption] {os.path.basename(k)} (attempt {attempt}/2"
                                f"{', detailed' if attempt >= 2 else ''}): \"{cap[:110]}\"")
                except Exception:
                    logger.warning(f"[auto-recaption] captioning failed for {os.path.basename(k)} - retry next "
                                   f"boundary", exc_info=True)
        finally:
            enc.to("cpu")
            del enc
            gc.collect()
            torch.cuda.empty_cache()
        return rows


def _cleanup(output_dir):
    """Fresh run: clear the previous run's Problem Images artifacts; rotate (not delete) the research JSONL."""
    ll = os.path.join(output_dir, "loss_log")
    for f in ("problem_images.json", "problem_images.json.tmp", "caption_updates_applied.json",
              "caption_updates_applied.json.tmp", "caption_updates.json", "caption_updates.json.processing"):
        _remove(os.path.join(ll, f))
    jl = os.path.join(ll, "per_image_loss.jsonl")
    if os.path.exists(jl):
        try:
            os.replace(jl, jl + "." + time.strftime("%Y%m%d%H%M%S") + ".bak")
        except OSError:
            pass


def _remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass                     # the GUI may hold it open; it is consumed next boundary


def _requeue(path, processing, updates):
    """Put a claimed batch back, merged with anything the GUI queued meanwhile (newer edits win)."""
    if not updates:
        _remove(processing)
        return
    try:
        newer = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                newer = json.load(f)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({**updates, **newer}, f, indent=2)
        os.replace(path + ".tmp", path)
        _remove(processing)
    except Exception:
        pass


def _ack(output_dir, todo, epoch):
    """Per-fix history per key (the GUI's row badge, and the resume replay's reset timeline)."""
    path = os.path.join(output_dir, "loss_log", "caption_updates_applied.json")
    applied = {}
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                applied = {k: (v if isinstance(v, list) else [v]) for k, v in json.load(f).items()}
    except Exception:
        applied = {}
    for k, _, cap, attempt in todo:
        applied.setdefault(k, []).append({"epoch": epoch, "caption": cap, "attempt": attempt, "auto": attempt > 0})
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            json.dump(applied, f, indent=2)
        os.replace(path + ".tmp", path)
    except Exception:
        logger.warning("[caption-fix] could not write caption_updates_applied.json", exc_info=True)
