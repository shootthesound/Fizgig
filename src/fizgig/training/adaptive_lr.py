"""Epoch-level adaptive learning rate: the bi-directional plateau tracker (probe up on steady descent, halve on a
plateau or a weight-norm jump, with rollback). Used by the standard-layer trainer (families/train.py)."""
import logging

import torch

logger = logging.getLogger(__name__)


class AdaptiveLR:
    """Bi-directional plateau LR tracker — a faithful port of Klein's adaptive_lr logic.

    Each epoch boundary: probe UP ×1.25 on steady loss descent (patience 2); reduce DOWN ×0.5
    on loss plateau (patience ramp) or a stability signal. On a stability event it blends the
    LoRA weights 70/30 toward the previous epoch's snapshot and restores the optimizer state
    (kills bad Adam momentum). Klein's stability signals are grad-clip ratio + weight-norm
    growth; krea2 has no grad clipping, so weight-norm growth (>30%) is the stability signal.

    State (streaks/best_loss/prev_weight_norm) is JSON round-trippable for pause/resume; the
    CPU rollback snapshot is in-memory only (too big to persist) — so the first post-resume
    epoch can't roll back, exactly as in Klein. Call epoch_boundary() at each epoch end.

    Learned ceiling: a cut (plateau or stability) within CEILING_EPOCHS of a probe up, from the rate that probe set,
    marks that rate too high for this run; later probes stop one step (x1.25) below it. Without it, a run that
    plateaued after 2.83e-4 -> 3.54e-4 and was cut to 2e-4 would climb back past 3.54e-4 (Peter, 6 Nov - Qwen Image 2.1
    on Muon). It persists with the rest of the state, so pause / resume / Move keep it."""

    BLEND = 0.7
    WEIGHT_GROWTH_THRESHOLD = 0.30

    CLIP_RATIO_THRESHOLD = 0.5
    CEILING_EPOCHS = 3  # a cut this many epochs or fewer after a probe up blames that probe's rate
    SMOOTH_EPOCHS = 3   # plateau / improving is judged on the mean loss of the last 3 epochs: an epoch's loss jitters
                        # by about ±0.005, which on its own read as a plateau every other epoch and walked the LR to Min

    def __init__(self, min_lr, max_lr, clip_signal=False):
        """clip_signal: Klein's rules - a grad-clip ratio over 50% of the epoch's steps is a stability signal
        (checked before the weight-norm growth; record_clip() counts the steps), and, as the old Klein trainer, the
        weight-norm baseline starts at the first comparison and the first epoch's clip counts carry into it."""
        self.min_lr = float(min_lr)
        self.max_lr = float(max_lr)
        self.clip_signal = bool(clip_signal)
        self.clip_steps = 0
        self.clip_events = 0
        self.best_loss = None
        self.good_streak = 0
        self.bad_streak = 0
        self.stability_streak = 0
        self.stability_triggered = False
        self.prev_weight_norm = None
        self.recent_losses = []
        self.ceiling = None           # the learned top: a probe stops here (None: max_lr only)
        self.probe_epoch = None       # the last probe up: its epoch and the rate it set
        self.probe_lr = None
        self.snapshot = None  # {"weights": {...cpu...}, "optim": cpu state} — not persisted

    def record_clip(self, pre_clip_norm, max_norm):
        """One optimizer step's clip, for clip_signal (the norm clip_grad_norm_ returned, before clipping)."""
        self.clip_steps += 1
        try:
            if float(pre_clip_norm) > float(max_norm):
                self.clip_events += 1
        except (TypeError, ValueError):
            pass

    def state_dict(self):
        return {"best_loss": self.best_loss, "good_streak": self.good_streak,
                "bad_streak": self.bad_streak, "stability_streak": self.stability_streak,
                "stability_triggered": self.stability_triggered,
                "prev_weight_norm": self.prev_weight_norm, "recent_losses": list(self.recent_losses),
                "ceiling": self.ceiling, "probe_epoch": self.probe_epoch, "probe_lr": self.probe_lr}

    def load_state_dict(self, d):
        if not d:
            return
        self.best_loss = d.get("best_loss")
        self.good_streak = int(d.get("good_streak", 0))
        self.bad_streak = int(d.get("bad_streak", 0))
        self.stability_streak = int(d.get("stability_streak", 0))
        self.stability_triggered = bool(d.get("stability_triggered", False))
        self.prev_weight_norm = d.get("prev_weight_norm")
        self.recent_losses = [float(x) for x in (d.get("recent_losses") or [])][-self.SMOOTH_EPOCHS:]
        self.ceiling = d.get("ceiling")
        self.probe_epoch = d.get("probe_epoch")
        self.probe_lr = d.get("probe_lr")

    def _top(self):
        """The highest rate a probe may set: max_lr, or the learned ceiling when it is lower."""
        return self.max_lr if self.ceiling is None else min(self.max_lr, self.ceiling)

    def _learn_ceiling(self, epoch, cur_lr):
        """A cut is being made at `epoch` from `cur_lr`: when it follows a probe up closely and comes from the rate
        that probe set, that rate was too high - cap later probes one step below it. Returns a log note or ''."""
        if (self.probe_epoch is None or self.probe_lr is None or epoch - self.probe_epoch > self.CEILING_EPOCHS
                or cur_lr < self.probe_lr * 0.999):
            return ""
        cap = max(self.min_lr, self.probe_lr / 1.25)
        self.probe_epoch = self.probe_lr = None
        if self.ceiling is not None and self.ceiling <= cap:
            return ""
        self.ceiling = cap
        return f"; ceiling {cap:.2e} learned (the probe to {cap * 1.25:.2e} led to this cut)"

    @staticmethod
    def _weight_norm(network):
        wn = 0.0
        with torch.no_grad():
            for p in network.parameters():
                if p.requires_grad:
                    wn += float(p.detach().float().norm().item()) ** 2
        return wn ** 0.5

    def _snapshot(self, network, optimizer):
        with torch.no_grad():
            weights = {n: p.detach().clone().to("cpu")
                       for n, p in network.named_parameters() if p.requires_grad}

        def _cpu(o):
            if isinstance(o, torch.Tensor):
                return o.detach().clone().to("cpu")
            if isinstance(o, dict):
                return {k: _cpu(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_cpu(v) for v in o]
            return o
        try:
            self.snapshot = {"weights": weights, "optim": _cpu(optimizer.state_dict())}
        except Exception:
            self.snapshot = {"weights": weights, "optim": None}

    def _rollback(self, network, optimizer):
        cur = dict(network.named_parameters())
        with torch.no_grad():
            for name, prev in self.snapshot["weights"].items():
                if name in cur and cur[name].requires_grad:
                    p = cur[name]
                    prev_d = prev.to(device=p.device, dtype=p.dtype)
                    p.copy_(self.BLEND * prev_d + (1.0 - self.BLEND) * p)
        if self.snapshot.get("optim") is not None:
            try:
                optimizer.load_state_dict(self.snapshot["optim"])
            except Exception:
                pass

    def epoch_boundary(self, epoch, current_loss, network, optimizer):
        """epoch is 0-indexed (global). epoch 0 arms the baseline; epoch >= 1 adjusts the LR."""
        raw_loss = current_loss
        self.recent_losses = (self.recent_losses + [float(current_loss)])[-self.SMOOTH_EPOCHS:]
        current_loss = sum(self.recent_losses) / len(self.recent_losses)      # smoothed
        if epoch == 0:
            self.best_loss = current_loss
            if not self.clip_signal:        # Klein's baseline starts at the first comparison
                self.prev_weight_norm = self._weight_norm(network)
            logger.info(f"[adaptive_lr] epoch 1: loss={raw_loss:.4f} "
                        f"lr={optimizer.param_groups[0]['lr']:.2e} | ARMED")
            self._snapshot(network, optimizer)
            return

        patience_up = 2
        patience_down = 2 if (self.stability_triggered or epoch == 1 or epoch >= 4) else 1
        cur_lr = optimizer.param_groups[0]["lr"]
        new_lr = cur_lr
        cur_wn = self._weight_norm(network)
        weight_growth = None
        if self.prev_weight_norm and self.prev_weight_norm > 0:
            weight_growth = (cur_wn - self.prev_weight_norm) / self.prev_weight_norm
        stability_reason = None
        clip_ratio = self.clip_events / max(self.clip_steps, 1)
        if self.clip_signal and clip_ratio > self.CLIP_RATIO_THRESHOLD:
            stability_reason = f"grad clip {clip_ratio*100:.0f}% > {self.CLIP_RATIO_THRESHOLD*100:.0f}%"
        elif weight_growth is not None and weight_growth > self.WEIGHT_GROWTH_THRESHOLD:
            stability_reason = f"wnorm_Δ {weight_growth*100:+.0f}% > {self.WEIGHT_GROWTH_THRESHOLD*100:.0f}%"

        action, reason = "HOLD", ""
        if stability_reason is not None:
            self.stability_streak += 1
            stability_patience = 1 if not self.stability_triggered else 2
            if self.stability_streak >= stability_patience:
                candidate = max(cur_lr * 0.5, self.min_lr)
                note = ""
                if self.snapshot is not None:
                    self._rollback(network, optimizer)
                    note = f"; blended {int(self.BLEND*100)}/{int((1-self.BLEND)*100)} + optim restored"
                if candidate < cur_lr:
                    note += self._learn_ceiling(epoch, cur_lr)
                    new_lr = candidate
                    action = "REDUCE+ROLLBACK" if self.snapshot is not None else "REDUCE"
                else:
                    action = "HOLD (floored)"
                reason = f"stability: {stability_reason}{note}"
                self.good_streak = self.bad_streak = self.stability_streak = 0
                self.stability_triggered = True
            else:
                action = "WAIT"
                reason = f"stability: {stability_reason}, streak {self.stability_streak}/{stability_patience}"
        elif self.best_loss is None or current_loss < self.best_loss:
            self.stability_streak = 0
            self.best_loss = current_loss
            self.good_streak += 1
            self.bad_streak = 0
            if self.good_streak >= patience_up:
                candidate = min(cur_lr * 1.25, self._top())
                if candidate > cur_lr * 1.001:
                    new_lr = candidate
                    action = "PROBE UP"
                    reason = f"loss improving, streak {self.good_streak}"
                    if self.ceiling is not None and candidate >= self.ceiling * 0.999:
                        reason += f", to the learned ceiling {self.ceiling:.2e}"
                    self.probe_epoch, self.probe_lr = epoch, candidate
                else:
                    action = "HOLD (capped)"
                    reason = ("loss improving, at max_lr" if self._top() >= self.max_lr
                              else f"loss improving, at the learned ceiling {self.ceiling:.2e}")
                self.good_streak = 0
            else:
                reason = f"loss improving, streak {self.good_streak}/{patience_up}"
        else:
            self.stability_streak = 0
            self.bad_streak += 1
            self.good_streak = 0
            if self.bad_streak >= patience_down:
                candidate = max(cur_lr * 0.5, self.min_lr)
                if candidate < cur_lr:
                    reason = f"loss plateau, streak {self.bad_streak}" + self._learn_ceiling(epoch, cur_lr)
                    new_lr = candidate
                    action = "REDUCE"
                else:
                    action = "HOLD (floored)"
                    reason = "loss plateau, at min_lr"
                self.bad_streak = 0
            else:
                reason = f"loss plateau, streak {self.bad_streak}/{patience_down}"

        if new_lr != cur_lr:
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr
        lr_str = f"{cur_lr:.2e}" if new_lr == cur_lr else f"{cur_lr:.2e}->{new_lr:.2e}"
        wn_str = f"{weight_growth*100:+.0f}%" if weight_growth is not None else "—"
        clip_str = f"clip={clip_ratio*100:.0f}% " if self.clip_signal else ""
        logger.info(f"[adaptive_lr] epoch {epoch + 1}: loss={raw_loss:.4f} (avg {current_loss:.4f}) lr={lr_str} "
                    f"{clip_str}wnorm_Δ={wn_str} | {action} ({reason})")
        self.prev_weight_norm = cur_wn
        self.clip_steps = self.clip_events = 0
        self._snapshot(network, optimizer)
