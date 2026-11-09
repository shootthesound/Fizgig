"""Optimizer selection, shared by the trainers.

Krea 2 hardcoded `bnb.optim.AdamW8bit` — a good default, but the only choice, which is the one
place the community comparison against OneTrainer (~40 optimizers) landed a fair hit.

Two things matter for a LoRA and pull against the "more optimizers is better" instinct:

* **Optimizer state is tiny here.** Krea 2 trains 264 small factors, so AdamW's two moments cost
  ~tens of MB against a 13-19 GB base. Choosing an 8-bit or factored optimizer to save memory is
  nearly pointless — the base dominates. What the choice actually buys is *update behaviour*.
* **Learning rates are not comparable across families.** Lion's update is a sign, so it wants
  roughly a tenth of AdamW's LR. Handing someone a dropdown without saying that is how you
  produce a fried LoRA and a bug report, so `create_optimizer` warns loudly on the record when
  the LR looks wrong for the family. Prodigy estimates the rate itself (`d`) and takes `lr=1.0` as a
  multiplier; Muon with `adjust_lr_fn=match_rms_adamw` matches AdamW's step size, so AdamW's rates
  apply. (CAME stays out; see the note above _CATALOG.)

Free-form `module.path.ClassName` is accepted too, so a user who pip-installs something exotic
does not need a Fizgig release to use it.
"""

from __future__ import annotations

import ast
import importlib
import logging

import torch

logger = logging.getLogger(__name__)

# What the Optimizer Args box is filled with when Prodigy or Muon is picked (and the defaults
# create_optimizer applies under anything typed there).
PRODIGY_DEFAULT_ARGS = "decouple=True weight_decay=0.01 d_coef=1.0 use_bias_correction=True safeguard_warmup=True"
MUON_DEFAULT_ARGS = "adjust_lr_fn=match_rms_adamw weight_decay=0.01 momentum=0.95 nesterov=True ns_steps=5"
NEW_OPTIMIZERS = ("prodigy", "muon", "automagic4")    # added with #292; they never fall back to AdamW


# name -> (import to test, one-line description shown in the GUI/CLI help)
# Removed by decision (2026-07-28): prodigy, came, adafactor — the "manage their own LR"
# family. They fight Adaptive LR by design (prodigy wants lr=1.0 as a multiplier, adafactor
# relative_step stores lr=None, came is an Adafactor variant), two of the three needed
# external packages, and both prodigy and adafactor caused real bugs (silent AdamW fallback
# at lr=1.0; TypeError against the adaptive watcher). Exotic optimizers remain available
# via the full module.path.ClassName form.
# Prodigy came back with #292 (@Murf-Code): vendored (fizgig/training/prodigy.py, no package),
# Adaptive LR and the scheduler stand down for it (owns_its_rate), and it never falls back to
# AdamW. CAME stays out; Adafactor came back at a fixed rate (#312, see its entry).
_CATALOG = {
    "adamw8bit":          ("bitsandbytes", "AdamW, 8-bit state (default — the validated recipe)"),
    "adamw":              (None,           "AdamW, fp32 state, CUDA-fused where available"),
    # Back 9 Nov 2026 (#312) at a FIXED rate (relative_step off, the rate is the Learning Rate box), so Adaptive LR
    # and the scheduler drive it like any other: the old bugs came from relative_step's lr=None. Every full
    # fine-tune trains on it by default.
    "adafactor":          ("transformers",  "Adafactor - factored state, far smaller than AdamW's; the fine-tune default"),
    "pagedadamw8bit":     ("bitsandbytes", "AdamW8bit that pages state to CPU under pressure"),
    "ademamix8bit":       ("bitsandbytes", "AdEMAMix — second slow EMA, aimed at long runs"),
    "pagedademamix8bit":  ("bitsandbytes", "AdEMAMix8bit with CPU paging"),
    "lion8bit":           ("bitsandbytes", "Lion — sign updates; use ~1/10 the AdamW LR"),
    # Ostris's Automagic v3, vendored (fizgig/training/automagic3.py, MIT). One learning rate per
    # parameter group, nudged up while update signs hold steady and down while they alternate;
    # the LR given is only its START (1e-6 is its own default). MiniMax H3 LoRA experiment,
    # 17 Sep 2026 — unmeasured against the AdamW recipe until an A/B says otherwise.
    "automagic3":         (None,           "Automagic v3 (Ostris) — sets its own learning rate from the update signs; "
                                            "the LR box is its start, 1e-6 recommended. Experiment"),
    "prodigy":            (None,           "Prodigy — estimates the learning rate itself; the Learning Rate box is "
                                            "locked to 1.0 (a multiplier)"),
    "muon":               (None,           "Muon — orthogonalized momentum, step size matched to AdamW's, so AdamW's "
                                            "learning rates apply"),
    # Automagic v4 (fizgig/training/automagic4.py, MIT): v3 plus a per-weight brake on the most
    # sensitive weights and a rate controller that settles instead of annealing to zero.
    "automagic4":         (None,           "Automagic v4 (Ostris) — like v3, sets its own learning rate, and evens "
                                            "out how hard each weight is pushed; the LR box is its start, 1e-6 "
                                            "recommended"),
}

DEFAULT_OPTIMIZER = "adamw8bit"

# What a full fine-tune can train with (families/ft.py make_optimizer, one optimizer per weight at the fine-tune's
# fixed rate), Adafactor first - the default. Muon, Prodigy and Automagic are LoRA optimizers: a fine-tune given one
# trains with Adafactor and says so. The number is the optimizer state per GB of the window's bf16 weights, which
# the fine-tune planner adds to each window (Adafactor's factored state is in every family's measured overhead).
FT_OPTIMIZERS = {"adafactor": 0.0, "adamw8bit": 1.0, "pagedadamw8bit": 1.0, "ademamix8bit": 1.5,
                 "pagedademamix8bit": 1.5, "lion8bit": 0.5, "adamw": 2.0}
FT_DEFAULT_OPTIMIZER = "adafactor"


def ft_optimizer(name) -> str:
    """The optimizer a fine-tune runs for this choice: itself when a fine-tune can use it, else Adafactor."""
    n = str(name or "").strip().lower()
    return n if n in FT_OPTIMIZERS else FT_DEFAULT_OPTIMIZER


def available_optimizers() -> list[str]:
    """Catalog entries whose backing package is actually importable on this machine."""
    out = []
    for name, (module, _desc) in _CATALOG.items():
        if module is None:
            out.append(name)
            continue
        try:
            importlib.import_module(module)
            out.append(name)
        except Exception:
            pass
    return out


def optimizer_lr(optimizer) -> float:
    """The rate the optimizer is actually applying: Automagic v3 keeps it in its state (the
    group's "lr" is only the start); Prodigy applies d x lr; everyone else keeps it on the group."""
    if optimizer is None:
        return 0.0
    if optimizer.__class__.__name__ == "Prodigy":
        g = optimizer.param_groups[0]
        return float(g["lr"]) * float(g.get("d", 1.0))
    fn = getattr(optimizer, "get_avg_learning_rate", None)
    if callable(fn):
        try:
            return float(fn())
        except Exception:
            pass
    return float(optimizer.param_groups[0]["lr"])


# Krea 2's LoRA covers 264 Linears whose families converge at very different speeds. An
# optimizer that keeps one rate per param GROUP (Automagic v3) finds a compromise nobody wants
# when handed all of them as one group — and txtfusion is measured to be under-trained by our
# recipe, so it is exactly the minority that gets outvoted. These patterns split the network
# into families, each of which then finds its own rate.
KREA2_LORA_FAMILIES = (
    ("txtfusion", ("txtfusion",)),                  # layerwise + refiner blocks + the projector
    ("attn",      ("_attn_",)),                     # wq/wk/wv/wo/gate on the 28 stream blocks
    ("mlp",       ("_mlp_",)),                      # up/gate/down on the same blocks
)
KREA2_LORA_FAMILY_OTHER = "io"                      # first/last, tmlp, tproj, txtmlp — 7 singletons


def family_of(lora_name: str) -> str:
    """Which Krea 2 family a LoRA module belongs to. txtfusion wins over attn/mlp because its
    own blocks carry attn_/mlp_ names too."""
    n = str(lora_name or "")
    for fam, needles in KREA2_LORA_FAMILIES:
        if any(x in n for x in needles):
            return fam
    return KREA2_LORA_FAMILY_OTHER


def family_param_groups(network, lr: float):
    """-> ([{"params": [...], "lr": lr, "family": name}, ...], {name: n_modules}) or (None, {})
    when the network exposes no named modules (then the caller keeps its flat list)."""
    loras = list(getattr(network, "unet_loras", None) or [])
    if not loras:
        return None, {}
    buckets, counts = {}, {}
    for mod in loras:
        fam = family_of(getattr(mod, "lora_name", ""))
        ps = [p for p in mod.parameters() if p.requires_grad]
        if not ps:
            continue
        buckets.setdefault(fam, []).extend(ps)
        counts[fam] = counts.get(fam, 0) + 1
    if len(buckets) < 2:
        return None, {}
    order = [f for f, _ in KREA2_LORA_FAMILIES] + [KREA2_LORA_FAMILY_OTHER]
    groups = [{"params": buckets[f], "lr": float(lr), "family": f} for f in order if f in buckets]
    return groups, counts


def group_rates(optimizer) -> str:
    """"attn 1.8e-04  mlp 2.4e-04  …" — the per-group rates for a log line, or "" when the
    optimizer keeps one rate."""
    fn = getattr(optimizer, "get_learning_rates", None)
    if not callable(fn) or optimizer is None or len(optimizer.param_groups) < 2:
        return ""
    try:
        rates = fn()
    except Exception:
        return ""
    return "  ".join(f"{g.get('family', i)} {r:.2e}"
                     for i, (g, r) in enumerate(zip(optimizer.param_groups, rates)))


def owns_its_rate(optimizer) -> bool:
    """True for an optimizer that sets its own learning rate (Automagic v3 / v4, Prodigy): schedulers
    and adaptive watchers that write the group rate have no effect on it and should stand down."""
    return optimizer is not None and optimizer.__class__.__name__ in ("Automagic3", "Automagic4", "Prodigy")


def describe(name: str) -> str:
    return _CATALOG.get(name.lower(), (None, "custom optimizer"))[1]


def parse_optimizer_args(raw: str) -> dict:
    """`"weight_decay=0.01 betas=0.9,0.99"` -> `{"weight_decay": 0.01, "betas": (0.9, 0.99)}`.

    Values go through `ast.literal_eval`, so tuples, bools and None all survive; anything that
    isn't a literal is kept as a plain string rather than being executed.
    """
    kwargs = {}
    for tok in (raw or "").split():
        if "=" not in tok:
            raise ValueError(f"optimizer arg {tok!r} is not key=value")
        key, value = tok.split("=", 1)
        try:
            kwargs[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            kwargs[key] = value
    return kwargs


def _bnb(cls_name: str):
    import bitsandbytes as bnb
    return getattr(bnb.optim, cls_name)


def _warn_lr(name: str, lr: float) -> None:
    """A wrong-family LR is silent at step 1 and obvious only hours later. Say it now."""
    if name == "lion8bit" and lr > 5e-5:
        logger.warning("[optimizer] Lion applies the SIGN of the update, so it needs roughly a "
                       "TENTH of an AdamW LR. %.2e will likely overbake — try %.2e.", lr, lr / 10)
    elif name == "prodigy":
        if lr < 0.5:
            logger.warning("[optimizer] Prodigy estimates the rate itself and takes lr=1.0 as a multiplier; "
                           "at %.2e it will barely train.", lr)
    elif name == "automagic4" and lr > 1e-3:
        logger.warning("[optimizer] Automagic v4 starts at the LR you give and finds its own rate — "
                       "%.2e is a high start (its default is 1e-6); the controller will walk it down, "
                       "but the first steps run at %.2e.", lr, lr)
    elif name == "automagic3" and lr > 1e-3:
        logger.warning("[optimizer] Automagic v3 starts at the LR you give and finds its own rate — "
                       "%.2e is a high start (its default is 1e-6); the controller will walk it down, "
                       "but the first steps run at %.2e.", lr, lr)
    elif name != "lion8bit" and lr > 1e-2:
        logger.warning("[optimizer] LR %.2e is very high for %s.", lr, name)


def create_optimizer(name: str, params, lr: float, args_str: str = "",
                     eps_floor_8bit: bool = False) -> tuple:
    """Build an optimizer. Returns `(optimizer, label)`; the label goes into LoRA metadata.

    Falls back to plain AdamW if the requested one cannot be constructed — a training run should
    not die at minute one over a dropdown, but the substitution is logged as a warning, never
    silently.

    `eps_floor_8bit` raises the 8-bit Adam family's eps to 1e-6. OFF by default: it is a
    MiniMax H3 workaround and every other family keeps the library default. See the note below.
    """
    name = (name or DEFAULT_OPTIMIZER).strip()
    if name.lower() not in ("muon", "prodigy") and (args_str or "").strip() in (MUON_DEFAULT_ARGS, PRODIGY_DEFAULT_ARGS):
        # Muon's / Prodigy's own default args left over from a preset (every picture preset is on Muon): they'd make
        # this optimizer fail to build and fall back to plain AdamW - they don't belong to it, so they're dropped
        logger.info("[optimizer] %s: ignoring the Optimizer Args left from Muon / Prodigy (%s)", name, args_str.strip())
        args_str = ""
    kwargs = parse_optimizer_args(args_str)
    key = name.lower()
    _warn_lr(key, lr)

    # eps 1e-6, not the library defaults' 1e-8 (matches ai-toolkit, which passes eps=1e-6 to
    # every Adam-family optimizer). This is a REAL stability bound, not a nicety: the 8-bit
    # optimizers store the second moment blockwise-quantized, and for heavily structured
    # gradients the small v entries quantize to ZERO — the update then degrades to lr*m/eps.
    # Measured on a MiniMax H3 epoch (46 steps @ 1e-4, eps=1e-8): lora_up drift reached 0.81
    # against an Adam bound of ~0.005 — the optimizer was applying ~100x the configured LR to
    # the most structured tensors (adaln worst, fc1 next), which presented as melted anatomy
    # at epoch 1. eps=1e-6 caps that amplification two orders of magnitude lower. Explicit
    # "eps=..." in Optimizer Args still wins.
    # NOTE the two conditions. 8-BIT Adam family only, NOT full-precision adam/adamw: full
    # precision has no quantized state, so v is whatever it really is, and a 1e-6 floor there
    # would DAMP the tensors with genuinely small second moments — the ones converging on fine
    # detail — while looking like a stability measure.
    #
    # And OPT-IN per caller, never global. This began as a MiniMax fix (bafb4e6) applied to the
    # whole Adam family, which silently moved Krea 2's DEFAULT optimizer off the library eps and
    # shipped that way in v3.3.0. Krea 2 never had the failure this works around and never asked
    # for the change. A workaround for one model family does not get to alter another's defaults;
    # the caller that needs it asks for it. Explicit "eps=..." in Optimizer Args still wins.
    if eps_floor_8bit and "8bit" in key and "lion" not in key:
        kwargs.setdefault("eps", 1e-6)

    try:
        if key == "adamw8bit":
            opt = _bnb("AdamW8bit")(params, lr=lr, **kwargs)
        elif key == "pagedadamw8bit":
            opt = _bnb("PagedAdamW8bit")(params, lr=lr, **kwargs)
        elif key == "ademamix8bit":
            opt = _bnb("AdEMAMix8bit")(params, lr=lr, **kwargs)
        elif key == "pagedademamix8bit":
            opt = _bnb("PagedAdEMAMix8bit")(params, lr=lr, **kwargs)
        elif key == "lion8bit":
            opt = _bnb("Lion8bit")(params, lr=lr, **kwargs)
        elif key == "adafactor":
            from transformers.optimization import Adafactor
            for k, v in (("scale_parameter", False), ("relative_step", False), ("warmup_init", False)):
                kwargs.setdefault(k, v)
            opt = Adafactor(params, lr=lr, **kwargs)
        elif key == "adamw":
            # Fused AdamW is one CUDA kernel over all 264 factors instead of a Python loop —
            # the "CUDA optimizer" the comparison thread was pointing at. Requires every param
            # on CUDA and floating point, which LoRA factors are.
            kwargs.setdefault("fused", torch.cuda.is_available())
            opt = torch.optim.AdamW(params, lr=lr, **kwargs)
        elif key == "automagic3":
            from fizgig.training.automagic3 import Automagic3
            # NON-fused: the MiniMax H3 loop clips, masks and snapshots grads between backward
            # and step, and a fused optimizer would update inside backward, ahead of all of it.
            kwargs.setdefault("fused", False)
            opt = Automagic3(params, lr=lr, **kwargs)
        elif key == "prodigy":
            from fizgig.training.prodigy import Prodigy        # the vendored copy only, never a package
            for k, v in parse_optimizer_args(PRODIGY_DEFAULT_ARGS).items():
                kwargs.setdefault(k, v)
            opt = Prodigy(params, lr=lr, **kwargs)
        elif key == "muon":
            for k, v in parse_optimizer_args(MUON_DEFAULT_ARGS).items():
                kwargs.setdefault(k, v)
            params = list(params)
            flat = ([p for g in params for p in g.get("params", ())]
                    if params and isinstance(params[0], dict) else params)
            if all(getattr(p, "ndim", 0) == 2 for p in flat):
                # torch.optim.Muon's update with the per-matrix work batched (a LoRA's hundreds of small matrices
                # made its per-parameter loop a third of a MOSS step, #221)
                from fizgig.training.muon import BatchedMuon
                opt = BatchedMuon(params, lr=lr, **kwargs)
            else:
                from fizgig.training.muon import Muon
                opt = Muon(params, lr=lr, **kwargs)
        elif key == "automagic4":
            from fizgig.training.automagic4 import Automagic4
            kwargs.setdefault("fused", False)     # as v3: the H3 loop works on grads between backward and step
            opt = Automagic4(params, lr=lr, **kwargs)
        elif "." in name:
            module_path, cls_name = name.rsplit(".", 1)
            opt = getattr(importlib.import_module(module_path), cls_name)(params, lr=lr, **kwargs)
        else:
            raise ValueError(f"unknown optimizer {name!r} — use one of {available_optimizers()} "
                             "or a full module.path.ClassName")
    except Exception as e:
        if key in NEW_OPTIMIZERS:        # no AdamW at Prodigy's lr=1.0: say what's wrong and stop
            raise ValueError(f"could not create {name}: {e}") from e
        logger.warning("[optimizer] could not create %s (%s) — falling back to AdamW", name, e)
        return torch.optim.AdamW(params, lr=lr), "adamw (fallback)"

    label = name + (f"({args_str.strip()})" if args_str.strip() else "")
    logger.info("optimizer: %s — %s", label, describe(key))
    return opt, label
