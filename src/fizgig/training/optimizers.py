"""Optimizer selection, shared by the trainers.

Krea 2 hardcoded `bnb.optim.AdamW8bit` — a good default, but the only choice, which is the one
place the community comparison against OneTrainer (~40 optimizers) landed a fair hit.

Two things matter for a LoRA and pull against the "more optimizers is better" instinct:

* **Optimizer state is tiny here.** Krea 2 trains 264 small factors, so AdamW's two moments cost
  ~tens of MB against a 13-19 GB base. Choosing an 8-bit or factored optimizer to save memory is
  nearly pointless — the base dominates. What the choice actually buys is *update behaviour*.
* **Learning rates are not comparable across families.** Lion's update is a sign, so it wants
  roughly a tenth of AdamW's LR; Prodigy estimates the LR itself (`d`) and wants `lr=1.0` as a
  multiplier; Muon with `adjust_lr_fn=match_rms_adamw` matches AdamW's step RMS so standard AdamW
  learning rates (`1e-4` to `2e-4`) apply. `create_optimizer` warns loudly on the record when the
  LR looks wrong for the family, and `owns_its_rate` stands Adaptive LR and the scheduler down for
  self-tuning optimizers.

Free-form `module.path.ClassName` is accepted too, so a user who pip-installs something exotic
does not need a Fizgig release to use it.
"""

from __future__ import annotations

import ast
import importlib
import logging

import torch

logger = logging.getLogger(__name__)

# Default Prodigy LoRA kwargs (auto-filled in the UI's Optimizer Args box when prodigy is selected,
# and applied as defaults in create_optimizer unless overridden in Optimizer Args).
PRODIGY_DEFAULT_ARGS = "decouple=True weight_decay=0.01 d_coef=1.0 use_bias_correction=True safeguard_warmup=True"
PRODIGY_ONLY_KEYS = ("decouple", "d_coef", "use_bias_correction", "safeguard_warmup",
                     "d0", "beta3", "growth_rate", "slice_p", "fsdp_in_use")

# Default Muon LoRA kwargs (RMS-matched to AdamW so standard 1e-4..2e-4 LRs and schedulers apply).
MUON_DEFAULT_ARGS = "adjust_lr_fn=match_rms_adamw weight_decay=0.01 momentum=0.95 nesterov=True ns_steps=5"
MUON_ONLY_KEYS = ("adjust_lr_fn", "momentum", "nesterov", "ns_steps", "ns_coefficients")


def _strip_keys(raw: str, default_str: str, only_keys: tuple[str, ...]) -> str:
    s = (raw or "").strip()
    if not s or s == default_str:
        return ""
    kept, had_special = [], False
    for tok in s.split():
        key = tok.partition("=")[0].strip()
        if key in only_keys:
            had_special = True
        else:
            kept.append(tok)
    if had_special and all(k.partition("=")[0].strip() == "weight_decay" for k in kept):
        return ""
    return " ".join(kept)


def strip_prodigy_args(raw: str) -> str:
    """Remove Prodigy-specific key=value tokens from an Optimizer Args string when switching to
    another optimizer."""
    return _strip_keys(raw, PRODIGY_DEFAULT_ARGS, PRODIGY_ONLY_KEYS)


def strip_muon_args(raw: str) -> str:
    """Remove Muon-specific key=value tokens from an Optimizer Args string when switching to
    another optimizer."""
    return _strip_keys(raw, MUON_DEFAULT_ARGS, MUON_ONLY_KEYS)


# name -> (import to test, one-line description shown in the GUI/CLI help)
# Prodigy is back (it was dropped on 2026-07-28 for fighting Adaptive LR): it is vendored now (no
# extra package), and `owns_its_rate` makes Adaptive LR and the scheduler stand down for it, so they
# no longer write over its rate. CAME and Adafactor stay out.
_CATALOG = {
    "adamw8bit":          ("bitsandbytes", "AdamW, 8-bit state (default — the validated recipe)"),
    "adamw":              (None,           "AdamW, fp32 state, CUDA-fused where available"),
    "pagedadamw8bit":     ("bitsandbytes", "AdamW8bit that pages state to CPU under pressure"),
    "ademamix8bit":       ("bitsandbytes", "AdEMAMix — second slow EMA, aimed at long runs"),
    "pagedademamix8bit":  ("bitsandbytes", "AdEMAMix8bit with CPU paging"),
    "lion8bit":           ("bitsandbytes", "Lion — sign updates; use ~1/10 the AdamW LR"),
    "prodigy":            (None,           "Prodigy — estimates the LR itself (d); Learning Rate is locked to 1.0"),
    "muon":               (None,           "Muon — Newton-Schulz orthogonalized momentum (RMS-matched to AdamW LR)"),
    # Ostris's Automagic v3, vendored (fizgig/training/automagic3.py, MIT). One learning rate per
    # parameter group, nudged up while update signs hold steady and down while they alternate;
    # the LR given is only its START (1e-6 is its own default). MiniMax H3 LoRA experiment,
    # 17 Sep 2026 — unmeasured against the AdamW recipe until an A/B says otherwise.
    "automagic3":         (None,           "Automagic v3 (Ostris) — sets its own learning rate from the update signs; "
                                            "the LR box is its start, 1e-6 recommended. Experiment"),
    # Automagic v4 (fizgig/training/automagic4.py, MIT): v3 plus a per-weight brake on the most
    # sensitive weights and a rate controller that settles instead of annealing to zero.
    "automagic4":         (None,           "Automagic v4 (Ostris) — like v3, sets its own learning rate, but evens out "
                                            "how hard each weight is pushed; LR box is its start, 1e-6 recommended. Experiment"),
}

DEFAULT_OPTIMIZER = "adamw8bit"


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
    group's "lr" is only the start); Prodigy keeps its adapted step size in "d" (scaled by
    "lr"); everyone else keeps it on the group."""
    if optimizer is None:
        return 0.0
    fn = getattr(optimizer, "get_avg_learning_rate", None)
    if callable(fn):
        try:
            return float(fn())
        except Exception:
            pass
    g = optimizer.param_groups[0]
    if "d" in g:
        return float(g["lr"]) * float(g["d"])
    return float(g["lr"])


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
    """True for an optimizer that sets its own learning rate (Automagic v3, Prodigy): schedulers and
    adaptive watchers that write the group rate have no effect on it and should stand down."""
    if optimizer is None:
        return False
    if optimizer.__class__.__name__ in ("Automagic3", "Automagic4", "Prodigy"):
        return True
    groups = getattr(optimizer, "param_groups", None)
    return bool(groups and isinstance(groups[0], dict) and "d" in groups[0])


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
        if value.lower() in ("true", "false"):
            kwargs[key] = value.lower() == "true"
            continue
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
    elif name in ("automagic3", "automagic4") and lr > 1e-3:
        logger.warning("[optimizer] Automagic starts at the LR you give and finds its own rate — "
                       "%.2e is a high start (its default is 1e-6); the controller will walk it down, "
                       "but the first steps run at %.2e.", lr, lr)
    elif (name == "prodigy" or "prodigy" in name) and lr < 0.5:
        logger.warning("[optimizer] Prodigy estimates the LR itself and expects lr=1.0 as a "
                       "multiplier. At %.2e it will barely train.", lr)
    elif name not in ("lion8bit", "prodigy") and "prodigy" not in name and lr > 1e-2:
        logger.warning("[optimizer] LR %.2e is very high for %s.", lr, name)


def create_optimizer(name: str, params, lr: float, args_str: str = "",
                     eps_floor_8bit: bool = False) -> tuple:
    """Build an optimizer. Returns `(optimizer, label)`; the label goes into LoRA metadata.

    Falls back to plain AdamW if the requested one cannot be constructed — a training run should
    not die at minute one over a dropdown, but the substitution is logged as a warning, never
    silently, and an LR >= 0.01 (such as Prodigy's 1.0) is clamped to 1e-4 on fallback.

    `eps_floor_8bit` raises the 8-bit Adam family's eps to 1e-6. OFF by default: it is a
    MiniMax H3 workaround and every other family keeps the library default. See the note below.
    """
    name = (name or DEFAULT_OPTIMIZER).strip()
    kwargs = parse_optimizer_args(args_str)
    key = name.lower()
    is_prodigy = key == "prodigy" or key.endswith(".prodigy")
    is_muon = key == "muon" or key.endswith(".muon")
    if is_prodigy and lr < 0.1:
        logger.info("[optimizer] Prodigy expects lr=1.0 as a multiplier (got %.2e) — using lr=1.0", lr)
        lr = 1.0
    _warn_lr(key, lr)

    if not is_prodigy:
        for k in PRODIGY_ONLY_KEYS:
            kwargs.pop(k, None)
    if not is_muon:
        for k in MUON_ONLY_KEYS:
            kwargs.pop(k, None)

    if eps_floor_8bit and "8bit" in key and "lion" not in key:
        kwargs.setdefault("eps", 1e-6)

    params_list = list(params)
    try:
        if key == "adamw8bit":
            opt = _bnb("AdamW8bit")(params_list, lr=lr, **kwargs)
        elif key == "pagedadamw8bit":
            opt = _bnb("PagedAdamW8bit")(params_list, lr=lr, **kwargs)
        elif key == "ademamix8bit":
            opt = _bnb("AdEMAMix8bit")(params_list, lr=lr, **kwargs)
        elif key == "pagedademamix8bit":
            opt = _bnb("PagedAdEMAMix8bit")(params_list, lr=lr, **kwargs)
        elif key == "lion8bit":
            opt = _bnb("Lion8bit")(params_list, lr=lr, **kwargs)
        elif key == "adamw":
            # Fused AdamW is one CUDA kernel over all 264 factors instead of a Python loop —
            # the "CUDA optimizer" the comparison thread was pointing at. Requires every param
            # on CUDA and floating point, which LoRA factors are.
            kwargs.setdefault("fused", torch.cuda.is_available())
            opt = torch.optim.AdamW(params_list, lr=lr, **kwargs)
        elif key == "prodigy":
            for k, v in (("decouple", True), ("weight_decay", 0.01), ("d_coef", 1.0),
                         ("use_bias_correction", True), ("safeguard_warmup", True)):
                kwargs.setdefault(k, v)
            try:
                from prodigyopt import Prodigy
            except ImportError:
                from fizgig.training.prodigy import Prodigy
            opt = Prodigy(params_list, lr=lr, **kwargs)
        elif key == "muon":
            for k, v in (("adjust_lr_fn", "match_rms_adamw"), ("weight_decay", 0.01),
                         ("momentum", 0.95), ("nesterov", True), ("ns_steps", 5)):
                kwargs.setdefault(k, v)
            flat = ([p for g in params_list for p in g.get("params", ())]
                    if (params_list and isinstance(params_list[0], dict)) else params_list)
            if hasattr(torch.optim, "Muon") and all(getattr(p, "ndim", 0) == 2 for p in flat):
                opt = torch.optim.Muon(params_list, lr=lr, **kwargs)
            else:
                from fizgig.training.muon import Muon
                opt = Muon(params_list, lr=lr, **kwargs)
        elif key == "automagic3":
            from fizgig.training.automagic3 import Automagic3
            # NON-fused: the MiniMax H3 loop clips, masks and snapshots grads between backward
            # and step, and a fused optimizer would update inside backward, ahead of all of it.
            kwargs.setdefault("fused", False)
            opt = Automagic3(params_list, lr=lr, **kwargs)
        elif key == "automagic4":
            from fizgig.training.automagic4 import Automagic4
            kwargs.setdefault("fused", False)     # same reason as v3
            opt = Automagic4(params_list, lr=lr, **kwargs)
        elif "." in name:
            module_path, cls_name = name.rsplit(".", 1)
            opt = getattr(importlib.import_module(module_path), cls_name)(params_list, lr=lr, **kwargs)
        else:
            raise ValueError(f"unknown optimizer {name!r} — use one of {available_optimizers()} "
                             "or a full module.path.ClassName")
    except Exception as e:
        fallback_lr = 1e-4 if lr >= 0.01 else lr
        if fallback_lr != lr:
            logger.warning("[optimizer] could not create %s (%s) — falling back to AdamW at lr=%.1e "
                           "(clamped from %.2g)", name, e, fallback_lr, lr)
        else:
            logger.warning("[optimizer] could not create %s (%s) — falling back to AdamW", name, e)
        return torch.optim.AdamW(params_list, lr=fallback_lr), "adamw (fallback)"

    label = name + (f"({args_str.strip()})" if args_str.strip() else "")
    logger.info("optimizer: %s — %s", label, describe(key))
    return opt, label
