"""The families described through FamilyDescription - every family Fizgig trains. Their order is the Base Model
dropdown's and the workbench tabs' radio order.
"""
from typing import Optional

from fizgig.families.description import FamilyDescription
from fizgig.families.klein import KLEIN
from fizgig.families.krea2 import KREA2
from fizgig.families.minimax import MINIMAX
from fizgig.families.qwen_image import QWEN_IMAGE_21
from fizgig.families.sdxl import SDXL
from fizgig.families.anima import ANIMA
from fizgig.families.zimage import ZIMAGE

FAMILIES = {d.key: d for d in (KLEIN, MINIMAX, KREA2, QWEN_IMAGE_21, SDXL, ANIMA, ZIMAGE)}

for _d in FAMILIES.values():
    _problems = _d.validate()
    if _problems:
        raise ValueError(f"family description {_d.key!r} is inconsistent: {'; '.join(_problems)}")


def get(key: str) -> Optional[FamilyDescription]:
    return FAMILIES.get(key)


def by_gui_label(label: str) -> Optional[FamilyDescription]:
    """The description behind a Base Model selector entry, or None."""
    for d in FAMILIES.values():
        if label == d.gui_label or label in d.aliases:
            return d
    return None


def by_arch_id(arch_id: str) -> Optional[FamilyDescription]:
    """The description whose architecture id (cache filenames, metadata) is arch_id, or None.
    Shared code (metadata, dataset buckets) asks this instead of carrying per-family entries."""
    for d in FAMILIES.values():
        if d.arch_id == arch_id:
            return d
    return None


def training_families() -> list:
    """Descriptions whose training entry points exist (shown in the Base Model selector)."""
    return [d for d in FAMILIES.values() if d.training_ready and not d.hidden]


def workbench_families(tool: str) -> list:
    """Descriptions whose driver is built and whose description enables this workbench tool ("repair", ...)."""
    return [d for d in FAMILIES.values() if d.training_ready and not d.hidden and tool in d.workbench]


def family_of_lora(path: str) -> Optional[FamilyDescription]:
    """The described family a LoRA file was written for, from its header alone (key names, no tensor data), or
    None: the family whose block map takes the largest share of the file's modules, when that share is at least half
    (every naming the readers accept: own keys, kohya-flattened, another trainer's via alias_flat, fused tensors)."""
    try:
        from safetensors import safe_open
        with safe_open(path, "pt") as f:
            keys = list(f.keys())
    except Exception:
        return None
    import re
    from fizgig.families.lorafile import loha_modules, lokr_modules, lora_pairs
    stem_of = re.compile(r"(.+)\.(?:lora_A|lora_down|lora\.down)\.weight$|(.+)\.(?:lokr_w1(?:_a)?|hada_w1_a)$")
    # a LoRA's text-encoder parts (kohya lora_te1_ / lora_te2_, diffusers text_encoder.) aren't the diffusion model's
    stems = {m.group(1) or m.group(2) for m in map(stem_of.match, keys) if m}
    stems = {s for s in stems if not s.startswith(("lora_te", "text_encoder", "te1.", "te2.", "te."))}
    if not stems:
        return None
    # the family placing the largest share of the file's modules inside its block map: several can place some
    # (Krea 2 maps other trainers' transformer_blocks names, so a Qwen file half-fits it too)
    best, best_share = None, 0.0
    for d in training_families():
        try:
            found = [(m, stem_of.match(down).group(1)) for m, down, *_ in lora_pairs(d, keys)]
        except ValueError:
            continue
        found += lokr_modules(d, keys) + loha_modules(d, keys)
        drv = d.load_driver()
        share = len({s for m, s in found if m is not None and drv.block_of(m) is not None}) / len(stems)
        if share > best_share:
            best, best_share = d, share
    return best if best_share >= 0.5 else None
