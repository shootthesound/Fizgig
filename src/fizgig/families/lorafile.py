"""Reading a LoRA file for a described family without loading the model (Profiler, Extract).

Resolves every LoRA pair in the file - the family's own keys, kohya (`lora_unet_<flattened>`), or PEFT / diffusers
(`lora_A/B` or `lora_down/up`, bare or under a common prefix) - to the dotted module name the family uses. Dotted names
resolve for any module; kohya's flattened names can only be un-flattened for modules the block map knows.
"""
import re

_DOWN = re.compile(r"(.+)\.(lora_A|lora_down|lora\.down)\.weight$")
_PREFIXES = ("transformer.", "unet.", "diffusion_model.", "model.diffusion_model.", "base_model.model.")
_UP = {"lora_A": "lora_B", "lora_down": "lora_up", "lora.down": "lora.up"}
_LOHA = re.compile(r"(.+)\.hada_w1_a$")
_LOKR = re.compile(r"(.+)\.lokr_w1(_a)?$")


def _resolver(drv, known, flat):
    """stem -> dotted module name: kohya-flattened names through the block map, dotted names with their prefix
    stripped, and another trainer's naming through the driver's renames (alias_flat)."""
    def resolve(stem):
        if stem.startswith("lora_unet_"):
            f = stem[len("lora_unet_"):]
            return flat.get(f) or flat.get(drv.alias_flat(f) or "")
        mod = next((stem[len(p):] for p in _PREFIXES if stem.startswith(p)), stem)
        if mod in known:
            return mod
        return flat.get(drv.alias_flat(mod.replace(".", "_")) or "") or mod
    return resolve


def fused_parts(drv, stem):
    """[(model stem, part, parts)] when `stem` names a tensor the model file fuses and the model splits
    (FTSpec.file_layout - Qwen's img_mlp.gate_up is [gate_layer; proj]): a LoRA extracted from such a file adapts
    the fused tensor, and each model Linear takes its rows of the up matrix. [] for any other stem."""
    try:
        spec = drv.ft_spec(None)
    except Exception:
        spec = None
    out = []
    for msuf, fsuf, pt, n in getattr(spec, "file_layout", ()) or ():
        f, m = fsuf[:-len(".weight")], msuf[:-len(".weight")]
        for fs, ms in ((f, m), (f.replace(".", "_"), m.replace(".", "_"))):
            if stem.endswith(fs):
                out.append((stem[:-len(fs)] + ms, int(pt), int(n)))
                break
    return out


def get_up(f, up):
    """An up weight from lora_pairs: a key, or (key, part, parts) for a fused tensor's rows."""
    if isinstance(up, tuple):
        key, part, parts = up
        return f.get_tensor(key).chunk(parts, dim=0)[part]
    return f.get_tensor(up)


def lora_pairs(desc, keys):
    """-> [(module or None, down key, up key, alpha key or None)] for every down/up pair in `keys`. module is None
    for a kohya-flattened name outside the block map (it cannot be un-flattened without the model). A pair on a fused
    file tensor (fused_parts) gives one entry per model Linear, its up as (key, part, parts) - read it with get_up."""
    keys = set(keys)
    drv = desc.load_driver()
    known = {m for g in drv.block_map() for b in g.blocks for m in b.modules}
    flat = {m.replace(".", "_"): m for m in known}
    resolve = _resolver(drv, known, flat)
    out = []
    for key in sorted(keys):
        m = _DOWN.match(key)
        if not m:
            continue
        stem = m.group(1)
        up = f"{stem}.{_UP[m.group(2)]}.weight"
        if up not in keys:
            continue
        alpha = f"{stem}.alpha"
        alpha = alpha if alpha in keys else None
        mod = resolve(stem)
        if mod not in known:
            split = [(resolve(ms), pt, n) for ms, pt, n in fused_parts(drv, stem)]
            if split and all(m in known for m, _pt, _n in split):
                out += [(m, key, (up, pt, n), alpha) for m, pt, n in split]
                continue
        out.append((mod, key, up, alpha))
    return out


def lokr_modules(desc, keys):
    """-> [(module or None, stem)] for every LoKR module in `keys` (stem = the key prefix before .lokr_*)."""
    keys = set(keys)
    drv = desc.load_driver()
    known = {m for g in drv.block_map() for b in g.blocks for m in b.modules}
    resolve = _resolver(drv, known, {m.replace(".", "_"): m for m in known})
    out = []
    for key in sorted(keys):
        m = _LOKR.match(key)
        if not m:
            continue
        stem = m.group(1)
        out.append((resolve(stem), stem))
    return out


def lokr_factors(f, stem):
    """(w1, w2, scale) of a LoKR module from an open safetensors file, low-rank factors multiplied out."""
    from fizgig.networks.lora import lycoris_scale_from_keys
    keys = {k[len(stem) + 1:]: f.get_tensor(k) for k in f.keys() if k.startswith(stem + ".")}
    w1 = keys["lokr_w1"] if "lokr_w1" in keys else keys["lokr_w1_a"].float() @ keys["lokr_w1_b"].float()
    w2 = keys["lokr_w2"] if "lokr_w2" in keys else keys["lokr_w2_a"].float() @ keys["lokr_w2_b"].float()
    return w1.float(), w2.float(), lycoris_scale_from_keys(keys)


def loha_modules(desc, keys):
    """-> [(module or None, stem)] for every LoHa module in `keys` (stem = the key prefix before .hada_*)."""
    drv = desc.load_driver()
    known = {m for g in drv.block_map() for b in g.blocks for m in b.modules}
    resolve = _resolver(drv, known, {m.replace(".", "_"): m for m in known})
    return [(resolve(m.group(1)), m.group(1)) for m in (_LOHA.match(k) for k in sorted(set(keys))) if m]


def loha_delta(f, stem):
    """A LoHa module's whole weight change (out, in), float32: (w1_a @ w1_b) * (w2_a @ w2_b) times the LyCORIS
    scale - the same product the workbench applies."""
    from fizgig.networks.lora import lycoris_scale_from_keys
    keys = {k[len(stem) + 1:]: f.get_tensor(k) for k in f.keys() if k.startswith(stem + ".")}
    w1 = keys["hada_w1_a"].float() @ keys["hada_w1_b"].float()
    w2 = keys["hada_w2_a"].float() @ keys["hada_w2_b"].float()
    return (w1 * w2).reshape(w1.shape[0], -1) * lycoris_scale_from_keys(keys)


def block_of(desc):
    """{dotted module: block id} for the family's block map."""
    return {m: b.id for g in desc.load_driver().block_map() for b in g.blocks for m in b.modules}


def family_keys(desc, module):
    """(down, up, alpha) keys of a module in the family's own file format."""
    f = desc.lora
    # a kohya family flattens the path (lora_unet_blocks_0_attn_wq), as FamilyLoRA saves it
    stem = f"lora_unet_{module.replace('.', '_')}" if f.kohya else f"{f.file_prefix}{module}"
    return f"{stem}.{f.down}.weight", f"{stem}.{f.up}.weight", f.alpha_key.format(prefix=stem)
