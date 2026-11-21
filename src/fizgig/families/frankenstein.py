"""Frankenstein mode: two compatible base DiTs blended block by block (Studio's sliders as the blend weights).

A blend weight is 0 (the loaded DiT) to 1 (the other). Dense weights blend linearly in fp32. An int8 ConvRot layer
(H3's pruned int8 files) stores codes of rotate(W) with one scale per output row; the rotation is the same fixed
block regular Hadamard in both files (group 256 in each), so the blend is done in the stored basis - no rotation
either way - and requantised per row, exactly as the files were made (quantize_int8_convrot's nearest rounding). At 0
and 1 the source codes are returned untouched (no requantising error at the ends).

A pruned H3 file's AdaLN is a per-timestep table (adaln_t_table, outside the blocks) through each block's own
[hidden * 6, 8] Linear. Each model learned its own table (H3 Image's and H3 video's differ completely), so a block's
AdaLN weights mean nothing beside the other model's table: a blended block's AdaLN mixes the two OUTPUTS instead, each
through its own model's table.

The other side may carry a LoRA baked in (H3 video's Turbo LoRA, which makes its blocks few-step like H3 Image's): its
Linears' updates added to the other file's weights at load, its AdaLN rows added to the other side's AdaLN output.
"""
from __future__ import annotations

import os
import re

import torch
import torch.nn.functional as F


def blend_int8_rows(qa: torch.Tensor, sa: torch.Tensor, qb: torch.Tensor, sb: torch.Tensor, w: float):
    """Two int8 ConvRot layers ([out, in] codes, [out, 1] or [out] scales) at weight w -> (codes int8, scale fp32
    [out, 1]). w = 0 gives a's own, w = 1 b's own."""
    if w <= 0.0:
        return qa, sa.reshape(-1, 1).to(torch.float32)
    if w >= 1.0:
        return qb, sb.reshape(-1, 1).to(torch.float32)
    dev = qa.device
    m = (1.0 - w) * (qa.to(torch.float32) * sa.to(dev, torch.float32).reshape(-1, 1)) \
        + w * (qb.to(dev, torch.float32) * sb.to(dev, torch.float32).reshape(-1, 1))
    return _requant(m)


def _requant(m):
    s = m.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 127.0
    return torch.round(m / s).clamp(-127, 127).to(torch.int8), s


def blend_dense(a: torch.Tensor, b: torch.Tensor, w: float) -> torch.Tensor:
    """Two dense tensors at weight w, in fp32, back in a's dtype (a's own / b's own at the ends)."""
    if w <= 0.0:
        return a
    if w >= 1.0:
        return b.to(a.device, a.dtype)
    return torch.lerp(a.to(torch.float32), b.to(a.device, torch.float32), float(w)).to(a.dtype)


def _prefix(names):
    """The block a list of its Linears lives in ("blocks.3."): their shared dotted prefix, cut after its last number -
    everything of the block lives under it, however many of its Linears the map names."""
    parts = [n.split(".") for n in names]
    common = []
    for seg in zip(*parts):
        if len(set(seg)) != 1:
            break
        common.append(seg[0])
    num = [i for i, s in enumerate(common) if s.isdigit()]
    if num:
        common = common[: num[-1] + 1]
    return ".".join(common) + "." if common else ""


def _lora_pairs(path, strength):
    """A PEFT / kohya LoRA file -> {module name: (A [r, in], B [out, r] with strength and alpha / rank folded in)}."""
    from fizgig.utils.safetensors import MemoryEfficientSafeOpen
    out = {}
    with MemoryEfficientSafeOpen(path) as f:
        keys = set(f.keys())
        for k in keys:
            for down, up in ((".lora_A.weight", ".lora_B.weight"), (".lora_down.weight", ".lora_up.weight")):
                if k.endswith(down):
                    name = k[: -len(down)]
                    a, b = f.get_tensor(k).float(), f.get_tensor(name + up).float()
                    ak = name + ".alpha"
                    scale = float(f.get_tensor(ak).float().item()) / a.shape[0] if ak in keys else 1.0
                    out[name] = (a, b * (scale * float(strength)))
    return out


class DiTBlend:
    """The loaded DiT's blocks blended live with another file's: `set({block id: weight})` rewrites only the blocks
    whose weight changed, in place, on the card. The loaded DiT's own tensors are kept on the CPU (weight 0 puts them
    back byte for byte); the other file's are read once, on the CPU. A block's tensors are everything under its
    Linears' shared prefix that both have with the same shape: int8 ConvRot Linears (qdata / wscale, with the same
    rotation group) blend in their stored basis; the rest (norms) linearly; a pruned AdaLN by its output (see the
    module docstring). Whatever only one side has, or lives outside the blocks (embedders, H3 Image's head bank), stays
    the loaded DiT's. other_lora: (path, strength) baked into the other side. own_path: the loaded DiT's file - its
    int8 layers' own codes are then read back from it when a block returns to 0 instead of kept in RAM (checked equal
    to the loaded ones first), and the other file's int8 layers are read only when their block changes: about one
    block's worth of RAM instead of both models' (~42 GB for H3 - with H3's text encoder loading, a 128 GB machine ran
    short)."""

    def __init__(self, dit, groups, other_path, other_lora=None, log=None, own_path=None):
        from fizgig.minimax.convrot import ConvRotInt8Linear, parse_comfy_quant
        from fizgig.utils.safetensors import MemoryEfficientSafeOpen
        try:
            from fizgig.minimax.model import AdalnProj
        except ImportError:                       # another family's model: no pruned AdaLN to treat apart
            AdalnProj = ()
        self.dit, self.other_path = dit, other_path
        self.items, self.w = {}, {}
        self.skipped_names = []
        lora = _lora_pairs(*other_lora) if other_lora else {}
        self.lora_used = 0
        dev = next(iter([*dit.parameters(), *dit.buffers()])).device
        self.dev = dev
        self.other_f = f = MemoryEfficientSafeOpen(other_path)
        self.own_f = MemoryEfficientSafeOpen(own_path) if own_path and os.path.isfile(own_path) else None
        own_checked = False
        if f is not None:                      # (the other file, kept open for reads on demand)
            hdr = f.header
            # the other model's own AdaLN table (a pruned H3), for its blocks' AdaLN outputs
            self.other_table = f.get_tensor("adaln_t_table").float().to(dev) \
                if "adaln_t_table" in hdr and getattr(dit, "pruned_adaln", False) else None
            self.egrid = None
            for g in groups:
                for b in g.blocks:
                    pre = _prefix(b.modules)
                    if not pre:
                        continue
                    items, adaln_lin = [], set()
                    mods = [(m, n, mod) for n, mod in dit.named_modules() if n.startswith(pre) and ".adapters" not in n
                            for m in [re.sub(r"\.base(?=\.|$)", "", n)]]
                    for name, _raw, mod in mods:
                        if self.other_table is not None and isinstance(mod, AdalnProj) \
                                and f"{name}.linear.weight" in hdr:
                            ld = lora.get(f"{name}.linear")
                            items.append(("a", mod, f.get_tensor(f"{name}.linear.weight").float().to(dev),
                                          f.get_tensor(f"{name}.linear.bias").float().to(dev)
                                          if f"{name}.linear.bias" in hdr else None, ld, f"{name}.linear"))
                            adaln_lin.add(f"{name}.linear")
                            self.lora_used += ld is not None
                    for name, _raw, mod in mods:
                        if name in adaln_lin:
                            continue
                        ld = lora.get(name)
                        if isinstance(mod, ConvRotInt8Linear):
                            wk, sk, ck = name + ".weight", name + ".weight_scale", name + ".comfy_quant"
                            ok = wk in hdr and sk in hdr and list(hdr[wk]["shape"]) == list(mod.qdata.shape) \
                                and hdr[wk]["dtype"] == "I8"
                            if ok and ck in hdr:
                                conf = parse_comfy_quant(f.get_tensor(ck))
                                ok = int(conf.get("convrot_groupsize", 256)) == int(mod.rot) and bool(conf.get("convrot"))
                            if not ok:
                                self.skipped_names.append(name)
                                continue
                            if not own_checked and self.own_f is not None:
                                # the loaded codes are the file's own (H3's loader keeps them): else keep copies
                                ok_own = wk in self.own_f.header and torch.equal(
                                    self.own_f.get_tensor(wk).to(mod.qdata.device), mod.qdata)
                                if not ok_own:
                                    self.own_f.file.close()
                                    self.own_f = None
                                own_checked = True
                            own = None if self.own_f is not None else                                 (mod.qdata.detach().cpu().clone(), mod.wscale.detach().cpu().clone())
                            self.lora_used += ld is not None
                            items.append(("q", mod, own, ld, name))
                            if getattr(mod, "bias", None) is not None and name + ".bias" in hdr \
                                    and list(hdr[name + ".bias"]["shape"]) == list(mod.bias.shape):
                                items.append(("d", mod, "bias", mod.bias.detach().cpu().clone(),
                                              f.get_tensor(name + ".bias").to(mod.bias.dtype), name + ".bias"))
                            continue
                        own = list(mod.named_parameters(recurse=False)) + list(mod.named_buffers(recurse=False))
                        for pname, t in own:
                            key = f"{name}.{pname}"
                            if t is None or not t.is_floating_point():
                                continue
                            if key not in hdr or list(hdr[key]["shape"]) != list(t.shape):
                                self.skipped_names.append(key)
                                continue
                            bt = f.get_tensor(key).float()
                            if pname == "weight" and ld is not None and tuple((ld[1] @ ld[0]).shape) == tuple(bt.shape):
                                bt = bt + (ld[1] @ ld[0])
                                self.lora_used += 1
                            items.append(("d", mod, pname, t.detach().cpu().clone(), bt.to(t.dtype), key))
                    if items:
                        self.items[b.id] = items
                        self.w[b.id] = 0.0
        if self.other_table is not None and any(it[0] == "a" and it[4] is not None for v in self.items.values()
                                                for it in v):
            from fizgig.minimax.common import _load_h3_egrid
            self.egrid = _load_h3_egrid().to(dev)
        self._wrap_time_embedding()
        if log:
            n = sum(len(v) for v in self.items.values())
            log(f"[frankenstein] {len(self.items)} blocks blendable with {os.path.basename(other_path)} "
                f"({n} tensors; {len(self.skipped_names)} not in both, kept as loaded"
                + (f"; {self.lora_used} LoRA updates baked into its side" if other_lora else "") + ")")

    def _wrap_time_embedding(self):
        """The DiT's timestep -> table lookup, noting the cleanness values: a blended AdaLN looks the same t up in the
        other model's table."""
        dit = self.dit
        if self.other_table is None or getattr(dit, "_fr_wrapped", False):
            return
        orig = dit._time_embedding

        def time_embedding(t):
            dit._fr_t = t
            return orig(t)
        dit._time_embedding = time_embedding
        dit._fr_wrapped = True

    def _other_t_emb(self, t):
        table = self.other_table
        pos = t.to(torch.float32).clamp(0.0, 1.0).to(table.device) * (table.shape[0] - 1)
        lo = pos.floor().long()
        hi = (lo + 1).clamp(max=table.shape[0] - 1)
        frac = (pos - lo.float()).unsqueeze(1)
        return table[lo] * (1.0 - frac) + table[hi] * frac, pos

    def _adaln_forward(self, mod, Wv, bv, ld, w):
        dit = self

        def forward(t_emb):
            x = mod.linear(F.silu(t_emb) if mod.apply_silu else t_emb)
            tv, pos = dit._other_t_emb(dit.dit._fr_t)
            xv = F.linear(tv, Wv, bv)
            if ld is not None and dit.egrid is not None:    # the baked LoRA's AdaLN rows, as the Turbo patch adds them
                st = dit.egrid[pos.round().long().clamp(0, dit.egrid.shape[0] - 1)]
                xv = xv + (ld[1].to(xv.device) @ (ld[0].to(xv.device) @ st.float().T)).T
            x = (x.float() * (1.0 - w) + xv * w).to(x.dtype)
            x = x.view(x.shape[0] * mod.modalities, mod.expand * mod.hidden)
            return x.chunk(mod.expand, dim=-1)
        return forward

    @torch.no_grad()
    def set(self, weights):
        """{block id: weight 0..1} -> the changed blocks rewritten in place. Returns how many blocks changed."""
        changed = 0
        for bid, items in self.items.items():
            w = min(1.0, max(0.0, float(weights.get(bid, 0.0))))
            if abs(w - self.w[bid]) < 1e-6:
                continue
            for it in items:
                if it[0] == "q":
                    _, mod, own, ld, name = it
                    dev = mod.qdata.device
                    qa, sa = self._own_codes(own, name, dev) if w < 1.0 else (None, None)
                    qb, sb = self._other_codes(mod, ld, name, dev) if w > 0.0 else (None, None)
                    q, s = (qa, sa) if w <= 0.0 else (qb, sb) if w >= 1.0 else blend_int8_rows(qa, sa, qb, sb, w)
                    mod.qdata.copy_(q)
                    mod.wscale.copy_(s.reshape(mod.wscale.shape))
                elif it[0] == "d":
                    _, mod, pname, a, b, _k = it
                    t = getattr(mod, pname)
                    t.data.copy_(blend_dense(a.to(t.device), b.to(t.device), w))
                else:
                    _, mod, Wv, bv, ld, _k = it
                    if "forward" in mod.__dict__:
                        del mod.forward
                    if w > 0.0:
                        mod.forward = self._adaln_forward(mod, Wv, bv, ld, w)
            self.w[bid] = w
            changed += 1
        return changed

    def weights(self):
        return dict(self.w)

    def _own_codes(self, own, name, dev):
        """The loaded model's own int8 codes and scales for a layer: kept copies, or read back from its file."""
        if own is not None:
            return own[0].to(dev), own[1].to(dev, torch.float32).reshape(-1, 1)
        f = self.own_f
        return (f.get_tensor(name + ".weight").to(dev),
                f.get_tensor(name + ".weight_scale").to(dev, torch.float32).reshape(-1, 1))

    def _other_codes(self, mod, ld, name, dev):
        """The other file's codes and scales for a layer, read now, with the baked LoRA in its stored basis."""
        from fizgig.minimax.convrot import rotate
        f = self.other_f
        qb = f.get_tensor(name + ".weight").to(dev)
        sb = f.get_tensor(name + ".weight_scale").to(dev, torch.float32).reshape(-1, 1)
        if ld is None:
            return qb, sb
        m = qb.to(torch.float32) * sb + rotate((ld[1].to(dev) @ ld[0].to(dev)).float(), int(mod.rot))
        return _requant(m)

    @torch.no_grad()
    def _adaln_refit(self, it, w):
        """A blended pruned AdaLN as ONE Linear over the loaded model's own table (an exported file has one table):
        the blend's output at every table row, refitted by least squares onto [table, 1]. Measured on H3 Image /
        H3 video with the Turbo LoRA: 0.07-0.08% relative error at weight 1 (the two tables span nearly the same
        space) - the exported model renders the blend. -> (weight, bias) on the CPU, fp32."""
        _, mod, Wv, bv, ld, _k = it
        dev = Wv.device
        Ti = self.dit.adaln_t_table.float().to(dev)
        rows = torch.linspace(0.0, 1.0, Ti.shape[0], device=dev)
        own = F.linear(Ti, mod.linear.weight.float().to(dev),
                       mod.linear.bias.float().to(dev) if mod.linear.bias is not None else None)
        tv, pos = self._other_t_emb(rows)
        other = F.linear(tv, Wv, bv)
        if ld is not None and self.egrid is not None:
            st = self.egrid[pos.round().long().clamp(0, self.egrid.shape[0] - 1)].float()
            other = other + (ld[1].to(dev) @ (ld[0].to(dev) @ st.T)).T
        target = own * (1.0 - w) + other * w
        X = torch.cat([Ti, torch.ones(Ti.shape[0], 1, device=dev)], 1)
        sol = torch.linalg.lstsq(X, target).solution            # [9, out]
        return sol[:-1].T.contiguous().cpu(), sol[-1].contiguous().cpu()

    def export(self, src_path, out_path, metadata=None, log=None):
        """The blend as a new base file, in the loaded model's own format (ComfyUI loads it as it loads src_path):
        every tensor of src_path, with the blended blocks' current values in place - int8 codes and scales as they
        are on the card, blended norms, each blended AdaLN refitted onto the file's own table. Untouched tensors are
        copied byte for byte. Written to out_path + ".tmp", then moved into place."""
        import json

        from fizgig.krea2.safetensors_utils import stream_save_file
        from fizgig.utils.safetensors import MemoryEfficientSafeOpen
        repl = {}                              # file key -> producer of its blended value
        for bid, items in self.items.items():
            w = self.w[bid]
            if w <= 0.0:
                continue
            for it in items:
                if it[0] == "q":
                    mod, name = it[1], it[4]
                    repl[name + ".weight"] = (lambda m=mod: m.qdata.detach().cpu())
                    repl[name + ".weight_scale"] = (lambda m=mod: m.wscale.detach().cpu())
                elif it[0] == "d":
                    mod, pname, key = it[1], it[2], it[5]
                    repl[key] = (lambda m=mod, p=pname: getattr(m, p).detach().cpu())
                else:
                    key = it[5]
                    fit = {}

                    def get(part, it=it, w=w, fit=fit):
                        if not fit:
                            fit["w"], fit["b"] = self._adaln_refit(it, w)
                        return fit[part]
                    repl[key + ".weight"] = (lambda g=get: g("w"))
                    repl[key + ".bias"] = (lambda g=get: g("b"))
        src = MemoryEfficientSafeOpen(src_path)
        specs = {}
        for key in src.keys():
            info = src.header[key]
            dt = src._get_torch_dtype(info["dtype"])
            shape = tuple(info["shape"])
            if key in repl:
                specs[key] = (dt, shape, (lambda f=repl[key], dt=dt, shape=shape: f().to(dt).reshape(shape)))
            else:
                specs[key] = (dt, shape, (lambda k=key: src.get_tensor(k)))
        meta = dict(src.metadata() or {})
        meta["fizgig_frankenstein"] = json.dumps({"base": os.path.basename(src_path),
                                                   "blended_with": os.path.basename(self.other_path),
                                                   "weights": {b: round(v, 4) for b, v in self.w.items() if v > 0},
                                                   **(metadata or {})})
        tmp = out_path + ".tmp"
        try:
            stream_save_file(specs, tmp, meta)
            os.replace(tmp, out_path)
        finally:
            src.file.close()
            if os.path.exists(tmp):
                os.remove(tmp)
        if log:
            log(f"[frankenstein] exported {os.path.basename(out_path)}: {len(repl)} tensors blended, "
                f"{len(specs) - len(repl)} copied")
        return out_path
