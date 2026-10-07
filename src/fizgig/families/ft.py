"""Full fine-tune for described families - opt-in per driver.

A family trains its whole base model by rotation: the base stays frozen in NF4, and one component window at a time
(e.g. every block's attention, then every block's MLP gate ...) is swapped up to trainable bf16 from a CPU bf16
master, trained, and written back. A full rotation trains every component once. Checkpoints (and the previews that
ride them) happen only at whole rotations, so a saved file never has some components trained more than others.

A driver opts in by returning an FTSpec from `ft_spec(dit)`; a driver that doesn't has no fine-tune and never meets
this module. Everything here is model-agnostic: the window schedule and planner (shared with Krea 2's and H3's
original trainers), the NF4 swap per Linear (the format families/quant.py writes), the master read from the model
file, streaming of out-of-window blocks through the model's own block-swap hooks, and the streamed full-checkpoint
save in the source file's own layout.

Decided (Peter, 30 Sep 2026): component windows only, an NF4 trunk only, biases frozen, resume = continue from a
saved checkpoint (no optimizer state; the optimizer is rebuilt every window anyway).
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from fizgig.krea2.rotation import (RotationSchedule, component_entry_matches, component_gb_per_block,
                                   plan_component_windows, snap_ft_epochs)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FTSpec:
    """What a driver declares to offer a full fine-tune.

    blocks: the DiT attribute holding the numbered blocks (an nn.ModuleList), or a tuple of them numbered one after
        another as one cycle (Klein: double_blocks then single_blocks; a component names Linears in whichever
        blocks have them).
    components: Linear-name prefixes within a block, in rotation order; each is one window spanning every block (the
        planner depth-splits a window that doesn't fit). Balance them by size - a window's bf16 weights, grads and
        optimizer state are what the card holds. A card that holds several parts at once trains them together (the
        fewest windows that fit; every part in one window and nothing rotates).
    always_on: dotted module names trained for the whole run (e.g. a text-fusion stack outside the blocks). Their
        Linears must be bf16 (outside the family's quant targets).
    overhead_gb: VRAM with the NF4 trunk resident plus activations and margin - the planner's base. None = the
        measured NF4 trunk + 3.5 GB.
    trunk_gb_per_block: NF4 trunk share per block (what streaming a block reclaims). None = measured.
    file_layout: where a weight lives in the model file when the loader renames or splits it:
        ((module suffix, file suffix, part, parts), ...) - e.g. ("img_mlp.gate_layer.weight", "img_mlp.gate_up.weight",
        0, 2): the Linear's weight is the first of 2 equal row-chunks of the file's gate_up tensor. The master reads
        that slice and the checkpoint writes the parts back as the file had them. A "diffusion_model." prefix in the
        file is found on its own.
    file_prefix: a prefix every DiT key in the model file carries besides that one (Anima's "net."), so weights are
        read from and written back under the file's own names. Keys without it (a checkpoint's text encoders and VAE)
        are not the model's.
    file_names: ((model prefix, file prefix), ...) where the file names a block differently from the loaded model
        (SDXL: diffusers' "down_blocks.1.attentions.0." is the checkpoint's "input_blocks.4.1.").
    """
    blocks: str = "blocks"
    components: tuple = ()
    always_on: tuple = ()
    overhead_gb: Optional[float] = None
    trunk_gb_per_block: Optional[float] = None
    slots_gb: float = 1.5
    file_layout: tuple = ()
    file_prefix: str = ""
    file_names: tuple = ()
    # streaming: what a step needs beyond its own resident blocks and window (activations, streamed blocks in flight,
    # fragmentation), in the planner's budget frame. None = derived from overhead_gb (conservative)
    stream_base_gb: Optional[float] = None
    # activation growth with resolution: overhead_gb and stream_base_gb are measured at calib_mp megapixels, and a
    # run's largest bucket above that adds act_gb_per_mp for each extra megapixel (4 GB/MP above 0.25 MP when the
    # driver has measured nothing - cautious)
    calib_mp: float = 0.25
    act_gb_per_mp: float = 4.0
    # what a window costs per GB of its bf16 weights: 1.0 = the weights alone (the calibrated overhead absorbs the
    # rest - every family before Z-Image); 2.0 = weights plus a gradient held for every trained weight at once, for a
    # model whose measured peaks grow by twice the window's weights (Z-Image Turbo: one window of all four parts peaked
    # 23.7 GB vs 10.5 for attention alone - base 3.4 + 2 x weights fits both, no single base under 1.0 does)
    window_factor: float = 1.0


_PREFIX = "diffusion_model."


def _lists(spec: FTSpec) -> tuple:
    """The spec's block lists in cycle order."""
    return (spec.blocks,) if isinstance(spec.blocks, str) else tuple(spec.blocks)


def _file_loc(spec: FTSpec, mkey: str, have) -> Optional[tuple]:
    """(file key, part, parts) of a model weight in a file whose keys are `have`, or None."""
    fkey, part, parts = mkey, 0, 1
    for msuf, fsuf, pt, n in spec.file_layout:
        if mkey.endswith(msuf):
            fkey, part, parts = mkey[:-len(msuf)] + fsuf, int(pt), int(n)
            break
    for mp, fp in spec.file_names:
        if fkey.startswith(mp):
            fkey = fp + fkey[len(mp):]
            break
    for k in (fkey, _PREFIX + fkey) + ((spec.file_prefix + fkey,) if spec.file_prefix else ()):
        if k in have:
            return k, part, parts
    return None


def _model_name(spec: FTSpec, fkey: str) -> str:
    """A file key (prefix already stripped) under the loaded model's name (FTSpec.file_names, reversed)."""
    for mp, fp in spec.file_names:
        if fkey.startswith(fp):
            return mp + fkey[len(fp):]
    return fkey


def _read(src, loc):
    """A master weight in the file's own precision: bf16, or fp16 / fp32 (most SDXL checkpoints are fp16), so a
    weight training leaves alone is written back exactly."""
    fkey, part, parts = loc
    t = src.get_tensor(fkey)
    if parts > 1:
        t = t.chunk(parts, dim=0)[part]
    return t if t.dtype in (torch.bfloat16, torch.float16, torch.float32) else t.to(torch.bfloat16)


def _master_bytes(header, fkey, parts=1):
    """Bytes the master of one file tensor takes (one part of a fused tensor): its own precision as _read keeps it,
    so fp32 counts 4 bytes a value - everything else is held at 2 (bf16 / fp16)."""
    n = int(torch.Size(header[fkey]["shape"]).numel()) // parts
    return n * (4 if header[fkey].get("dtype") == "F32" else 2)


def _merge(base: torch.Tensor, trained: torch.Tensor) -> torch.Tensor:
    """The master after training: `base` (the master, in the file's precision) plus what training changed in the bf16
    copy it trained - the whole of `trained` for a bf16 file, and for an fp16 / fp32 one the change alone, so the
    bf16 round trip's rounding never reaches the weights."""
    trained = trained.detach().to("cpu")
    if base.dtype == torch.bfloat16:
        return trained.to(torch.bfloat16).clone()
    return (base.float() + (trained.float() - base.to(torch.bfloat16).float())).to(base.dtype)


def source_unfit_reason(path: str) -> Optional[str]:
    """Why a model file cannot be fine-tuned (header read only), or None: a pre-quantised or fp8 file has no bf16
    layout to build the master from or write the checkpoint into."""
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
    with MemoryEfficientSafeOpen(path) as f:
        keys = f.keys()
        if any(k.endswith(".weight_scale") or k.endswith(".scale_weight") for k in keys):
            return "is a pre-quantized checkpoint (weight scale tensors)"
        low = [k for k in keys if str(f.header[k].get("dtype", "")).startswith(("F8", "I8", "U8"))]
    if low:
        return f"stores {len(low)} tensor(s) below 16 bits (e.g. {low[0]})"
    return None


class DiskMaster:
    """The bf16 master on disk instead of in RAM - for machines whose memory can't hold it (Krea 2's is ~24 GB).

    An untouched weight is read straight from the bf16 model file when it's needed (no build step, nothing held);
    only a weight that has trained is written, one raw bf16 file per weight plus a manifest, each replaced atomically
    (H3's MasterStore, generalised). Bytes are bytes, so the master stays exact. The rotation's access pattern is
    coarse - one window read in and written back per boundary - which sequential files serve far better than
    letting the OS page a RAM master. Dict surface: get / [] / []= / in / keys."""

    def __init__(self, src_path: str, loc: dict, scratch_dir: str):
        from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
        import json
        import shutil
        self.src, self.dir, self._loc = src_path, scratch_dir, dict(loc)
        self._keys = list(self._loc)
        # Every run builds its master from the model file. A scratch left here is a crashed run's partly trained
        # weights (a finished run deletes it once its checkpoint is written, and continuing starts from a checkpoint),
        # so a new run clears it rather than training on top of it.
        if os.path.isdir(scratch_dir) and os.listdir(scratch_dir):
            import logging
            logging.getLogger(__name__).warning(f"[finetune] clearing an earlier run's on-disk master at {scratch_dir} "
                                                f"(a run that stopped before its checkpoint) - this run starts from "
                                                f"{os.path.basename(src_path)}")
            shutil.rmtree(scratch_dir, ignore_errors=True)
        os.makedirs(scratch_dir, exist_ok=True)
        self._f = MemoryEfficientSafeOpen(src_path)
        self._shape = {}
        for k, (fk, part, parts) in self._loc.items():
            sh = list(self._f.header[fk]["shape"])
            sh[0] //= parts
            self._shape[k] = tuple(sh)
        need = sum(_master_bytes(self._f.header, fk, parts) for fk, _part, parts in self._loc.values())
        self.est_gb = need / 1e9
        free = shutil.disk_usage(scratch_dir).free
        if free < need * 1.1:
            raise RuntimeError(f"[finetune] the drive holding {scratch_dir} has {free / 1e9:.0f} GB free; the "
                               f"on-disk master needs up to ~{self.est_gb:.0f} GB - free space there, or run where "
                               f"system memory can hold it")
        self._manifest = os.path.join(scratch_dir, "manifest.json")
        self._trained = {}

    def keys(self):
        return list(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._shape

    def get(self, key, default=None):
        import numpy as np
        rec = self._trained.get(key)
        if rec is not None:
            fn, dt = rec
            dt = getattr(torch, dt)
            arr = np.fromfile(os.path.join(self.dir, fn), dtype=np.uint32 if dt == torch.float32 else np.uint16)
            return torch.from_numpy(arr).view(dt).reshape(self._shape[key])
        if key in self._shape:
            return _read(self._f, self._loc[key])
        return default

    def __getitem__(self, key):
        t = self.get(key)
        if t is None:
            raise KeyError(key)
        return t

    def __setitem__(self, key, t):
        import hashlib
        import json
        if key not in self._shape:
            raise KeyError(f"not a master weight: {key}")
        fn = hashlib.sha1(key.encode()).hexdigest()[:16] + ".bin"
        tmp = os.path.join(self.dir, fn + ".tmp")
        t = t.detach().to("cpu")
        if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            t = t.to(torch.bfloat16)
        t.contiguous().view(torch.int32 if t.dtype == torch.float32 else torch.int16).numpy().tofile(tmp)
        os.replace(tmp, os.path.join(self.dir, fn))          # the old bytes stay valid until this instant
        self._trained[key] = [fn, str(t.dtype).replace("torch.", "")]
        mt = self._manifest + ".tmp"
        with open(mt, "w", encoding="utf-8") as f:
            json.dump({"source": os.path.basename(self.src), "trained": self._trained}, f)
        os.replace(mt, self._manifest)

    def cleanup(self):
        """Delete the scratch - only once a checkpoint holding its contents is safely written."""
        import shutil
        try:
            self._f.file.close()
        except Exception:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)


class _TrainedView:
    """The checkpoint's trained tensors BY FILE KEY, produced one at a time as the save writes them: the master (RAM
    or disk) with the live GPU weights (active window, always-on) laid over it - never the whole master in RAM at
    once - and a file tensor the model splits (e.g. a fused gate_up) put back together from its parts."""

    def __init__(self, master, live, loc, src_path):
        self._m, self._live, self._src = master, live, src_path     # live: model key -> Linear with a current .weight
        self._loc, self._srcf = loc, None
        self._parts = {}                                             # file key -> {part: model key}, and its count
        for mk, (fk, part, parts) in loc.items():
            self._parts.setdefault(fk, [parts, {}])[1][part] = mk

    def keys(self):
        return list(self._parts)

    def __iter__(self):
        return iter(self._parts)

    def __contains__(self, key):
        return key in self._parts

    def _value(self, mk):
        lin = self._live.get(mk)
        if lin is not None:
            return _merge(self._base(mk), lin.weight)
        return self._m[mk]

    def _base(self, mk):
        """The weight before training: the master's, or for an always-on Linear (no master copy) the model file's."""
        if mk in self._m:
            return self._m[mk]
        if self._srcf is None:
            from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
            self._srcf = MemoryEfficientSafeOpen(self._src)
        return _read(self._srcf, self._loc[mk])

    def __getitem__(self, fkey):
        parts, have = self._parts[fkey]
        if parts == 1:
            return self._value(have[0])
        out = []
        for i in range(parts):
            if i in have:
                out.append(self._value(have[i]))
            else:                                   # a part nothing trains: the file's own slice
                from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
                with MemoryEfficientSafeOpen(self._src) as f:
                    out.append(_read(f, (fkey, i, parts)))
        return torch.cat(out, dim=0)


def _unwrapped(name: str) -> str:
    """A Linear's name as the model file knows it: FamilyLoRA wraps each target and keeps the real Linear as `.base`."""
    if name == "base":                     # the wrapped module itself (an always-on Linear named on its own)
        return ""
    return name[:-len(".base")] if name.endswith(".base") else name


def model_linears(module: nn.Module):
    """(name as the model file knows it, Linear) for the model's own Linears - never an adapter's (FamilyLoRA keeps
    its adapters under `.adapters.`, and their factors are Linears too)."""
    for name, m in module.named_modules():
        if isinstance(m, nn.Linear) and ".adapters." not in f".{name}.":
            yield _unwrapped(name), m


class Rotator:
    """Swaps component windows between NF4-frozen and bf16-trainable, in place, on the real Linears.

    `master` (key -> CPU bf16) is the source of truth: a window activates FROM it and writes back TO it, so training
    never round-trips through NF4. FamilyLoRA's wrappers hold these Linears as `.base` and call them, so adapters
    (training adapter, context LoRA, speed LoRA) keep working while the weights underneath change.
    """

    def __init__(self, dit, spec: FTSpec, device):
        self.dit = dit
        self.spec = spec
        self.device = torch.device(device)
        # the blocks in cycle order and where each lives ((list name, index in it)); one list is its ModuleList
        self._where = [(name, li) for name in _lists(spec) for li in range(len(dit.get_submodule(name)))]
        self.blocks = (dit.get_submodule(spec.blocks) if isinstance(spec.blocks, str)
                       else [dit.get_submodule(name)[li] for name, li in self._where])
        # every NF4 Linear in the blocks under a component prefix: (model key, linear, block index, name in block),
        # found once while everything is still frozen
        self.targets = []
        for bi, block in enumerate(self.blocks):
            name, li = self._where[bi]
            for lname, m in model_linears(block):
                if not getattr(m, "_is_nf4", False):
                    continue
                if any(lname.startswith(p) for p in spec.components):
                    self.targets.append((f"{name}.{li}.{lname}.weight", m, bi, lname))
        self.always = []                      # (model key, linear): dense bf16, trainable all run
        for mod_name in spec.always_on:
            mod = dit.get_submodule(mod_name)
            for lname, m in model_linears(mod):
                if getattr(m, "_is_nf4", False) or getattr(m, "_is_int8", False):
                    raise RuntimeError(f"[finetune] always-on module {mod_name}.{lname} is quantised - always-on "
                                       f"modules must stay bf16 (outside the family's quant targets)")
                self.always.append((f"{mod_name}.{lname}.weight" if lname else f"{mod_name}.weight", m))
        self.master: Dict[str, torch.Tensor] = {}
        self.loc: Dict[str, tuple] = {}       # model key -> (file key, part, parts)
        self.src = None
        self.active: List = []
        self._forward = {}

    # ---- master ---------------------------------------------------------------------------------------------------
    def build_master(self, path: str, where: str = "auto", scratch_dir: Optional[str] = None):
        """The bf16 master of every rotating weight, read from the model file (bf16 on disk) - never dequantised
        from the GPU copy, which has been through NF4. `where`: "ram" (read now, one tensor at a time), "disk"
        (DiskMaster in scratch_dir), or "auto" - disk when the master would take more than 40% of the free system
        memory. Returns (GB, "ram" | "disk")."""
        from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
        self.src = path
        with MemoryEfficientSafeOpen(path) as src:
            have = set(src.keys())
            missing = []
            for k in [k for k, *_ in self.targets] + [k for k, _ in self.always]:
                loc = _file_loc(self.spec, k, have)
                if loc is None:
                    missing.append(k)
                else:
                    self.loc[k] = loc
            gb = sum(_master_bytes(src.header, fk, n)
                     for k, (fk, _p, n) in self.loc.items() if k in {t[0] for t in self.targets}) / 1e9
        if missing:
            raise RuntimeError(f"[finetune] {len(missing)} weights are not in {os.path.basename(path)} under the "
                               f"expected names, e.g. {missing[:3]} (the driver's FTSpec.file_layout maps them)")
        if where == "auto":
            try:
                import psutil
                where = "disk" if gb > 0.4 * psutil.virtual_memory().available / 1e9 else "ram"
            except Exception:
                where = "ram"
        if where == "disk":
            self.master = DiskMaster(path, {k: self.loc[k] for k, *_ in self.targets}, scratch_dir)
            return gb, "disk"
        with MemoryEfficientSafeOpen(path) as src:
            for key, *_ in self.targets:
                self.master[key] = _read(src, self.loc[key]).to("cpu").clone()
        gc.collect()
        return gb, "ram"

    # ---- windows --------------------------------------------------------------------------------------------------
    def _window(self, spec) -> list:
        return [(k, m) for k, m, bi, ln in self.targets if any(component_entry_matches(c, ln, bi) for c in spec)]

    def resident_blocks(self, spec) -> set:
        """Blocks holding trainable Linears under a window (the streamer's resident set)."""
        return {bi for _k, _m, bi, ln in self.targets if any(component_entry_matches(e, ln, bi) for e in spec)}

    def _activate(self, pairs):
        for key, lin in pairs:
            self._forward[id(lin)] = lin.__dict__.pop("forward", None)    # the NF4 forward, restored verbatim
            lin.weight = nn.Parameter(self.master[key].to(self.device, dtype=torch.bfloat16), requires_grad=True)
            lin._nf4_packed = lin._nf4_state = None     # pure duplication while it trains
        return len(pairs)

    def _deactivate(self, pairs):
        from bitsandbytes.functional import quantize_nf4
        from fizgig.modules.nf4 import nf4_linear_forward_patch
        for key, lin in pairs:
            # the gradient first: the last step's autograd graph keeps this Parameter alive past the swap, and with
            # it a window-sized .grad (~6 GB on Krea 2's MLP windows) that freeing the weight storage does not free
            lin.weight.grad = None
            trained = lin.weight.detach()
            self.master[key] = _merge(self.master[key], trained)                 # before the lossy re-encode
            packed, state = quantize_nf4(trained.contiguous())
            lin._nf4_packed, lin._nf4_state = packed, state
            lin.weight = nn.Parameter(torch.empty(0, device=packed.device, dtype=torch.bfloat16), requires_grad=False)
            saved = self._forward.pop(id(lin), None)
            lin.forward = saved if saved is not None else nf4_linear_forward_patch.__get__(lin, type(lin))
            # release the orphan: a C++-side autograd referrer keeps the old bf16 storage alive otherwise, and a
            # rotation would hold two windows (measured on H3 and Krea 2). Guarded: never free a storage still read.
            orphan = trained.untyped_storage()
            del trained
            if all(t is None or t.numel() == 0 or t.untyped_storage().data_ptr() != orphan.data_ptr()
                   for t in (lin.weight, lin._nf4_packed)):
                try:
                    orphan.resize_(0)
                except Exception:
                    pass
        return len(pairs)

    def rotate_to(self, spec) -> int:
        spec = list(spec)
        if spec == self.active:
            return 0
        if self.active:
            self._deactivate(self._window(self.active))
            gc.collect()
            torch.cuda.empty_cache()          # the outgoing window back to the allocator before the next one lands
        n = self._activate(self._window(spec)) if spec else 0
        self.active = spec
        if not spec:
            for _key, lin in self.always:     # parked: no gradient outlives the window it came from
                lin.weight.grad = None
        return n

    def start_always(self) -> int:
        for _key, lin in self.always:
            lin.weight.requires_grad_(True)   # weights only: biases stay frozen
        return len(self.always)

    def trainable_params(self) -> List[nn.Parameter]:
        return [lin.weight for _k, lin in self._window(self.active)] + [lin.weight for _k, lin in self.always]

    def state_dict(self):
        """Every trained weight, produced as the save writes it: the master with the active window and the
        always-on Linears laid over it."""
        return _TrainedView(self.master, dict(self._window(self.active)) | {k: lin for k, lin in self.always},
                            self.loc, self.src)


def _plan(comp_gb, n_blocks, trunk, spec: FTSpec, usable, allow_stream=True, mp=None, spans=None, max_parts=0):
    if spec.window_factor != 1.0:      # a family whose windows cost more than their weights (FTSpec.window_factor)
        comp_gb = {p: g * spec.window_factor for p, g in comp_gb.items()}
    overhead = spec.overhead_gb if spec.overhead_gb is not None else trunk * n_blocks + 3.5
    extra = spec.act_gb_per_mp * max(0.0, float(mp or spec.calib_mp) - spec.calib_mp)
    overhead += extra                  # a bigger bucket's activations, resident and streaming alike
    # the shared planner's streaming base is overhead - trunk + slots; a measured stream_base_gb sets it directly
    slots = (spec.slots_gb if spec.stream_base_gb is None
             else spec.stream_base_gb + extra - (overhead - trunk * n_blocks))
    return plan_component_windows(usable, range(n_blocks), n_blocks, comp_gb, overhead_gb=overhead,
                                  trunk_gb_per_block=trunk, slots_gb=slots, allow_stream=allow_stream, spans=spans,
                                  max_parts=max_parts)


def plan_windows(dit, spec: FTSpec, rotator: Rotator, free_gb: float, allow_stream: bool = True, mp=None,
                 max_parts=0):
    """(windows, stream, reasons, usable GB) for this card, with the model loaded: component sizes measured from it,
    `free_gb` read after the NF4 trunk landed (so the trunk is added back into the budget)."""
    comp_gb = {p: 0.0 for p in spec.components}
    if isinstance(spec.blocks, str):
        block0 = dit.get_submodule(spec.blocks)[0]
        for ln, m in model_linears(block0):
            for p in spec.components:
                if ln.startswith(p):
                    comp_gb[p] += m.out_features * m.in_features * 2 / 1e9     # logical size: NF4 empties .weight
                    break
        n = len(dit.get_submodule(spec.blocks))
        spans = None
    else:
        # blocks that differ (Klein's double and single): each part sized per block of the blocks that hold it, and
        # split only across those
        n = len(rotator.blocks)
        spans = {p: set() for p in spec.components}
        for bi, blk in enumerate(rotator.blocks):
            for ln, m in model_linears(blk):
                for p in spec.components:
                    if ln.startswith(p):
                        comp_gb[p] += m.out_features * m.in_features * 2 / 1e9
                        spans[p].add(bi)
                        break
        comp_gb = {p: v / max(1, len(spans[p])) for p, v in comp_gb.items()}
    if spec.trunk_gb_per_block is not None:
        trunk = float(spec.trunk_gb_per_block)
    else:
        packed = sum(lin._nf4_packed.numel() for _k, lin, _b, _l in rotator.targets
                     if getattr(lin, "_nf4_packed", None) is not None)
        trunk = packed / 1e9 / max(1, n)
    usable = free_gb + trunk * n - 1.5
    windows, stream, why = _plan(comp_gb, n, trunk, spec, usable, allow_stream, mp, spans=spans, max_parts=max_parts)
    return windows, stream, why, usable


def plan_from_file(path: str, spec: FTSpec, free_gb: float, mp=None, max_parts=0):
    """The same plan before anything loads (the Training tab's "on this card" line): component sizes from the model
    file's header, `free_gb` read on the idle card. The trainer budgets after its model and preview VAE are in, so
    the non-block layers (kept bf16) and ~1.05 GB (VAE + runtime; measured on Krea 2 and Qwen 2.1) come off here. Returns (windows, stream, usable) or
    None when the file does not show the spec's blocks."""
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
    import re
    with MemoryEfficientSafeOpen(path) as f:
        pre = tuple(p for p in (_PREFIX, spec.file_prefix) if p)
        keys = [k for k in f.keys() if not spec.file_prefix or k.startswith(pre)]
        hdr = {_model_name(spec, next((k[len(p):] for p in pre if k.startswith(p)), k)): f.header[k] for k in keys}
    if not isinstance(spec.blocks, str):
        return _plan_from_header_lists(hdr, spec, free_gb, mp, max_parts)
    pat = re.compile(rf"^{re.escape(spec.blocks)}\.(\d+)\.(.+)$")
    blocks, comp_gb, block0_params, nonblock = set(), {p: 0.0 for p in spec.components}, 0, 0.0
    for k, info in hdr.items():
        shape = info.get("shape") or []
        numel = 1
        for d in shape:
            numel *= int(d)
        m = pat.match(k)
        if not m:
            nonblock += numel * 2 / 1e9
            continue
        blocks.add(int(m.group(1)))
        if m.group(1) != "0" or len(shape) != 2 or not k.endswith(".weight"):
            continue
        block0_params += numel
        rest = m.group(2)
        names = [(rest[:-len(fsuf)] + msuf, numel / n) for msuf, fsuf, _pt, n in spec.file_layout
                 if rest.endswith(fsuf)] or [(rest, numel)]
        for name, n_el in names:
            for p in spec.components:
                if name.startswith(p):
                    comp_gb[p] += n_el * 2 / 1e9
                    break
    if not blocks or not all(comp_gb.values()):
        return None
    n = len(blocks)
    trunk = float(spec.trunk_gb_per_block) if spec.trunk_gb_per_block is not None else block0_params * 0.53 / 1e9
    usable = free_gb - nonblock - 1.05 - 1.5
    windows, stream, _why = _plan(comp_gb, n, trunk, spec, usable, mp=mp, max_parts=max_parts)
    return windows, stream, usable


def _plan_from_header_lists(hdr, spec: FTSpec, free_gb: float, mp=None, max_parts=0):
    """plan_from_file for several block lists: each part sized per block of the blocks that hold it, numbered as
    one cycle (the lists end to end)."""
    import re
    pats = [re.compile(rf"^{re.escape(name)}\.(\d+)\.(.+)$") for name in _lists(spec)]
    seen, comp_gb, params, nonblock = set(), {p: 0.0 for p in spec.components}, 0, 0.0
    held = {p: set() for p in spec.components}          # (list, block) pairs holding the part
    for k, info in hdr.items():
        shape = info.get("shape") or []
        numel = 1
        for d in shape:
            numel *= int(d)
        hit = None
        for li, pat in enumerate(pats):
            m = pat.match(k)
            if m:
                hit = (li, m)
                break
        if hit is None:
            nonblock += numel * 2 / 1e9
            continue
        li, m = hit
        seen.add((li, int(m.group(1))))
        if len(shape) != 2 or not k.endswith(".weight"):
            continue
        params += numel
        rest = m.group(2)
        names = [(rest[:-len(fsuf)] + msuf, numel / n) for msuf, fsuf, _pt, n in spec.file_layout
                 if rest.endswith(fsuf)] or [(rest, numel)]
        for name, n_el in names:
            for p in spec.components:
                if name.startswith(p):
                    comp_gb[p] += n_el * 2 / 1e9
                    held[p].add((li, int(m.group(1))))
                    break
    n = len(seen)
    if not n or not all(comp_gb.values()):
        return None
    sizes = [1 + max((b for l, b in seen if l == li), default=-1) for li in range(len(pats))]
    start = [sum(sizes[:li]) for li in range(len(pats))]
    spans = {p: {start[li] + b for li, b in held[p]} for p in spec.components}
    comp_gb = {p: v / len(spans[p]) for p, v in comp_gb.items()}
    trunk = float(spec.trunk_gb_per_block) if spec.trunk_gb_per_block is not None else params * 0.53 / 1e9 / n
    usable = free_gb - nonblock - 1.05 - 1.5
    windows, stream, _why = _plan(comp_gb, n, trunk, spec, usable, mp=mp, spans=spans, max_parts=max_parts)
    return windows, stream, usable


def make_optimizer(params, lr):
    """Adafactor first (factored state ~10x smaller than Adam's - what keeps a full fine-tune on the card), then
    AdamW8bit, then AdamW. Returns (optimizer, label)."""
    try:
        from transformers.optimization import Adafactor
        return Adafactor(params, lr=lr, scale_parameter=False, relative_step=False, warmup_init=False), "adafactor"
    except Exception:
        pass
    try:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=lr), "adamw8bit"
    except Exception:
        return torch.optim.AdamW(params, lr=lr), "adamw"


class FusedSteps:
    """Optimizer in backward: one optimizer per parameter, stepped from its grad hook and the grad freed at once,
    so only one parameter's gradient is ever live (~5 GB saved on Krea 2). No clipping, no accumulation."""

    def __init__(self, lr):
        self.lr = lr
        self.opts = {}
        self.handles = []

    def attach(self, params):
        self.detach()
        for p in params:
            self.opts[p] = make_optimizer([p], self.lr)[0]

        def _hook(param):
            opt = self.opts.get(param)
            if opt is not None:
                opt.step()
                opt.zero_grad(set_to_none=True)
        for p in params:
            self.handles.append(p.register_post_accumulate_grad_hook(_hook))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        self.opts.clear()


def save_checkpoint(state: Dict[str, torch.Tensor], src_path: str, path: str, metadata: dict):
    """The fine-tuned model: the source file with the trained tensors replaced, in the source's own tensor order,
    streamed one tensor at a time (peak RAM ~ the master + one tensor, #143) to a .tmp renamed on completion (an
    interrupted save never leaves a truncated file under the real name)."""
    from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen, stream_save_file
    with MemoryEfficientSafeOpen(src_path) as src:
        header = {k: src.header[k] for k in src.keys()}
    extra = [k for k in state if k not in header]
    if extra:
        logger.warning("[finetune] %d trained tensor(s) have no slot in the source file and are NOT saved, e.g. %s",
                       len(extra), extra[:3])
    _dt = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
           "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8,
           "BOOL": torch.bool}
    reader = {"f": None}

    def producer(key):
        info = header[key]
        dt, shape = _dt[info["dtype"]], tuple(info["shape"])
        if key in state:
            return dt, shape, (lambda key=key, dt=dt, shape=shape: state[key].to(dt).reshape(shape))
        return dt, shape, (lambda key=key: reader["f"].get_tensor(key))

    specs = {k: producer(k) for k in header}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    try:
        reader["f"] = MemoryEfficientSafeOpen(src_path)
        try:
            stream_save_file(specs, tmp, metadata={str(k): str(v) for k, v in metadata.items()})
        finally:
            reader["f"].file.close()
        os.replace(tmp, path)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    del specs
    gc.collect()
    return sum(1 for k in state if k in header), len(header)


class SharedBackend:
    """The fine-tune's model-side pieces for a family whose base is a bf16 file quantised to NF4 at load (Krea 2,
    Qwen): the Rotator and its master, the window planner, the block streamer, the streamed checkpoint. A driver
    whose base differs (H3's int8 ConvRot file, its own 4-bit trunk and ring) brings its own backend with the same
    surface (driver.ft_backend)."""

    always_label = None

    def __init__(self, driver, dit, device, src, group=None):
        self.driver, self.dit, self.device, self.src = driver, dit, torch.device(device), src
        self.spec = driver.ft_spec(dit)
        if self.spec is None:
            raise RuntimeError(f"{driver.description.display_name}'s driver declares no fine-tune (ft_spec)")
        self.rot = Rotator(dit, self.spec, device)
        self.streamer = None
        self.always_label = ", ".join(self.spec.always_on) or "none"

    def build_master(self, where, scratch_dir):
        return self.rot.build_master(self.src, where, scratch_dir)

    def start_always(self):
        return self.rot.start_always()

    def plan(self, free_gb, allow_stream=True, mp=None, max_parts=0):
        windows, stream, why, usable = plan_windows(self.dit, self.spec, self.rot, free_gb, allow_stream, mp,
                                                    max_parts)
        if windows is not None and stream and any(isinstance(w, tuple) for w in windows):
            from fizgig.krea2.rotation import RotationOffloader
            n = len(self.rot.blocks)
            self.streamer = RotationOffloader(self.rot.blocks, self.device, range(n))
            self.driver.install_ft_streamer(self.dit, self.streamer)
            logger.info("[finetune] frozen blocks outside the window stream from CPU")
        return windows, stream, why, usable

    def cycle_len(self):
        return len(self.rot.blocks)

    def rotate(self, want):
        if self.streamer is not None:
            self.streamer.set_resident(self.rot.resident_blocks(want))
        return self.rot.rotate_to(want)

    def trainable_params(self):
        return self.rot.trainable_params()

    def park(self):
        self.rot.rotate_to([])

    @property
    def active(self):
        return self.rot.active

    @property
    def master(self):
        return self.rot.master

    def params_in_blocks(self, block_ids):
        want = set(block_ids)
        return [lin.weight for k, lin, bi, ln in self.rot.targets
                if self.driver.block_of(k[:-len(".weight")]) in want and lin.weight.requires_grad]

    def save(self, path, meta):
        return save_checkpoint(self.rot.state_dict(), self.src, path, meta)

    def cleanup(self):
        if hasattr(self.rot.master, "cleanup"):
            self.rot.master.cleanup()


def schedule(windows, n_blocks, rotate_every=1, start_window=0) -> RotationSchedule:
    return RotationSchedule(n_blocks, mode="component", components=tuple(windows), rotate_every=rotate_every,
                            start_window=start_window)


__all__ = ["FTSpec", "Rotator", "SharedBackend", "FusedSteps", "make_optimizer", "plan_windows", "save_checkpoint", "schedule",
           "snap_ft_epochs", "source_unfit_reason", "component_gb_per_block"]
