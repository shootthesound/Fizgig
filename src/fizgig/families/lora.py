"""The standard layer's LoRA: wraps a family's Linears, trains one adapter and runs any number of frozen ones.

Which Linears (driver.block_map / lora_target_names) and how files are keyed (description.lora: file prefix, down/up
names, alpha key) come from the family, so every described family gets the same adapter machinery and writes its own
ComfyUI-compatible format.

Each wrapped Linear computes W x + sum_i s_i * B_i(A_i(x)). For a frozen adapter
    s = alpha / rank * load_strength * block_strength * (adapter on) * (block on)
where block_strength / block on belong to the block the module is in (driver.block_of); modules outside the block
map (e.g. a speed LoRA's modulation layers) follow the adapter's load strength and on/off only. The trainable adapter
is named "lora"; frozen ones (training adapter, context LoRA, a workbench primary/donor) get their own names.

Files in any common layout are accepted: the family's own keys, kohya (`lora_unet_<flattened>.lora_down/up`), or
PEFT / diffusers (`lora_A/lora_B` or `lora_down/lora_up`, bare or under `transformer.` / `diffusion_model.`).
LyCORIS (LoKR / LoHa) is not handled by the standard layer yet and is refused with a clear message.
"""
import math
import re

import torch
import torch.nn as nn

TRAINABLE = "lora"
_PREFIXES = ("transformer.", "unet.", "diffusion_model.", "model.diffusion_model.", "base_model.model.", "")


class LoRAFactor(nn.Linear):
    """One adapter matrix (A or B). Its own class so block-swap offloaders, which stream every module whose class name
    ends in "Linear", leave the adapters resident: trainable weights must never be moved between steps."""


class LoKR(nn.Module):
    """A LoKR (Kronecker) adapter: delta = scale * kron(w1, w2), applied without materialising it (Krea 2's trainable
    recipe: full-matrix w2, w1 about factor x factor). Frozen files may carry w1/w2 as low-rank factors; they are
    multiplied out at load."""

    def __init__(self, in_f, out_f, factor=8, w1=None, w2=None):
        super().__init__()
        from fizgig.networks.lora import factorization
        if w1 is not None:
            self.a, self.b = w1.shape
            self.c, self.d = w2.shape
        else:
            self.a, self.c = factorization(out_f, int(factor))
            self.b, self.d = factorization(in_f, int(factor))
        if self.a * self.c != out_f or self.b * self.d != in_f:
            raise ValueError(f"LoKR factors {self.a}x{self.b} (x) {self.c}x{self.d} do not tile {out_f}x{in_f}")
        self.lokr_w1 = nn.Parameter(torch.empty(self.a, self.b) if w1 is None else w1.clone())
        self.lokr_w2 = nn.Parameter(torch.zeros(self.c, self.d) if w2 is None else w2.clone())
        if w1 is None:
            nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))   # w2 = 0: delta is exactly 0 at step 0

    def forward(self, x):
        """Computed in the input's dtype (bf16 in the DiT): the Kronecker product runs as thousands of small batched
        matmuls, which in fp32 made LoKR 3.7x slower than LoRA on Qwen (7.46 vs 2.04 s/step). The weights stay fp32
        for the optimizer; autograd casts the gradients back."""
        from fizgig.networks.lora import _lokr_forward_update
        return _lokr_forward_update(x, self.lokr_w1.to(x.dtype), self.lokr_w2.to(x.dtype), self.a, self.b, self.c,
                                    self.d)

    def delta(self):
        return torch.kron(self.lokr_w1.float(), self.lokr_w2.float())


class LoHa(nn.Module):
    """A frozen LoHa (Hadamard) adapter: delta = (w1_a @ w1_b) * (w2_a @ w2_b), materialised per forward as the old
    loaders' LoHaInfModule does (the Hadamard product does not factor); the scale is applied by the LoRALinear."""

    def __init__(self, w1a, w1b, w2a, w2b):
        super().__init__()
        self.hada_w1_a, self.hada_w1_b = nn.Parameter(w1a.clone()), nn.Parameter(w1b.clone())
        self.hada_w2_a, self.hada_w2_b = nn.Parameter(w2a.clone()), nn.Parameter(w2b.clone())

    def delta(self):
        return (self.hada_w1_a.float() @ self.hada_w1_b.float()) * (self.hada_w2_a.float() @ self.hada_w2_b.float())

    def forward(self, x):
        w = (self.hada_w1_a @ self.hada_w1_b) * (self.hada_w2_a @ self.hada_w2_b)
        return x @ w.to(x.dtype).transpose(-1, -2)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.scales = {}

    def _base_device(self):
        """Where the base weight lives, read off its storage: a ConvRot int8 Linear's .weight decodes the whole
        matrix."""
        q = getattr(self.base, "qdata", None)
        return q.device if q is not None else self.base.weight.device

    def add(self, name, rank, alpha, trainable, A=None, B=None, strength=1.0, trainable_dtype=torch.float32):
        a = LoRAFactor(self.base.in_features, rank, bias=False)
        b = LoRAFactor(rank, self.base.out_features, bias=False)
        if A is not None:
            a.weight.data.copy_(A)
            b.weight.data.copy_(B)
        else:
            nn.init.kaiming_uniform_(a.weight, a=math.sqrt(5))
            nn.init.zeros_(b.weight)
        dev = getattr(self, "home", None) or self._base_device()   # a swapped block's base may sit on CPU
        dt = trainable_dtype if trainable else torch.bfloat16
        a.to(dev, dt).requires_grad_(trainable)
        b.to(dev, dt).requires_grad_(trainable)
        self.adapters[name] = nn.Sequential(a, b)
        self.scales[name] = alpha / rank * strength

    def add_lokr(self, name, trainable, factor=8, w1=None, w2=None, scale=1.0, trainable_dtype=torch.float32):
        ad = LoKR(self.base.in_features, self.base.out_features, factor, w1, w2)
        dev = getattr(self, "home", None) or self._base_device()
        ad.to(dev, trainable_dtype if trainable else torch.bfloat16).requires_grad_(trainable)
        self.adapters[name] = ad
        self.scales[name] = scale

    def add_loha(self, name, w1a, w1b, w2a, w2b):
        ad = LoHa(w1a, w1b, w2a, w2b)
        dev = getattr(self, "home", None) or self._base_device()
        ad.to(dev, torch.bfloat16).requires_grad_(False)
        self.adapters[name] = ad
        self.scales[name] = 1.0

    def forward(self, x):
        out = self.base(x)
        for n, ad in self.adapters.items():
            s = self.scales.get(n, 0.0)
            if s:
                if isinstance(ad, (LoKR, LoHa)):
                    out = out + (s * ad(x)).to(out.dtype)
                    continue
                lx = ad(x.to(ad[0].weight.dtype))
                if lx.dtype == out.dtype:
                    # a frozen adapter in the model's dtype: ONE fused add, the strength formed in fp32 and the sum
                    # rounded once - the old loaders' LoRAInfModule epilogue, so a preview at strength 0.75 matches them
                    out = torch.add(out, lx, alpha=float(s))
                else:                                # the fp32 trainable adapter: as before
                    out = out + (s * lx).to(out.dtype)
        return out


class LoRAConv(nn.Module):
    """A frozen LoRA on a Conv2d (LoCon, and speed LoRAs such as LCM / Lightning that also patch a UNet's resnets): down
    = a conv with the base's kernel, stride and padding to `rank` channels, up = a 1x1 conv back, kohya's layout. The
    same adapters / scales interface as LoRALinear, so loading, strengths, block switches and baking treat it alike.
    Never trained: training targets the block map's Linears."""

    def __init__(self, base: nn.Conv2d):
        super().__init__()
        self.base = base
        self.adapters = nn.ModuleDict()
        self.scales = {}

    @property
    def in_features(self):
        return self.base.in_channels

    @property
    def out_features(self):
        return self.base.out_channels

    def add(self, name, rank, alpha, trainable, A=None, B=None, strength=1.0, trainable_dtype=torch.float32):
        if trainable or A is None:
            raise ValueError("a conv LoRA is frozen-only (loaded from a file)")
        bc = self.base
        down = nn.Conv2d(bc.in_channels, rank, tuple(A.shape[2:]) if A.dim() == 4 else 1, stride=bc.stride,
                         padding=bc.padding if A.dim() == 4 and tuple(A.shape[2:]) == tuple(bc.kernel_size) else 0,
                         dilation=bc.dilation, bias=False)
        up = nn.Conv2d(rank, bc.out_channels, 1, bias=False)
        down.weight.data.copy_(A.reshape(down.weight.shape))
        up.weight.data.copy_(B.reshape(up.weight.shape))
        dev = getattr(self, "home", None) or bc.weight.device
        ad = nn.Sequential(down, up).to(dev, torch.bfloat16).requires_grad_(False)
        self.adapters[name] = ad
        self.scales[name] = alpha / rank * strength

    def forward(self, x):
        out = self.base(x)
        for n, ad in self.adapters.items():
            s = self.scales.get(n, 0.0)
            if s:
                out = torch.add(out, ad(x.to(ad[0].weight.dtype)).to(out.dtype), alpha=float(s))
        return out


class FamilyLoRA:
    """A DiT's adapter set. `wrapped` maps the module name (relative to the DiT) -> LoRALinear."""

    def __init__(self, dit, driver, device=None):
        """device: where adapters live. Needed when block swap has parked some base weights on the CPU; by default
        each adapter follows its base weight."""
        self.dit = dit
        self.driver = driver
        self.device = torch.device(device) if device is not None else None
        self.desc = driver.description
        self.targets = set(driver.lora_target_names(dit))
        # every module a LoRA file can adapt: Linears, and Conv2d for frozen files (LoCon / UNet speed LoRAs)
        self.linears = {n for n, m in dit.named_modules() if isinstance(m, (nn.Linear, nn.Conv2d))}
        self._flat = {n.replace(".", "_"): n for n in self.linears}
        self.wrapped = {}
        for full in sorted(self.targets):
            self._wrap(full)
        if not self.wrapped:
            raise RuntimeError(f"{self.desc.display_name}: none of the driver's LoRA targets exist in this model")
        # per frozen adapter: {"alpha_rank": {module: alpha/rank}, "load": float, "on": bool,
        #                      "block_mult": {block_id: float}, "block_on": {block_id: bool}}
        self._frozen = {}

    def _keys(self, full):
        """(down key, up key, alpha key) of any wrapped module, in the family's file format. Built from the module
        path alone, so modules outside the blocks never need a block id."""
        f = self.desc.lora
        stem = self._stem(full)
        return f"{stem}.{f.down}.weight", f"{stem}.{f.up}.weight", f.alpha_key.format(prefix=stem)

    def _stem(self, full, lokr=False):
        """A module's key stem in the family's format: '<file_prefix><dotted path>', or for a kohya family
        'lora_unet_<path with dots as underscores>' (a LoKR there uses the LyCORIS standard 'diffusion_model.<path>',
        as Fizgig's Krea 2 always saved it)."""
        f = self.desc.lora
        if f.kohya:
            return f"diffusion_model.{full}" if lokr and not f.lokr_kohya_stems else f"lora_unet_{full.replace('.', '_')}"
        return f"{f.file_prefix}{full}"

    def _wrap(self, full):
        """Wrap one Linear by dotted name (targets at init; frozen files may reach beyond them, e.g. a speed LoRA
        that also patches the modulation / timestep layers). Returns the LoRALinear or None."""
        if full in self.wrapped:
            return self.wrapped[full]
        parent_name, _, leaf = full.rpartition(".")
        parent = self.dit.get_submodule(parent_name) if parent_name else self.dit
        child = getattr(parent, leaf, None)
        if isinstance(child, nn.Conv2d):
            w = LoRAConv(child)
        elif isinstance(child, nn.Linear):
            w = LoRALinear(child)
        else:
            return None
        if self.device is not None:
            w.home = self.device
        setattr(parent, leaf, w)
        self.wrapped[full] = w
        return w

    # ---- trainable ------------------------------------------------------------------------------
    def add_trainable(self, rank, alpha, blocks=None, kind="lora", factor=8):
        """blocks: optional set of block ids to train (None = every target). kind "lokr": a Kronecker adapter per
        Linear (w1 about factor x factor, full w2); rank / alpha do not apply to it. The adapter is fp32 unless the
        description's trainable_dtype says otherwise (H3's old trainer trained its LoRA in bf16)."""
        tdt = {"bf16": torch.bfloat16}.get(getattr(self.desc, "trainable_dtype", "fp32"), torch.float32)
        for full, w in self.wrapped.items():
            if full not in self.targets:
                continue                    # extra Linears wrapped for a frozen file are never trained
            if blocks is None or self.driver.block_of(full) in blocks:
                if kind == "lokr":
                    w.add_lokr(TRAINABLE, True, factor, trainable_dtype=tdt)
                else:
                    w.add(TRAINABLE, rank, alpha, True, trainable_dtype=tdt)
        self.rank, self.alpha, self.kind, self.factor = rank, alpha, kind, factor
        self._trainable_scale = {full: w.scales[TRAINABLE] for full, w in self.wrapped.items()
                                 if TRAINABLE in w.adapters}

    def set_trainable_multiplier(self, m):
        """Signed strength of the trainable adapter (sliders train it at +1 and -1, previews show -1 / 0 / +1).
        Takes effect on the next forward - and on a checkpointed block's recompute, so backward each pole
        before flipping."""
        for full, base in getattr(self, "_trainable_scale", {}).items():
            self.wrapped[full].scales[TRAINABLE] = base * float(m)

    def trainable_modules(self):
        return nn.ModuleList([w.adapters[TRAINABLE] for w in self.wrapped.values() if TRAINABLE in w.adapters])

    def parameters(self):
        return [p for m in self.trainable_modules() for p in m.parameters()]

    # ---- reading LoRA files in any common layout ------------------------------------------------
    def _module_for(self, stem):
        """A file's module stem (dotted, prefixed, or kohya-flattened) -> a Linear name in this model, or None."""
        if stem.startswith("lora_unet_"):
            flats = [stem[len("lora_unet_"):]]
        else:
            flats = []
            for p in _PREFIXES:
                if p and not stem.startswith(p):
                    continue
                name = stem[len(p):]
                if name in self.linears:
                    return name
                flats.append(name.replace(".", "_"))
        for flat in flats:
            if flat in self._flat:
                return self._flat[flat]
        for flat in flats:                  # another trainer's naming (the driver's renames)
            alias = self.driver.alias_flat(flat)
            if alias in self._flat:
                return self._flat[alias]
        return None

    def read_file(self, path):
        """-> {module name: entry} for every Linear the file adapts in this model. A LoRA entry is
        ("lora", A, B, scale) with scale = alpha / rank; a LoKR entry is ("lokr", w1, w2, scale) with low-rank factors
        multiplied out and the LyCORIS scale rule (lycoris_scale_from_keys). A LoHa entry is
        ("loha", (w1_a, w1_b), (w2_a, w2_b), scale), the LyCORIS scale rule as for LoKR."""
        from safetensors.torch import load_file
        sd = self.driver.convert_lora_state_dict(load_file(path))
        out = {}
        for key in sd:
            m = re.match(r"(.+)\.hada_w1_a$", key)
            if m:
                stem = m.group(1)
                full = self._module_for(stem)
                if full is None:
                    continue
                keys = {k[len(stem) + 1:]: v for k, v in sd.items() if k.startswith(stem + ".")}
                from fizgig.networks.lora import lycoris_scale_from_keys
                out[full] = ("loha", (keys["hada_w1_a"], keys["hada_w1_b"]), (keys["hada_w2_a"], keys["hada_w2_b"]),
                             lycoris_scale_from_keys(keys))
                continue
            m = re.match(r"(.+)\.lokr_w1(_a)?$", key)
            if m:
                stem = m.group(1)
                full = self._module_for(stem)
                if full is None:
                    continue
                keys = {k[len(stem) + 1:]: v for k, v in sd.items() if k.startswith(stem + ".")}
                w1 = keys["lokr_w1"] if "lokr_w1" in keys else keys["lokr_w1_a"].float() @ keys["lokr_w1_b"].float()
                w2 = keys["lokr_w2"] if "lokr_w2" in keys else keys["lokr_w2_a"].float() @ keys["lokr_w2_b"].float()
                from fizgig.networks.lora import lycoris_scale_from_keys
                out[full] = ("lokr", w1, w2, lycoris_scale_from_keys(keys))
                continue
            m = re.match(r"(.+)\.(lora_A|lora_down|lora\.down)\.weight$", key)
            if not m:
                continue
            stem, down = m.group(1), m.group(2)
            up = {"lora_A": "lora_B", "lora_down": "lora_up", "lora.down": "lora.up"}[down]
            if f"{stem}.{up}.weight" not in sd:
                continue
            A, B = sd[key], sd[f"{stem}.{up}.weight"]
            alpha = sd.get(f"{stem}.alpha")
            scale = (float(alpha.item()) if alpha is not None else float(A.shape[0])) / A.shape[0]
            full = self._module_for(stem)
            if full is None:
                # a tensor the model file fuses (FTSpec.file_layout): each Linear takes its rows of the up matrix
                from fizgig.families.lorafile import fused_parts
                for ms, pt, n in fused_parts(self.driver, stem):
                    fm = self._module_for(ms)
                    if fm is not None:
                        out[fm] = ("lora", A, B.chunk(n, dim=0)[pt], scale)
                continue
            out[full] = ("lora", A, B, scale)
        return out

    # ---- frozen adapters ------------------------------------------------------------------------
    def add_file(self, path, name, strength=1.0):
        """Attach a LoRA file frozen under `name` on every Linear it adapts. Returns the number of Linears covered
        (0 = nothing in the file matches this model)."""
        n = 0
        ar = {}
        for full, (kind, P, Q, scale) in self.read_file(path).items():
            w = self._wrap(full)
            if w is None:
                continue
            if isinstance(w, LoRAConv):
                if kind != "lora" or P.shape[1] != w.in_features or Q.shape[0] != w.out_features:
                    continue                          # LoKR / LoHa on convs are not read
                w.add(name, P.shape[0], P.shape[0], False, P, Q)
            elif kind == "loha":
                if P[0].shape[0] != w.base.out_features or P[1].shape[1] != w.base.in_features:
                    continue
                w.add_loha(name, P[0], P[1], Q[0], Q[1])
            elif kind == "lokr":
                if P.shape[0] * Q.shape[0] != w.base.out_features or P.shape[1] * Q.shape[1] != w.base.in_features:
                    continue
                w.add_lokr(name, False, w1=P, w2=Q)
            else:
                if P.shape[1] != w.base.in_features or Q.shape[0] != w.base.out_features:
                    continue
                w.add(name, P.shape[0], P.shape[0], False, P, Q)
            ar[full] = scale
            n += 1
        self._frozen[name] = {"alpha_rank": ar, "load": float(strength), "on": True, "block_mult": {},
                              "block_on": {}, "outside_on": True, "path": path, "biases": self._read_biases(path),
                              "bias_saved": []}
        self._apply(name)
        return n

    def _read_biases(self, path):
        """A file's bias deltas (`<module>.diff_b`, e.g. Krea 2's turbo LoRA on its input, timestep and output
        layers - a low-rank pair cannot carry them): [(module name, delta on CPU)] for Linears with a matching bias."""
        from safetensors import safe_open
        out = []
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                if not k.endswith(".diff_b"):
                    continue
                full = self._module_for(k[:-len(".diff_b")])
                bias = self._bias(full) if full else None
                delta = f.get_tensor(k)
                if bias is not None and tuple(bias.shape) == tuple(delta.shape):
                    out.append((full, delta.cpu()))
        return out

    def _bias(self, full):
        w = self.wrapped.get(full)
        mod = w.base if w is not None else self.dit.get_submodule(full)
        return getattr(mod, "bias", None)

    def _apply_biases(self, name):
        """Put back the biases as they were, then (adapter on) add the deltas at the load strength. Snapshot and
        restore, not += and -=, which in bf16 does not land back on the same values."""
        st = self._frozen[name]
        for bias, snap in st["bias_saved"]:
            bias.data.copy_(snap)
        st["bias_saved"] = []
        if st["on"]:
            for full, delta in st["biases"]:
                bias = self._bias(full)
                st["bias_saved"].append((bias, bias.detach().clone()))
                d = delta if st["load"] == 1.0 else delta.float() * st["load"]
                bias.data.add_(d.to(device=bias.device, dtype=bias.dtype))

    def has(self, name):
        return name in self._frozen

    def _apply(self, name):
        st = self._frozen[name]
        for full, ar in st["alpha_rank"].items():
            b = self.driver.block_of(full)
            on = st["on"] and (st["outside_on"] if b is None else st["block_on"].get(b, True))
            mult = 1.0 if b is None else st["block_mult"].get(b, 1.0)
            self.wrapped[full].scales[name] = ar * st["load"] * mult if on else 0.0
        if st.get("biases"):
            self._apply_biases(name)

    def set_enabled(self, name, enabled: bool):
        """Switch a frozen adapter on (with its load strength and block settings) or fully off - every module,
        inside or outside the block map."""
        if name in self._frozen:
            self._frozen[name]["on"] = bool(enabled)
            self._apply(name)

    def set_strength(self, name, strength):
        """The adapter's whole-file (load) strength."""
        self._frozen[name]["load"] = float(strength)
        self._apply(name)

    def set_outside(self, name, enabled: bool):
        """On/off for the adapter's modules outside the block map (no slider reaches them)."""
        self._frozen[name]["outside_on"] = bool(enabled)
        self._apply(name)

    def set_blocks(self, name, mult=None, enabled=None):
        """Per-block controls for one adapter: mult {block_id: strength}, enabled {block_id: bool}. Blocks not
        named keep their current values."""
        st = self._frozen[name]
        st["block_mult"].update(mult or {})
        st["block_on"].update(enabled or {})
        self._apply(name)

    def adapter_blocks(self, name):
        """Block ids the adapter touches (the workbench greys out the rest)."""
        return {b for b in (self.driver.block_of(f) for f in self._frozen.get(name, {}).get("alpha_rank", {}))
                if b is not None}

    @torch.no_grad()
    def swap_file(self, name, path):
        """Replace an adapter's weights with another file's (epoch scrubbing). In place when the file adapts the
        same modules at the same ranks; otherwise the adapter is rebuilt. Block settings and strength carry over.
        Returns the number of Linears covered."""
        st = self._frozen[name]
        new = self.read_file(path)

        def params(full):
            ad = self.wrapped[full].adapters[name]
            return (ad.lokr_w1, ad.lokr_w2) if isinstance(ad, LoKR) else (ad[0].weight, ad[1].weight)
        same = set(new) == set(st["alpha_rank"]) and not any(
            new[f][0] == "loha" or isinstance(self.wrapped[f].adapters[name], LoHa) for f in new) and all(
            (new[f][0] == "lokr") == isinstance(self.wrapped[f].adapters[name], LoKR)
            and params(f)[0].shape == new[f][1].shape and params(f)[1].shape == new[f][2].shape for f in new)
        if same:
            for full, (_kind, P, Q, scale) in new.items():
                a, b = params(full)
                a.copy_(P.to(a.dtype))
                b.copy_(Q.to(b.dtype))
                st["alpha_rank"][full] = scale
            st["path"] = path
            self._apply(name)
            return len(new)
        keep = {k: st[k] for k in ("load", "on", "block_mult", "block_on", "outside_on")}
        self.remove(name)
        n = self.add_file(path, name, keep["load"])
        self._frozen[name].update(keep)
        self._apply(name)
        return n

    def remove(self, name):
        """Drop a frozen adapter's weights entirely."""
        st = self._frozen.get(name)
        if st and st.get("bias_saved"):
            st["on"] = False
            self._apply_biases(name)
        for w in self.wrapped.values():
            if name in w.adapters:
                del w.adapters[name]
                w.scales.pop(name, None)
        self._frozen.pop(name, None)

    def move_adapter(self, name, device):
        """Move one frozen adapter's weights (e.g. a speed LoRA parked on CPU between previews)."""
        for w in self.wrapped.values():
            if name in w.adapters:
                w.adapters[name].to(device)

    # ---- bake: what is live now, as one standard LoRA in the family format ------------------------
    @torch.no_grad()
    def bake(self, names, dtype=torch.bfloat16):
        """One LoRA state dict equal to the named adapters as currently set (strengths, block settings, on/off),
        in the family's key format. Several adapters on a module are rank-concatenated with their scales folded
        into the up weights, so alpha = total rank (scale 1). Returns (state_dict, {module: rank})."""
        sd, ranks = {}, {}
        for full, w in self.wrapped.items():
            live = [(n, w.scales[n]) for n in names if n in w.adapters and w.scales.get(n, 0.0)]
            if not live:
                continue
            if len(live) == 1 and isinstance(w.adapters[live[0][0]], LoKR):
                # a lone LoKR stays a LoKR: the scale folds into w2 (alpha 1, full matrices -> scale 1 everywhere)
                ad, s1 = w.adapters[live[0][0]], live[0][1]
                stem = self._stem(full, lokr=True)
                sd[f"{stem}.lokr_w1"] = ad.lokr_w1.detach().to("cpu", dtype).contiguous()
                sd[f"{stem}.lokr_w2"] = (ad.lokr_w2.detach().float() * s1).to("cpu", dtype).contiguous()
                sd[f"{stem}.alpha"] = torch.tensor(1.0)
                ranks[full] = 0
                continue
            As, Bs = [], []
            for n, sc in live:
                ad = w.adapters[n]
                if isinstance(ad, (LoKR, LoHa)):   # Kronecker / Hadamard delta: SVD to a rank <= 64 LoRA
                    U, S, Vh = torch.linalg.svd(ad.delta(), full_matrices=False)
                    k = min(64, S.numel())
                    root = S[:k].sqrt()
                    As.append(root[:, None] * Vh[:k])
                    Bs.append(U[:, :k] * root * sc)
                    continue
                a, b = ad
                As.append(a.weight.float())
                Bs.append(b.weight.float() * sc)
            A, B = torch.cat(As, 0), torch.cat(Bs, 1)
            ka, kb, kal = self._keys(full)
            sd[ka] = A.to("cpu", dtype).contiguous()
            sd[kb] = B.to("cpu", dtype).contiguous()
            sd[kal] = torch.tensor(float(A.shape[0]))
            ranks[full] = A.shape[0]
        return sd, ranks

    # ---- save / load the trainable adapter (approved key format) --------------------------------
    def state_dict(self, dtype=torch.bfloat16):
        sd = {}
        for full, w in self.wrapped.items():
            if TRAINABLE in w.adapters and isinstance(w.adapters[TRAINABLE], LoKR):
                ad = w.adapters[TRAINABLE]
                stem = self._stem(full, lokr=True)
                sd[f"{stem}.lokr_w1"] = ad.lokr_w1.detach().to("cpu", dtype).contiguous()
                sd[f"{stem}.lokr_w2"] = ad.lokr_w2.detach().to("cpu", dtype).contiguous()
                sd[f"{stem}.alpha"] = torch.tensor(1.0)     # full matrices: LyCORIS scale = alpha = 1
            elif TRAINABLE in w.adapters:
                a, b = w.adapters[TRAINABLE]
                ka, kb, kal = self._keys(full)
                sd[ka] = a.weight.detach().to("cpu", dtype).contiguous()
                sd[kb] = b.weight.detach().to("cpu", dtype).contiguous()
                sd[kal] = torch.tensor(float(self.alpha))
        return sd

    def save(self, path, metadata=None, dtype=torch.bfloat16):
        from safetensors.torch import save_file
        save_file(self.state_dict(dtype), path, metadata={k: str(v) for k, v in (metadata or {}).items()})

    @torch.no_grad()
    def load_trainable(self, path):
        """Restore the trainable adapter from a file saved by save(). Returns modules matched."""
        from safetensors.torch import load_file
        sd = load_file(path)
        n = 0
        for full, w in self.wrapped.items():
            if TRAINABLE not in w.adapters:
                continue                # frozen-only wraps (training adapter, speed LoRA extras) hold nothing to load
            ad = w.adapters[TRAINABLE]
            if isinstance(ad, LoKR):
                stem = self._stem(full, lokr=True)
                if f"{stem}.lokr_w1" in sd:
                    ad.lokr_w1.copy_(sd[f"{stem}.lokr_w1"].to(ad.lokr_w1.dtype))
                    ad.lokr_w2.copy_(sd[f"{stem}.lokr_w2"].to(ad.lokr_w2.dtype))
                    n += 1
                continue
            ka, kb, _ = self._keys(full)
            if ka in sd:
                a, b = w.adapters[TRAINABLE]
                a.weight.copy_(sd[ka].to(a.weight.dtype))
                b.weight.copy_(sd[kb].to(b.weight.dtype))
                n += 1
        return n
