"""MiniMax H3 Image driver for Fizgig's standard layer (families/driver.py): a stills family on MiniMax H3's model code.

* base: H3's loader (the checkpoint's own int8 ConvRot codes, or NF4 / HQQ) and its H2D block-swap rings, converted to
  the image model (minimax_image/model.py: no audio stream, the per-step heads)
* conditioning: H3's Qwen3-VL-32B text encoder unchanged, the raw caption, layer-50 states -> {"hidden_states" (L, 5120)};
  edits put "Picture i: " + vision blocks ahead of the caption, as ComfyUI's MiniMaxH3ImageConditioning does
* latents: the image VAE (minimax_image/vae.py), tiled, the posterior sample ComfyUI takes -> (24, h, w) at /16
* training: flow matching on the checkpoint's own 6 distilled noise levels, each through its own head (Peter, 10 Nov
  2026: "t drawn around the 6 fixed sigmas, each through its own head, and plain flow-matching loss"); the head
  returns x0 - noise (H3's convention), so the target is x0 - noise
* sampling: 6-step Euler on the PDD grid, CFG-free (minimax_image/sampling.py)
"""
import gc
import logging

import torch

from fizgig.families.driver import (Block, BlockGroup, FamilyDriver, picture_losses, prior_base, weighted_loss,
                                    with_prior)
from fizgig.minimax_image import model as M
from fizgig.minimax_image import sampling as S

logger = logging.getLogger(__name__)

DTYPE = torch.bfloat16
_BLOCK_MODULES = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")      # minimax/driver.py, H3's LoRA targets
# ComfyUI MiniMaxH3ImageConditioning (comfy_extras/nodes_minimax_h3.py): the text encoder sees an edit's pictures at a
# 1024 short edge on the 32 px canvas grid, each after the label "Picture i: "
_REF_VISION_SHORT_EDGE = 1024
_CANVAS = 32
_PICTURE_LABEL = "Picture %d: "


def _lat5(x):
    return x if x.dim() == 5 else x.unsqueeze(2)


class _VAEBox(torch.nn.Module):
    """The image VAE and the device it works on. A module, so the workbench can park it (`.to("cpu")`) like any
    family's VAE on a small card; vae["vae"] / vae["device"] as the driver reads it."""

    def __init__(self, vae, device):
        super().__init__()
        self.vae = vae
        self.device = device

    def __getitem__(self, key):
        return getattr(self, key)

    def to(self, *args, **kwargs):
        """A move to the GPU (the workbench's, before an encode or decode on a small card) is where the jobs run from
        then on; parking on the CPU leaves that as it is - the driver moves the halves for each job itself."""
        out = super().to(*args, **kwargs)
        dev = next(self.vae.parameters()).device
        if dev.type != "cpu":
            self.device = dev
        return out


class MiniMaxH3ImageDriver(FamilyDriver):

    _grid = M.PDD_SIGMAS

    # ---- the base: H3's tiers and H2D rings --------------------------------------------------------
    def load_dit(self, path, device):
        return self.load_planned(path, device, "int8", 0)[0]

    def max_blocks_to_swap(self, dit=None):
        return 40                     # H3's cap (50 blocks, >= 10 resident)

    def load_planned(self, path, device, precision, blocks_to_swap):
        """The base at int8 (the file's ConvRot codes), NF4 or HQQ, the last n blocks streamed host->device through
        the ring for their type, as MiniMax H3 loads it (minimax/driver.py load_planned)."""
        import os
        mode = precision if precision in ("int8", "nf4", "hqq") else "int8"
        n = max(0, min(int(blocks_to_swap or 0), self.max_blocks_to_swap()))
        dit = M.load_dit(path, device, base_quant=mode, blocks_to_swap=n, compute_dtype=DTYPE)
        if n > 0:
            ring = mode == "int8" or os.environ.get("FIZGIG_NO_NF4_H2D") != "1"
            n = dit.enable_block_swap(n, h2d_only=ring, ring_size=2)
            logger.info(f"[vram] block swap: the last {n} blocks stream host->device ({mode})")
        self._n_swap, self._grid = n, dit.final_layer.pdd_sigmas
        logger.info(f"[precision] {os.path.basename(path)}: {mode}, {n} blocks streamed; noise levels "
                    f"{', '.join(f'{s:.3f}' for s in self._grid[:-1])}, one head each")
        return dit, n

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.enable_gradient_checkpointing(bool(on))

    def compile_targets(self, dit):
        return dit.blocks

    def compile_blocks(self, dit, boundary="inside", blocks_to_swap=0):
        """The shared compile, with H3's int8 ConvRot Linears left eager: their custom autograd Function (Triton
        W8A16 kernels, a fallback path, class state) can't be traced, so dynamo steps around each one and compiles
        the rest of the block (norms, rotary, attention, modulation)."""
        from fizgig.minimax.convrot import ConvRotInt8Linear
        for m in dit.blocks.modules():
            if isinstance(m, ConvRotInt8Linear) and not getattr(m, "_fizgig_eager", False):
                m.forward = torch.compiler.disable(m.forward, recursive=True)
                m._fizgig_eager = True
        # the shared compile reads gradient_checkpointing; H3 keeps it as _gradient_checkpointing
        dit.gradient_checkpointing = bool(getattr(dit, "_gradient_checkpointing", False))
        super().compile_blocks(dit, boundary=boundary, blocks_to_swap=blocks_to_swap)

    def park_for(self, dit, device, need_gb, purpose):
        """Only the missing gigabytes (+1) of tail blocks go to CPU (H3's partial park, ring-aware); a decode wants
        ~3 GB (the 24-layer decoder in fp32 and one row of 256 px tiles)."""
        from fizgig.minimax.common import park_dit_partial
        from fizgig.utils.device import plannable_free_vram
        need_gb = 3.0 if need_gb is None else need_gb
        gc.collect()
        torch.cuda.empty_cache()
        free = plannable_free_vram()
        if free >= need_gb:
            return False
        logger.info(f"[preview] {free:.1f} GB free is too tight for {purpose} - parking ~{need_gb - free + 1:.1f} GB "
                    f"of tail blocks for this pass.")
        park_dit_partial(dit, need_gb=need_gb - free + 1.0)
        gc.collect()
        torch.cuda.empty_cache()
        return True

    def unpark(self, dit, device, token):
        if token:
            from fizgig.minimax.common import restore_parked_dit
            restore_parked_dit(dit, device, getattr(self, "_n_swap", 0))
            torch.cuda.empty_cache()

    # ---- text encoder: H3's ----------------------------------------------------------------------
    def load_text_encoder(self, path, device):
        from fizgig.minimax.embedder import load_minimax_h3_te_planned
        return load_minimax_h3_te_planned(path, device=device, compute_dtype=DTYPE, quantize=True)

    supports_references = True
    # training previews render with comfy-kitchen's INT8 attention, as the checkpoint asks (every block's
    # comfy_attention.config) and ComfyUI does: 22 dB from ComfyUI without it, 27 dB with (10 Nov 2026, 2048 px)
    preview_int8_attention = True

    def load_reference_text_encoder(self, path, device):
        """With its vision tower: an edit's pictures are encoded with the caption."""
        from fizgig.minimax.embedder import load_minimax_h3_te_planned
        return load_minimax_h3_te_planned(path, device=device, compute_dtype=DTYPE, quantize=True, with_vision=True)

    def unload_text_encoder(self, te):
        del te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode_text(self, te, captions):
        # H3's batched encode: right padding on a causal stack, so each row equals its single encode (embedder.py
        # encode_batch, verified to 4.5e-8); one forward per batch instead of per caption
        return [{"hidden_states": h[0].detach().cpu()} for h in te.encode_batch(list(captions), batch_size=16)]

    @staticmethod
    def _vision_size(w, h):
        s = _REF_VISION_SHORT_EDGE / min(w, h)
        return tuple(max(_CANVAS, round(v * s / _CANVAS) * _CANVAS) for v in (w, h))

    @torch.no_grad()
    def encode_text_with_references(self, te, captions, references):
        """"Picture 1: " <vision block> ["Picture 2: " ...] <caption> (ComfyUI's tokenize_with_weights with
        picture_label "Picture %d: "): each picture resized to a 1024 short edge (Lanczos) for the vision tower."""
        from PIL import Image
        from fizgig.minimax.embedder import TEXT_TAG, VIDEO_TAG, frame_tensor, reference_patches
        tok = te.tokenizer
        v_start, v_end = tok.convert_tokens_to_ids("<|vision_start|>"), tok.convert_tokens_to_ids("<|vision_end|>")
        img_pad = tok.convert_tokens_to_ids("<|image_pad|>")
        out = []
        for cap, imgs in zip(captions, references):
            ids, tags, pix, grids = [], [], [], []
            for i, a in enumerate(imgs):
                im = Image.fromarray(a[..., :3])
                im = im.resize(self._vision_size(*im.size), Image.LANCZOS)
                t = tok(_PICTURE_LABEL % (i + 1), add_special_tokens=False)["input_ids"]
                ids += t
                tags += [TEXT_TAG] * len(t)
                flat, grid = reference_patches(frame_tensor(im).unsqueeze(0))
                run = [v_start] + [img_pad] * (int(grid[0].prod()) // 4) + [v_end]
                ids += run
                tags += [VIDEO_TAG] * len(run)
                pix.append(flat)
                grids.append(grid)
            t = tok(cap, add_special_tokens=False)["input_ids"]
            ids += t
            tags += [TEXT_TAG] * len(t)
            kw = {}
            if pix:
                kw = {"pixel_values": torch.cat(pix).to(te.device, torch.bfloat16),
                      "image_grid_thw": torch.cat(grids).to(te.device)}
            ids = torch.tensor([ids], dtype=torch.long)
            h = te.model(input_ids=ids.to(te.device), attention_mask=torch.ones_like(ids).to(te.device),
                         **kw).last_hidden_state.to(DTYPE)
            out.append({"hidden_states": h[0].cpu(), "token_tags": torch.tensor(tags, dtype=torch.long)})
        return out

    # ---- VAE ----------------------------------------------------------------------------------------
    def load_vae(self, path, device):
        """Both halves in fp32 on the CPU (2.3 GB): each rides to the GPU for its own job only."""
        from fizgig.minimax_image.vae import load_image_vae
        return _VAEBox(load_image_vae(path, "cpu"), device)

    @torch.no_grad()
    def encode_images(self, vae, images):
        import numpy as np
        m, device = vae["vae"], vae["device"]
        m.to(device)
        try:
            out = []
            for im in images:             # one at a time: the tiles bound the memory, the batch would not
                x = torch.from_numpy(np.ascontiguousarray(np.asarray(im)[..., :3])).permute(2, 0, 1)[None]
                z = m.encode(x.float().div(127.5).sub(1.0).to(device))                    # (1, 24, 1, h, w)
                out.append(z[0, :, 0].cpu())
            return out
        finally:
            if not self.keep_vae_resident:
                m.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        m, device = vae["vae"], vae["device"]
        m.to(device)
        try:
            px = m.decode(_lat5(latents).to(device, torch.float32))[0, :, 0]
            return Image.fromarray((px.permute(1, 2, 0) * 255).round().byte().cpu().numpy())
        finally:
            if not self.keep_vae_resident:
                m.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ---- training -----------------------------------------------------------------------------------
    def _sigma(self, generator, min_t, max_t):
        """One of the checkpoint's noise levels (its 6 grid points above 0), uniformly; the Timestep Range keeps the
        levels inside it, or the nearest one when none is."""
        levels = list(self._grid[:-1])
        keep = [s for s in levels if min_t - 1e-6 <= s <= max_t + 1e-6]
        if not keep:
            mid = 0.5 * (min_t + max_t)
            keep = [min(levels, key=lambda s: abs(s - mid))]
        pct = self._clean_share()                  # that share of the steps at the levels below sigma 0.5, the rest
        if pct is not None and len(keep) > 1:       # spread evenly above
            p = min(max(float(str(pct).rstrip("%")) / 100.0, 0.0), 1.0)
            lo, hi = [s for s in keep if s < 0.5], [s for s in keep if s >= 0.5]
            if lo and hi:
                w = torch.tensor([p / len(lo) if s < 0.5 else (1.0 - p) / len(hi) for s in keep])
                return float(keep[int(torch.multinomial(w, 1, generator=generator))])
        return float(keep[int(torch.randint(len(keep), (1,), generator=generator))])

    def _clean_share(self):
        """The clean-end share for this epoch (percent) or None (the model's even draw). Likeness optimization:
        likeness_before (75) for epochs 1 to likeness_hinge (8), likeness_after (50) after it; likeness_alternate
        (after / before / off) makes that side switch between the two shares every epoch, starting with its own.
        likeness_slide=N: the N epochs after the hinge move evenly from Before to After (After's alternating after).
        lownoise_pct=P (the CLI) still sets one share."""
        if str(self.options.get("likeness_opt", "")).strip() in ("1", "true", "True"):
            def num(key, default):
                try:
                    return float(str(self.options.get(key, "") or default).rstrip("%").strip())
                except ValueError:
                    return float(default)
            before, after = num("likeness_before", 75), num("likeness_after", 50)
            hinge, slide = int(num("likeness_hinge", 8)), max(int(num("likeness_slide", 0)), 0)
            e = int(getattr(self, "epoch", 1) or 1)
            alt = str(self.options.get("likeness_alternate", "") or "after").strip().lower()
            if e <= hinge:
                pct = after if alt == "before" and (e - 1) % 2 == 1 else before
            elif e <= hinge + slide:            # the slide: evenly from Before to After, at After on its last epoch
                pct = round(before + (after - before) * (e - hinge) / slide, 1)
            else:
                pct = before if alt == "after" and (e - hinge - slide - 1) % 2 == 1 else after
            pct = int(pct) if float(pct).is_integer() else pct
            if getattr(self, "_share_said", None) != pct:
                self._share_said = pct
                logger.info(f"MiniMax H3 Image: Likeness optimization - {pct}% of steps at the clean end from epoch {e}")
            return pct
        pct = self.options.get("lownoise_pct")
        return None if pct in (None, "") else float(str(pct).rstrip("%"))

    def _noised(self, latents, generator, min_t, max_t):
        """(x0, noise, sigma): the latent cropped to the 2x2 patch grid, the step's noise level, then its noise."""
        lat = _lat5(latents)
        x0 = lat[..., :lat.shape[-2] // 2 * 2, :lat.shape[-1] // 2 * 2].float()
        s = self._sigma(generator, min_t, max_t)
        noise = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(x0.device)
        return x0, noise, s

    @staticmethod
    def _txt(cond, device):
        h = cond["hidden_states"]
        return (h[None] if h.dim() == 2 else h).to(device, DTYPE)

    def _forward(self, dit, x, s, cond, refs=None, seed=0):
        kw = {}
        if refs:
            kw["ref_latents"] = [_lat5(r).to(x.device, torch.float32) for r in refs]
            kw["seed"] = int(seed)
        if "token_tags" in cond:
            tt = cond["token_tags"]
            kw["text_token_tags"] = tt[0] if tt.dim() == 2 else tt
        t = torch.tensor([1.0 - s], device=x.device, dtype=torch.float32)
        return dit(x.to(DTYPE), t, self._txt(cond, x.device), **kw).float()

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0, loss_mask=None, prior=None):
        import torch.nn.functional as F
        if loss_mask is not None and prior is not None and float(loss_mask.min()) >= 1.0:
            prior = None                  # an all-white mask leaves nothing outside it to preserve
        x0, noise, s = self._noised(latents, generator, min_t, max_t)
        seed = int(torch.randint(2 ** 31 - 1, (1,), generator=generator)) if refs else 0   # the references' noise aug
        xt = (1.0 - s) * x0 + s * noise
        target = x0 - noise
        mw = None if loss_mask is None else loss_mask.float().to(x0.device)[..., :x0.shape[-2], :x0.shape[-1]]   # (1, 1, h, w)
        base = prior_base(prior, loss_mask, lambda: self._forward(dit, xt, s, cond, refs, seed))
        pred = self._forward(dit, xt, s, cond, refs, seed)
        wt = None
        if diff_ref is not None and diff_weight > 0.0:
            # slider disentanglement: each patch token weighted by how much the poles differ there (Krea 2 / Qwen)
            ref = _lat5(diff_ref)[..., :x0.shape[-2], :x0.shape[-1]].to(x0.device).float()
            d = (x0 - ref).abs().mean(dim=1)                                           # (1, 1, h, w)
            d = F.interpolate(F.avg_pool2d(d, 2), scale_factor=2, mode="nearest")
            dm = d.mean()
            wt = ((1.0 - float(diff_weight)) + float(diff_weight) * (d / dm).clamp(max=8.0)) if float(dm) > 1e-6 \
                else torch.ones_like(d)
        if wt is None and mw is None:
            return F.mse_loss(pred, target), {"t": s}
        loss = weighted_loss((pred - target).pow(2).mean(dim=1), wt, mw)
        if base is not None:
            loss = with_prior(loss, prior, base, (pred - base).pow(2).mean(dim=1), mw, wt)
        return loss, {"t": s}

    def stack_conds(self, conds):
        """Batch Size above 1: each picture's own caption, packed one after another by forward_packed (nothing
        padded); an edit's references stay at batch 1."""
        if any("token_tags" in c for c in conds):
            return None
        return {"items": conds}

    def training_loss_batch(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, loss_mask=None):
        """Each picture's level and noise drawn as training_loss draws them, picture by picture, then one
        forward_packed (each picture through its own head); the loss is the pictures' mean."""
        items, parts = [], []
        for i, c in enumerate(cond["items"]):
            x0, noise, s = self._noised(latents[i:i + 1], generator, min_t, max_t)
            items.append((((1.0 - s) * x0 + s * noise).to(DTYPE),
                          torch.tensor([1.0 - s], device=x0.device, dtype=torch.float32), self._txt(c, x0.device)))
            mw = None if loss_mask is None else \
                loss_mask[i:i + 1].float().to(x0.device)[..., :x0.shape[-2], :x0.shape[-1]]
            parts.append((x0, noise, mw, s))
        preds = dit.forward_packed(items)
        per = torch.stack([picture_losses((p.float() - (x0 - n)).pow(2).mean(dim=1), mw)[0]
                           for p, (x0, n, mw, _s) in zip(preds, parts)])
        ts = [p[3] for p in parts]
        return per.mean(), {"t": sum(ts) / len(ts), "per": per, "ts": ts}

    # ---- prompt sliders and training maps -------------------------------------------------------------
    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        x0, noise, s = self._noised(latents, generator, min_t, max_t)
        return {"x": ((1.0 - s) * x0 + s * noise), "t": s}

    def predict(self, dit, state, cond):
        from fizgig.training import attn_probe
        if attn_probe._ACTIVE is not None:   # the trigger map: picture rows after the text, over the joint sequence
            L = cond["hidden_states"].shape[-2]
            n = (state["x"].shape[-2] // 2) * (state["x"].shape[-1] // 2)
            attn_probe.layout((L, L + n), 0, L + n)
        return self._forward(dit, state["x"], state["t"], cond)

    def map_target(self, latents, state):
        """predict() returns x0 - noise: (x0 - x_t) / t at this state."""
        x0 = _lat5(latents).float().to(state["x"].device)[..., :state["x"].shape[-2], :state["x"].shape[-1]]
        return (x0 - state["x"].float()) / float(state["t"])

    def trigger_text(self, tokenizer, caption):
        """H3's encoder reads the raw caption: no template, nothing dropped."""
        return caption, ""

    def trigger_tokenizer(self):
        from fizgig.families import trigger_check
        return trigger_check.fast_tokenizer("minimax")          # the same Qwen3-VL tokenizer as MiniMax H3's

    def trigger_attention_hooks(self, dit):
        """Each block's attention wrapped to hand the probe its queries and keys after QK-norm and RoPE (recomputed
        from the attention's own input, as minimax/model.py Attention.forward builds them), only while a probe is
        capturing."""
        import contextlib
        from fizgig.minimax.model import apply_rope_split_half
        from fizgig.training import attn_probe

        def wrap(attn):
            orig = attn.forward

            def fwd(x, cos=None, sin=None, segs=None):
                if attn_probe._ACTIVE is not None and not segs:
                    s = x.shape[0]
                    q, k, _v = attn.qkv_proj(x).split(attn.heads * attn.head_dim, dim=-1)
                    q = attn.q_norm(q.view(s, attn.heads, attn.head_dim))
                    k = attn.k_norm(k.view(s, attn.heads, attn.head_dim))
                    if cos is not None:
                        q, k = apply_rope_split_half(q, cos, sin), apply_rope_split_half(k, cos, sin)
                    attn_probe.see(q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0))
                return orig(x, cos, sin, segs)
            return orig, fwd

        @contextlib.contextmanager
        def hooks():
            def inner(b):        # Compile Blocks wraps each block (torch.compile's module, then CheckpointedBlock)
                while not hasattr(b, "attn"):
                    b = getattr(b, "_orig_mod", None) or getattr(b, "block")
                return b
            mods = [inner(b).attn for b in dit.blocks]
            saved = []
            for a in mods:
                orig, fwd = wrap(a)
                saved.append((a, orig))
                a.forward = fwd
            try:
                yield
            finally:
                for a, orig in saved:
                    a.forward = orig
        return hooks()

    # ---- sampling -----------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        return S.initial_noise(seed, width, height)

    @staticmethod
    def _real_tokens(cond):
        """A prompt-travel blend (pad_conditioning) keeps only the tokens either waypoint has, so each waypoint renders
        exactly as its own prompt (H3's attention has no key mask)."""
        h = cond["hidden_states"]
        h = h[0] if h.dim() == 3 else h
        if "mask" in cond:
            m = cond["mask"]
            h = h[(m[0] if m.dim() == 2 else m).to(h.device) > 0]
        return h

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        """CFG-free (the checkpoint is guidance-distilled: cfg and the negative are not used). -> (1, 24, h, w)."""
        if sigmas is not None and len(sigmas) not in (steps, steps + 1):
            sigmas = None
        tags = cond.get("token_tags")
        if tags is not None:
            tags = tags[0] if tags.dim() == 2 else tags
        z = S.sample(dit, self._real_tokens(cond), width, height, steps=steps, seed=seed, sigmas=sigmas,
                     noise=None if noise is None else _lat5(noise), on_step=on_step,
                     refs=[_lat5(r) for r in refs] if refs else None, text_token_tags=tags)
        return z[:, :, 0]

    def pad_conditioning(self, conds):
        """Prompt travel (LoRA Royale): prompts zero-padded to one length with a mask of each one's real tokens;
        generate keeps only the masked-in tokens."""
        L = max(c["hidden_states"].shape[-2] for c in conds)
        out = []
        for c in conds:
            h = c["hidden_states"][0] if c["hidden_states"].dim() == 3 else c["hidden_states"]
            pad = L - h.shape[0]
            out.append({"hidden_states": torch.cat([h, h.new_zeros(pad, h.shape[1])]) if pad else h,
                        "mask": torch.cat([torch.ones(h.shape[0], dtype=torch.bool),
                                           torch.zeros(pad, dtype=torch.bool)])})
        return out

    # ---- the fine-tune: MiniMax H3's rotation fine-tune (component windows on an NF4 trunk, int8 ConvRot saves), the
    #      same DiT layout and file format - its methods as they are
    from fizgig.minimax.driver import MiniMaxDriver as _H3
    ft_spec, ft_card_plan, ft_source_unfit, ft_backend = _H3.ft_spec, _H3.ft_card_plan, _H3.ft_source_unfit, _H3.ft_backend
    del _H3

    # ---- LoRA and the block map ----------------------------------------------------------------------
    def block_map(self, dit=None):
        """H3's map (minimax/driver.py block_map): the 50 blocks and the two text refiner blocks, the same ids."""
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if names is None or m in names]
        main = [Block(f"h3blk_{i}", f"Block {i}", keep([f"blocks.{i}.{m}" for m in _BLOCK_MODULES]))
                for i in range(self.description.n_blocks)]
        refiner = [Block(f"h3_rf_{i}", f"Refiner {i}",
                         keep([f"token_refiner.blocks.{i}.{m}" for m in _BLOCK_MODULES])) for i in range(2)]
        return [BlockGroup("Blocks", main), BlockGroup("Token Refiner", refiner)]

    def lora_target_names(self, dit):
        """The 50 main blocks, as MiniMax H3 trains (the token refiner left out: training it softens output and makes
        previews judder between epochs, measured on H3)."""
        return [m for b in self.block_map(dit)[0].blocks for m in b.modules]

    def convert_lora_state_dict(self, sd):
        """A diffusers-named H3 Image LoRA (transformer_blocks.N / token_refiner.refiner_blocks.N, separate to_q /
        to_k / to_v, to_out.0, ff.net.0.proj / ff.net.2; witcherderivia's and akhaliq's diffusers files) -> ComfyUI's
        names. q/k/v fuse into qkv_proj as akhaliq's own ComfyUI conversion does: A stacked [q; k; v], B block
        diagonal, alpha the summed ranks (each part keeps its alpha / rank scale). Any other file passes as is."""
        import re
        pat = re.compile(r"^(?:transformer\.)?(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)\."
                         r"(lora_A|lora_B)\.weight$")
        if not any(pat.match(k) for k in sd):
            return sd
        rename = {"attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}
        out, qkv = {}, {}
        for k, v in sd.items():
            m = pat.match(k)
            if m is None:
                out[k] = v
                continue
            where = ("blocks" if m.group(1) == "transformer_blocks" else "token_refiner.blocks") + "." + m.group(2)
            mod, ab = m.group(3), m.group(4)
            if mod in ("attn.to_q", "attn.to_k", "attn.to_v"):
                qkv.setdefault(where, {})[(mod[-1], ab)] = v
            elif mod in rename:
                out[f"diffusion_model.{where}.{rename[mod]}.{ab}.weight"] = v
        for where, parts in qkv.items():
            if not all((p, ab) in parts for p in "qkv" for ab in ("lora_A", "lora_B")):
                continue
            a = [parts[(p, "lora_A")] for p in "qkv"]
            b = [parts[(p, "lora_B")] for p in "qkv"]
            out[f"diffusion_model.{where}.attn.qkv_proj.lora_A.weight"] = torch.cat(a, 0)
            out[f"diffusion_model.{where}.attn.qkv_proj.lora_B.weight"] = torch.block_diag(*b)
            out[f"diffusion_model.{where}.attn.qkv_proj.alpha"] = torch.tensor(float(sum(x.shape[0] for x in a)))
        return out
