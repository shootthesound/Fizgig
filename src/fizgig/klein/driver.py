"""Klein 9B driver for Fizgig's standard layer (families/driver.py).

The old Klein trainer's rules (training/trainer.py, scripts/cache_latents.py / cache_text.py) behind the FamilyDriver
interface, so the generic cache / train / preview code trains Klein the way the old trainer does:
* latents: FLUX.2 AE in float32, images in [-1, 1], 2x2-packed -> (128, h, w) at /16, stored float32
* conditioning: Qwen3-8B layers 9 / 18 / 27 through the final RMSNorm, 512 padded tokens -> {"ctx_vec": (512, 12288)}
* training: flow matching, x_t = (1 - t) x0 + t noise, target = noise - x0; flux2_shift timesteps (logit-normal, shift
  exp(mu), mu linear in the latent token count, 0.5 at 256 to 1.15 at 4096), rescaled into the min / max window; the
  DiT sees t + 0.001 (the old trainer's 1..1000 timesteps / 1000), guidance 1, under bf16 autocast; unweighted MSE
* sampling: Euler; Base on the empirical-mu schedule (CFG when cfg > 1), the Distilled checkpoint on ComfyUI's simple
  schedule (shift 2.02), as the old trainer's previews
* edit: reference latents ride after the image tokens at time offsets 10, 20, ... (pack_control_latent)
* LoRA: the Linears of the double and single blocks (attention qkv / proj, MLPs, linear1 / linear2), kohya keys
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from fizgig.families.driver import Block, BlockGroup, FamilyDriver, picture_losses, prior_base, weighted_loss, with_prior

DTYPE = torch.bfloat16
_DOUBLE_MODULES = ("img_attn.qkv", "img_attn.proj", "img_mlp.0", "img_mlp.2",
                   "txt_attn.qkv", "txt_attn.proj", "txt_mlp.0", "txt_mlp.2")
_SINGLE_MODULES = ("linear1", "linear2")
N_DOUBLE, N_SINGLE = 8, 24


def _mu(n_tokens):
    """training/train_utils.get_lin_function() through (256, 0.5) and (4096, 1.15) - flux2_shift's mu."""
    m = (1.15 - 0.5) / (4096 - 256)
    return m * n_tokens + (0.5 - m * 256)


def _is_fp8_file(path) -> bool:
    """True when the DiT file stores weights in fp8 (BFL's / Comfy-Org's pre-quantised Klein files - BFL's base
    keeps the attention weights bf16 and the MLPs fp8)."""
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            return any(f.get_slice(k).get_dtype().startswith("F8") for k in f.keys() if k.endswith(".weight"))
    except Exception:
        return False


class KleinDriver(FamilyDriver):

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        """bf16, or a pre-quantised fp8 file kept fp8 with its per-Linear scales (the old loader's passthrough);
        INT8 / NF4 requantise from it."""
        from fizgig.klein.model_utils import KLEIN_MODEL_INFO, load_dit
        dit = load_dit(device=device, model_version_info=KLEIN_MODEL_INFO["klein-base-9b"], dit_path=path,
                       attn_mode="torch", split_attn=False, loading_device=device, dit_weight_dtype=DTYPE)
        return dit.eval().requires_grad_(False)

    def auto_uncompiled_precision(self, dit_path, precision):
        # measured 3 Oct 2026: uncompiled, BFL's fp8 file as it is beats requantising it to INT8 (0.77 vs 1.14 s/step
        # on a 5090, 439 vs ~688 s on a 4090 laptop) at less memory (11.1 vs 12.8 GB) - compiled INT8 is the fastest
        if precision != "int8" or not _is_fp8_file(dit_path):
            return None
        return "bf16"

    def max_blocks_to_swap(self, dit=None):
        # the old Training tab's maximum: KleinDiT.enable_block_swap splits 16 into 6 double + 18 single streamed
        # blocks (2 + 6 resident); above 16 its split overruns the single blocks
        return 16

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, torch.device(device), supports_backward)
        dit.move_to_device_except_swap_blocks(torch.device(device))
        dit.switch_block_swap_for_training()

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    def load_vae(self, path, device):
        from fizgig.klein.model_utils import load_vae
        return load_vae(path, dtype=torch.float32, device=device, disable_mmap=True).to(device).eval()

    def load_text_encoder(self, path, device):
        """Qwen3-8B, bf16 when it fits the free VRAM with room to run, else fp8 weights (the old trainer's
        --fp8_text_encoder, its default): about 8 GB instead of 16."""
        from fizgig.families.quant import free_vram_gb
        from fizgig.klein.model_utils import load_text_encoder
        dtype = torch.float8_e4m3fn if free_vram_gb() < 19.5 else DTYPE
        return load_text_encoder(path, dtype=dtype, device=device, disable_mmap=True)

    def unload_text_encoder(self, te):
        try:
            te.to("cpu")
        except Exception:
            pass
        del te
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def compile_blocks(self, dit, boundary="inside", blocks_to_swap=0):
        """Both block lists through the shared compile (families/compile.py). Klein's blocks checkpoint themselves;
        once a list is accepted for compiling, the shared wrapper does the checkpoint and each block's own is switched
        off (before the first trace) - a refused list keeps its own, so it never runs without one."""
        from fizgig.families.compile import compile_blocks
        if boundary == "outside" and any(getattr(m, "scale_weight", None) is not None
                                         and getattr(m, "weight", None) is not None
                                         and m.weight.dtype == torch.float8_e4m3fn for m in dit.modules()):
            # measured 3 Oct 2026: an fp8-resident base (BFL's fp8 file on "bf16") compiled with the checkpoint
            # outside the graph stops at the first backward - the recompute sees different tensor metadata. Inside
            # trains (0.69 vs 0.77 s/step eager at 0.25 MP, +0.8 GB)
            logging.getLogger(__name__).info("[compile] fp8 base: compiling with the checkpoint inside the graph "
                                             "(outside does not work with fp8 weights)")
            boundary = "inside"
        for blocks in (dit.double_blocks, dit.single_blocks):
            originals = list(blocks)
            compile_blocks(dit, blocks, blocks_to_swap, boundary=boundary,
                           fullgraph=self.description.compile_fullgraph)
            if blocks[0] is originals[0]:
                return                      # refused (and logged): run eager, checkpointing as before
            for b in originals:
                b.gradient_checkpointing = False

    def enable_gradient_checkpointing(self, dit, on=True):
        if on:
            dit.enable_gradient_checkpointing()
        else:
            dit.disable_gradient_checkpointing()

    # ---- encoding (the old cache scripts, per batch) --------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2) / 127.5 - 1.0
        z = vae.encode(x.to(vae.device, dtype=vae.dtype))                     # (B, 128, H/16, W/16)
        return [zi.cpu() for zi in z]

    @torch.no_grad()
    def encode_text(self, te, captions):
        dev = te.device
        ac = DTYPE if te.dtype.itemsize == 1 else te.dtype
        with torch.autocast(device_type=dev.type, dtype=ac):
            ctx = te(list(captions)).cpu()                                        # (B, 512, 12288)
        return [{"ctx_vec": c} for c in ctx]

    supports_references = True

    def load_reference_text_encoder(self, path, device):
        return self.load_text_encoder(path, device)

    def encode_text_with_references(self, te, captions, references):
        """Klein's text encoder never sees the references: they reach the DiT as latents only."""
        return self.encode_text(te, captions)

    # ---- training -------------------------------------------------------------------------------
    @staticmethod
    def _sample_t(n_tokens, generator, min_t=0.0, max_t=1.0):
        """flux2_shift (sigmoid scale 1), then t * (max - min) + min - the old trainer's order."""
        shift = float(np.exp(_mu(n_tokens)))
        t = torch.randn(1, generator=generator).sigmoid()
        t = (t * shift) / (1 + (shift - 1) * t)
        return t * (float(max_t) - float(min_t)) + float(min_t)

    @staticmethod
    def _refs(refs, device):
        if not refs:
            return None, None
        from fizgig.klein.position import pack_control_latent
        tok, ids = pack_control_latent([r.to(device) for r in refs])
        return tok.to(device=device, dtype=torch.float32), ids.to(device)

    def _forward(self, dit, noisy, t, cond, refs=None):
        """The old call_dit: pack, float32 inputs, guidance 1, timesteps (t * 1000 + 1) / 1000, bf16 autocast,
        image tokens back to (B, C, h, w)."""
        from fizgig.klein.position import prc_img, prc_txt
        device = noisy.device
        b, _, h, w = noisy.shape
        x, x_ids = prc_img(noisy)
        ctx = cond["ctx_vec"]
        if ctx.dim() == 2:
            ctx = ctx[None]
        ctx, ctx_ids = prc_txt(ctx)
        x = x.to(device=device, dtype=torch.float32)
        ctx = ctx.to(device=device, dtype=torch.float32)
        ctx_ids = ctx_ids.to(device)
        n = x.shape[1]
        ref_tok, ref_ids = self._refs(refs, device)
        if ref_tok is not None:
            x, x_ids = torch.cat((x, ref_tok), dim=1), torch.cat((x_ids, ref_ids), dim=1)
        guidance = torch.full((b,), 1.0, device=device, dtype=torch.float32)
        ts = (t.to(device=device, dtype=torch.float32) * 1000.0 + 1) / 1000.0
        with torch.autocast(device_type=device.type, dtype=DTYPE):
            pred = dit(x=x, x_ids=x_ids, timesteps=ts, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)
        return rearrange(pred[:, :n], "b (h w) c -> b c h w", h=h, w=w)

    def loss_at(self, dit, latents, noise, t, cond, *, refs=None, diff_ref=None, diff_weight=0.0, loss_mask=None,
                prior=None, per=False):
        """The training loss for a given latent, noise and t - the old trainer's arithmetic in its dtype order.
        training_loss draws noise and t; tests call this directly to compare with the old trainer. per: each
        picture's own loss, (B,) (training_loss_batch)."""
        t = t.to(latents.device)
        t_ = t.view(-1, 1, 1, 1)
        noise = noise.to(device=latents.device, dtype=latents.dtype)
        noisy = (1 - t_) * latents + t_ * noise
        base = prior_base(prior, loss_mask, lambda: self._forward(dit, noisy, t, cond, refs))
        pred = self._forward(dit, noisy, t, cond, refs)
        target = noise - latents.to(torch.float32)
        wt = None
        if diff_ref is not None and diff_weight > 0.0:
            # slider disentanglement, per latent token (Krea 2's formula)
            d = (latents.float() - diff_ref.to(latents.device).float()).abs().mean(dim=1).flatten(1)   # (1, N)
            dm = d.mean(dim=1, keepdim=True)
            r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
            wt = (1.0 - float(diff_weight)) + float(diff_weight) * r
            wt = wt / wt.mean(dim=1, keepdim=True).clamp_min(1e-8)
            wt = torch.where(dm > 1e-6, wt, torch.ones_like(wt))
        mw = None if loss_mask is None else loss_mask.float().flatten(1)              # (1, N): one cell per token
        if per:
            return picture_losses((pred.float() - target).pow(2).mean(dim=1).flatten(1), wt, mw)
        if wt is None and mw is None:
            return F.mse_loss(pred.to(torch.float32), target)
        loss = weighted_loss((pred.float() - target).pow(2).mean(dim=1).flatten(1), wt, mw)
        if base is not None:
            loss = with_prior(loss, prior, base, (pred.float() - base).pow(2).mean(dim=1).flatten(1), mw, wt)
        return loss

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0, loss_mask=None, prior=None):
        h, w = latents.shape[-2:]
        noise = torch.randn(latents.shape, generator=generator)
        t = self._sample_t(h * w, generator, min_t, max_t)
        loss = self.loss_at(dit, latents, noise, t, cond, refs=refs, diff_ref=diff_ref, diff_weight=diff_weight,
                            loss_mask=loss_mask, prior=prior)
        return loss, {"t": float(t.mean())}

    def training_loss_batch(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, loss_mask=None):
        """Batch Size above 1: each picture's noise and t drawn as training_loss draws them, picture by picture, then
        one loss_at over the batch (its MSE, and a mask's per-picture weights, are the pictures' mean)."""
        h, w = latents.shape[-2:]
        noises, ts = [], []
        for i in range(latents.shape[0]):
            noises.append(torch.randn(latents[i:i + 1].shape, generator=generator))
            ts.append(self._sample_t(h * w, generator, min_t, max_t))
        t = torch.cat(ts)
        per = self.loss_at(dit, latents, torch.cat(noises), t, cond, loss_mask=loss_mask, per=True)
        return per.mean(), {"t": float(t.mean()), "per": per, "ts": [float(v) for v in t]}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        h, w = latents.shape[-2:]
        noise = torch.randn(latents.shape, generator=generator).to(latents.device, latents.dtype)
        t = self._sample_t(h * w, generator, min_t, max_t).to(latents.device)
        t_ = t.view(-1, 1, 1, 1)
        return {"xt": (1 - t_) * latents + t_ * noise, "t": t}

    def predict(self, dit, state, cond):
        return self._forward(dit, state["xt"], state["t"], cond)

    # ---- sampling (the old trainer's previews, split at the decode) -----------------------------
    @staticmethod
    def _grid(width, height):
        return max(16, (int(width) // 16) * 16), max(16, (int(height) // 16) * 16)

    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        width, height = self._grid(width, height)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        g = torch.Generator(device=dev).manual_seed(int(seed))
        return torch.randn((1, 128, height // 16, width // 16), generator=g, device=dev, dtype=DTYPE).float().cpu()

    def pad_conditioning(self, conds):
        """Prompt travel: every Klein prompt is the encoder's fixed 512 padded tokens, so the shapes already match."""
        return [dict(c) for c in conds]

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        """options: ("schedule", "simple") - ComfyUI's Euler simple at ("shift", 2.02), the Distilled previews;
        otherwise the empirical-mu schedule (("flow_shift", s) pins a fixed shift). ("guidance", g) is the guidance
        vector (1 by default)."""
        from fizgig.klein.model_utils import get_schedule, get_simple_euler_schedule
        from fizgig.klein.position import prc_img, prc_txt, scatter_ids
        device = next(p for p in dit.parameters() if p.device.type != "meta").device
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
        width, height = self._grid(width, height)
        opts = dict(options)
        lat = (noise if noise is not None else self.initial_noise(seed, width, height)).to(device, DTYPE)
        x, x_ids = prc_img(lat)
        ctx = cond["ctx_vec"]
        ctx, ctx_ids = prc_txt((ctx[None] if ctx.dim() == 2 else ctx).to(device, DTYPE))
        guided = cfg > 1.0 and neg_cond is not None
        if guided:
            nctx = neg_cond["ctx_vec"]
            nctx, nctx_ids = prc_txt((nctx[None] if nctx.dim() == 2 else nctx).to(device, DTYPE))
        ref_tok, ref_ids = self._refs(refs, device)
        if ref_tok is not None:
            ref_tok = ref_tok.to(DTYPE)
        if sigmas is not None and len(sigmas) == steps:
            ts = [float(s) for s in sigmas] + [0.0]
        elif opts.get("schedule") == "simple":
            ts = get_simple_euler_schedule(steps, float(opts.get("shift", 2.02)))
        else:
            ts = get_schedule(steps, x.shape[1], opts.get("flow_shift"))
        g = torch.full((1,), float(opts.get("guidance", 1.0)), device=device, dtype=DTYPE)
        if hasattr(dit, "prepare_block_swap_before_forward"):
            dit.prepare_block_swap_before_forward()
        n = x.shape[1]
        for i, (tc, tp) in enumerate(zip(ts[:-1], ts[1:])):
            if on_step is not None:
                on_step(i, len(ts) - 1)
            tv = torch.full((1,), tc, dtype=DTYPE, device=device)
            xin, xin_ids = (torch.cat((x, ref_tok), 1), torch.cat((x_ids, ref_ids), 1)) if ref_tok is not None \
                else (x, x_ids)
            with torch.autocast(device_type=device.type, dtype=DTYPE):
                pred = dit(x=xin, x_ids=xin_ids, timesteps=tv, ctx=ctx, ctx_ids=ctx_ids,
                           guidance=None if guided else g)
                if guided:
                    un = dit(x=xin, x_ids=xin_ids, timesteps=tv, ctx=nctx, ctx_ids=nctx_ids, guidance=None)
            pred = pred[:, :n]
            if guided:
                pred = un[:, :n] + cfg * (pred - un[:, :n])
            x = x + (tp - tc) * pred
        return torch.cat(scatter_ids(x, x_ids)).squeeze(2)                     # (1, 128, h, w)

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        dev = next(vae.parameters()).device
        px = vae.decode(latents.to(dev, vae.dtype)).float().cpu()
        px = (px / 2 + 0.5).clamp(0, 1)
        return Image.fromarray((px[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))

    # ---- Distilled training previews: the old trainer's model handoff (training/trainer.py sample_images) ----------
    def park_for_preview(self, dit, device):
        """Max block swap on the training DiT so the Distilled fits beside it: 6 double + 22 single, per type (the
        ratio split stops at 24). An NF4 base cannot swap and is small - left as it is (token None)."""
        if getattr(dit, "_nf4_quantized", False):
            return None
        orig = int(dit.blocks_to_swap or 0)
        nd, ns = dit.num_double_blocks - 2, dit.num_single_blocks - 2
        dit.enable_block_swap(nd + ns, torch.device(device), True, double_blocks_to_swap=nd,
                              single_blocks_to_swap=ns)
        dit.prepare_block_swap_before_forward()
        return orig

    @staticmethod
    def _preview_swap(nd=N_DOUBLE, ns=N_SINGLE):
        """The Distilled's own swap by card (_auto_distilled_sample_swap): 23 GB+ none, 15-22 GB 16 (ratio split),
        under 15 GB the maximum per type. FIZGIG_SIM_VRAM_GB simulates a card."""
        import os
        sim = os.environ.get("FIZGIG_SIM_VRAM_GB", "").strip()
        if sim:
            gb = float(sim)
        elif torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        else:
            return 0, None, None
        if gb >= 23:
            return 0, None, None
        if gb >= 15:
            return 16, None, None
        return (nd - 2) + (ns - 2), nd - 2, ns - 2

    def load_preview_checkpoint(self, path, device, int8=False):
        """The Distilled DiT: loaded on the CPU when it streams, optionally INT8 (before the swap and the LoRA),
        forward-only block swap."""
        from fizgig.klein.model_utils import KLEIN_MODEL_INFO, load_dit
        n, nd, ns = self._preview_swap()
        device = torch.device(device)
        loading = torch.device("cpu") if n else device
        m = load_dit(device=device, model_version_info=KLEIN_MODEL_INFO["klein-9b"], dit_path=path,
                     attn_mode="torch", split_attn=False, loading_device=loading, dit_weight_dtype=DTYPE)
        if int8:
            from fizgig.klein.model import FP8_OPTIMIZATION_EXCLUDE_KEYS, FP8_OPTIMIZATION_TARGET_KEYS
            from fizgig.modules.int8 import apply_int8_quantization
            apply_int8_quantization(m, target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
                                    exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS, compute_device=loading)
        if n:
            m.enable_block_swap(n, device, supports_backward=False, double_blocks_to_swap=nd,
                                single_blocks_to_swap=ns)
            m.move_to_device_except_swap_blocks(device)
        m.prepare_block_swap_before_forward()
        return m.eval().requires_grad_(False), n

    def unpark_after_preview(self, dit, device, token):
        """The run's own swap back (re-placing the parked blocks), or, with none, the sample-time offloaders torn
        down - their backward hooks would otherwise fire on the next backward - and the model back on the GPU."""
        if token is None:
            return
        device = torch.device(device)
        if token > 0:
            dit.enable_block_swap(token, device, True)
            dit.move_to_device_except_swap_blocks(device)
        else:
            from fizgig.families import quant
            dit.disable_block_swap()
            quant.move(dit, device)
        dit.switch_block_swap_for_training()

    # ---- full fine-tune (families/ft.py) ------------------------------------------------------------------------
    def ft_spec(self, dit):
        # the double blocks then the single blocks as one cycle; each stream's attention and MLP a window, the single
        # blocks' fused linear1 (qkv + MLP in) and linear2 (attention out + MLP out) theirs (linear1 is the largest:
        # the planner splits it by depth where it does not fit). The modulation, embedders and final layer stay frozen.
        from fizgig.families.ft import FTSpec
        return FTSpec(blocks=("double_blocks", "single_blocks"),
                      components=("img_attn", "img_mlp", "txt_attn", "txt_mlp", "linear1", "linear2"))

    def install_ft_streamer(self, dit, streamer):
        # Klein's forward swaps the double and the single blocks through two offloaders, each counting from 0; the
        # streamer holds the double blocks then the single blocks (ft_spec's cycle), so each list gets its view
        dit.disable_block_swap()
        nd = len(dit.double_blocks)
        dit.offloader_double = streamer.view(0, nd)
        dit.offloader_single = streamer.view(nd, len(dit.single_blocks))
        dit.blocks_to_swap = 1

    # ---- LoRA and the block map -------------------------------------------------------------------
    def convert_lora_state_dict(self, sd):
        """Every Klein LoRA layout the old loaders took (networks/lora.py ensure_kohya_lora_state_dict): kohya,
        OneTrainer's lora_transformer_, PEFT, diffusers Flux with split q/k/v fused into Klein's qkv / linear1."""
        from fizgig.networks.lora import ensure_kohya_lora_state_dict
        return ensure_kohya_lora_state_dict(sd)

    def block_map(self, dit=None):
        """Klein's own ids (double_N / single_N), as the old Repair Studio presets name them."""
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def keep(mods):
            return [m for m in mods if names is None or m in names]
        return [BlockGroup("Double blocks", [Block(f"double_{i}", f"Double {i}",
                                                   keep([f"double_blocks.{i}.{m}" for m in _DOUBLE_MODULES]))
                                             for i in range(N_DOUBLE)]),
                BlockGroup("Single blocks", [Block(f"single_{i}", f"Single {i}",
                                                   keep([f"single_blocks.{i}.{m}" for m in _SINGLE_MODULES]))
                                             for i in range(N_SINGLE)])]
