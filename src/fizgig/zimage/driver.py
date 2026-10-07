"""Z-Image Turbo driver for Fizgig's standard layer (families/driver.py) - Tongyi-MAI's 6B S3-DiT.

* model: zimage/model.py (Tongyi's Apache-2.0 reference transformer), loaded from ComfyUI's single file (fused qkv
  split into to_q / to_k / to_v); parity with diffusers 0.36's ZImageTransformer2DModel measured exact in fp32
* conditioning: Qwen3-4B on the chat-templated caption, hidden_states[-2], real tokens only -> {"cap": (L, 2560)}
* latents: the FLUX.1 VAE, (z - 0.1159) * 0.3611, (16, h, w) at /8
* training: flow matching, x_t = (1 - s) x0 + s noise; the model runs at t = 1 - s and predicts x0 - noise; s is
  logit-normal with the model's static shift 3 (its own sampling schedule), unweighted MSE
* sampling: Euler on the shift-3 schedule; Turbo samples 8 steps with no CFG. Fizgig's CFG scale is the usual
  uncond + cfg * (cond - uncond), i.e. the reference's guidance scale plus 1
* LoRA: attention (to_q / to_k / to_v / to_out.0) and MLP (w1 / w2 / w3) of the 30 main layers, written as
  diffusion_model.layers.N.* lora_A / lora_B, which ComfyUI slices onto its fused qkv
"""
import numpy as np
import torch
import torch.nn.functional as F

from fizgig.families.driver import FamilyDriver
from fizgig.zimage import pipeline as P

DTYPE = torch.bfloat16


def _cap(cond, i=0):
    """-> (caption (L, 2560), travel weights (L,) or None). A prompt-travel caption (pad_conditioning, then Royale's
    blend) carries a mask: True / 1 on its real tokens, the blend weight on the tail only one waypoint has. The blend
    scaled that tail toward the other waypoint's zero padding, so it is divided back out here and the model weights
    the token instead (exact for lerp and norm blends)."""
    h = cond["cap"]
    h = h[i] if h.dim() == 3 else h
    m = cond.get("mask")
    if m is None:
        return h, None
    m = m[i] if m.dim() == 2 else m
    w = m.float().to(h.device)
    part = (w > 1e-6) & (w < 1 - 1e-6)
    if bool(part.any()):
        h = torch.where(part[:, None], h.float() / w.clamp_min(1e-6)[:, None], h.float()).to(h.dtype)
    return h, w


class ZImageDriver(FamilyDriver):

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from fizgig.zimage.model import load_zimage_dit
        return load_zimage_dit(path, device, DTYPE).eval().requires_grad_(False)

    def max_blocks_to_swap(self, dit=None):
        return (len(dit.layers) if dit is not None else self.description.n_blocks) - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    def load_vae(self, path, device):
        return P.load_vae(path, device)

    def load_text_encoder(self, path, device):
        return P.TextEncoder(path, device)

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.gradient_checkpointing = bool(on)

    def compile_targets(self, dit):
        return dit.layers

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
        return [z.to(DTYPE).cpu() for z in P.encode_image(vae, x)]

    @torch.no_grad()
    def encode_text(self, te, captions):
        return [{"cap": h.to(DTYPE).cpu()} for h in te.encode(captions)]

    # ---- the model call -------------------------------------------------------------------------
    @staticmethod
    def _velocity(dit, x, sigma, caps):
        """x (B, 16, h, w) at noise level sigma (B,), caps: list of _cap() pairs -> velocity noise - x0, (B, 16, h, w)."""
        t = (1.0 - sigma).to(x.device, torch.float32)
        weights = [w for _c, w in caps]
        out = dit([xi.unsqueeze(1).to(DTYPE) for xi in x], t, [c.to(x.device, DTYPE) for c, _w in caps],
                  cap_weights=weights if any(w is not None for w in weights) else None)
        return -torch.stack([o.float() for o in out]).squeeze(2)

    @staticmethod
    def _sample_sigma(n, generator, min_t, max_t):
        s = torch.sigmoid(torch.randn(n, generator=generator))
        s = P.DEFAULT_SHIFT * s / (1 + (P.DEFAULT_SHIFT - 1) * s)
        return min_t + (max_t - min_t) * s

    # ---- training -------------------------------------------------------------------------------
    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        x0 = latents.float()
        n = x0.shape[0]
        sig = self._sample_sigma(n, generator, min_t, max_t).to(x0.device)
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        sb = sig.view(-1, 1, 1, 1)
        xt = (1 - sb) * x0 + sb * noise
        pred = self._velocity(dit, xt, sig, [_cap(cond, i) for i in range(n)])
        target = noise - x0
        if diff_ref is not None and diff_weight > 0.0:
            # slider pairs: positions where the two poles differ count more (Krea 2's formula, as Qwen and Anima)
            d = (x0 - diff_ref.to(x0.device).float()).abs().mean(dim=1).flatten(1)     # (B, h*w)
            dm = d.mean(dim=1, keepdim=True)
            r = (d / dm.clamp_min(1e-8)).clamp(max=8.0)
            w = (1.0 - float(diff_weight)) + float(diff_weight) * r
            w = w / w.mean(dim=1, keepdim=True).clamp_min(1e-8)
            w = torch.where(dm > 1e-6, w, torch.ones_like(w))      # identical pair: uniform, never all-zero
            se = (pred - target).pow(2).mean(dim=1).flatten(1)
            return (se * w).mean(), {"t": float(sig.mean())}
        return F.mse_loss(pred, target), {"t": float(sig.mean())}

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        x0 = latents.float()
        sig = self._sample_sigma(x0.shape[0], generator, min_t, max_t).to(x0.device)
        noise = torch.randn(x0.shape, generator=generator).to(x0.device)
        sb = sig.view(-1, 1, 1, 1)
        return {"xt": (1 - sb) * x0 + sb * noise, "t": sig}

    def predict(self, dit, state, cond):
        n = state["xt"].shape[0]
        return self._velocity(dit, state["xt"], state["t"], [_cap(cond, i) for i in range(n)])

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        g = torch.Generator("cpu").manual_seed(int(seed))
        return torch.randn((1, 16, height // 8, width // 8), generator=g, dtype=torch.float32)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None):
        device = next(dit.parameters()).device
        x = (noise if noise is not None else self.initial_noise(seed, width, height)).to(device, torch.float32)
        shift = float(dict(options).get("shift", P.DEFAULT_SHIFT))
        sig = (torch.tensor(list(sigmas) + [0.0]) if sigmas is not None and len(sigmas) == steps
               else P.sigmas(steps, shift))
        cap = [_cap(cond)]
        neg = [_cap(neg_cond)] if (neg_cond is not None and cfg > 1.0) else None
        for i in range(len(sig) - 1):
            if on_step is not None:
                on_step(i, len(sig) - 1)
            s = torch.full((1,), float(sig[i]))
            v = self._velocity(dit, x, s, cap)
            if neg is not None:
                u = self._velocity(dit, x, s, neg)
                v = u + cfg * (v - u)
            x = x + (float(sig[i + 1]) - float(sig[i])) * v
        return x

    def pad_conditioning(self, conds):
        """Prompt travel (LoRA Royale): captions zero-padded to one length with a mask of their real tokens. The model
        lays a padded caption out as the reference does (real tokens, then its pad token to the next multiple of 32),
        so each waypoint renders exactly as its own prompt, and a blend weights the tokens only one side has."""
        L = max(c["cap"].shape[-2] for c in conds)
        out = []
        for c in conds:
            h = c["cap"][0] if c["cap"].dim() == 3 else c["cap"]
            pad = L - h.shape[0]
            out.append({"cap": torch.cat([h, h.new_zeros(pad, h.shape[1])]) if pad else h,
                        "mask": torch.cat([torch.ones(h.shape[0], dtype=torch.bool),
                                           torch.zeros(pad, dtype=torch.bool)])})
        return out

    # ---- other trainers' LoRAs ------------------------------------------------------------------
    def convert_lora_state_dict(self, sd):
        """LoRAs keyed by ComfyUI's own Z-Image names (fused attention.qkv, attention.out - e.g. extracted from a
        fine-tune, or trained in ComfyUI): the qkv pair becomes to_q / to_k / to_v, each with the shared down weight
        and its third of the up rows (exact: ComfyUI applies the fused pair the same way), out becomes to_out.0."""
        if not any(".attention.qkv." in k or ".attention.out." in k for k in sd):
            return sd
        out = {}
        for k, v in sd.items():
            if ".attention.qkv." in k:
                stem, tail = k.split(".attention.qkv.", 1)
                for part, name in enumerate(("to_q", "to_k", "to_v")):
                    nk = f"{stem}.attention.{name}.{tail}"
                    up = any(u in tail for u in ("lora_B", "lora_up", "lora.up"))
                    out[nk] = v.chunk(3, dim=0)[part].contiguous() if up else v
            elif ".attention.out." in k:
                out[k.replace(".attention.out.", ".attention.to_out.0.")] = v
            else:
                out[k] = v
        return out

    def alias_flat(self, flat):
        """kohya-flattened ComfyUI names: layers_N_attention_out -> layers_N_attention_to_out_0 (the fused qkv is split
        by convert_lora_state_dict)."""
        if flat.endswith("_attention_out"):
            return flat[:-len("_out")] + "_to_out_0"
        return None

    # ---- full fine-tune -------------------------------------------------------------------------
    def ft_spec(self, dit):
        """Fine-tuning: the 30 layers' attention and MLP Linears on the NF4 trunk, in four windows of similar size per
        layer (attention ~59M parameters, each MLP matrix ~39M). The refiners, embedders, AdaLN modulation and final
        layer stay frozen (as the LoRA). ComfyUI's single file fuses q / k / v into attention.qkv and names the output
        attention.out; the checkpoint is written back in that layout, so ComfyUI loads it as it loads the base."""
        from fizgig.families.ft import FTSpec
        return FTSpec(blocks="layers", components=("attention", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3"),
                      file_layout=(("attention.to_q.weight", "attention.qkv.weight", 0, 3),
                                   ("attention.to_k.weight", "attention.qkv.weight", 1, 3),
                                   ("attention.to_v.weight", "attention.qkv.weight", 2, 3),
                                   ("attention.to_out.0.weight", "attention.out.weight", 0, 1)),
                      # measured 7 Oct 2026 on a 5090 (6 photos, 0.5 / 1 MP, 1024 preview, adapter on): one window of
                      # all four parts 23.7 / 24.3 GB, attention alone (3 windows under a simulated 16 GB card) 10.5 GB
                      # -> base 3.4 GB + 2 x the window's bf16 weights fits both; +0.6 GB from 0.5 to 1 MP
                      window_factor=2.0, overhead_gb=3.4, calib_mp=0.5, act_gb_per_mp=1.25)

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        img = P.decode_latents(vae, latents)[0].float()
        return Image.fromarray(((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8))
