"""The Vimoo 1.0 Lite family driver: the transformer (vimoo/model.py), Wan 2.2's TI2V VAE (vimoo/vae.py) and the
encoders (vimoo/text.py) behind Fizgig's family interface.

Photos train as one-frame clips (the VAE encodes a still to a single latent frame); clips are 24 fps, 4n + 1 frames.
Training is text-to-video: the 49 conditioning channels (a reference latent and its mask) stay zero, as the pipeline
leaves them for text-to-video, and the timestep is drawn from the distribution the Lite schedule is built on,
sigmoid(ln 8 + 2.5 z). The target is the velocity noise - clean, which the pipeline's Euler steps integrate.
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from fizgig.families.driver import FamilyDriver

logger = logging.getLogger(__name__)

SHIFT, SCHEDULE_STD = 8.0, 2.5            # Lite's lognormal grid (scheduler_config.json)
GUIDANCE_RESCALE = 0.85                   # inference_defaults.json, common
FPS = 24.0
COND_CHANNELS = 49                        # reference latent (48) + mask (1)
PEFT_SCALE = 0.5                          # Vimoo's adapters: lora_alpha = r / 2
ANCHOR_TAU = 0.001                        # image-to-video: the first frame's noise share (i2v_anchor_tau)
REF_GUIDANCE = 3.0                        # image-to-video: the reference scale with CFG (Lite i2v defaults)


def lognorm_sigmas(steps):
    """Lite's schedule: the quantiles of sigmoid(ln shift + std z) from 1 to 0, FP64 then rounded to FP32, the ends
    pinned (VimooTI2VFlowMatchEulerDiscreteScheduler.set_timesteps)."""
    q = torch.linspace(1, 0, int(steps) + 1, dtype=torch.float64)
    grid = torch.sigmoid(math.log(SHIFT) + SCHEDULE_STD * torch.special.ndtri(q)).to(torch.float32)
    grid[0], grid[-1] = 1.0, 0.0
    return grid


def rescale_cfg(cfg_pred, pos, rescale):
    """diffusers' rescale_noise_cfg: the guided prediction's per-sample std brought back toward the positive's."""
    dims = list(range(1, pos.ndim))
    std_pos = pos.std(dim=dims, keepdim=True)
    std_cfg = cfg_pred.std(dim=dims, keepdim=True)
    rescaled = cfg_pred * (std_pos / std_cfg)
    return rescale * rescaled + (1 - rescale) * cfg_pred


class VimooDriver(FamilyDriver):
    keep_vae_resident = False
    # INT8 is weight-only for Vimoo: per-token activation rounding (the shared W8A8 path) mottled plain backgrounds,
    # flickering frame to frame (6 Nov 2026); weight-only keeps INT8's memory with bf16 activations
    int8_weight_only = True

    def __init__(self):
        super().__init__()
        self._glyph_path = ""
        self._siglip_path = ""
        self._frames = 49

    # ---- options ------------------------------------------------------------------------------------------------
    def set_options(self, options):
        super().set_options(options)
        self._glyph_path = options.get("glyph", "") or self._glyph_path
        self._siglip_path = options.get("siglip", "") or self._siglip_path
        try:
            f = int(options.get("sample_frames") or self._frames)
            self._frames = f if f == 1 or (f - 1) % 4 == 0 else self._frames
        except ValueError:
            pass

    # ---- the transformer ------------------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from safetensors.torch import load_file
        from fizgig.vimoo.model import VimooLite
        with torch.device("meta"):
            model = VimooLite()
        sd = load_file(str(path), device=str(device) if str(device) != "meta" else "cpu")
        sd = {k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k: v
              for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if missing or unexpected:
            raise ValueError(f"{path} isn't Vimoo 1.0 Lite: {len(missing)} missing, {len(unexpected)} unexpected "
                             f"keys (e.g. {(missing or unexpected)[:3]})")
        logger.info("Vimoo load_dit: %d tensors from %s", len(sd), os.path.basename(str(path)))
        return model.eval().requires_grad_(False)

    def dit_gb(self, path):
        return os.path.getsize(path) / 1e9 if os.path.isfile(path) else 19.3

    def max_blocks_to_swap(self, dit=None):
        return (len(dit.double_blocks) if dit is not None else self.description.n_blocks) - 2

    def enable_block_swap(self, dit, num_blocks, device, supports_backward=True):
        dit.enable_block_swap(num_blocks, device, supports_backward)
        dit.move_to_device_except_swap_blocks(device)

    def block_swap_mode(self, dit, inference):
        if inference:
            dit.switch_block_swap_for_inference()
        else:
            dit.switch_block_swap_for_training()

    def enable_gradient_checkpointing(self, dit, on=True):
        dit.gradient_checkpointing = bool(on)

    def compile_targets(self, dit):
        return dit.double_blocks

    # ---- the VAE ----------------------------------------------------------------------------------------------------
    def load_vae(self, path, device):
        from fizgig.vimoo import vae as V
        return V.load_vae(path, device, torch.bfloat16)

    @staticmethod
    def _pixels(item):
        """A still (PIL / HWC array) or a clip (a list of them) -> [1, 3, F, H, W] in -1..1, float32."""
        frames = item if isinstance(item, (list, tuple)) else [item]
        arr = np.stack([np.asarray(f.convert("RGB") if hasattr(f, "convert") else f)[..., :3] for f in frames])
        x = torch.from_numpy(arr).float().div_(127.5).sub_(1.0)          # F, H, W, 3
        return x.permute(3, 0, 1, 2).unsqueeze(0)

    def encode_images(self, vae, images):
        """Stills -> (48, h, w); clips -> (48, T, h, w) (the cache's two layouts)."""
        from fizgig.vimoo import vae as V
        out = []
        # on the GPU, as decode() runs: a VAE decode() parked back on the CPU (a card under 20 GB) would otherwise
        # encode in fp32 on the CPU - a prompt slider's practice pictures took tens of minutes each (6 Nov 2026)
        p = next(vae.parameters())
        moved = p.device.type == "cpu" and torch.cuda.is_available()
        if moved:
            vae.to("cuda")
        try:
            for item in images:
                z = V.encode(vae, self._pixels(item))[0].cpu()                   # 48, T, h, w
                out.append(z[:, 0] if not isinstance(item, (list, tuple)) else z)
        finally:
            if moved:
                vae.to("cpu")
                torch.cuda.empty_cache()
        return out

    def still_renders(self):
        """generate() renders stills inside: a prompt slider's practice pictures are stills, whatever the previews'
        length (as H3's)."""
        import contextlib

        @contextlib.contextmanager
        def _stills():
            had = self._frames
            self._frames = 1
            try:
                yield
            finally:
                self._frames = had
        return _stills()

    def cache_stage(self, stage, datasets, args, device, aux):
        """Latents: the shared loop's, with a clip encoded whole (the shared one takes a still's first item). Text:
        the shared loop, with the glyph encoder's path from aux."""
        if stage == "text":
            self._glyph_path = aux.get("glyph", "") or self._glyph_path
            return False
        from fizgig.families.cache import save_latents
        from fizgig.scripts.cache_latents import encode_datasets
        vae = self.load_vae(args.model, device)
        args.keep_alpha = False
        siglip = None
        sp = aux.get("siglip", "") or self._siglip_path

        def needs_reencode(path):
            """A clip cached before SigLIP was set (no first-frame tokens) re-encodes once when it is."""
            if not (sp and os.path.isfile(sp)):
                return False
            from safetensors import safe_open
            try:
                with safe_open(path, framework="pt") as f:
                    keys = list(f.keys())
            except Exception:
                return True
            clip = any(k.startswith("latent_") and k.count("x") == 2 for k in keys)
            return clip and "cond__first_vision" not in keys
        args.needs_reencode = needs_reencode
        if sp and os.path.isfile(sp):
            # each clip's first frame through SigLIP, stored beside its latent: the image-to-video share trains a clip
            # from its own start (its first latent frame is that frame encoded alone - the VAE is causal)
            from fizgig.vimoo.text import SigLIP
            siglip = SigLIP(sp, device)

        def encode(batch):
            from PIL import Image
            for it in batch:
                c = it.content
                clip = isinstance(c, list) and len(c) > 1
                item = c if clip else (c[0] if isinstance(c, list) else c)
                z = self.encode_images(vae, [item])[0]
                extra = None
                if clip and siglip is not None:
                    f0 = c[0] if hasattr(c[0], "convert") else Image.fromarray(np.asarray(c[0])[..., :3])
                    extra = {"cond__first_vision": siglip.encode(f0, f0.width, f0.height).to(torch.float16)}
                logger.info(f"latent cache: {it.item_key} -> {tuple(z.shape)}" + (" + first-frame SigLIP" if extra else ""))
                save_latents(self.description, it, z, extra=extra)
        encode_datasets(datasets, encode, args)
        return True

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        """A one-frame latent -> PIL; a clip -> {"frames": [3, F, H, W] in 0..1, "image": the middle frame, "wave"}."""
        from PIL import Image
        from fizgig.vimoo import vae as V
        z = latents["latent"] if isinstance(latents, dict) else latents
        if z.dim() == 4:
            z = z.unsqueeze(2) if z.shape[0] == 1 else z.unsqueeze(0)
        p = next(vae.parameters())
        try:
            vae.to(p.device if p.device.type != "cpu" or not torch.cuda.is_available() else torch.device("cuda"))
            px = V.decode(vae, z)[0].cpu()                                  # 3, F, H, W in -1..1
        finally:
            if not self.keep_vae_resident:
                vae.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        px = (px / 2 + 0.5).clamp(0, 1)
        if px.shape[1] == 1:
            return Image.fromarray((px[:, 0].permute(1, 2, 0) * 255).byte().numpy())
        mid = (px[:, px.shape[1] // 2].permute(1, 2, 0) * 255).byte().numpy()
        return {"frames": px, "image": Image.fromarray(mid), "wave": None}

    def save_preview(self, result, path):
        """A still: the PNG. A clip: every 2nd frame as JPEG in <stem>.clip/, a silent mp4, then the middle frame as the
        PNG - last, the gallery's 'finished' signal (H3's contract)."""
        from PIL import Image
        if not isinstance(result, dict):
            result.save(path)
            return [path]
        stem, px, out = path[:-4], result["frames"], []
        clip_dir = stem + ".clip"
        os.makedirs(clip_dir, exist_ok=True)
        keep = list(range(0, px.shape[1], 2))
        if keep[-1] != px.shape[1] - 1:
            keep.append(px.shape[1] - 1)
        for k in keep:
            Image.fromarray((px[:, k].permute(1, 2, 0) * 255).byte().numpy()).save(
                os.path.join(clip_dir, f"f{k:03d}.jpg"), quality=87)
        try:
            from fizgig.minimax.common import write_preview_mp4
            write_preview_mp4(stem + ".mp4", px, None, fps=int(FPS))
            out.append(stem + ".mp4")
        except Exception:
            pass
        result["image"].save(path)
        return out + [path]

    # ---- the encoders ---------------------------------------------------------------------------------------------
    # A reference picture is image-to-video's first frame: SigLIP reads it beside the prompt (encode_text_with_references)
    # and its latent anchors the clip's first latent frame (generate / training_loss refs=). Edit-pair datasets train
    # image-to-video the same way: the "before" picture is the clip's start.
    supports_references = True

    def load_text_encoder(self, path, device):
        from fizgig.vimoo.text import GlyphByT5, QwenText
        glyph = GlyphByT5(self._glyph_path) if self._glyph_path and os.path.isfile(self._glyph_path) else None
        if glyph is None:
            logger.warning("Vimoo: no Glyph-ByT5 file set - quoted text in prompts won't be read as lettering")
        return {"qwen": QwenText(path, device), "glyph": glyph}

    def load_reference_text_encoder(self, path, device):
        from fizgig.vimoo.text import SigLIP
        if not (self._siglip_path and os.path.isfile(self._siglip_path)):
            raise ValueError("Vimoo image-to-video needs SigLIP (Preferences: SigLIP so400m) for its reference picture")
        te = self.load_text_encoder(path, device)
        te["siglip"] = SigLIP(self._siglip_path, device)
        return te

    def unload_text_encoder(self, te):
        if isinstance(te, dict):
            te.clear()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def text_encoder_gb(self, path):
        return 15.0

    def encode_text(self, te, captions):
        states = te["qwen"].encode(captions)
        out = []
        for cap, s in zip(captions, states):
            c = {"text": s}
            g = te["glyph"].encode(cap) if te.get("glyph") is not None else None
            if g is not None:
                c["glyph"] = g.to(torch.float16)
            out.append(c)
        return out

    def encode_text_with_references(self, te, captions, references):
        """The prompt as encode_text, plus SigLIP's 729 tokens of each caption's first reference picture (its start
        frame, already at the clip's size)."""
        from PIL import Image
        out = self.encode_text(te, captions)
        for c, refs in zip(out, references):
            if refs:
                im = Image.fromarray(np.asarray(refs[0])[..., :3])
                c["vision"] = te["siglip"].encode(im, im.width, im.height).to(torch.float16)
        return out

    @staticmethod
    def _cond(cond, device):
        """A cached / encoded dict -> the model's text and vision arguments, batched."""
        def hard(m):
            # a travel blend's mask is a weight per token (0..1): a token either side has stays in - its states are
            # already blended toward the zero padding - since the model takes masks as kept / dropped
            if m is None:
                return None
            m = (m > 0) if m.is_floating_point() else m
            return (m.unsqueeze(0) if m.dim() == 1 else m).to(torch.int64)
        text = cond["text"]
        text = text.unsqueeze(0) if text.dim() == 2 else text
        mask = hard(cond.get("text_mask"))
        glyph = cond.get("glyph")
        gmask = None
        if glyph is not None:
            glyph = glyph.unsqueeze(0) if glyph.dim() == 2 else glyph
            gmask = hard(cond.get("glyph_mask"))
            if gmask is None:
                gmask = torch.ones(glyph.shape[:2], dtype=torch.int64)
            glyph, gmask = glyph.to(device), gmask.to(device)
        vision = cond.get("vision")
        if vision is not None and not torch.is_tensor(vision):
            vision = None
        if vision is not None:
            vision = (vision.unsqueeze(0) if vision.dim() == 2 else vision).to(device, torch.bfloat16)
        return dict(text=text.to(device), text_mask=None if mask is None else mask.to(device), glyph=glyph,
                    glyph_mask=gmask, vision=vision)

    # ---- image-to-video: the first frame --------------------------------------------------------------------------
    @staticmethod
    def _ref5(ref, device):
        r = ref if ref.dim() == 5 else ref.unsqueeze(2)                    # (1, 48, 1, h, w)
        return r[:, :, :1].to(device, torch.bfloat16)

    @staticmethod
    def _condition(ref, frames):
        """The 49 conditioning channels: the reference latent in the first frame, zeros after, and the mask."""
        b, c, _, h, w = ref.shape
        cond = ref.new_zeros(b, c + 1, frames, h, w)
        cond[:, :c, :1] = ref
        cond[:, c, :1] = 1.0
        return cond

    @staticmethod
    def _anchor(ref, first_noise):
        """The first frame held almost clean: (1 - tau) reference + tau noise, the coefficient kept in FP32."""
        return ((1.0 - ANCHOR_TAU) * ref.float() + ANCHOR_TAU * first_noise.float()).to(ref.dtype)

    # ---- training ------------------------------------------------------------------------------------------------------
    def _sample_t(self, generator, min_t, max_t):
        z = torch.randn((), generator=generator)
        t = float(torch.sigmoid(math.log(SHIFT) + SCHEDULE_STD * z))
        return min_t + (max_t - min_t) * t

    @staticmethod
    def _as5(latents):
        return latents.unsqueeze(2) if latents.dim() == 4 else latents

    def _model_input(self, x, condition=None):
        if condition is not None:
            return torch.cat([x, condition.to(x.dtype)], dim=1)
        b, _, t, h, w = x.shape
        return torch.cat([x, x.new_zeros(b, COND_CHANNELS, t, h, w)], dim=1)

    def predict(self, dit, state, cond):
        device = state["hidden_states"].device
        x = self._model_input(state["hidden_states"].to(torch.bfloat16))
        return dit(x, state["timestep"].float() * 1000.0, fps=FPS, **self._cond(cond, device))

    def noise_latents(self, latents, generator, *, min_t=0.0, max_t=1.0):
        lat = self._as5(latents)
        t = self._sample_t(generator, min_t, max_t)
        noise = torch.randn(lat.shape, generator=generator).to(lat.device, lat.dtype)
        x_t = (1 - t) * lat + t * noise
        return {"hidden_states": x_t, "timestep": torch.full((lat.shape[0],), t, device=lat.device), "noise": noise,
                "t": t}

    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0, loss_mask=None, prior=None):
        lat = self._as5(latents)
        state = self.noise_latents(lat, generator, min_t=min_t, max_t=max_t)
        share = float(self.options.get("i2v_share") or 0)
        if not refs and share > 0 and lat.shape[2] > 1 and cond.get("first_vision") is not None                 and float(torch.rand((), generator=generator)) < share:
            # the image-to-video share: the clip's own first latent frame is its start picture, SigLIP's tokens of
            # that frame (cached) read beside the prompt - as Vimoo trained text- and image-to-video together
            refs = [lat[:, :, :1]]
            cond = {**cond, "vision": cond["first_vision"]}
        if refs and lat.shape[2] > 1:
            # image-to-video: the pair's picture is the clip's start - anchored almost clean in the first latent frame,
            # read through the conditioning channels and SigLIP, modulated at its own near-zero time; the loss is the
            # frames after it (the pipeline holds the first frame's flow at zero)
            device = lat.device
            ref = self._ref5(refs[0], device)
            x = state["hidden_states"].to(torch.bfloat16)
            x[:, :, :1] = self._anchor(ref, state["noise"][:, :, :1].to(torch.bfloat16))
            xin = self._model_input(x, self._condition(ref, lat.shape[2]))
            pred = dit(xin, state["timestep"].float() * 1000.0, fps=FPS, observed_t=torch.tensor([ANCHOR_TAU * 1000.0]),
                       **self._cond(cond, device)).float()
            target = (state["noise"] - lat).float()
            return F.mse_loss(pred[:, :, 1:], target[:, :, 1:]), {"t": state["t"]}
        from fizgig.families.driver import prior_base, weighted_loss, with_prior
        target = (state["noise"] - lat).float()
        wgt = None
        if diff_ref is not None and diff_weight > 0.0:
            # photo- or clip-pair sliders (Krea 2's weighting, per latent cell): each cell's error weighted by how much
            # the two poles differ there, so the slider learns what changes between them, not what they share
            ref = self._as5(diff_ref).float().to(lat.device)[:, :, :lat.shape[2], :lat.shape[3], :lat.shape[4]]
            d = (lat.float() - ref).abs().mean(dim=1)                          # (1, T, h, w)
            dm = d.mean().clamp_min(1e-8)
            wgt = (1.0 - float(diff_weight)) + float(diff_weight) * (d / dm).clamp(max=8.0)
            wgt = torch.ones_like(d) if float(dm) <= 1e-6 else wgt / wgt.mean().clamp_min(1e-8)
        mw = None
        if loss_mask is not None and lat.shape[2] == 1:
            # masked training (a clip is never masked - the dataset gives masks to stills only, as H3's): one weight
            # per latent cell, the mask sized from the cached latent; the prior's LoRA-off forward runs first, and not
            # at all for an all-white mask (nothing outside it to keep)
            h, w = lat.shape[-2:]
            mw = loss_mask.float().reshape(loss_mask.shape[-2], loss_mask.shape[-1])[:h, :w].to(lat.device)[None, None]
            if prior is not None and float(mw.min()) >= 1.0:
                prior = None
            if not getattr(self, "_mask_logged", False):
                self._mask_logged = True
                logger.info("[masks] Vimoo weighting this step per cell: mask on %s, x%.2f..x%.2f, prior %s",
                            tuple(mw.shape[-2:]), float(mw.min()), float(mw.max()),
                            f"x{float(prior.weight):g}" if prior is not None else "off")
        else:
            prior = None
        if wgt is None and mw is None:
            pred = self.predict(dit, state, cond).float()
            return F.mse_loss(pred, target), {"t": state["t"]}
        base = prior_base(prior, mw, lambda: self.predict(dit, state, cond).float())
        pred = self.predict(dit, state, cond).float()
        loss = weighted_loss((pred - target).pow(2).mean(dim=1), wgt, mw)
        if base is not None:
            loss = with_prior(loss, prior, base, (pred - base).pow(2).mean(dim=1), mw, wgt)
        return loss, {"t": state["t"]}

    # ---- rendering ---------------------------------------------------------------------------------------------------
    def _latent_shape(self, width, height, frames):
        w, h = (int(width) // 32) * 32, (int(height) // 32) * 32
        return (1, 48, (int(frames) - 1) // 4 + 1, h // 16, w // 16)

    @torch.no_grad()
    def initial_noise(self, seed, width, height, frames=None):
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        return torch.randn(self._latent_shape(width, height, frames or self._frames), generator=g,
                           dtype=torch.float32)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None, frames=None, audio=False, **kw):
        """Euler on Lite's lognormal grid (or the distilled LoRA's own timesteps), the state rounded to bf16 after each
        step as the pipeline does. Text-to-video: CFG with guidance rescale when cfg > 1 and a negative is given.
        Image-to-video (refs = the start picture's latent): the first frame anchored and its flow held at zero; with
        CFG, the reference's three branches - unreferenced, negative, positive - mixed at reference scale 3 and the text
        scale. Returns {"latent": (1, 48, T, h, w)}."""
        device = dit.img_in.proj.weight.device          # the input layers stay resident under block swap
        frames = int(frames or self._frames)
        x = (noise if noise is not None else self.initial_noise(seed, width, height, frames)).to(device,
                                                                                                 torch.bfloat16)
        # a schedule made for another step count (the distilled LoRA's 12 timesteps, asked for 15) gives way to Lite's own
        # grid at the steps asked for - before, the steps were silently ignored while the LoRA was on
        if sigmas is not None and len(sigmas) != int(steps) + 1:
            sigmas = None
        sched = torch.tensor(sigmas, dtype=torch.float32) if sigmas is not None else lognorm_sigmas(steps)
        pos = self._cond(cond, device)
        guided = cfg > 1.0 and neg_cond is not None
        neg = self._cond(neg_cond, device) if guided else None
        i2v = bool(refs) and x.shape[2] > 1
        extra, cond_ch, anchor = {}, None, None
        if i2v:
            ref = self._ref5(refs[0], device)
            first_noise = x[:, :, :1].clone()
            anchor = self._anchor(ref, first_noise)
            cond_ch = self._condition(ref, x.shape[2])
            extra = {"observed_t": torch.tensor([ANCHOR_TAU * 1000.0], device=device)}
            if neg is not None and neg.get("vision") is None:
                neg["vision"] = pos.get("vision")
        n = len(sched) - 1
        for i in range(n):
            if on_step is not None:
                on_step(i, n)
            s, s_next = float(sched[i]), float(sched[i + 1])
            t = torch.full((1,), s * 1000.0, device=device)
            if i2v:
                x[:, :, :1] = anchor
            xin = self._model_input(x, cond_ch)
            v = dit(xin, t, fps=FPS, **pos, **extra).float()
            if guided:
                u = dit(xin, t, fps=FPS, **neg, **extra).float()
                if i2v and REF_GUIDANCE != 1.0:
                    # the unreferenced branch: no picture anywhere - its first frame noise, no conditioning channels,
                    # the first frame's time at 1000, SigLIP's tokens masked out
                    xu = self._model_input(x, torch.zeros_like(cond_ch))
                    xu[:, :48, :1] = first_noise
                    un = dict(neg)
                    vis = pos.get("vision")
                    if vis is not None:
                        un["vision"] = torch.zeros_like(vis)
                        un["vision_mask"] = torch.zeros(vis.shape[:2], dtype=torch.bool, device=device)
                    u0 = dit(xu, t, fps=FPS, observed_t=torch.tensor([1000.0], device=device), **un).float()
                    g = u0 + REF_GUIDANCE * (u - u0) + max(cfg, 1.0) * (v - u)
                else:
                    g = u + cfg * (v - u)
                if GUIDANCE_RESCALE > 0:
                    st = 1 if i2v else 0
                    g[:, :, st:] = rescale_cfg(g[:, :, st:], v[:, :, st:], GUIDANCE_RESCALE)
                v = g
            if i2v:
                v[:, :, :1] = 0
            x = (x.float() + v * (s_next - s)).to(torch.bfloat16)
        return {"latent": x.float()}

    # ---- memory plan: a clip counts as its token load ---------------------------------------------------------
    def _heaviest_item(self, group):
        """(effective MP, latent frames) of the dataset's heaviest cached item: a still's area, a clip's area x its
        latent frames (one sequence of tokens either way). Header-only: the cache keys carry the shapes."""
        from safetensors import safe_open
        suffix = f"_{self.description.arch_id}.safetensors"
        best, best_t = 0.0, 1
        for ds in getattr(group, "datasets", []):
            d = getattr(ds, "cache_directory", None)
            if not d or not os.path.isdir(d):
                continue
            for name in os.listdir(d):
                if not name.endswith(suffix):
                    continue
                try:
                    with safe_open(os.path.join(d, name), framework="pt") as f:
                        for k in f.keys():
                            if k.startswith("latent_") and not k.startswith("latent_control_"):
                                dims = [int(x) for x in k[len("latent_"):].split("x")]
                                t = dims[0] if len(dims) == 3 else 1
                                mp = t * dims[-2] * 16 * dims[-1] * 16 / 1e6
                                if mp > best:
                                    best, best_t = mp, t
                except Exception:
                    continue
        return best, best_t

    def plan_run(self, precision, blocks_to_swap, *, group, run):
        """The shared Auto plan (description train_memory), asked at the heaviest item's token load: a 49-frame clip at
        0.4 MP is 13 latent frames x 0.4 = 5.2 MP of tokens, which is what its activations scale with."""
        from fizgig.families import quant
        mp = 0.25
        try:
            mp = max(k[-2] * k[-1] / 1e6 for ds in group.datasets for k in ds.batch_manager.bucket_resos)
        except Exception:
            pass
        eff, t = self._heaviest_item(group)
        if t > 1:
            logger.info(f"[vram] the heaviest item is a {t}-latent-frame clip: planning at its {eff:.2f} MP of tokens")
        return quant.plan(self.description, self, precision, blocks_to_swap, megapixels=max(mp, eff), pictures=1)

    # ---- full fine-tune ---------------------------------------------------------------------------------------------
    def ft_spec(self, dit):
        """The 40 double blocks' Linears in six parts of ~3 GB each in bf16 across the depth (image and text attention,
        qkv + proj; each MLP's up and down halves). The rank-16 modulations, embedders, the text refiner, the glyph and
        vision projections and the final layer stay frozen, as for a LoRA. Memory, measured 6 Nov 2026 on a 16 GB RTX
        4090 Laptop GPU at 0.25 MP (one window per part, no streaming, NF4 trunk): peaks 9.0-9.9 GB, 6.3 GB allocated
        at planning, so overhead = 9.9 - 6.3 + the 4.9 GB 4-bit trunk - the 3.0 GB bf16 window = 5.5 -> 6.0 with slack.
        With it the 16 GB card plans 3 windows of 2 parts: they peaked 11.7-12.5 GB (1.0-1.1 s a step). act_gb_per_mp
        stays the shared cautious 4.0 until a 1 MP run measures it. A rotation at learning rate 1e-30 saved a file equal
        to the source, all 1031 tensors (both plans)."""
        from fizgig.families.ft import FTSpec
        return FTSpec(blocks="double_blocks",
                      components=("img_attn", "img_mlp.0", "img_mlp.2", "txt_attn", "txt_mlp.0", "txt_mlp.2"),
                      overhead_gb=6.0, calib_mp=0.25)

    def ft_load_mp(self, group):
        """A clip plans at its token load (latent frames x megapixels), as plan_run does."""
        mp = super().ft_load_mp(group) or 0.25
        eff, _t = self._heaviest_item(group)
        return max(mp, eff)

    # ---- prompt travel ---------------------------------------------------------------------------------------------
    def pad_conditioning(self, conds):
        """Prompts zero-padded to one length, each with a mask of its own tokens (the model drops masked keys and
        takes the refiner's mean over the real ones), so each waypoint renders as its own prompt."""
        def pad(c, key, mkey, n):
            x = c.get(key)
            if x is None:
                return None, None
            x = x if x.dim() == 2 else x[0]
            m = c.get(mkey)
            m = torch.ones(x.shape[0], dtype=torch.bool) if m is None else (m if m.dim() == 1 else m[0]).bool()
            p = n - x.shape[0]
            return (torch.cat([x, x.new_zeros(p, x.shape[1])]) if p else x,
                    torch.cat([m, m.new_zeros(p)]) if p else m)
        n_text = max((c["text"] if c["text"].dim() == 2 else c["text"][0]).shape[0] for c in conds)
        glyphs = [c["glyph"] if c["glyph"].dim() == 2 else c["glyph"][0] for c in conds if c.get("glyph") is not None]
        n_glyph = max((g.shape[0] for g in glyphs), default=0)
        g_dim = glyphs[0].shape[1] if glyphs else 0
        out = []
        for c in conds:
            t, tm = pad(c, "text", "text_mask", n_text)
            o = {"text": t, "text_mask": tm}
            if n_glyph:
                g, gm = pad(c, "glyph", "glyph_mask", n_glyph)
                if g is None:
                    g = torch.zeros(n_glyph, g_dim, dtype=t.dtype)
                    gm = torch.zeros(n_glyph, dtype=torch.int64)
                o["glyph"], o["glyph_mask"] = g, gm
            if c.get("vision") is not None:
                o["vision"] = c["vision"]
            out.append(o)
        return out

    # ---- LoRA files ------------------------------------------------------------------------------------------------
    _DIFFUSERS = {"attn.to_out.0.projection": "img_attn.proj", "attn.to_add_out.projection": "txt_attn.proj",
                  "img_mlp.in_proj.projection": "img_mlp.0", "img_mlp.out_proj.projection": "img_mlp.2",
                  "txt_mlp.in_proj.projection": "txt_mlp.0", "txt_mlp.out_proj.projection": "txt_mlp.2"}
    _QKV = {"img_attn.qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
            "txt_attn.qkv": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj")}

    def convert_lora_state_dict(self, sd):
        """Vimoo's own adapter format (the distilled LoRA: transformer.double_blocks.N.attn.to_q.lora_A ..., PEFT A/B,
        split q/k/v) -> this family's kohya keys, q/k/v fused into qkv as the model has them (A stacked, B
        block-diagonal - exactly the same product). Its alpha lives in the file's metadata, not in tensors: Vimoo's
        adapters are alpha = rank / 2 (the distilled LoRA: r 128, lora_alpha 64, its sidecar's effective_scale 0.5).
        Other files pass through."""
        if not any(k.startswith("transformer.double_blocks.") and ".lora_A." in k for k in sd):
            return sd
        out = {}
        blocks = sorted({int(k.split(".")[2]) for k in sd if k.startswith("transformer.double_blocks.")})
        for b in blocks:
            pre = f"transformer.double_blocks.{b}."
            for src, dst in self._DIFFUSERS.items():
                a, bb = sd.get(pre + src + ".lora_A.weight"), sd.get(pre + src + ".lora_B.weight")
                if a is None:
                    continue
                stem = f"lora_unet_double_blocks_{b}_{dst.replace('.', '_')}"
                out[stem + ".lora_down.weight"], out[stem + ".lora_up.weight"] = a, bb
                out[stem + ".alpha"] = torch.tensor(PEFT_SCALE * a.shape[0])
            for dst, parts in self._QKV.items():
                As = [sd.get(pre + p + ".lora_A.weight") for p in parts]
                Bs = [sd.get(pre + p + ".lora_B.weight") for p in parts]
                if any(t is None for t in As + Bs):
                    continue
                r = As[0].shape[0]
                down = torch.cat(As, dim=0)                                  # 3r, in
                up = torch.zeros(sum(t.shape[0] for t in Bs), 3 * r, dtype=Bs[0].dtype)
                o = 0
                for i, t in enumerate(Bs):
                    up[o:o + t.shape[0], i * r:(i + 1) * r] = t
                    o += t.shape[0]
                stem = f"lora_unet_double_blocks_{b}_{dst.replace('.', '_')}"
                out[stem + ".lora_down.weight"], out[stem + ".lora_up.weight"] = down, up
                # the same scale alpha / rank on the fused 3r-wide pair
                out[stem + ".alpha"] = torch.tensor(PEFT_SCALE * 3 * r)
        return out
