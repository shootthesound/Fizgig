"""Repair-Studio engine for MiniMax H3 — the parallel of `krea2_engine.Krea2RepairEngine`.

H3's `sampling.sample_image` is a complete sampler (audio carried-variable math included), so
the preview is thin: apply the slider state to the LoRA networks (the model-agnostic
`set_module_*_by_pattern` API) and call `sample_image`. The per-block slider config is the
shared `SliderState`; only the block ids/regex are H3-specific (`h3_blocks`).

**The preview regime is a 22-frame 768x768 clip judged by its MIDDLE frame.** A still is the
most out-of-distribution render H3 has (the trainer's preview-regime lesson: single-frame
previews misrepresent a video model), so the workbench renders the shortest clip that shows
real behaviour and decodes one frame of it.

Base precision is planned from free VRAM: >=28 GB free loads the accurate int8 base (~21 GB,
the checkpoint's own storage); anything less loads NF4-of-pruned (~10.5 GB) with NO block
swap — deliberately, so a future activation-cache resume never has to reason about the H2D
ring's walk-from-swap_from assumption. The community Turbo LoRA (6-step @ 0.75) is applied
once at load when configured — it conditions baseline and tweaked renders equally, and the
bake path reads the LoRA files, never the live network, so saves stay exact.

Public surface mirrors RepairEngine / Krea2RepairEngine so the shared UI and LoRA Royale
workers can drive any of the three.
"""

import gc
import hashlib
import logging
import os
import threading
from typing import Optional, Set

import torch
from PIL import Image

from fizgig.repair_studio.h3_blocks import block_regex_h3, extract_block_ids_h3

logger = logging.getLogger(__name__)

H3_PREVIEW_FRAMES = 22          # shortest clip with real motion; ~7x a still, 1/5th of 124


class _Loaded:
    def __init__(self):
        self.is_loaded = True


def _apply_lora(target, sd, multiplier, device, dtype):
    """Normalize foreign formats, build the network, apply it live, load the weights.
    Mirrors the verified Context-LoRA / preview path (ensure_kohya -> apply_to ->
    load_state_dict — create_network_from_weights only builds STRUCTURE; skipping the
    load leaves lora_up at zero and the LoRA contributes nothing)."""
    from fizgig.networks.lora import create_network_from_weights, ensure_kohya_lora_state_dict
    sd = ensure_kohya_lora_state_dict(sd)
    net = create_network_from_weights(None, float(multiplier), sd, None, target, for_inference=True)
    net.apply_to(text_encoders=None, unet=target, apply_text_encoder=False, apply_unet=True)
    net.load_state_dict(sd, strict=False)
    net.to(device=device, dtype=dtype).eval()
    # The pruned base keeps its AdaLN projections in fp32 (ComfyUI's curve-checkpoint dtype)
    # and Repair Studio never asks the loader to drop that. A LoRA that carries AdaLN keys
    # (any run trained with AdaLN on; AI-Toolkit's 516-key files) then feeds an fp32
    # activation to a bf16 adapter and dies in F.linear. Match each module to the Linear it
    # wraps — the trainer sidesteps this by loading bf16 AdaLN when AdaLN is a target, which
    # a workbench that loads the base once and many LoRAs after cannot do.
    for m in getattr(net, "unet_loras", []):
        org = getattr(getattr(m, "org_forward", None), "__self__", None)
        w = getattr(org, "weight", None)
        if w is not None and w.dtype == torch.float32:
            m.to(torch.float32)
    return net


class H3RepairEngine:
    def __init__(self):
        self.pipeline: Optional[_Loaded] = None
        self.dit = None
        self.decoder = None            # fp16 video VAE decoder, parked on CPU between decodes
        self.te_path: Optional[str] = None
        self.device = "cuda"
        self.dtype = torch.bfloat16

        self.primary_network = None
        self.donor_network = None
        self.primary_path: Optional[str] = None
        self.donor_path: Optional[str] = None
        self.primary_block_ids: Set[str] = set()
        self.donor_block_ids: Set[str] = set()
        self.primary_hash: Optional[str] = None
        self.donor_hash: Optional[str] = None

        # The community Turbo LoRA (6-step previews). Applied ONCE at ensure_pipeline and left
        # enabled — it conditions every render identically, so baseline-vs-tweaked comparisons
        # are apples to apples. (This is the ~780 MB preview accelerator, not Phase C's
        # activation-cache "Turbo Preview".)
        self._turbo_net = None
        self._turbo_adaln = []
        self._steps = 20               # 6 when the Turbo LoRA loads
        self._turbo_strength = 0.75    # the strength the Turbo was loaded at (Confirm regime)
        self._turbo_load_strength = 0.75   # what the AdaLN injection pairs were folded at
        self._turbo_adaln_on = 0.0         # the strength the AdaLN rows are installed at now
        # Lazily-loaded, CPU-parked between uses: the audio VAE decoder (previews with
        # sound) and the video VAE ENCODER (first/last-frame keyframes). Both optional.
        self._audio_vae_path = None
        self.audio_decoder = None
        self.encoder = None

        # Encoded-prompt caches. The TE is Qwen3-VL-32B and takes ~2 minutes to load, so
        # prompts are cached at TWO levels: in-memory for the session, and on DISK keyed by
        # sha256(prompt + te_path) so later sessions never load the TE at all.
        self._prompt_cache_key = None
        self._prompt_cache = None      # CPU [1, L, 5120]
        self._te_cache_dir = None
        self._cancel_event = threading.Event()
        # Optional progress hook: called (step_done, total_steps) once per denoising step,
        # from the render thread. The GUI sets it to drive a determinate progress bar.
        self.on_step = None
        self._baseline_cache_key = None
        self._baseline_cache_image: Optional[Image.Image] = None
        self._baseline_clip_key = None
        self._baseline_clip: Optional[dict] = None
        self._nolora_clip_key = None
        self._nolora_clip: Optional[dict] = None
        self._last_frame_latent = None   # Klein-only chain; kept None for Royale's workers

        # Activation cache. `_turbo_enabled` is the OLD whole-render resume behind
        # generate_preview (measured misleading on H3 — stays False). `resume_enabled` is the
        # EXACT pass-1 resume in render_latent: only step 0 is recorded (~2.3 GB at 768²×22f,
        # ~1 GB at the ⅔ dial canvas) and only step 0 resumes. `_cache_device` is decided at
        # load: "cuda" when the card has the headroom, else "cpu" (a PCIe round trip per
        # render — worth it only if it still nets a saving; see ensure_pipeline).
        self._turbo_enabled = False
        self.resume_enabled = True
        self._cache_device = "cpu"
        self.last_resume_from = None   # what the last render skipped up to (None = full pass)
        self._act_cache = None
        self._act_cache_key = None
        self._act_cache_state = None

    # ----- pipeline + LoRA loading -------------------------------------------
    def ensure_pipeline(self, dit_path: str, vae_path: str, text_encoder_path: str,
                        device: str = "cuda", turbo_lora_path: str = "",
                        turbo_lora_strength: float = 0.75, te_cache_dir: str = "",
                        audio_vae_path: str = "", **_ignored) -> None:
        """Load the DiT (base precision auto-planned from free VRAM) + the Turbo LoRA once.
        The video VAE decoder loads lazily at first decode; the TE loads only on a prompt-cache
        miss (and is freed straight after)."""
        if self.pipeline is not None and self.pipeline.is_loaded:
            return
        from fizgig.minimax.loader import load_minimax_h3_dit
        self.device = device
        self.te_path = text_encoder_path
        self._vae_path = vae_path
        self._audio_vae_path = audio_vae_path or None
        self._te_cache_dir = te_cache_dir or None

        try:
            from fizgig.utils.device import plannable_free_vram
            free = plannable_free_vram()
        except Exception:
            free = 0.0
        base_quant = "int8" if free >= 28.0 else "nf4"
        logger.info("[h3-workbench] %.1f GB free -> %s base, no block swap "
                    "(int8 needs ~21 GB resident + decode headroom; NF4-of-pruned is ~10.5 GB)",
                    free, base_quant)
        self.dit = load_minimax_h3_dit(dit_path, device=device, compute_dtype=self.dtype,
                                       quantize=True, blocks_to_swap=0, base_quant=base_quant)
        self.dit.eval()

        if turbo_lora_path and os.path.exists(turbo_lora_path):
            try:
                from fizgig.minimax.trainer import load_preview_turbo, turbo_adaln_patch
                self._turbo_net, self._turbo_adaln = load_preview_turbo(
                    self.dit, turbo_lora_path, float(turbo_lora_strength))
                self._turbo_net.to(device=device, dtype=self.dtype)
                for _m in self._turbo_net.unet_loras:
                    _m.enabled = True
                n_ad = turbo_adaln_patch(self.dit, self._turbo_adaln, device, self.dtype)
                self._steps = 6
                self._turbo_strength = float(turbo_lora_strength)
                self._turbo_load_strength = float(turbo_lora_strength)
                self._turbo_adaln_on = float(turbo_lora_strength) if n_ad else 0.0
                logger.info("[h3-workbench] Turbo LoRA on for all previews — 6 steps at %g"
                            + (", %d adaln injected" % n_ad if n_ad else ""),
                            float(turbo_lora_strength))
            except Exception:
                logger.exception("[h3-workbench] Turbo LoRA failed to load — previews run "
                                 "the standard 20 steps")
                self._turbo_net, self._turbo_adaln, self._steps = None, [], 20
        else:
            logger.info("[h3-workbench] no Turbo LoRA configured — previews run 20 steps "
                        "(set it in Preferences for 6-step previews)")
        # Pass-1 resume cache placement. The step-0 block inputs are 1.7 GB at 768×640×22f
        # (~0.8 GB at the ⅔ dial canvas). Measured 3 Sep on the 5090 (int8 plan): a
        # CPU-parked cache costs +0.2 s per render for the PCIe round trip and the resumed
        # render saves exactly what the GPU-parked one does (4.22 s vs 4.18 s at block 45,
        # plain 5.71 s), so the resume is ON everywhere; the GPU only hosts the cache when
        # there is headroom left for the clip decoder beside it.
        try:
            from fizgig.utils.device import plannable_free_vram
            free_after = plannable_free_vram()
        except Exception:
            free_after = 0.0
        self.resume_enabled = True
        self._cache_device = "cuda" if free_after >= 8.0 else "cpu"
        logger.info("[h3-workbench] exact pass-1 resume ON, step-0 cache on the %s "
                    "(%.1f GB free after the DiT)", "GPU" if self._cache_device == "cuda"
                    else "CPU", free_after)
        # Warm the clip decoder now, on the CPU: the first decode of a session otherwise
        # pays ~10 s (4.8 GB off disk + first CUDA calls) in the middle of the first preview
        # — measured 3 Sep: 11.6 s cold, 1.8 s every decode after.
        try:
            self._ensure_decoder()
        except Exception:
            logger.exception("[h3-workbench] decoder pre-load failed (it loads at first decode)")
        self.pipeline = _Loaded()

    def _ensure_decoder(self):
        if self.decoder is not None:
            return self.decoder
        from safetensors import safe_open
        from fizgig.minimax.vae import MiniMaxH3VideoVAEDecoder
        dec = MiniMaxH3VideoVAEDecoder()
        with safe_open(self._vae_path, framework="pt", device="cpu") as f:
            dec.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, strict=False)
        # FP16, not bf16 — the weights ship fp16 and this decoder was excluded from bf16 by
        # ComfyUI on purpose (see the trainer's decode-phase comment for the full story).
        self.decoder = dec.to(torch.float16).eval()
        return self.decoder

    def _ensure_audio_decoder(self):
        """The audio VAE's decoder half (~0.45 GB), or None when no audio VAE is configured."""
        if self.audio_decoder is not None:
            return self.audio_decoder
        if not self._audio_vae_path or not os.path.exists(self._audio_vae_path):
            return None
        from fizgig.minimax.audio_vae import load_minimax_h3_audio_vae_decoder
        self.audio_decoder = load_minimax_h3_audio_vae_decoder(
            self._audio_vae_path, device="cpu", dtype=torch.float32).eval()
        return self.audio_decoder

    def _ensure_encoder(self):
        """The video VAE ENCODER (first/last-frame keyframes). fp32 like the caching script;
        parked on CPU between uses."""
        if self.encoder is not None:
            return self.encoder
        from safetensors import safe_open
        from fizgig.minimax.vae import MiniMaxH3VideoVAEEncoder
        enc = MiniMaxH3VideoVAEEncoder()
        with safe_open(self._vae_path, framework="pt", device="cpu") as f:
            enc.load_state_dict({k: f.get_tensor(k) for k in f.keys()}, strict=False)
        self.encoder = enc.to(torch.float32).eval()
        return self.encoder

    def set_turbo_strength(self, strength: float) -> None:
        """Re-dial the built-in Turbo LoRA live. 0 switches it OFF — every module disabled
        and the AdaLN injection removed — so the render is the base plus your LoRAs at the
        steps you chose (how a Turbo LoRA loaded as the primary is edited on its own). Any
        other strength scales the modules AND re-installs the injected AdaLN rows at that
        strength: they were folded at load strength, so they are rescaled by strength /
        load strength (until 4 Sep they stayed at load strength whatever the dial said)."""
        if self._turbo_net is None:
            return
        s = float(strength)
        for _m in self._turbo_net.unet_loras:
            _m.enabled = abs(s) > 1e-9
            _m.multiplier = s
        self._turbo_strength = s
        self._apply_turbo_adaln(s)

    def _apply_turbo_adaln(self, strength: float) -> None:
        """Install the Turbo AdaLN injection at `strength` (0 = remove it). Instance-attribute
        forwards, so removal is a plain delete — see trainer.turbo_adaln_patch."""
        pairs = getattr(self, "_turbo_adaln", None) or []
        cur = float(getattr(self, "_turbo_adaln_on", 0.0) or 0.0)
        s = float(strength)
        if not pairs or abs(s - cur) < 1e-9:
            return
        for mod, _a, _b in pairs:
            if "forward" in mod.__dict__:
                del mod.forward
        self._turbo_adaln_on = 0.0
        if abs(s) < 1e-9:
            return
        load = float(getattr(self, "_turbo_load_strength", 0.0) or 0.0)
        if abs(load) < 1e-9:
            return                      # folded at 0 — nothing to rescale
        try:
            from fizgig.minimax.trainer import turbo_adaln_patch
            f = s / load
            n = turbo_adaln_patch(self.dit, [(m, a, b * f) for m, a, b in pairs],
                                  self.device, self.dtype)
            self._turbo_adaln_on = s if n else 0.0
        except Exception:
            logger.exception("Turbo AdaLN re-injection failed — rows stay off at this strength")

    # ----- keyframes ---------------------------------------------------------
    @torch.no_grad()
    def encode_keyframe(self, image: Image.Image, width: int, height: int):
        """A PIL image -> keyframe latent [1, 24, 1, H/16, W/16] at the clip's OWN canvas.

        The caller crops (aspect-locked box); this only resizes to the exact canvas so the
        latent grid matches the target's — the DiT refuses anything else. Encoder to GPU for
        the call, parked back to CPU after (it is only ever used at conditioning time)."""
        w, h = int(width), int(height)
        w, h = (w // 32) * 32, (h // 32) * 32          # even latent grid, like the sampler
        img = image.convert("RGB")
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)
        import numpy as np
        arr = torch.from_numpy(np.asarray(img).copy()).float().permute(2, 0, 1) / 127.5 - 1.0
        enc = self._ensure_encoder().to(self.device)
        try:
            z = enc.encode(arr.unsqueeze(0).to(self.device))     # [1, 24, 1, h, w]
        finally:
            self.encoder = enc.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return z.float().cpu()

    def load_primary(self, path: str) -> None:
        if self.pipeline is None or not self.pipeline.is_loaded:
            raise RuntimeError("Pipeline not loaded; call ensure_pipeline() first.")
        if self.primary_network is not None:
            raise RuntimeError("Primary already loaded — call reset() to swap.")
        from safetensors.torch import load_file
        self.primary_network = _apply_lora(self.dit, load_file(path), 1.0, self.device, self.dtype)
        self.primary_path = path
        self.primary_block_ids = extract_block_ids_h3(self.primary_network)
        self._invalidate_baseline_cache()
        try:
            from fizgig.profiler.visualize import compute_lora_hash
            self.primary_hash = compute_lora_hash(path)
        except Exception:
            self.primary_hash = None
        logger.info("H3 primary loaded: %s (%d blocks)", path, len(self.primary_block_ids))

    def swap_primary_weights(self, path: str) -> bool:
        """LoRA Royale fast path: same-structure weight swap in place (epochs of one run).
        Returns False on a structure mismatch — caller resets and reloads."""
        if self.primary_network is None:
            raise RuntimeError("No primary loaded; call load_primary() first.")
        from fizgig.networks.lora import ensure_kohya_lora_state_dict
        from safetensors.torch import load_file
        try:
            sd = ensure_kohya_lora_state_dict(load_file(path))
        except Exception:
            logger.exception("swap_primary_weights: failed to load %s", path)
            return False
        try:
            info = self.primary_network.load_state_dict(sd, strict=False)
        except Exception as e:
            logger.info("swap_primary_weights: structure/shape mismatch, needs full reload (%s)", e)
            return False
        if sd and len(info.unexpected_keys) > 0.5 * len(sd):
            return False
        self.primary_network.to(device=self.device, dtype=self.dtype)
        self.primary_path = path
        self.primary_block_ids = extract_block_ids_h3(self.primary_network)
        try:
            from fizgig.profiler.visualize import compute_lora_hash
            self.primary_hash = compute_lora_hash(path)
        except Exception:
            self.primary_hash = None
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()
        return True

    def load_donor(self, path: str) -> None:
        if self.primary_network is None:
            raise RuntimeError("Load primary LoRA before donor.")
        if self.donor_network is not None:
            raise RuntimeError("Donor already loaded — unload_donor() or reset() first.")
        from safetensors.torch import load_file
        net = _apply_lora(self.dit, load_file(path), 1.0, self.device, self.dtype)
        net.set_enabled(False)  # donor blocks are opt-in per-slider
        self.donor_network = net
        self.donor_path = path
        self.donor_block_ids = extract_block_ids_h3(net)
        try:
            from fizgig.profiler.visualize import compute_lora_hash
            self.donor_hash = compute_lora_hash(path)
        except Exception:
            self.donor_hash = None
        logger.info("H3 donor loaded: %s (%d blocks)", path, len(self.donor_block_ids))

    def unload_donor(self) -> None:
        if self.donor_network is not None:
            self.donor_network.set_enabled(False)
            self.donor_network = None
            self.donor_path = None
            self.donor_hash = None
            self.donor_block_ids = set()

    def cache_key_for(self, state, *, frames, regime, steps=None, turbo_strength=None,
                      **_ignored) -> Optional[str]:
        """The render-cache setup key for this state's render setup, or None before a
        primary is loaded. Sound doesn't enter the key: audio rows are part of every entry."""
        if self.primary_network is None or not self.primary_hash:
            return None
        from fizgig.repair_studio.h3_render_cache import setup_key
        steps, strength = self.regime_params(regime, steps, turbo_strength)
        frames = int(frames or getattr(state, "preview_frames", 0) or H3_PREVIEW_FRAMES)
        return setup_key(primary_hash=self.primary_hash, donor_hash=self.donor_hash or "",
                         prompt=state.prompt, seed=int(state.seed), frames=frames,
                         width=int(state.preview_width), height=int(state.preview_height),
                         steps=int(steps), turbo_strength=strength,
                         keyframe_sig=self.keyframe_signature(state),
                         primary_scale=float(getattr(state, "primary_scale", 1.0)),
                         donor_scale=float(getattr(state, "donor_scale", 1.0)))

    # ----- slider state ------------------------------------------------------
    def apply_state(self, state) -> None:
        """Push the per-block slider config into the live networks (regex-based, no reload)."""
        if self.primary_network is None:
            return
        # Load strength: each slider is relative to it (see SliderState.primary_scale).
        ps = float(getattr(state, "primary_scale", 1.0))
        ds = float(getattr(state, "donor_scale", 1.0))
        for bid, bs in state.blocks.items():
            try:
                pat = block_regex_h3(bid)
            except ValueError:
                continue
            self.primary_network.set_module_enabled_by_pattern(pat, bool(bs.primary_enabled))
            self.primary_network.set_module_multiplier_by_pattern(pat, float(bs.primary_strength) * ps)
            if self.donor_network is not None:
                self.donor_network.set_module_enabled_by_pattern(pat, bool(bs.donor_enabled))
                self.donor_network.set_module_multiplier_by_pattern(pat, float(bs.donor_strength) * ds)

    # ----- cancellation ------------------------------------------------------
    def request_cancel(self) -> None:
        self._cancel_event.set()

    def clear_cancel(self) -> None:
        self._cancel_event.clear()

    # ----- prompt encoding (two-level cache) ---------------------------------
    def _prompt_disk_path(self, prompt: str) -> Optional[str]:
        if not self._te_cache_dir:
            return None
        h = hashlib.sha256((prompt + "\x00" + os.path.basename(self.te_path or ""))
                           .encode("utf-8")).hexdigest()
        return os.path.join(self._te_cache_dir, f"{h}.safetensors")

    def _encode_prompt(self, prompt: str):
        """[1, L, 5120] on CPU. In-memory hit -> free; disk hit -> milliseconds; miss -> the
        32B TE loads once (couple of minutes), encodes, frees, and the result persists so no
        future session pays again."""
        if self._prompt_cache_key == prompt and self._prompt_cache is not None:
            return self._prompt_cache
        disk = self._prompt_disk_path(prompt)
        if disk and os.path.exists(disk):
            from safetensors.torch import load_file
            emb = load_file(disk)["hidden_states"].unsqueeze(0)
            self._prompt_cache_key, self._prompt_cache = prompt, emb
            return emb
        from fizgig.minimax.sampling import encode_sample_prompts
        logger.info("[h3-workbench] encoding prompt with the 32B TE (one-off per prompt — "
                    "cached to disk after this)")
        # The TE (~15.7 GB) and the resident base must never be co-resident (the trainer's
        # rule; the int8 base alone is ~21 GB). Park the DiT + Turbo net for the encode and
        # restore after — safe as a whole-model .to because this engine never block-swaps.
        _parked = False
        if self.dit is not None:
            self.dit.to("cpu")
            if self._turbo_net is not None:
                self._turbo_net.to("cpu")
            _parked = True
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            emb = encode_sample_prompts(self.te_path, [prompt], device=self.device)[0]
        finally:
            if _parked:
                self.dit.to(self.device)
                if self._turbo_net is not None:
                    self._turbo_net.to(device=self.device, dtype=self.dtype)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        emb = emb.cpu()
        if disk:
            try:
                os.makedirs(self._te_cache_dir, exist_ok=True)
                from safetensors.torch import save_file
                save_file({"hidden_states": emb[0].contiguous()}, disk,
                          metadata={"prompt": prompt[:512], "te": os.path.basename(self.te_path or "")})
            except Exception:
                logger.exception("prompt disk-cache write failed (render continues)")
        self._prompt_cache_key, self._prompt_cache = prompt, emb
        return emb

    # ----- preview -----------------------------------------------------------
    def generate_preview(self, state, *, seed: Optional[int] = None,
                         prompt: Optional[str] = None, width: Optional[int] = None,
                         height: Optional[int] = None, steps: Optional[int] = None,
                         num_frames: Optional[int] = None,
                         seed_b: Optional[int] = None, travel_t: float = 0.0,
                         override_ctx=None, override_neg_ctx=None,
                         prev_latent=None, prev_latent_strength: float = 1.0) -> Image.Image:
        """Apply the slider state, render a 22-frame clip, return its MIDDLE frame.

        seed_b / travel_t / override_neg_ctx / prev_latent are accepted for call-signature
        parity with the other engines (LoRA Royale's shared workers); seed-travel and the
        Klein latent chain have no H3 wiring yet and are ignored. override_ctx (a precomputed
        [1, L, 5120] embed) drives prompt-travel."""
        from fizgig.minimax import sampling
        self.apply_state(state)
        prompt = prompt if prompt is not None else state.prompt
        seed = seed if seed is not None else state.seed
        width = width or state.preview_width
        height = height or state.preview_height
        steps = steps or self._steps
        frames = int(num_frames or H3_PREVIEW_FRAMES)

        emb = override_ctx if override_ctx is not None else self._encode_prompt(prompt)
        if isinstance(emb, tuple):          # travel waypoints carry (txt, None)
            emb = emb[0]

        def _abort_check(_seconds, _step, _total):
            cb = self.on_step
            if cb is not None:
                try:
                    cb(_step, _total)
                except Exception:
                    pass
            return True if self._cancel_event.is_set() else None

        # Activation-cache resume — OFF by default and NOT exposed in the H3 GUI. Measured on
        # the real 33B (18 Aug): the resume is 3-4x faster (3-4s vs 12.5s) but a resumed
        # render retains only ~6% of a block tweak's visible effect — on a 50-block model
        # most of the effect comes from the perturbation re-entering the EARLY blocks on
        # later steps, which a per-step resume never re-runs (Krea 2 hit the same wall at 28
        # blocks / 8 steps). Kept for programmatic use and as the base for a future
        # multi-step-aware cache; _turbo_enabled stays False for previews.
        cache_key = (self.primary_path, self.donor_path, int(seed), prompt,
                     width, height, frames)
        ctx = None
        if self._turbo_enabled and override_ctx is None and seed_b is None:
            resume = None
            if self._act_cache_key == cache_key and self._act_cache:
                resume = self._resume_from_diff(state)
            ctx = sampling.BlockCacheContext(
                entries=self._act_cache if self._act_cache_key == cache_key else {},
                resume_from=resume, cache_device="cpu")

        def _render(block_cache):
            with torch.no_grad():
                return sampling.sample_image(
                    self.dit, emb.to(self.device, self.dtype),
                    width=width, height=height, steps=steps, cfg_scale=1.0,
                    seed=int(seed), device=self.device, dtype=self.dtype,
                    num_frames=frames, on_slow_step=_abort_check, slow_step_s=0.0,
                    block_cache=block_cache, keyframes=getattr(state, "keyframes", None),
                    exact_frames=True)

        if ctx is not None:
            try:
                lat = _render(ctx)
                self._act_cache = ctx.new_entries
                self._act_cache_key = cache_key
                self._act_cache_state = state.copy()
            except sampling.PreviewAborted:
                raise
            except Exception:
                logger.exception("Turbo Preview failed — falling back to a full forward")
                self._invalidate_activation_cache()
                lat = _render(None)
        else:
            lat = _render(None)

        dec = self._ensure_decoder().to(self.device)
        try:
            px = dec.decode_middle_frame(lat.float())[0]      # [3, H, W] in [0, 1]
        finally:
            self.decoder = dec.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        arr = (px.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        return Image.fromarray(arr)

    # ----- clip primitives (Repair Studio video mode / effect lattice) --------
    def render_latent(self, state, *, seed: Optional[int] = None,
                      prompt: Optional[str] = None, width: Optional[int] = None,
                      height: Optional[int] = None, frames: Optional[int] = None,
                      steps: Optional[int] = None, turbo_strength: Optional[float] = None,
                      keyframes=None, on_denoised=None, no_lora: bool = False):
        """Apply the slider state and sample ONE clip: returns (latent [1,24,T,H/16,W/16] on
        CPU fp32, audio_rows [2*A, 32] on CPU fp32 or None). No decode — the caller decides
        (decode_clip_frames / decode_audio, or store it in the render cache).

        keyframes: [(pixel_frame_index, latent)] from encode_keyframe at THIS width/height;
        defaults to state.keyframes when the state carries them. turbo_strength re-dials the
        Turbo modules for this render (Dial 1.0 / Confirm 0.75) and restores nothing — the
        caller owns the regime; see set_turbo_strength. on_denoised(step, n, x0_estimate)
        fires after every pass (the "show early" hook).

        **Exact pass-1 resume.** When the previous render had the same setup (LoRA, donor,
        seed, prompt, canvas, length, steps, Turbo strength) and only blocks >= k changed,
        the FIRST pass skips blocks < k by restoring their cached input — bit-identical to a
        full pass, because nothing before k saw a different input yet. Passes 2..n always run
        in full: their input latent already carries the change, so nothing there is reusable
        (resuming them anyway was measured to keep ~6% of a tweak's effect — never again).
        Only pass 1 is recorded (one step's block inputs, not a whole render's). Off with
        keyframes (forward_cached doesn't lay them out). Measured 3 Sep at 768×640×22 Dial:
        bit-exact, saves 2% / 12% / 22% of the render for a change at block 5 / 25 / 45.

        no_lora: the base model on its own — every primary / donor module off for this render
        (the Turbo regime stays: it is the sampler, not the LoRA under test); the state is
        re-applied afterwards so the next render is unaffected, and the resume sits out."""
        from fizgig.minimax import sampling
        self.apply_state(state)
        if no_lora and self.primary_network is not None:
            self.primary_network.set_enabled(False)
            if self.donor_network is not None:
                self.donor_network.set_enabled(False)
        prompt = prompt if prompt is not None else state.prompt
        seed = seed if seed is not None else state.seed
        width = width or state.preview_width
        height = height or state.preview_height
        steps = steps or self._steps
        frames = int(frames or getattr(state, "preview_frames", 0) or H3_PREVIEW_FRAMES)
        if keyframes is None:
            keyframes = getattr(state, "keyframes", None)
        if turbo_strength is not None:
            self.set_turbo_strength(turbo_strength)
        emb = self._encode_prompt(prompt)

        def _abort_check(_seconds, _step, _total):
            cb = self.on_step
            if cb is not None:
                try:
                    cb(_step, _total)
                except Exception:
                    pass
            return True if self._cancel_event.is_set() else None

        ctx = None
        key = (self.primary_path, self.donor_path, int(seed), prompt, int(width), int(height),
               int(frames), int(steps), self._turbo_strength,
               round(float(getattr(state, "primary_scale", 1.0)), 4),
               round(float(getattr(state, "donor_scale", 1.0)), 4))
        if (self.resume_enabled and not keyframes and not no_lora
                and hasattr(self.dit, "forward_cached")):
            resume = None
            entries = {}
            if self._act_cache_key == key and self._act_cache:
                entries = self._act_cache
                resume = self._resume_from_diff(state)
            elif self._act_cache:
                # Another setup's cache is dead weight — on the GPU it can be several GB
                # (56 frames at 768² records ~5.1 GB), and it must go BEFORE the free-VRAM
                # check below, not sit there while this render fights for room.
                self._invalidate_activation_cache()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            dev = self._resume_cache_device(width, height, frames)
            ctx = sampling.BlockCacheContext(entries=entries, resume_from=resume,
                                             cache_device=dev, record_steps={0})
            self.last_resume_from = resume

        def _render(block_cache):
            with torch.no_grad():
                return sampling.sample_image(
                    self.dit, emb.to(self.device, self.dtype),
                    width=width, height=height, steps=steps, cfg_scale=1.0,
                    seed=int(seed), device=self.device, dtype=self.dtype,
                    num_frames=frames, on_slow_step=_abort_check, slow_step_s=0.0,
                    return_audio=True, keyframes=keyframes, block_cache=block_cache,
                    on_denoised=on_denoised, exact_frames=True)

        try:
            resume_failed = False
            if ctx is not None:
                try:
                    lat, audio = _render(ctx)
                    e0 = ctx.new_entries.get(0)
                    if e0 is not None:
                        self._act_cache = {0: e0}
                        self._act_cache_key = key
                        self._act_cache_state = state.copy()
                except sampling.PreviewAborted:
                    raise
                except Exception:
                    logger.exception("pass-1 resume failed — falling back to a full render")
                    resume_failed = True
            if resume_failed:
                # OUTSIDE the except block on purpose: inside it the exception's traceback
                # keeps every frame of the failed pass alive — including the half-filled
                # cache entry (~5 GB at 56 frames × 768²) — so a fallback run from there
                # ran out of VRAM too (measured 4 Sep). Out here the traceback is gone;
                # drop what ctx still holds, collect, and give the allocator its room back.
                self._invalidate_activation_cache()
                ctx.new_entries.clear()
                ctx.entries = {}
                ctx = None
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                lat, audio = _render(None)
            elif ctx is None:
                lat, audio = _render(None)
        finally:
            if no_lora and self.primary_network is not None:
                self.primary_network.set_enabled(True)
                self.apply_state(state)               # per-block flags back as they were
        lat = lat.detach().float().cpu()
        audio = audio.detach().float().cpu() if audio is not None else None
        return lat, audio

    @torch.no_grad()
    def decode_clip_frames(self, latent) -> "list[Image.Image]":
        """A clip latent -> PIL frames (all of them). At 22 frames the whole clip is one
        decoder chunk, so this costs what the middle-frame decode already did."""
        dec = self._ensure_decoder().to(self.device)
        try:
            z = latent.to(self.device).float()
            if z.shape[2] == 1:
                # A still: the single-latent decode (the 5n+2 clip grid has no T=1 rung).
                px = dec.decode(z)[0].unsqueeze(1)                    # [3, 1, H, W]
            else:
                px = dec.decode_clip(z)[0]                            # [3, F, H, W] in [0, 1]
            px = (px.permute(1, 2, 3, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        finally:
            self.decoder = dec.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return [Image.fromarray(px[i]) for i in range(px.shape[0])]

    @torch.no_grad()
    def decode_audio(self, audio_rows):
        """[2*A, 32] denoised audio rows -> stereo waveform [2, L] float in [-1, 1] on CPU,
        or None when no audio VAE is configured (previews then play silent)."""
        if audio_rows is None:
            return None
        dec = self._ensure_audio_decoder()
        if dec is None:
            return None
        from fizgig.minimax.audio_vae import unpack_audio
        dec = dec.to(self.device)
        try:
            z = unpack_audio(audio_rows.to(self.device).float())   # [1, 32, 2, A]
            wav = dec.decode(z)[0].float().clamp(-1, 1).cpu()      # [2, L]
        finally:
            self.audio_decoder = dec.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return wav

    # ----- clip bundles (the Repair Studio video preview) --------------------
    # A "clip" is what the H3 tab actually shows: every decoded frame (for the in-app player),
    # the soundtrack when an audio VAE is configured, and the middle frame (what the main
    # panel, the metrics strip and the Royale workers still judge). Dial = 4 steps at Turbo
    # 1.0 (the fast loop), Confirm = 6 at 0.75 (the render that matches training previews).
    REGIMES = {"dial": (4, 1.0), "confirm": (6, 0.75)}

    def regime_params(self, regime: str, steps=None, turbo_strength=None):
        """(steps, turbo_strength) for a regime name — the preset (Dial 4 @ 1.0, Confirm
        6 @ 0.75) unless the caller dials its own numbers: `steps` and `turbo_strength`
        override (0 = the Turbo switched off for the render). Without a Turbo LoRA loaded
        the strength has nothing to dial (None) and the steps default to the plain 20."""
        if self._turbo_net is None:
            return (int(steps) if steps else self._steps), None
        st, tu = self.REGIMES.get(regime, self.REGIMES["confirm"])
        if steps:
            st = max(1, int(steps))
        if turbo_strength is not None:
            tu = float(turbo_strength)
        return st, tu

    @staticmethod
    def keyframe_signature(state):
        """A hashable stand-in for state.keyframes (index + tensor fingerprint per entry) —
        cheap enough for a cache key, specific enough that a different crop re-renders."""
        kf = getattr(state, "keyframes", None)
        if not kf:
            return ()
        sig = []
        for idx, lat in kf:
            t = lat.float()
            sig.append((int(idx), tuple(t.shape), round(float(t.sum()), 3),
                        round(float(t.abs().mean()), 5)))
        return tuple(sig)

    def clip_key(self, state, *, frames, steps, turbo_strength, with_audio):
        return (self.primary_path, self.donor_path, int(state.seed), state.prompt,
                int(state.preview_width), int(state.preview_height), int(frames), int(steps),
                turbo_strength, bool(with_audio), self.keyframe_signature(state),
                round(float(getattr(state, "primary_scale", 1.0)), 4),
                round(float(getattr(state, "donor_scale", 1.0)), 4))

    def render_clip(self, state, *, frames: Optional[int] = None, regime: str = "confirm",
                    with_audio: bool = True, cache=None, early_step: int = 0,
                    on_early=None, no_lora: bool = False, steps=None, turbo_strength=None,
                    **_ignored) -> dict:
        """Render + decode one clip for the slider state. Returns
        {"latent", "audio_rows", "frames": [PIL...], "wav": [2, L] or None, "middle": PIL,
         "regime", "steps", "turbo_strength", "frames_n", "cached": bool}.

        cache: a RenderCache for this setup — a state rendered before is served from it
        (decode only, "cached": True); a fresh render is stored into it under the state's
        signature. early_step + on_early: "show early" — after pass `early_step` the
        clean-latent estimate's middle frame is decoded and handed to on_early(pil, step, n)
        while the remaining passes run. Never fires on a cache hit. no_lora renders the
        base model alone (see render_latent) under the "nolora" signature."""
        from fizgig.repair_studio.h3_render_cache import signature, NOLORA_SIG
        steps, strength = self.regime_params(regime, steps, turbo_strength)
        frames = int(frames or getattr(state, "preview_frames", 0) or H3_PREVIEW_FRAMES)
        sig = NOLORA_SIG if no_lora else signature(state)
        hit = cache.get(sig) if cache is not None else None
        if hit is not None:
            lat, aud = hit
            cached = True
        else:
            def _on_denoised(step, n, x0):
                if on_early is None or step != int(early_step):
                    return
                img = self.decode_middle_frame_image(x0)
                on_early(img, step, n)

            lat, aud = self.render_latent(state, frames=frames, steps=steps,
                                          turbo_strength=strength,
                                          on_denoised=_on_denoised if early_step > 0 else None,
                                          no_lora=no_lora)
            cached = False
        imgs = self.decode_clip_frames(lat)
        wav = self.decode_audio(aud) if (with_audio and frames > 1) else None
        clip = {"latent": lat, "audio_rows": aud, "frames": imgs, "wav": wav,
                "middle": imgs[len(imgs) // 2], "regime": regime, "steps": steps,
                "turbo_strength": strength, "frames_n": frames, "cached": cached, "sig": sig}
        if cache is not None and not cached:
            try:
                cache.put(sig, lat, aud, middle=clip["middle"], regime=regime,
                          label="No LoRA" if no_lora else self.describe_state(state))
            except Exception:
                logger.exception("render cache: put failed (render still shown)")
        return clip

    def clip_from_cache(self, cache, sig: str, *, regime: str = "dial",
                        with_audio: bool = True, steps=None, turbo_strength=None) -> Optional[dict]:
        """A clip dict straight from a cached entry (history strip, peeks, pinned baseline)
        — decode only, no state needed. None when the entry isn't there."""
        hit = cache.get(sig) if cache is not None else None
        if hit is None:
            return None
        lat, aud = hit
        steps, strength = self.regime_params(regime, steps, turbo_strength)
        imgs = self.decode_clip_frames(lat)
        wav = self.decode_audio(aud) if (with_audio and len(imgs) > 1) else None
        return {"latent": lat, "audio_rows": aud, "frames": imgs, "wav": wav,
                "middle": imgs[len(imgs) // 2], "regime": regime, "steps": steps,
                "turbo_strength": strength, "frames_n": len(imgs), "cached": True,
                "sig": sig, "label": cache.info(sig).get("label", "")}

    @torch.no_grad()
    def decode_middle_frame_image(self, latent) -> Image.Image:
        """One PIL frame (the clip's middle) from a latent — the early-look decode."""
        dec = self._ensure_decoder().to(self.device)
        try:
            px = dec.decode_middle_frame(latent.to(self.device).float())[0]
            arr = (px.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        finally:
            self.decoder = dec.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return Image.fromarray(arr)

    @staticmethod
    def describe_state(state) -> str:
        """A short human label for a slider state ("Block 30 off", "3 blocks moved")."""
        from fizgig.repair_studio.h3_render_cache import signature, BASE_SIG
        sig = signature(state)
        if sig == BASE_SIG:
            return "Baseline"
        if sig.startswith("off:"):
            bid = sig[4:]
            return ("Refiner " + bid.split("_")[2] if bid.startswith("h3_rf_")
                    else "Block " + bid.split("_")[1]) + " off"
        moved = [b for b, bs in state.blocks.items()
                 if not (bs.primary_enabled and abs(bs.primary_strength - 1.0) < 1e-6
                         and abs(bs.donor_strength) < 1e-6)]
        return f"{len(moved)} block{'s' if len(moved) != 1 else ''} moved"

    def baseline_clip(self, state, *, frames: Optional[int] = None, regime: str = "confirm",
                      with_audio: bool = True, cache=None, steps=None, turbo_strength=None,
                      **_ignored) -> dict:
        """The clip for the primary at its load strength / all on, donor off — cached in
        memory on everything the render depends on (a slider move never re-renders it; a
        regime, length, size, seed, prompt, scale or keyframe change does) and, through
        `cache`, on disk."""
        from fizgig.repair_studio.state import SliderState
        steps, strength = self.regime_params(regime, steps, turbo_strength)
        frames = int(frames or getattr(state, "preview_frames", 0) or H3_PREVIEW_FRAMES)
        key = self.clip_key(state, frames=frames, steps=steps, turbo_strength=strength,
                            with_audio=with_audio)
        if self._baseline_clip_key == key and self._baseline_clip is not None:
            return self._baseline_clip
        base = SliderState.default_h3()
        base.seed = state.seed
        base.prompt = state.prompt
        base.preview_width = state.preview_width
        base.preview_height = state.preview_height
        base.preview_frames = frames
        base.keyframes = getattr(state, "keyframes", None)
        base.primary_scale = float(getattr(state, "primary_scale", 1.0))
        base.donor_scale = float(getattr(state, "donor_scale", 1.0))
        clip = self.render_clip(base, frames=frames, regime=regime, with_audio=with_audio,
                                cache=cache, steps=steps, turbo_strength=strength)
        self._baseline_clip_key = key
        self._baseline_clip = clip
        return clip

    def nolora_clip(self, state, *, frames: Optional[int] = None, regime: str = "confirm",
                    with_audio: bool = True, cache=None, steps=None, turbo_strength=None,
                    **_ignored) -> dict:
        """The same seed / prompt / canvas / length / keyframes rendered by the base model
        with no LoRA at all — the player's third pane. Cached in memory like the baseline
        (a slider move never re-renders it) and on disk under "nolora"."""
        from fizgig.repair_studio.state import SliderState
        steps, strength = self.regime_params(regime, steps, turbo_strength)
        frames = int(frames or getattr(state, "preview_frames", 0) or H3_PREVIEW_FRAMES)
        key = self.clip_key(state, frames=frames, steps=steps, turbo_strength=strength,
                            with_audio=with_audio)
        if (getattr(self, "_nolora_clip_key", None) == key
                and getattr(self, "_nolora_clip", None) is not None):
            return self._nolora_clip
        base = SliderState.default_h3()
        base.seed = state.seed
        base.prompt = state.prompt
        base.preview_width = state.preview_width
        base.preview_height = state.preview_height
        base.preview_frames = frames
        base.keyframes = getattr(state, "keyframes", None)
        clip = self.render_clip(base, frames=frames, regime=regime, with_audio=with_audio,
                                cache=cache, no_lora=True, steps=steps, turbo_strength=strength)
        self._nolora_clip_key = key
        self._nolora_clip = clip
        return clip

    def generate_baseline(self, state) -> Image.Image:
        """Baseline = primary at default 1.0 / all enabled, donor off. Cached on
        (primary_path, seed, prompt, w, h) — slider tweaks don't invalidate it."""
        from fizgig.repair_studio.state import SliderState
        key = (self.primary_path, state.seed, state.prompt,
               state.preview_width, state.preview_height)
        if self._baseline_cache_key == key and self._baseline_cache_image is not None:
            return self._baseline_cache_image
        base = SliderState.default_h3()
        base.seed = state.seed
        base.prompt = state.prompt
        base.preview_width = state.preview_width
        base.preview_height = state.preview_height
        img = self.generate_preview(base)
        self._baseline_cache_key = key
        self._baseline_cache_image = img
        return img

    # ----- prompt travel (Royale) --------------------------------------------
    def encode_travel_prompts(self, prompts):
        """Encode waypoint prompts in one TE cycle. Returns ([(emb, None), ...], None) to match
        the other engines' (waypoints, negative) shape."""
        from fizgig.minimax.sampling import encode_sample_prompts
        embs = encode_sample_prompts(self.te_path, list(prompts), device=self.device)
        self._prompt_cache_key = None      # the TE was cycled; session cache is stale
        self._prompt_cache = None
        return [(e.cpu(), None) for e in embs], None

    @staticmethod
    def interp_waypoints(vecs, t, mode="lerp"):
        """Piecewise lerp across ordered (emb, None) waypoints — H3 embeds are unpadded and can
        differ in L, so segments interpolate over the shorter length and keep the tail of the
        LONGER end (tokens fade in/out at segment boundaries rather than truncating hard)."""
        if len(vecs) == 1:
            return vecs[0]
        t = min(max(float(t), 0.0), 1.0)
        segs = len(vecs) - 1
        pos = t * segs
        i = min(int(pos), segs - 1)
        local = pos - i
        a, b = vecs[i][0].float(), vecs[i + 1][0].float()
        L = min(a.shape[1], b.shape[1])
        blended = torch.lerp(a[:, :L], b[:, :L], local)
        tail_src = a if a.shape[1] > b.shape[1] else b
        if tail_src.shape[1] > L:
            w = (1.0 - local) if tail_src is a else local
            blended = torch.cat([blended, tail_src[:, L:].float() * w], dim=1)
        return blended.to(vecs[i][0].dtype), None

    # ----- caches / shims ----------------------------------------------------
    def _invalidate_baseline_cache(self) -> None:
        self._baseline_cache_key = None
        self._baseline_cache_image = None
        self._baseline_clip_key = None
        self._baseline_clip = None
        self._nolora_clip_key = None
        self._nolora_clip = None

    def _invalidate_activation_cache(self) -> None:
        self._act_cache = None
        self._act_cache_key = None
        self._act_cache_state = None

    # VRAM the pass-1 cache must leave free beside itself for the render's own working set
    # — which grows with the cache (both scale with tokens): 3 GB plus half the cache.
    # Measured 4 Sep at 768×768×56 with 8.4 GB free: a 5.1 GB cache + the render did NOT
    # fit (needed ~3 GB of working set + fragmentation on top), so that size parks on the
    # CPU on a 32 GB card; the 22-frame dial (0.9 GB) stays on the GPU.
    _RESUME_HEADROOM_GB = 3.0
    _RESUME_HEADROOM_FRAC = 0.5

    def resume_cache_gb(self, width, height, frames) -> float:
        """Size of one pass-1 cache for this canvas / length: every main block's input row
        block, [tokens, hidden] bf16. 512×416×22 ≈ 0.9 GB; 768×640×22 ≈ 1.9 GB;
        768×768×22 ≈ 2.3 GB; 768×768×56 ≈ 5.1 GB."""
        from fizgig.minimax.model import (latent_frames_for_pixels, audio_latents_for_frames,
                                          AUDIO_CHANNELS)
        cfg = self.dit.config
        lt = latent_frames_for_pixels(int(frames), exact=True)
        toks = (lt * (int(height) // 16 // 2) * (int(width) // 16 // 2)
                + audio_latents_for_frames(int(frames)) * AUDIO_CHANNELS + 256)
        return len(self.dit.blocks) * toks * int(cfg.hidden_size) * 2 / float(2 ** 30)

    def _resume_cache_device(self, width, height, frames) -> str:
        """Where THIS render's pass-1 cache lives: the GPU when it fits beside the model with
        headroom, else the CPU (parked over PCIe — measured to net the same saving at 22
        frames). Decided per render because the cache scales with tokens: the load-time
        "cuda" tier was sized for the 22-frame dial, and a 56-frame Confirm at 768² is
        2.4× that — recording it on the GPU is what ran the 5090 out of VRAM (4 Sep)."""
        try:
            est = self.resume_cache_gb(width, height, frames)
        except Exception:
            return "cpu"
        self.last_resume_cache_gb = est
        if self._cache_device != "cuda":
            return "cpu"
        try:
            from fizgig.utils.device import plannable_free_vram
            free = float(plannable_free_vram())
        except Exception:
            return "cpu"
        need = est * (1.0 + self._RESUME_HEADROOM_FRAC) + self._RESUME_HEADROOM_GB
        dev = "cuda" if need <= free else "cpu"
        if getattr(self, "_last_cache_plan", None) != (round(est, 2), dev):
            self._last_cache_plan = (round(est, 2), dev)
            logger.info("[h3-workbench] pass-1 cache %.2f GB for %dx%d x%d -> %s (%.1f GB free)",
                        est, int(width), int(height), int(frames), dev, free)
        return dev

    def mark_blocks_changed(self, blocks) -> None:
        pass          # the resume point derives from a state diff at render time (race-free)

    @staticmethod
    def _block_index(block_id):
        return int(block_id.split("_")[1]) if str(block_id).startswith("h3blk_") else None

    def _resume_from_diff(self, state):
        """Earliest main-block index whose primary/donor differs from the cached state, or
        None (full recompute) if a refiner block changed (it feeds block 0) or there is no
        cached state. Diffing the FULL state guarantees no edit made during an in-flight
        render is ever missed."""
        if self._act_cache_state is None:
            return None
        changed = state.diff_blocks(self._act_cache_state)
        if not changed:
            return None
        idxs = [self._block_index(b) for b in changed]
        if any(i is None for i in idxs):     # h3_rf_* changed -> full pass
            return None
        return min(idxs)

    # ----- teardown ----------------------------------------------------------
    def reset(self) -> None:
        """Full unload — drop networks (break forward-hook ref cycles), unpatch the Turbo's
        AdaLN forwards, then the DiT + decoder."""
        from fizgig.utils.device import release_module_tensors as _strip
        for net in (self.primary_network, self.donor_network, self._turbo_net):
            if net is not None:
                try:
                    # Strip the LoRA weights themselves, not just our references: the field
                    # census showed a full turbo network (208 x up/down/alpha = 625 tensors)
                    # surviving reset via externally-held forward closures, its small weights
                    # pinning ~6 GB of allocator segments. Husks can't pin anything.
                    _strip(net)
                    for lora in net.unet_loras:
                        lora.org_forward = None
                    net.unet_loras.clear()
                except Exception:
                    pass
        try:
            from fizgig.minimax.trainer import turbo_adaln_unpatch
            turbo_adaln_unpatch(self._turbo_adaln)
        except Exception:
            pass
        self.primary_network = None
        self.donor_network = None
        self._turbo_net = None
        self._turbo_adaln = []
        # Field leak (19 Aug): the whole DiT survived reset, pinned from outside the engine
        # (~1657 params alive, 20 GB). Dropping our reference isn't enough for a pinned
        # module — strip its storages so the VRAM comes back regardless of the holder.
        from fizgig.utils.device import release_module_tensors
        release_module_tensors(self.dit)
        release_module_tensors(self.decoder)
        release_module_tensors(self.audio_decoder)
        release_module_tensors(self.encoder)
        self.dit = None
        self.decoder = None
        self.audio_decoder = None
        self.encoder = None
        self.pipeline = None
        self.primary_path = None
        self.donor_path = None
        self.primary_block_ids = set()
        self.donor_block_ids = set()
        self.primary_hash = None
        self.donor_hash = None
        self._prompt_cache_key = None
        self._prompt_cache = None
        self._invalidate_baseline_cache()
        self._invalidate_activation_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Field leak (19 Aug): ~21 GB survived a reset with every attribute above nulled —
        # the holder was OUTSIDE the engine. If that happens again, name it in the console.
        from fizgig.utils.device import report_cuda_leak, flush_reserved_vram
        report_cuda_leak("h3-repair-reset")
        # Second field case: allocated 0.01 GB but 6 GB reserved — pinned allocator segments.
        flush_reserved_vram("h3-repair-reset")
