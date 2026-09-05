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
import re
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
    # Only standard-LoRA modules whose target Linear exists on THIS base with matching shapes
    # become modules — the same prefilter load_preview_turbo applies. A Turbo LoRA's 2688-wide
    # AdaLN rows fail that on the pruned base (they are injected at render time instead,
    # see _collect_adaln_pairs); until 4 Sep they were built as modules anyway and blew up in
    # the first forward. LyCORIS modules (no lora_down) pass through untouched.
    linears = {f"lora_unet_{n.replace('.', '_')}": m
               for n, m in target.named_modules() if isinstance(m, torch.nn.Linear)}
    keep, dropped = {}, []
    for name in sorted({k.split(".")[0] for k in sd if "." in k}):
        down = sd.get(f"{name}.lora_down.weight")
        up = sd.get(f"{name}.lora_up.weight")
        if down is not None and up is not None:
            m = linears.get(name)
            if not (m is not None and down.shape[1] == m.in_features
                    and up.shape[0] == m.out_features):
                dropped.append(name)
                continue
        for k, v in sd.items():
            if k.split(".")[0] == name:
                keep[k] = v
    if dropped:
        logger.info("%d LoRA modules don't fit the loaded base and are not wired as modules "
                    "(AdaLN rows are injected separately): %s%s", len(dropped),
                    ", ".join(dropped[:3]), "…" if len(dropped) > 3 else "")
    if not keep:
        raise RuntimeError("no module in this LoRA matches the loaded base — wrong file?")
    sd = keep
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


def _collect_adaln_pairs(dit, sd):
    """The full-model AdaLN rows of a LoRA that the pruned base cannot host as modules —
    the same matching load_preview_turbo does for the built-in Turbo. Returns
    [(lora_name, adaln_module, A, B)] with B UNscaled (the engine folds strength and the
    block's slider in at install time). Empty for a LoRA without AdaLN keys."""
    try:
        parents = {f"lora_unet_{n.replace('.', '_')}_linear": m
                   for n, m in dit.named_modules() if type(m).__name__ == "AdalnProj"}
    except Exception:
        return []
    if not parents:
        return []
    out = []
    for name in sorted({k.split(".")[0] for k in sd}):
        ap = parents.get(name)
        if ap is None:
            continue
        down = sd.get(f"{name}.lora_down.weight")
        up = sd.get(f"{name}.lora_up.weight")
        if down is None or up is None or up.shape[0] != ap.linear.out_features:
            continue
        if down.shape[1] == ap.linear.in_features:
            # Shaped for THIS base's projection: it is wired as an ordinary module by
            # _apply_lora (a LoRA trained on the pruned base with AdaLN as a target).
            # Injecting it as well doubled its effect (4 Sep). Only full-model-space rows
            # — the ones no module can host — are injected.
            continue
        out.append((name, ap, down.clone(), up.clone()))
    return out


class H3RepairEngine:
    def __init__(self):
        self.pipeline: Optional[_Loaded] = None
        self.dit = None
        self.decoder = None            # fp16 video VAE decoder, parked on CPU between decodes
        self.te_path: Optional[str] = None
        self.device = "cuda"
        self.dtype = torch.bfloat16

        self.on_status = None          # callable(str) — the GUI's status line, if it wants it
        self._te_parked = None         # the layer-streamed text encoder kept in system RAM
        self._turbo_lora_path = ""
        self._turbo_lora_strength = 0.75
        self.base_mode = "auto"          # the Base picker: auto / stream / nf4
        self.int8_attention = False      # the Clip row tick (comfy-kitchen kernel), per render
        self.base_plan = None            # (base_quant, blocks streamed) actually loaded
        self._te_parked_vision = False # ...built with the vision tower (serves text too)
        self.dit_path: Optional[str] = None
        self._prompt_cache_tags = None # token tags that go with _prompt_cache (ref mode only)
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
        self._primary_adaln = []           # (name, module, A, B) — a Turbo LoRA loaded as primary
        self._donor_adaln = []
        self._adaln_installed = None       # signature of the injection installed right now
        self._adaln_no_lora = False        # a no-LoRA render is in progress: LoRA rows stay out
        self._adaln_bid = {}               # lora_name -> block id (memo)
        self._last_state = None
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
    # Which base the studio runs, from free VRAM at load. 32 GB-class: the int8 base
    # resident. 24 GB-class (and 20 GB): the SAME int8 file with the last n blocks parked on
    # the CPU and streamed through per pass (0.39 GB each, rintic-13's H2D offloader) — the
    # int8 weights' 0.2% error instead of the NF4 base's 9.5%, which matters when you are
    # judging a block's effect. Below ~18 GB free the streamed count gets silly and the NF4
    # base (10.5 GB) takes over — 16 GB cards.
    _INT8_RESIDENT_GB = 21.3
    _INT8_BLOCK_GB = 0.39
    # Headroom over the resident int8 weights: the LoRA nets (~1.3) + the ring (0.8) + the
    # larger of the two phases — a render's working set (≤3.5 at 56 f) or the DECODE, which
    # is the 4.5 GB fp16 decoder + ~1 GB of tiled activations + ~2 GB the allocator loses to
    # fragmentation while it churns. 6 GB covered the renders and lost a 56-frame decode at
    # 768×640 by 316 MB (simulated 24 GB card, 4 Sep); 9 GB also leaves the decoder resident
    # between 22-frame renders (two round-trips saved per Dial move) — at 9 GB the decoder
    # was still parked after every decode (6.9 GB free beside the loaded base, 2.4 with the
    # decoder on it — under the working-set + 2 GB rule), so 10: three more blocks streamed
    # (~0.4 s per Dial move) against ~2 s of decoder round-trips saved.
    _STUDIO_HEADROOM_GB = 10.0
    _INT8_SWAP_MIN_FREE_GB = 18.0
    # "Stream blocks" (the Base picker): the exact int8 base on ANY card with enough blocks
    # streamed to leave room for the biggest clips — 1024×1024 × 56 frames with keyframes is
    # ~18k tokens, about 2.4× the 768×640 clip's working set and decode (Peter's OOM on the
    # 5090, 5 Sep). 18 GB over the resident weights covers it; on a 5090 that is 18 blocks.
    _STREAM_HEADROOM_GB = 18.0

    @classmethod
    def plan_base(cls, free_gb: float, mode: str = "auto"):
        """(base_quant, blocks_to_swap) for `free_gb` of plannable VRAM.
        mode: "auto" (by free VRAM), "stream" (int8, blocks streamed for big-clip room on any
        card), "nf4" (the 4-bit base, smallest — 9.5% base error)."""
        import math
        free_gb = float(free_gb or 0.0)
        mode = (mode or "auto").lower()
        if mode == "nf4":
            return "nf4", 0
        if mode == "stream":
            n = math.ceil((cls._INT8_RESIDENT_GB + cls._STREAM_HEADROOM_GB - free_gb) / cls._INT8_BLOCK_GB)
            return "int8", max(1, min(40, n))
        if free_gb >= 28.0:
            return "int8", 0
        if free_gb >= cls._INT8_SWAP_MIN_FREE_GB:
            n = math.ceil((cls._INT8_RESIDENT_GB + cls._STUDIO_HEADROOM_GB - free_gb) / cls._INT8_BLOCK_GB)
            return "int8", max(1, min(40, n))
        return "nf4", 0

    def _load_dit_and_turbo(self) -> None:
        """The DiT (base precision planned from free VRAM) + the Turbo LoRA. Called once by
        ensure_pipeline — and again by _dit_restore on a small-RAM machine that unloads
        the base for a text-encoder pass instead of parking 21 GB of it in RAM."""
        from fizgig.minimax.loader import load_minimax_h3_dit
        try:
            from fizgig.utils.device import plannable_free_vram
            free = plannable_free_vram()
        except Exception:
            free = 0.0
        base_quant, n_swap = self.plan_base(free, getattr(self, "base_mode", "auto"))
        self._blocks_swapped = int(n_swap)
        self.base_plan = (base_quant, int(n_swap))
        if getattr(self, "base_mode", "auto") != "auto":
            logger.info("[h3-workbench] Base picker: %s -> %s base, %d blocks streamed (%.1f GB free)",
                        self.base_mode, base_quant, n_swap, free)
        elif n_swap:
            logger.info("[h3-workbench] %.1f GB free -> int8 base with the last %d blocks streamed "
                        "from CPU per pass (24 GB-class card: the int8 weights, 0.2%% error, "
                        "rather than the NF4 base's 9.5%%; the pass-1 resume sits out)",
                        free, n_swap)
        else:
            logger.info("[h3-workbench] %.1f GB free -> %s base, no block swap "
                        "(int8 needs ~21 GB resident + headroom; NF4-of-pruned is ~10.5 GB)",
                        free, base_quant)
        self.dit = load_minimax_h3_dit(self.dit_path, device=self.device, compute_dtype=self.dtype,
                                       quantize=True, blocks_to_swap=int(n_swap),
                                       base_quant=base_quant)
        if n_swap:
            # The loader only PARKS the tail blocks' weights; this sets the swap boundary
            # and installs the int8 ring offloader (static weights — norms, AdaLN — back
            # on the card, the int8 flats streamed per pass), exactly as the trainer does.
            self._blocks_swapped = int(self.dit.enable_block_swap(int(n_swap), h2d_only=True,
                                                                  ring_size=2))
        self.dit.eval()
        self.dit._abort_event = self._cancel_event      # a cancel lands within one block

        if self._turbo_lora_path and os.path.exists(self._turbo_lora_path):
            try:
                from fizgig.minimax.trainer import load_preview_turbo, turbo_adaln_patch
                self._turbo_net, _folded = load_preview_turbo(
                    self.dit, self._turbo_lora_path, float(self._turbo_lora_strength))
                self._turbo_net.to(device=self.device, dtype=self.dtype)
                for _m in self._turbo_net.unet_loras:
                    _m.enabled = True
                # Keep the AdaLN rows UNSCALED (load_preview_turbo folds the load strength
                # into B; rescaling that bf16 fold to another strength is not exact) — the
                # composer folds the dialled strength in at install time, exactly.
                try:
                    from safetensors.torch import load_file as _lf
                    from fizgig.networks.lora import ensure_kohya_lora_state_dict as _ek
                    self._turbo_adaln = [(m, a, b) for _nm, m, a, b in
                                         _collect_adaln_pairs(self.dit, _ek(_lf(self._turbo_lora_path)))]
                except Exception:
                    logger.exception("Turbo AdaLN rows: unscaled collection failed — using the folded ones")
                    self._turbo_adaln = [(m, a, b / float(self._turbo_lora_strength)) for m, a, b in _folded]                         if abs(float(self._turbo_lora_strength)) > 1e-9 else []
                self._steps = 6
                self._turbo_strength = float(self._turbo_lora_strength)
                self._turbo_load_strength = float(self._turbo_lora_strength)
                self._adaln_installed = None
                self._reinstall_adaln()
                n_ad = len(self._adaln_installed or ())
                logger.info("[h3-workbench] Turbo LoRA on for all previews — 6 steps at %g"
                            + (", %d adaln injected" % n_ad if n_ad else ""),
                            float(self._turbo_lora_strength))
            except Exception:
                logger.exception("[h3-workbench] Turbo LoRA failed to load — previews run "
                                 "the standard 20 steps")
                self._turbo_net, self._turbo_adaln, self._steps = None, [], 20
        else:
            logger.info("[h3-workbench] no Turbo LoRA configured — previews run 20 steps "
                        "(set it in Preferences for 6-step previews)")

    def ensure_pipeline(self, dit_path: str, vae_path: str, text_encoder_path: str,
                        device: str = "cuda", turbo_lora_path: str = "",
                        turbo_lora_strength: float = 0.75, te_cache_dir: str = "",
                        audio_vae_path: str = "", base_mode: str = "auto", **_ignored) -> None:
        """Load the DiT (base precision auto-planned from free VRAM, or forced by the Base
        picker: `base_mode` auto / stream / nf4) + the Turbo LoRA once. The video VAE decoder
        loads lazily at first decode; the TE loads only on a prompt-cache miss (and is freed
        straight after)."""
        if self.pipeline is not None and self.pipeline.is_loaded:
            return
        self.base_mode = (base_mode or "auto").lower()
        from fizgig.minimax.loader import load_minimax_h3_dit
        self.device = device
        self.te_path = text_encoder_path
        self.dit_path = dit_path
        self._vae_path = vae_path
        self._audio_vae_path = audio_vae_path or None
        self._te_cache_dir = te_cache_dir or None

        self._turbo_lora_path = turbo_lora_path or ""
        self._turbo_lora_strength = float(turbo_lora_strength)
        self._load_dit_and_turbo()
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
        if getattr(self, "_blocks_swapped", 0):
            logger.info("[h3-workbench] pass-1 resume sits out on the streamed-block plan "
                        "(%.1f GB free after the DiT)", free_after)
        else:
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
        self._reinstall_adaln()

    def _block_factor(self, state, name, which):
        """The slider factor a LoRA's AdaLN row gets: its block's strength (0 when the block is
        off), 1.0 for a row no block pattern claims. Memoised name -> block id."""
        if state is None:
            return 1.0
        memo = getattr(self, "_adaln_bid", None)
        if memo is None:
            memo = self._adaln_bid = {}
        bid = memo.get(name)
        if bid is None:
            bid = ""
            for cand in state.blocks:
                try:
                    if re.search(block_regex_h3(cand), name):
                        bid = cand
                        break
                except ValueError:
                    continue
            memo[name] = bid
        bs = state.blocks.get(bid) if bid else None
        if bs is None:
            # A row no block owns (the final layer's): on while any block of that LoRA is
            # on, off when every block is off — so "all off" really is the LoRA out.
            for row in state.blocks.values():
                on = row.primary_enabled if which == "primary" else row.donor_enabled
                strength = row.primary_strength if which == "primary" else row.donor_strength
                if on and abs(float(strength)) > 1e-9:
                    return 1.0
            return 0.0
        on = bs.primary_enabled if which == "primary" else bs.donor_enabled
        strength = bs.primary_strength if which == "primary" else bs.donor_strength
        return float(strength) if on else 0.0

    def _adaln_pairs_now(self, no_lora=False):
        """Everything the AdaLN injection should add right now, as (module, A, B·factor):
        the built-in Turbo's rows at the dialled strength, plus the primary's / donor's
        rows at their load strength × block slider —
        none of the LoRAs' rows for a no-LoRA render."""
        pairs, sig = [], []
        turbo = getattr(self, "_turbo_adaln", None) or []          # unscaled rows
        ts = float(getattr(self, "_turbo_strength", 0.0) or 0.0)
        if turbo and abs(ts) > 1e-9:
            f = ts
            for m, a, b in turbo:
                pairs.append((m, a, b * f))
                sig.append((id(m), id(a), round(f, 6)))
        if not no_lora:
            st = getattr(self, "_last_state", None)
            for plist, which in ((getattr(self, "_primary_adaln", None) or [], "primary"),
                                 (getattr(self, "_donor_adaln", None) or [], "donor")):
                scale = float(getattr(st, f"{which}_scale", 1.0)) if st is not None else 1.0
                for name, m, a, b in plist:
                    f = scale * self._block_factor(st, name, which)
                    if abs(f) < 1e-9:
                        continue
                    pairs.append((m, a, b * f))
                    sig.append((id(m), id(a), round(f, 6)))
        return pairs, tuple(sig)

    def _reinstall_adaln(self, no_lora=None):
        """(Re)install the AdaLN injection when what it should contain changed — see
        _adaln_pairs_now. Instance-attribute forwards, so removal is a plain delete.
        no_lora=None reads the render's sticky flag (set for a no-LoRA render)."""
        if getattr(self, "dit", None) is None:
            return
        if no_lora is None:
            no_lora = bool(getattr(self, "_adaln_no_lora", False))
        pairs, sig = self._adaln_pairs_now(no_lora)
        if sig == getattr(self, "_adaln_installed", None):
            return
        mods = {id(m): m for m, _a, _b in (getattr(self, "_turbo_adaln", None) or [])}
        for plist in (getattr(self, "_primary_adaln", None) or [], getattr(self, "_donor_adaln", None) or []):
            for _n, m, _a, _b in plist:
                mods[id(m)] = m
        for m in mods.values():
            if "forward" in m.__dict__:
                del m.forward
        self._adaln_installed = ()
        if not pairs:
            return
        try:
            from fizgig.minimax.trainer import turbo_adaln_patch
            turbo_adaln_patch(self.dit, pairs, self.device, self.dtype)
            self._adaln_installed = sig
        except Exception:
            logger.exception("AdaLN injection failed — rows stay off")

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

    @torch.no_grad()
    def encode_reference_image(self, image: Image.Image, width: int, height: int):
        """A PIL image (already cropped) -> (resized PIL, normalized reference latent
        [1, 24, 1, h, w]) for ref2va. Sized to the clip's pixel budget ("match": down-only,
        32-px multiples) and THE SAME resized image goes to the VAE here and to the vision
        blocks in _encode_prompt — the pairing the r2v workflow depends on."""
        from fizgig.minimax.reference import resize_reference
        import numpy as np
        img = resize_reference(image, int(width), int(height), "match")
        arr = torch.from_numpy(np.asarray(img).copy()).float().permute(2, 0, 1) / 127.5 - 1.0
        enc = self._ensure_encoder().to(self.device)
        try:
            z = enc.encode(arr.unsqueeze(0).to(self.device))     # [1, 24, 1, h, w]
        finally:
            self.encoder = enc.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return img, z.float().cpu()

    def load_primary(self, path: str) -> None:
        if self.pipeline is None or not self.pipeline.is_loaded:
            raise RuntimeError("Pipeline not loaded; call ensure_pipeline() first.")
        if self.primary_network is not None:
            raise RuntimeError("Primary already loaded — call reset() to swap.")
        self._wire_primary(path)

    def _wire_primary(self, path: str) -> None:
        from safetensors.torch import load_file
        from fizgig.networks.lora import ensure_kohya_lora_state_dict
        sd = ensure_kohya_lora_state_dict(load_file(path))
        self.primary_network = _apply_lora(self.dit, sd, 1.0, self.device, self.dtype)
        self.primary_path = path
        self.primary_block_ids = extract_block_ids_h3(self.primary_network)
        # A Turbo LoRA's AdaLN rows (full-model space) can't be modules on the pruned base —
        # they are injected at render time like the built-in Turbo's, scaled by the load
        # strength and each block's slider (missing until 4 Sep: a Turbo LoRA edited as the
        # primary lost its modulation and rendered badly at few steps).
        self._primary_adaln = _collect_adaln_pairs(self.dit, sd)
        if self._primary_adaln:
            logger.info("H3 primary carries %d AdaLN rows — injected at render time",
                        len(self._primary_adaln))
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
        self._wire_donor(path)

    def _wire_donor(self, path: str) -> None:
        from safetensors.torch import load_file
        from fizgig.networks.lora import ensure_kohya_lora_state_dict
        sd = ensure_kohya_lora_state_dict(load_file(path))
        net = _apply_lora(self.dit, sd, 1.0, self.device, self.dtype)
        net.set_enabled(False)  # donor blocks are opt-in per-slider
        self.donor_network = net
        self._donor_adaln = _collect_adaln_pairs(self.dit, sd)
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
            self._donor_adaln = []
            self._reinstall_adaln()

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
                         int8_attention=bool(getattr(self, "int8_attention", False)),
                         primary_scale=float(getattr(state, "primary_scale", 1.0)),
                         donor_scale=float(getattr(state, "donor_scale", 1.0)),
                         dit=os.path.basename(getattr(self, "dit_path", "") or ""))

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
        self._last_state = state
        self._reinstall_adaln()

    # ----- cancellation ------------------------------------------------------
    def request_cancel(self) -> None:
        self._cancel_event.set()

    def clear_cancel(self) -> None:
        self._cancel_event.clear()

    # ----- prompt encoding (two-level cache) ---------------------------------
    def _status(self, msg: str) -> None:
        cb = self.on_status
        if cb is not None:
            try:
                cb(msg)
            except Exception:
                pass

    @staticmethod
    def _ref_fingerprint(images) -> str:
        """A short hash of the reference images as they will be shown to the encoder (size +
        pixels) — part of the prompt cache key, since the embedding depends on them."""
        if not images:
            return ""
        h = hashlib.sha256()
        for im in images:
            im = im.convert("RGB")
            h.update(f"{im.width}x{im.height}:".encode())
            h.update(im.tobytes())
        return h.hexdigest()[:16]

    def _prompt_disk_path(self, prompt: str, ref_fp: str = "") -> Optional[str]:
        if not self._te_cache_dir:
            return None
        h = hashlib.sha256((prompt + "\x00" + os.path.basename(self.te_path or "")
                            + ("\x00ref:" + ref_fp if ref_fp else "")).encode("utf-8")).hexdigest()
        return os.path.join(self._te_cache_dir, f"{h}.safetensors")

    def _encode_prompt(self, prompt: str, images=None):
        """[1, L, 5120] on CPU. In-memory hit -> free; disk hit -> milliseconds; miss -> the
        32B TE loads once (couple of minutes), encodes, frees, and the result persists so no
        future session pays again. With `images` (ref2va): the vision-capable encoder sees
        `<Picture i>:` blocks ahead of the prompt and the per-row modality tags come back
        alongside (self._prompt_cache_tags) — cached under prompt + image fingerprint."""
        ref_fp = self._ref_fingerprint(images) if images else ""
        key = (prompt, ref_fp)
        if self._prompt_cache_key == key and self._prompt_cache is not None:
            return self._prompt_cache
        disk = self._prompt_disk_path(prompt, ref_fp)
        if disk and os.path.exists(disk):
            from safetensors.torch import load_file
            sd = load_file(disk)
            emb = sd["hidden_states"].unsqueeze(0)
            tags = sd.get("token_tags")
            if ref_fp and tags is None:
                pass                      # an old text-only file under a ref key: re-encode
            else:
                self._prompt_cache_key, self._prompt_cache = key, emb
                self._prompt_cache_tags = tags
                return emb
        logger.info("[h3-workbench] encoding prompt with the 32B TE (one-off per prompt — "
                    "cached to disk after this)")
        want_vision = bool(images)
        _parked = None
        if self.dit is not None and not self._te_can_park():
            # Small-RAM machine: no parked encoder, so the planned build is coming — the
            # resident one needs the whole card, and the planner decides with the card
            # empty. The DiT goes first, as it always did before parking existed.
            _parked = self._dit_offload()
        te, keep = self._te_get(want_vision)
        streamed = getattr(te, "layer_streamer", None) is not None
        if keep and streamed and getattr(self, "_te_was_parked", False):
            self._status("Encoding the prompt (text encoder parked in RAM — seconds)…")
        else:
            self._status("Encoding the prompt with the 32B text encoder — a couple of quiet "
                         "minutes, once" + (" (then it stays parked in RAM)" if keep else "")
                         + ("; reference set" if images else "") + "…")
        # The resident TE (~15.7 GB) and the resident base must never be co-resident (the
        # int8 base alone is ~21 GB): park the DiT + Turbo net for the encode and restore
        # after — a whole-model .to is safe because this engine never block-swaps. The
        # streamed build only needs its rings (~2 GB text, ~12.7 GB with vision): the DiT
        # stays put for a text encode when the card has the room.
        # The streamed build needs ~4.2 GB at most (measured 4 Sep: two references at
        # 768×640, or a 600-token prompt — rings + the resident embedding table), so the
        # DiT stays on the card for every encode; only the resident build (27 GB) parks
        # it. Parking the 21 GB base to the CPU and back was also leaving ~15 GB of host
        # memory retained per encode.
        need = 4.5 if streamed else 27.0             # streamed: measured 4.2 GB peak
        free = None
        try:
            from fizgig.utils.device import plannable_free_vram
            free = float(plannable_free_vram())
        except Exception:
            pass
        if _parked is None and self.dit is not None and (free is None or free < need + 1.0):
            _parked = self._dit_offload()
        tags = None
        try:
            if images:
                emb, tags = te.encode_with_reference(prompt, list(images))
            else:
                emb = te.encode(prompt)
        finally:
            if keep:
                self._te_park()
            else:
                del te
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if _parked:
                self._dit_restore(_parked)
            self._status("Rendering…")
        emb = emb.cpu()
        tags = tags.cpu() if tags is not None else None
        if disk:
            try:
                os.makedirs(self._te_cache_dir, exist_ok=True)
                from safetensors.torch import save_file
                blob = {"hidden_states": emb[0].contiguous()}
                if tags is not None:
                    blob["token_tags"] = tags.contiguous()
                save_file(blob, disk, metadata={"prompt": prompt[:512],
                                                "te": os.path.basename(self.te_path or ""),
                                                "refs": ref_fp})
            except Exception:
                logger.exception("prompt disk-cache write failed (render continues)")
        self._prompt_cache_key, self._prompt_cache = key, emb
        self._prompt_cache_tags = tags
        return emb

    # ----- the text encoder, parked in system RAM between prompts -------------------------
    # A new prompt used to stream the 32B encoder in from disk every time (a minute or two —
    # editing a long prompt word by word paid it per Update). The layer-streamed build (#79,
    # @mabseyuk / rintic-13) keeps the packed ~19 GB model in system RAM and only rings two
    # layers through the GPU, so it can simply be KEPT for the session: the next prompt costs
    # seconds. Only when the box has the RAM: ~24 GB available to stage it, and it is let go
    # again if available RAM drops under 8 GB. Otherwise the old path — resident build, freed.
    # Measured 4 Sep: the parked build costs ~35 GB of process RSS in the app (packed
    # weights + pinned staging), and freed pinned blocks are not all handed back to the OS.
    _TE_PARK_MIN_AVAIL_GB = 40.0
    _TE_KEEP_MIN_AVAIL_GB = 12.0

    @staticmethod
    def _ram_available_gb():
        try:
            import psutil
            return psutil.virtual_memory().available / 1e9
        except Exception:
            return None

    # Getting the base out of the way for a resident text-encoder pass (small-RAM machines
    # only — the streamed encoder never needs it). Parking = the 21 GB base copied to system
    # RAM and back (fast, ~6 s each way) — but on a 32 GB box that copy alone starts the
    # machine paging (Peter, 4 Sep). Below the threshold the base is UNLOADED instead and
    # reloaded from disk after (~25 s from a fast SSD / the page cache), the LoRAs re-wired
    # from their files: no RAM needed at all.
    _DIT_PARK_MIN_AVAIL_GB = 29.0            # 21 GB copy + headroom

    def _dit_offload(self) -> str:
        """Free the card of the DiT for an encode. Returns the token _dit_restore needs."""
        if self.dit is None:
            return "none"
        avail = self._ram_available_gb()
        park = (avail is not None and avail >= self._DIT_PARK_MIN_AVAIL_GB
                and os.environ.get("FIZGIG_DIT_UNLOAD") != "1"
                # a base with streamed blocks must never be moved with .to(): the H2D
                # offloader's ring bindings don't survive it (its field notes) — unload
                and not getattr(self, "_blocks_swapped", 0))
        if park:
            self.dit.to("cpu")
            if self._turbo_net is not None:
                self._turbo_net.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return "park"
        logger.info("[h3-workbench] %s GB RAM available — unloading the base for the encoder "
                    "pass and reloading it after (no RAM copy)",
                    "?" if avail is None else f"{avail:.0f}")
        self._status("Unloading the base for the text encoder (small-RAM machine), reloading after…")
        self._dit_release()
        return "unload"

    def _dit_release(self) -> None:
        """Drop the DiT, the Turbo net and the LoRA networks (paths kept for re-wiring)."""
        for mod, a, b in (self._turbo_adaln or []):
            if "forward" in mod.__dict__:
                del mod.forward
        self._turbo_adaln, self._primary_adaln, self._donor_adaln = [], [], []
        self._adaln_installed, self._adaln_bid = None, {}
        self.primary_network = None
        self.donor_network = None
        self._turbo_net = None
        try:
            _off = getattr(self.dit, "_h2d_offloader", None)
            if _off is not None:
                _off.release()                    # its pinned staging goes with it
        except Exception:
            pass
        self.dit = None
        self._invalidate_activation_cache()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _dit_restore(self, token: str) -> None:
        if token == "park" and self.dit is not None:
            self.dit.to(self.device)
            if self._turbo_net is not None:
                self._turbo_net.to(device=self.device, dtype=self.dtype)
        elif token == "unload":
            self._status("Reloading the base…")
            self._load_dit_and_turbo()
            if self.primary_path:
                self._wire_primary(self.primary_path)
            if self.donor_path:
                self._wire_donor(self.donor_path)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _te_can_park(self) -> bool:
        """A parked encoder exists, or the box has the RAM to hold one."""
        if self._te_parked is not None:
            return True
        if os.environ.get("FIZGIG_NO_TE_PARK") == "1":
            return False
        avail = self._ram_available_gb()
        return avail is not None and avail >= self._TE_PARK_MIN_AVAIL_GB

    def _te_get(self, with_vision: bool):
        """(encoder, keep): a parked streamed build (loading one if RAM allows), else the
        planned resident build to be freed after the encode. A parked vision build serves
        text prompts too; a text-only one is replaced when references arrive."""
        te = self._te_parked
        if te is not None and (self._te_parked_vision or not with_vision):
            self._te_was_parked = True
            self._te_wake(te)
            return te, True
        if te is not None:                        # text-only parked, vision needed now
            self._te_free()
        self._te_was_parked = False
        avail = self._ram_available_gb()
        if avail is not None and avail >= self._TE_PARK_MIN_AVAIL_GB \
                and os.environ.get("FIZGIG_NO_TE_PARK") != "1":
            try:
                from fizgig.minimax.embedderH2D import load_minimax_h3_te as _load_h2d
                # Always the vision build: it encodes plain prompts identically (same
                # language stack) and serves references without a second build — one
                # session, one parked encoder (two of them + a parked DiT is what pushed
                # a 137 GB box into paging when fl2va switched to ref mode, 4 Sep).
                te = _load_h2d(self.te_path, device=self.device, compute_dtype=torch.bfloat16,
                               quantize=True, with_vision=True, layer_streaming=True)
                self._te_parked, self._te_parked_vision = te, True
                logger.info("[h3-workbench] text encoder parked in system RAM for the session "
                            "(%.0f GB available) — the next prompt costs seconds", avail)
                return te, True
            except Exception:
                logger.exception("streamed text encoder failed — using the resident build")
        from fizgig.minimax.embedder import load_minimax_h3_te_planned
        te = load_minimax_h3_te_planned(self.te_path, device=self.device,
                                        compute_dtype=torch.bfloat16, quantize=True,
                                        with_vision=bool(with_vision))
        return te, False

    @staticmethod
    def _te_resident_modules(te):
        """The parts of a streamed encoder that live on the GPU between layers: everything
        except the streamed decoder layers (embeddings, norms, the vision tower)."""
        mods = []
        try:
            decoder, _emb = te._get_decoder_and_embeddings()
        except Exception:
            return mods
        for name, child in decoder.named_children():
            if name != "layers":
                mods.append(child)
        if te.model is not decoder:
            for _name, child in te.model.named_children():
                if child is not decoder:
                    mods.append(child)
        return mods

    def _te_park(self):
        """After an encode: rings off the GPU, resident parts to the CPU — unless the box is
        short of RAM now, in which case the encoder is let go."""
        te = self._te_parked
        if te is None:
            return
        avail = self._ram_available_gb()
        if avail is not None and avail < self._TE_KEEP_MIN_AVAIL_GB:
            logger.info("[h3-workbench] only %.1f GB RAM available — releasing the parked "
                        "text encoder", avail)
            self._te_free()
            return
        try:
            ls = getattr(te, "layer_streamer", None)
            if ls is not None:
                ls.unload_all()
            # Remember where each resident part LIVES (the text-only streamed build keeps
            # its embedding table on the CPU on purpose — waking it onto the GPU broke the
            # lookup, 4 Sep), then park them all on the CPU.
            if getattr(self, "_te_home", None) is None:
                home = []
                for m in self._te_resident_modules(te):
                    try:
                        dev = str(next(m.parameters()).device)
                    except StopIteration:
                        dev = str(self.device)
                    home.append((m, dev))
                self._te_home = home
            for m, _dev in self._te_home:
                m.to("cpu")
        except Exception:
            logger.exception("parking the text encoder failed — releasing it")
            self._te_free()
            return
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _te_wake(self, te):
        """Before an encode on a parked encoder: each resident part back to where it lived
        (the rings reload themselves on the first layer)."""
        for m, dev in (getattr(self, "_te_home", None) or []):
            m.to(dev)

    def _te_free(self):
        te = self._te_parked
        self._te_parked, self._te_parked_vision = None, False
        self._te_home = None
        if te is None:
            return
        try:
            ls = getattr(te, "layer_streamer", None)
            if ls is not None:
                ls.unload_all()
        except Exception:
            pass
        del te
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # The streamed build's weights are PINNED host memory; PyTorch's host caching
            # allocator keeps freed pinned blocks for reuse, so ~24 GB stayed "in use"
            # after the encoder was dropped (measured 4 Sep). Hand them back to the OS
            # (blocks with a copy event still pending are skipped — hence the sync first).
            try:
                torch.cuda.synchronize()
                torch._C._host_emptyCache()
            except Exception:
                pass

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
            from fizgig.minimax import model as _mm
            _mm.set_int8_attention(bool(getattr(self, "int8_attention", False)))
            try:
                with torch.no_grad():
                    return _sample(block_cache)
            finally:
                _mm.set_int8_attention(False)

        def _sample(block_cache):
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
            self._decoder_after_use(dec)
        arr = (px.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        return Image.fromarray(arr)

    # ----- clip primitives (Repair Studio video mode / effect lattice) --------
    def render_latent(self, state, *, seed: Optional[int] = None,
                      prompt: Optional[str] = None, width: Optional[int] = None,
                      height: Optional[int] = None, frames: Optional[int] = None,
                      steps: Optional[int] = None, turbo_strength: Optional[float] = None,
                      keyframes=None, on_denoised=None, no_lora: bool = False,
                      references=None):
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
            # Sticky for the whole render: set_turbo_strength below re-composes the
            # injection and must keep the LoRAs' rows out (it didn't — the "no LoRA"
            # render carried the primary's AdaLN rows, 4 Sep).
            self._adaln_no_lora = True
            self._reinstall_adaln()
        prompt = prompt if prompt is not None else state.prompt
        seed = seed if seed is not None else state.seed
        width = width or state.preview_width
        height = height or state.preview_height
        steps = steps or self._steps
        frames = int(frames or getattr(state, "preview_frames", 0) or H3_PREVIEW_FRAMES)
        if keyframes is None:
            keyframes = getattr(state, "keyframes", None)
        if references is None:
            references = getattr(state, "references", None)
        if turbo_strength is not None:
            self.set_turbo_strength(turbo_strength)
        ref_latents, text_tags = None, None
        if references:
            # ref2va: the prompt is encoded WITH the reference pictures (vision blocks), and
            # the same pictures' latents ride as condition rows.
            emb = self._encode_prompt(prompt, images=[im for im, _z in references])
            text_tags = self._prompt_cache_tags
            ref_latents = [z for _im, z in references]
        else:
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
               round(float(getattr(state, "donor_scale", 1.0)), 4),
               getattr(self, "dit_path", None), bool(getattr(self, "int8_attention", False)))
        resume_ok = (self.resume_enabled and not keyframes and not references and not no_lora
                     and not getattr(self, "_blocks_swapped", 0)
                     and hasattr(self.dit, "forward_cached"))
        if self._act_cache and not (resume_ok and self._act_cache_key == key):
            # Another setup's cache is dead weight — on the GPU it can be several GB
            # (56 frames at 768² records ~5.1 GB), and it must go BEFORE the free-VRAM
            # checks below, not sit there while this render fights for room. That holds
            # for a keyframe / reference / no-LoRA render too, which never resumes.
            self._invalidate_activation_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # Every render, resume or not: the canvas the decoder-residency rule sizes its
        # working set from, and the decoder / GPU cache giving way when the card is tight.
        # (These sat inside the resume branch until 4 Sep — on the streamed-block tier,
        # and for keyframe / reference / no-LoRA renders, the rule ran on a 2 GB guess.)
        self._last_canvas = (int(width), int(height), int(frames))
        self._park_decoder_if_tight(width, height, frames)
        self._evict_gpu_cache_if_tight(width, height, frames)
        if resume_ok:
            resume = None
            entries = {}
            if self._act_cache_key == key and self._act_cache:
                entries = self._act_cache
                resume = self._resume_from_diff(state)
            # The previous render's cache (same key) is replaced by this one: if it sits on
            # the GPU it is reclaimable, not "used" — without this the plan flip-flopped
            # cuda / cpu on alternate renders (Peter's log, 4 Sep).
            reclaim = (self.resume_cache_gb(width, height, frames)
                       if entries and getattr(self, "_act_cache_device", None) == "cuda" else 0.0)
            dev = self._resume_cache_device(width, height, frames, reclaim_gb=reclaim)
            ctx = sampling.BlockCacheContext(entries=entries, resume_from=resume,
                                             cache_device=dev, record_steps={0})
            self.last_resume_from = resume

        def _render(block_cache):
            from fizgig.minimax import model as _mm
            _mm.set_int8_attention(bool(getattr(self, "int8_attention", False)))
            try:
                with torch.no_grad():
                    return _sample(block_cache)
            finally:
                _mm.set_int8_attention(False)

        def _sample(block_cache):
            with torch.no_grad():
                return sampling.sample_image(
                    self.dit, emb.to(self.device, self.dtype),
                    width=width, height=height, steps=steps, cfg_scale=1.0,
                    seed=int(seed), device=self.device, dtype=self.dtype,
                    num_frames=frames, on_slow_step=_abort_check, slow_step_s=0.0,
                    return_audio=True, keyframes=keyframes, block_cache=block_cache,
                    on_denoised=on_denoised, exact_frames=True,
                    ref_latents=ref_latents, text_token_tags=text_tags)

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
                        self._act_cache_device = ctx.cache_device
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
                self._adaln_no_lora = False
                self.apply_state(state)               # per-block flags + AdaLN rows back
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
            self._decoder_after_use(dec)
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
        kf = getattr(state, "keyframes", None) or []
        refs = getattr(state, "references", None) or []
        if not kf and not refs:
            return ()
        sig = []
        for idx, lat in kf:
            t = lat.float()
            sig.append((int(idx), tuple(t.shape), round(float(t.sum()), 3),
                        round(float(t.abs().mean()), 5)))
        for i, (_img, lat) in enumerate(refs):
            t = lat.float()
            sig.append(("ref", i, tuple(t.shape), round(float(t.sum()), 3),
                        round(float(t.abs().mean()), 5)))
        return tuple(sig)

    def clip_key(self, state, *, frames, steps, turbo_strength, with_audio):
        return (self.primary_path, self.donor_path, int(state.seed), state.prompt,
                int(state.preview_width), int(state.preview_height), int(frames), int(steps),
                turbo_strength, bool(with_audio), self.keyframe_signature(state),
                round(float(getattr(state, "primary_scale", 1.0)), 4),
                round(float(getattr(state, "donor_scale", 1.0)), 4),
                bool(getattr(self, "int8_attention", False)))

    def render_clip(self, state, *, frames: Optional[int] = None, regime: str = "confirm",
                    with_audio: bool = True, cache=None, early_step: int = 0,
                    on_early=None, no_lora: bool = False, steps=None, turbo_strength=None,
                    decode: bool = True, **_ignored) -> dict:
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
        from fizgig.minimax.model import int8_kernel_available as _int8_active
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
        if decode:
            imgs = self.decode_clip_frames(lat)
            wav = self.decode_audio(aud) if (with_audio and frames > 1) else None
            middle = imgs[len(imgs) // 2]
        else:
            # The library builder: the latent is the entry, a thumb is enough — and not even
            # that if a cancel is already waiting (the render itself is never thrown away).
            imgs, wav = [], None
            _ev = getattr(self, "_cancel_event", None)
            middle = (None if (_ev is not None and _ev.is_set())
                      else self.decode_middle_frame_image(lat))
        clip = {"latent": lat, "audio_rows": aud, "frames": imgs, "wav": wav,
                "middle": middle, "regime": regime, "steps": steps,
                "turbo_strength": strength, "frames_n": frames, "cached": cached, "sig": sig,
                "int8_attention": bool(getattr(self, "int8_attention", False)) and _int8_active()}
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
                "int8_attention": bool(getattr(self, "int8_attention", False)) and _int8_active(),
                "sig": sig, "label": cache.info(sig).get("label", "")}

    @torch.no_grad()
    def decode_middle_frame_image(self, latent) -> Image.Image:
        """One PIL frame (the clip's middle) from a latent — the early-look decode."""
        dec = self._ensure_decoder().to(self.device)
        try:
            px = dec.decode_middle_frame(latent.to(self.device).float())[0]
            arr = (px.permute(1, 2, 0).clamp(0, 1) * 255).byte().cpu().numpy()
        finally:
            self._decoder_after_use(dec)
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
        base.references = getattr(state, "references", None)
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
        base.references = getattr(state, "references", None)
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
        self._act_cache_device = None

    # VRAM the pass-1 cache must leave free beside itself for the render's own working set
    # — which grows with the cache (both scale with tokens): 3 GB plus half the cache.
    # Measured 4 Sep at 768×768×56 with 8.4 GB free: a 5.1 GB cache + the render did NOT
    # fit (needed ~3 GB of working set + fragmentation on top), so that size parks on the
    # CPU on a 32 GB card; the 22-frame dial (0.9 GB) stays on the GPU.
    _RESUME_HEADROOM_GB = 4.0
    _RESUME_HEADROOM_FRAC = 0.5
    # Never on the GPU above this: a 56-frame cache (4-5 GB) beside the 21 GB base left a
    # render that "fitted" by the numbers paging through the Windows driver instead of
    # failing — All off at 768×640×56 took 4½ minutes (Peter, 4 Sep). The 22-frame caches
    # (≤ 2.3 GB) are the ones the resume pays off on anyway; the CPU nets the same saving.
    _RESUME_GPU_MAX_GB = 2.5

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

    # The 4.5 GB fp16 video decoder: moving it to the card and back costs ~1.1 s per decode
    # (measured 4 Sep), and a Dial render decodes twice (early look + clip) — ~2 s of a
    # 4–6 s render. It stays RESIDENT when the card has the next render's working set free
    # beside it, and is parked on the CPU only when the room is needed.
    _DECODER_GB = 4.5

    def _working_set_gb(self) -> float:
        """The render's own working set at the last canvas (~0.7 × the pass-1 cache size)."""
        try:
            w, h, f = self._last_canvas
            return 0.7 * self.resume_cache_gb(w, h, f)
        except Exception:
            return 2.0

    def _decoder_after_use(self, dec) -> None:
        """After a decode: keep the decoder on the card if the room is there, else park it."""
        free = None
        try:
            from fizgig.utils.device import plannable_free_vram
            if torch.cuda.is_available():
                # The decode's tiled churn leaves gigabytes reserved-but-free, and the
                # driver counts reserved as used: measured on the simulated 24 GB card,
                # 8 GB free with the decoder off the card read as 1.7 with it on. Judge
                # the true allocation.
                torch.cuda.empty_cache()
            free = float(plannable_free_vram())
        except Exception:
            pass
        if free is not None and free >= self._working_set_gb() + 2.0:
            self.decoder = dec                        # resident
            self._decoder_resident = True
            return
        self.decoder = dec.to("cpu")
        self._decoder_resident = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _park_decoder_if_tight(self, width, height, frames) -> None:
        """Before a render: a resident decoder gives way when the card lacks the working set."""
        if not getattr(self, "_decoder_resident", False) or self.decoder is None:
            return
        try:
            from fizgig.utils.device import plannable_free_vram
            if torch.cuda.is_available():
                torch.cuda.empty_cache()          # judge the allocation, not the churn
            free = float(plannable_free_vram())
            need = 0.7 * self.resume_cache_gb(width, height, frames) + 2.0
        except Exception:
            return
        if free >= need:
            return
        logger.info("[h3-workbench] %.1f GB free — parking the decoder for this render", free)
        self.decoder = self.decoder.to("cpu")
        self._decoder_resident = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _evict_gpu_cache_if_tight(self, width, height, frames) -> None:
        """A same-key cache sitting on the GPU is moved to the CPU when the card no longer
        has the render's working set free beside it (~0.7 × the cache size + 1.5 GB) —
        whatever took the room (a parked encoder's residue, another app). Paging is the
        alternative, and it is minutes."""
        entries = self._act_cache
        if not entries or getattr(self, "_act_cache_device", None) != "cuda":
            return
        try:
            from fizgig.utils.device import plannable_free_vram
            free = float(plannable_free_vram())
            est = self.resume_cache_gb(width, height, frames)
        except Exception:
            return
        if free >= 0.7 * est + 1.5:
            return
        logger.info("[h3-workbench] %.1f GB free beside a %.1f GB GPU cache — parking it on "
                    "the CPU for this render", free, est)
        for e in entries.values():
            try:
                e.block_inputs = [(t.to("cpu") if hasattr(t, "to") else t) for t in e.block_inputs]
            except Exception:
                pass
        self._act_cache_device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resume_cache_device(self, width, height, frames, reclaim_gb: float = 0.0) -> str:
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
            free = float(plannable_free_vram()) + float(reclaim_gb or 0.0)
        except Exception:
            return "cpu"
        # Fresh choice: the cache, half again for the render's working set, and the fixed
        # headroom. Staying: a same-key cache already on the GPU has proven the render fits
        # beside it, so only the cache plus the fixed headroom is needed (no flip-flop).
        need = est * (1.0 + self._RESUME_HEADROOM_FRAC) + self._RESUME_HEADROOM_GB
        if reclaim_gb and reclaim_gb > 0:
            need = est + self._RESUME_HEADROOM_GB
        dev = "cuda" if (need <= free and est <= self._RESUME_GPU_MAX_GB) else "cpu"
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
        try:
            _off = getattr(self.dit, "_h2d_offloader", None)
            if _off is not None:
                _off.release()                    # the ring's pinned staging goes with it
        except Exception:
            pass
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
            turbo_adaln_unpatch([(m, a, b) for _n, m, a, b in
                                 (self._primary_adaln or []) + (self._donor_adaln or [])])
        except Exception:
            pass
        self.primary_network = None
        self.donor_network = None
        self._turbo_net = None
        self._turbo_adaln = []
        self._primary_adaln = []
        try:
            self._te_free()
        except Exception:
            pass
        self._donor_adaln = []
        self._adaln_installed = None
        self._adaln_bid = {}
        self._last_state = None
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
