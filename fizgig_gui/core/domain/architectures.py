# Architecture configurations
ARCHITECTURES = {
    "Flux 2 Klein Base 9B": {
        "train_script": "FizgigIndependent/src/fizgig/scripts/train.py",
        "cache_latents_script": "FizgigIndependent/src/fizgig/scripts/cache_latents.py",
        "cache_text_script": "FizgigIndependent/src/fizgig/scripts/cache_text.py",
        "network_module": "fizgig.networks.lora_klein",
        "use_fizgig_venv": True,
        "timestep_sampling": "flux2_shift",
        "discrete_flow_shift": None,
        "weighting_scheme": "none",
        "blocks_swap_max": 16,
        "fp8_text_encoder_flag": "--fp8_text_encoder",
        "uses_clip": False,
        "uses_t5": False,
        "uses_text_encoder": True,
        "uses_model_type": False,
        "uses_model_version": True,
        "model_version": "klein-base-9b",
        "vae_label": "AE Model (ae.safetensors from FLUX.2-dev — NOT the Diffusers subfolder VAE)",
        "text_encoder_label": "Text Encoder (Qwen3-8B)",
        "is_distilled": False,  # Recommended for training
        "supports_weighting_scheme": False,  # Architecture uses hardcoded "none"
        "supports_discrete_flow_shift": False,  # Uses flux2_shift automatic
        # Sample generation settings
        "supports_samples": True,
        "sample_cfg_default": 4.5,
        "sample_flow_shift_default": None,
        "sample_steps_default": 40,  # Base is not step-distilled — BFL spec is ~50 steps; 40 balances quality vs sample time. Distilled path overrides to 4.
        "sample_width_default": 768,
        "sample_height_default": 768,
        # Trailing tag on the default LoRA name, so a file says which family made it. Swapped
        # when the family changes — see _apply_lora_name_suffix.
        "lora_name_suffix": "k9b",
    },
    "Krea 2": {
        # Krea 2 trains natively via fizgig.scripts.krea2_* (no accelerate launch — a
        # single-process script). The command builders branch on "is_krea2" and ignore
        # the Klein-shaped keys below; they're kept so start_training / the Samples tab /
        # validate_inputs can read the same config shape without KeyErrors.
        "train_script": "src/fizgig/scripts/krea2_train.py",
        "cache_latents_script": "src/fizgig/scripts/krea2_cache_latents.py",
        "cache_text_script": "src/fizgig/scripts/krea2_cache_text.py",
        "is_krea2": True,
        "network_module": "fizgig.networks.lora_klein",  # unused — krea2 trainer builds its own net
        "use_fizgig_venv": True,
        "timestep_sampling": "shift",
        "discrete_flow_shift": 2.5,
        "weighting_scheme": "none",
        "blocks_swap_max": 26,  # Krea 2 SingleStreamDiT has 28 blocks; offloader caps at 28-2
        "fp8_text_encoder_flag": None,
        "uses_clip": False,
        "uses_t5": False,
        "uses_text_encoder": True,
        "uses_model_type": False,
        "uses_model_version": False,
        "model_version": "krea-2",
        "vae_label": "Qwen-Image VAE",
        "text_encoder_label": "Qwen3-VL-4B",
        "is_distilled": False,
        "supports_weighting_scheme": False,
        "supports_discrete_flow_shift": False,
        # Sample generation: previews render on the fp8 Turbo (8-step; CFG optional —
        # raise Sample CFG Scale above 1 with a negative prompt for guided previews).
        "supports_samples": True,
        "sample_cfg_default": 1.0,
        "sample_flow_shift_default": None,
        "sample_steps_default": 8,
        "sample_width_default": 1024,
        "sample_height_default": 1024,
        "lora_name_suffix": "krea2",
    },
    "MiniMax H3": {
        # MiniMax H3 trains natively via fizgig.scripts.minimax_* (single-process). The command
        # builders branch on "is_minimax". Barebones IMAGE-ONLY: no samples, no preview, no
        # per-image loss watch, no LoKR — the most minimal training surface. The Klein-shaped keys
        # below are kept so start_training / Samples / validate_inputs read the same shape.
        "train_script": "src/fizgig/scripts/minimax_train.py",
        "cache_latents_script": "src/fizgig/scripts/minimax_cache_latents.py",
        "cache_text_script": "src/fizgig/scripts/minimax_cache_text.py",
        "is_minimax": True,
        "network_module": "fizgig.networks.lora_klein",  # unused — minimax trainer builds its own net
        "use_fizgig_venv": True,
        "timestep_sampling": "shift",
        "discrete_flow_shift": 12.0,   # H3 video sigma-shift (fixed; the Timestep section is hidden)
        "weighting_scheme": "none",
        "blocks_swap_max": 40,         # bnb NF4 blocks swap packed (uint8) — planner caps at 40 of 50
        "fp8_text_encoder_flag": None,
        "uses_clip": False,
        "uses_t5": False,
        "uses_text_encoder": True,
        "uses_model_type": False,
        "uses_model_version": False,
        "model_version": "minimax-h3",
        "vae_label": "MiniMax H3 Video VAE",
        "text_encoder_label": "Qwen3-VL-32B",
        "is_distilled": False,
        "supports_weighting_scheme": False,
        "supports_discrete_flow_shift": False,
        # In-training previews render one still per prompt on the resident training DiT
        # (latent_t=1, the training distribution). Per-epoch only, like Krea 2.
        "supports_samples": True,
        "sample_cfg_default": 1.0,
        "sample_flow_shift_default": None,
        # H3 samples run CFG-free on a fixed shift-12 schedule, exactly as every shipped ComfyUI
        # workflow does (BasicGuider, no negative conditioning). Reusing the existing distilled
        # flags greys out Negative Prompt and CFG Scale through the generic path rather than
        # adding a MiniMax branch — nothing about them is editable on this family.
        "sample_is_distilled": True,
        "sample_cfg_fixed": True,
        "sample_steps_default": 20,   # the reference pipeline default
        # 768x768 (Peter, 17 Aug — down from the 11 Aug 1024): H3's native canvas is a 768
        # short edge, and with clips-with-sound in the preview mix the smaller frame keeps
        # those affordable too. 1024 is still one dropdown away.
        "sample_width_default": 768,
        "sample_height_default": 768,
        "lora_name_suffix": "mmh3",
    },
}

# Saved configs written before 3.6.1 carry the old label. Every lookup here is a .get() that
# falls back to Klein, so without an alias a MiniMax preset would silently come back as a Klein
# one - wrong family, no error. The alias points at the same config; _canon_arch maps it forward
# so what the user then SEES is the current name.
_ARCH_ALIASES = {"MiniMax H3 (experimental)": "MiniMax H3",
                 "Krea 2 (experimental)": "Krea 2"}       # pre-rename saves (2026-07-28)
for _old, _new in _ARCH_ALIASES.items():
    ARCHITECTURES[_old] = ARCHITECTURES[_new]

# Aliases are readable, not offerable: the dropdown lists current names only.
ARCHITECTURE_LIST = [k for k in ARCHITECTURES if k not in _ARCH_ALIASES]


def _canon_arch(name):
    """Old label in, current label out."""
    return _ARCH_ALIASES.get(name, name)

# Every family suffix we recognise on a LoRA name. Used to swap ONE tag for another when the
# model family changes — matching this set (never "the last underscore segment") is what stops
# a name like portrait_v2 being mangled into portrait_krea2.
LORA_NAME_SUFFIXES = {c["lora_name_suffix"] for c in ARCHITECTURES.values()
                      if c.get("lora_name_suffix")}
