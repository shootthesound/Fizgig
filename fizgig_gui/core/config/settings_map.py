# Maps Training tab settings keys to prefs keys — entries with matching keys
# will be bound to the shared prefs StringVar (two-way sync with Preferences tab)
SETTING_TO_PREF = {
    "DIT_MODEL": "base_dit",
    "VAE_MODEL": "vae",
    "TEXT_ENCODER": "text_encoder",
    "LORA_OUTPUT_DIR": "lora_output_dir",
}


# Default preset for Klein Base 9B — the only supported architecture.
# Applied on first launch and when "Reset to Defaults" is used.
PRESETS = {
    "Flux 2 Klein Base 9B": {
        "LEARNING_RATE": 0.0004,       # Rank 4:4 optimal LR for 80+ image datasets
        "NETWORK_DIM": 4,              # Low rank — identity signal only
        "NETWORK_ALPHA": 4,            # 1:1 alpha:rank ratio
        "MAX_TRAIN_EPOCHS": 12,        # Fast convergence at low rank
        "OPTIMIZER_TYPE": "adamw8bit",
        "TIMESTEP_SAMPLING": "flux2_shift",
        "DISCRETE_FLOW_SHIFT": "0",
        "WEIGHTING_SCHEME": "none",
        "BLOCKS_SWAP": "auto",
        # Model paths come from Preferences at runtime — leave blank in the preset.
        "VAE_MODEL": "",
        "TEXT_ENCODER": "",
        "DIT_MODEL": "",
        "FP8": True,
        "SCALED": True,  # BF16 model, use fp8_scaled for memory efficiency
        "QUANT_4BIT": False,  # 4-bit NF4 base (low-VRAM); supersedes fp8 when on
        "COMPILE_BLOCKS": "auto",  # torch.compile the DiT blocks (krea2): auto | on | off
        "GRADIENT_CHECKPOINTING": True,  # ON by default — needed to fit 9B on most cards
    },
}
