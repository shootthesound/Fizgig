import os
import sys
import threading
import webbrowser

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY, BG_COLOR
from fizgig_gui.core.config.prefs import (PREFS_FILE, DEFAULT_PREFS,
                    _enumerate_gpus, _running_on_pod, _app_commit,
                    _pod_id, _pod_stop_key_env, load_prefs)
from fizgig_gui.core.ui_base.widgets import CollapsibleFrame, ToolTip

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT, FIZGIG_DIR as _FIZGIG_DIR

class PrefsTabMixin:
    # region Preferences Tab

    # The paths a family cannot train without. The optional rows (reference DiT, audio VAE,
    # Turbo LoRA) neither hold a section open nor count against its badge — a family whose only
    # gaps are optional is configured.
    _PREFS_FAMILY_KEYS = {
        "klein": ("base_dit", "distilled_dit", "vae", "text_encoder"),
        "krea2": ("krea2_raw_dit", "krea2_turbo_dit", "krea2_vae", "krea2_text_encoder"),
        "minimax": ("minimax_dit", "minimax_text_encoder", "minimax_vae"),
    }
    # Badge names — "2 paths needed if training Klein 9B" reads as advice, where a bare
    # "2 paths needed" reads as a problem: someone who never trains that family owes it nothing.
    _PREFS_FAMILY_NAMES = {"klein": "Klein 9B", "krea2": "Krea 2", "minimax": "MiniMax H3"}

    def _prefs_family_section(self, parent, family, title, description):
        """A collapsible model-path section for one model family on the Preferences tab.

        Starts collapsed only when the family's required paths are all filled: a new user sent
        here by the Start tab's setup prompt lands on the sections they still need already open,
        while a configured machine shows three closed headers instead of a wall of path rows.
        The header badge keeps a collapsed section honest — you can see whether a family needs
        attention without opening it. Returns (content_frame, first_free_grid_row).
        """
        keys = self._PREFS_FAMILY_KEYS[family]

        def _missing():
            return sum(1 for k in keys if not str(self.prefs_vars[k].get() or "").strip())

        section = CollapsibleFrame(parent, title, default_expanded=_missing() > 0)
        section.pack(fill=tk.X, padx=36, pady=(0, 16))
        content = section.get_content_frame()
        content.columnconfigure(1, weight=1)
        tk.Label(content, text=description, font=(FONT_FAMILY, 10),
                 fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT
                 ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(4, 10))

        def _refresh_badge(*_a):
            n = _missing()
            try:
                section.badge.config(
                    text="✓ configured" if n == 0
                    else (f"{n} path{'s' if n != 1 else ''} needed if training "
                          f"{self._PREFS_FAMILY_NAMES[family]}"),
                    fg=COLORS["text_secondary"] if n == 0 else COLORS["warning"])
            except tk.TclError:
                pass

        _refresh_badge()
        for k in keys:
            self.prefs_vars[k].trace_add("write", _refresh_badge)
        return content, 1

    def create_prefs_tab(self):
        """Create the Preferences tab (Start-tab styled)."""
        scrollable_frame, _ = self.create_scrollable_frame(self.prefs_tab)

        outer = tk.Frame(scrollable_frame, bg=COLORS["bg_deep"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._add_tab_banner(
            outer,
            "Preferences",
            "Centralised paths + inference performance knobs. Changes here propagate to every tab automatically "
            "and persist to prefs.json.",
        )

        # The three model-family sections are collapsible, and smart about it: a family with a
        # required path still blank starts open, a configured one starts closed. Click the
        # header to toggle; the badge says which state you're in without opening anything.
        models_card, next_row = self._prefs_family_section(
            outer, "klein", "Model Paths (Klein 9B)",
            "Absolute paths to the four model files. Each row has a Download link that opens the HuggingFace page "
            "in your browser.",
        )
        next_row = self._add_pref_row(
            models_card, next_row, "Base DiT:", "base_dit",
            "Klein 9B Base model (for training & precise profiling). "
            "Recommended: the fp8 version — same training quality, ~half the VRAM (stays resident at "
            "~9.6GB, fits 16GB cards). The bf16 version is the larger alternative.",
            download_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8/tree/main",
            download_label="Download fp8 (recommended)",
            download_note="~9.5GB fp8 — Black Forest Labs (flux-2-klein-base-9b-fp8.safetensors)",
            download_url2="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/tree/main",
            download_label2="Download bf16",
            download_note2="~17GB bf16 (flux-2-klein-base-9b.safetensors)",
        )
        next_row = self._add_pref_row(
            models_card, next_row, "Distilled DiT:", "distilled_dit",
            "Klein 9B Distilled model (for Repair Studio previews, fast profiling & diagnostics)",
            download_url="https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8/tree/main",
            download_note="~9GB fp8 quantised — Black Forest Labs (flux-2-klein-9b-fp8.safetensors)",
        )
        next_row = self._add_pref_row(
            models_card, next_row, "VAE / AE:", "vae",
            "Flux 2 AutoEncoder — use ae.safetensors from FLUX.2-dev root (NOT the vae/ subfolder Diffusers file)",
            download_url="https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors",
            download_note="~320MB  ·  get ae.safetensors from FLUX.2-dev root  ·  NOT vae/diffusion_pytorch_model.safetensors (Diffusers format, incompatible)",
        )
        next_row = self._add_pref_row(
            models_card, next_row, "Text Encoder:", "text_encoder",
            "Qwen3-8B text encoder (used by Klein 9B)",
            download_url="https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/blob/main/split_files/text_encoders/qwen_3_8b.safetensors",
            download_note="~15GB single-file safetensors — Qwen3-8B packaged for Klein 9B (Comfy-Org)",
        )
        # Klein users routinely skip the whole Krea 2 card, and would never learn that one file
        # in it is a better captioner than Florence-2 for THEIR datasets — captions are just
        # .txt, so the model family is irrelevant. Worth saying here, where they are already
        # filling in paths, rather than hoping they read the card below.
        _qwen_tip = tk.Label(
            models_card,
            text="💡 Training Klein only? The Krea 2 Qwen3-VL text encoder is still worth having — "
                 "the Captions tab can caption ANY dataset with it, following an instruction you "
                 "can edit, and it writes better training captions than Florence-2. The download "
                 "button below fetches it for you; nothing else about Krea 2 is needed.",
            font=(FONT_FAMILY, 9), fg=COLORS["text_secondary"], bg=COLORS["bg_surface"],
            wraplength=760, justify=tk.LEFT)
        _qwen_tip.grid(row=next_row, column=0, columnspan=3, sticky=tk.W, pady=(12, 2))
        next_row += 1
        self._add_fetch_models_row(
            models_card, next_row, "klein",
            "Fetches the four files above plus the Krea 2 Qwen3-VL captioning text encoder "
            "(~39 GB all in) and fills in these paths for you, plus the small helper models "
            "(Florence-2 captioner, face model for the Look Filter and likeness scoring, EN→ZH "
            "translator, Gizmo's Whisper transcriber — ~1.9 GB) so nothing stalls to download "
            "later and everything works offline. "
            "Black Forest Labs gate their downloads, so you'll need a free HuggingFace token — "
            "Fizgig asks for it and tells you which pages to accept the licence on.")
        next_row += 1

        # Krea 2 model paths
        krea_card, kr = self._prefs_family_section(
            outer, "krea2", "Model Paths (Krea 2)",
            "Krea 2 LoRA training + inference. Train on RAW; previews and inference use the pre-quant fp8 Turbo "
            "(8-step, CFG-free). The text encoder can be either Qwen3-VL-4B file — bf16, or the smaller "
            "fp8_scaled (~3.6 GB less VRAM; its vision tower is bf16 either way).",
        )
        kr = self._add_pref_row(
            krea_card, kr, "RAW DiT:", "krea2_raw_dit",
            "Krea 2 RAW (undistilled 12.9B base) — the training model (krea2_raw_bf16.safetensors)",
            download_url="https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_raw_bf16.safetensors",
            download_note="~26GB bf16 — Comfy-Org/Krea-2 → diffusion_models/krea2_raw_bf16.safetensors (train on this)",
        )
        kr = self._add_pref_row(
            krea_card, kr, "Turbo DiT (fp8):", "krea2_turbo_dit",
            "Krea 2 Turbo, pre-quantized fp8 (ComfyUI) — fast previews + inference (8-step, CFG-free)",
            download_url="https://huggingface.co/Comfy-Org/Krea-2/blob/main/diffusion_models/krea2_turbo_fp8_scaled.safetensors",
            download_note="~13GB fp8 — Comfy-Org/Krea-2 → diffusion_models/krea2_turbo_fp8_scaled.safetensors",
        )
        kr = self._add_pref_row(
            krea_card, kr, "Qwen-Image VAE:", "krea2_vae",
            "The Qwen-Image VAE used by Krea 2 (qwen_image_vae.safetensors)",
            download_url="https://huggingface.co/Comfy-Org/Krea-2/blob/main/vae/qwen_image_vae.safetensors",
            download_note="~250MB — Comfy-Org/Krea-2 → vae/qwen_image_vae.safetensors",
        )
        kr = self._add_pref_row(
            krea_card, kr, "Qwen3-VL TE:", "krea2_text_encoder",
            "Qwen3-VL-4B text encoder. fp8_scaled is recommended: it quantises only the language "
            "layers and ships the same full bf16 vision tower, so reference images and AI "
            "captioning are unaffected — measured 4.9 GB resident vs 8.3 GB, for captions we "
            "could not tell apart",
            download_url="https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
            download_label="Download fp8 (recommended)",
            download_note="~5.2GB fp8_scaled — 3.4 GB less VRAM than bf16, virtually identical output",
            download_url2="https://huggingface.co/Comfy-Org/Krea-2/blob/main/text_encoders/qwen3vl_4b_bf16.safetensors",
            download_label2="bf16",
            download_note2="~8.9GB bf16 — the full-precision original, if you would rather not "
                           "quantise at all",
        )
        kr = self._add_pref_row(
            krea_card, kr, "Turbo LoRA (rank 64):", "krea2_turbo_lora",
            "Turbo distillation as a LoRA — RAW + this at strength 1.0 samples like the Turbo model "
            "(same 8-step CFG-free settings), letting previews run on the training DiT without loading "
            "the separate Turbo checkpoint",
            download_url="https://huggingface.co/Comfy-Org/Krea-2/blob/main/loras/krea2_turbo_lora_rank_64_bf16.safetensors",
            download_note="~470MB bf16 — Comfy-Org/Krea-2 → loras/krea2_turbo_lora_rank_64_bf16.safetensors",
        )
        self._add_fetch_models_row(
            krea_card, kr + 1, "krea2",
            "Fetches every file above (~45 GB) and fills in these paths for you, plus the small "
            "helper models (Florence-2 captioner, face model for the Look Filter and likeness "
            "scoring, EN→ZH translator, Gizmo's Whisper transcriber — ~1.9 GB) so nothing "
            "stalls to download later and everything works offline. No HuggingFace account "
            "needed — none of these are gated.")
        _offline_tip = tk.Label(
            krea_card,
            text="💡 Already have these files for ComfyUI? Filling the paths in by hand works "
                 "perfectly — the download button is a convenience, not a requirement. The first "
                 "time you caption or train while online, Fizgig quietly fetches a few tiny "
                 "helper files and keeps them, and from then on everything runs fully offline. "
                 "Setting up a machine that will never see the internet? Paste a complete "
                 "HuggingFace model folder into the text encoder field instead of a single file "
                 "and nothing needs downloading at all.",
            font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
            wraplength=760, justify=tk.LEFT)
        _offline_tip.grid(row=kr + 2, column=0, columnspan=3, sticky=tk.W, pady=(12, 2))

        # MiniMax H3 model paths — third family, beside the other two now that clip+audio
        # training has outgrown its bottom-of-the-page beginnings.
        mm_card, mr = self._prefs_family_section(
            outer, "minimax", "Model Paths (MiniMax H3 — experimental)",
            "Image-only LoRA training for MiniMax's ~33B H3 omni DiT. Train on the pruned int8 DiT "
            "— the same file ComfyUI runs — quantized to NF4 at load, so the resident base is "
            "~11 GB. The Qwen3-VL-32B text encoder and the video VAE are only needed for the "
            "one-time caching pass; the compact nvfp4 TE is recommended. Trains on stills, or on "
            "short video clips — and with the audio VAE set, on their sound too.",
        )
        mr = self._add_pref_row(
            mm_card, mr, "DiT:", "minimax_dit",
            "MiniMax H3 DiT — the training base. Use the PRUNED int8 file "
            "(minimax_h3_fl2va_pruned_int8_convrot.safetensors, ~21 GB): it is the one ComfyUI "
            "runs, so your LoRA trains against the weights it will be deployed on, and its "
            "curve-table AdaLN is a target a LoRA can actually use. The ~66 GB bf16 file also "
            "works. The pruned file KEEPS its int8 weights (~21 GB on the GPU, what the reference "
            "trainer does); the bf16 file is quantized to NF4 at load (~11 GB, a little lossier).",
            download_url="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            download_note="~21GB — Comfy-Org/MiniMax-H3 → diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors (fl2va is the trainable variant; the 66GB bf16 file works too)",
        )
        mr = self._add_pref_row(
            mm_card, mr, "DiT (reference):", "minimax_ref_dit",
            "OPTIONAL — used when the Training tab's Training Base is set to Reference "
            "(ref2va), and for reference distillation ('Learn identity from'). This is the "
            "ref2va model, a DIFFERENT fine-tune from the fl2va one above and not just another "
            "quantization of it: it is what ComfyUI's Reference-to-Video workflow loads, and "
            "the only H3 build that accepts reference images. A LoRA trained on it is most "
            "faithful deployed on it. Leave blank if you only train on the standard base.",
            download_url="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            download_note="~21GB — Comfy-Org/MiniMax-H3 -> diffusion_models/"
                          "minimax_h3_ref2va_pruned_int8_convrot.safetensors (the pruned int8 "
                          "build, same shape as the fl2va one above; you may already have it if "
                          "you use the r2v workflow)",
        )
        mr = self._add_pref_row(
            mm_card, mr, "Qwen3-VL-32B TE:", "minimax_text_encoder",
            "Qwen3-VL-32B text encoder — nvfp4 (the compact ComfyUI file) or bf16 both work; the "
            "loader detects which you gave it. The nvfp4 file keeps its packed weights (~15.7 GB "
            "on the GPU); bf16 is NF4-quantized at load (~14 GB). Used only while caching caption "
            "embeddings, then offloaded before training. (The int8_convrot TE "
            "variant is NOT supported — its rotated weights can't be dequantized here.)",
            download_url="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            download_label="Download nvfp4 (recommended)",
            download_note="~15.7GB nvfp4-awq — the same TE ComfyUI uses, so you may already have it; "
                          "identical conditioning to bf16 (validated), just a slower one-off load",
            download_url2="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
            download_label2="bf16",
            download_note2="~51.5GB bf16 — the full-precision original; loads faster, 3.3x the disk",
        )
        mr = self._add_pref_row(
            mm_card, mr, "Video VAE:", "minimax_vae",
            "The H3 video VAE — encodes each training image to a 24-channel latent (used only "
            "during caching).",
            download_url="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_video_vae_fp16.safetensors",
            download_note="~4.9GB — Comfy-Org/MiniMax-H3 → vae/minimax_h3_video_vae_fp16.safetensors",
        )
        mr = self._add_pref_row(
            mm_card, mr, "Audio VAE:", "minimax_audio_vae",
            "OPTIONAL — set this to train on the sound in your video clips. H3 generates audio and "
            "video together, so a clip with sound can teach it a voice, and nothing else can. "
            "Used only during caching, and only by clips: a folder of stills never loads it, and "
            "neither does a clip you muted (a _mute on the filename trains that clip's video and "
            "ignores its sound). Leave blank and clips train silent, exactly as they did before.",
            download_url="https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors",
            download_note="~605MB — Comfy-Org/MiniMax-H3 → vae/minimax_h3_audio_vae_fp32.safetensors",
        )
        mr = self._add_pref_row(
            mm_card, mr, "Turbo LoRA:", "minimax_turbo_lora",
            "OPTIONAL — fast in-training previews. With this set, previews render in 6 steps "
            "with the community Turbo LoRA applied at 75% on top of your training LoRA — the "
            "same pairing fast ComfyUI inference uses — instead of the full 20-step pass. It "
            "touches PREVIEWS ONLY: the Turbo is switched in for the sample render and out "
            "again before the next training step, and your saved LoRA never contains it. "
            "Steps and strength are adjustable on the Samples tab.",
            download_url="https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/blob/main/minimax_h3_turbo_v4_step600.safetensors",
            download_note="~780MB — larryvrh/MiniMax-H3-Turbo-Lora → "
                          "minimax_h3_turbo_v4_step600.safetensors (you may already have it in "
                          "ComfyUI's loras folder)",
        )
        self._add_fetch_models_row(
            mm_card, mr, "minimax",
            "Fetches the DiT, text encoder, both VAEs and the Turbo LoRA above, plus the Krea 2 Qwen3-VL captioning "
            "text encoder (~47 GB all in), and fills in these paths for you — plus the small "
            "helper models (Florence-2 captioner, face model for the Look "
            "Filter and likeness scoring, EN→ZH translator, Gizmo's Whisper transcriber — "
            "~1.9 GB) so nothing stalls to download later and everything works offline. No "
            "HuggingFace account needed — none of these are gated. The "
            "reference DiT is left out unless you tick it above: another 21 GB, and it is only "
            "used by identity mode.",
            optional_label="Include the reference DiT (+21 GB)")

        # Card 1b: which GPU. Only when the machine actually has more than one - a chooser with a
        # single entry is noise, and the whole feature is a no-op there.
        _gpus = _enumerate_gpus()
        if len(_gpus) > 1:
            gpu_card = self._start_section_card(
                outer, "Graphics Card",
                "Which GPU Fizgig uses — for training and for the workbench tools alike. "
                "Everything else in the app then treats that card as the only one there is.",
            )
            gpu_card.columnconfigure(1, weight=1)
            ttk.Label(gpu_card, text="Use GPU:").grid(row=0, column=0, sticky=tk.W,
                                                      padx=(0, 10), pady=4)
            self._gpu_choice_labels = {"": "System default (GPU 0)"}
            self._gpu_info = {}
            for _i, _name, _gb, _uuid in _gpus:
                self._gpu_choice_labels[str(_i)] = f"{_i}: {_name} ({_gb:.0f} GB)"
                self._gpu_info[str(_i)] = (_i, _name, _gb, _uuid)
            self._nvml_init = False  # reset so first _read_vram() uses correct index
            _saved_gpu = str(self.prefs.get("cuda_device", "")).strip()
            # Match saved value: UUID (new format) or index (legacy prefs.json)
            _matched_key = ""
            if _saved_gpu:
                _matched_key = next(
                    (k for k, v in self._gpu_choice_labels.items()
                     if self._gpu_info.get(k, (None, None, None, None))[3] == _saved_gpu
                     or k == _saved_gpu),
                    "")
            self._gpu_choice_var = tk.StringVar(
                value=self._gpu_choice_labels.get(
                    _matched_key,
                    self._gpu_choice_labels[""]))
            _gpu_combo = ttk.Combobox(
                gpu_card, textvariable=self._gpu_choice_var,
                values=list(self._gpu_choice_labels.values()), width=44, state="readonly")
            _gpu_combo.grid(row=0, column=1, sticky=tk.W, pady=4)
            _gpu_combo.bind("<<ComboboxSelected>>", self._on_gpu_choice)
            _gpu_note = tk.Label(
                gpu_card,
                text=("Takes effect for the next training run straight away. The in-app tools "
                      "(Repair Studio, Explorer, Royale, Profiler, Extract) hold on to the card "
                      "they started with, so restart Fizgig to move those."),
                font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"],
                wraplength=720, justify=tk.LEFT)
            _gpu_note.grid(row=1, column=1, sticky=tk.W, pady=(0, 4))
            if os.environ.get("CUDA_VISIBLE_DEVICES") and not str(
                    self.prefs.get("cuda_device", "")).strip():
                tk.Label(gpu_card,
                         text=(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} is set "
                               f"in your environment and wins over this setting."),
                         font=(FONT_FAMILY, 9), fg=COLORS["warning"],
                         bg=COLORS["bg_surface"], wraplength=720, justify=tk.LEFT
                         ).grid(row=2, column=1, sticky=tk.W, pady=(0, 4))

        # Card 2: Inference Performance
        inf_card = self._start_section_card(
            outer, "Inference Performance",
            "DiT Block Swap moves transformer blocks to CPU during forward passes to cut VRAM, at the cost of PCIe "
            "latency per step. Affects the workbench tools — Repair Studio / Profiler / Extract / Explorer. "
            "Training (and its Distilled samples) manage their own swap automatically and ignore this setting.",
        )
        inf_card.columnconfigure(1, weight=1)

        ttk.Label(inf_card, text="DiT Block Swap (inference):").grid(row=0, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        # Workbench previews (Repair Studio / Profiler / Extract) default to the
        # Distilled fp8 model (light); loading Base is heavier. VRAM shown for each.
        inference_swap_options = [
            "Auto (detect from GPU)",
            "0  (Distilled fp8: 16GB / Base: 24GB)",
            "4  (Distilled fp8: 14GB / Base: 20GB)",
            "8  (Distilled fp8: 11GB / Base: 16GB)",
            "12 (Distilled fp8: 10GB / Base: 14GB)",
            "16 (Distilled fp8: 8GB / Base: 12GB)",
        ]
        _inf_combo = ttk.Combobox(
            inf_card, textvariable=self.prefs_vars["inference_blocks_to_swap"],
            values=inference_swap_options, width=40, state="readonly",
        )
        _inf_combo.grid(row=0, column=1, sticky=tk.W, pady=4)
        # Snap whatever's saved to the matching new label by extracting the leading integer.
        import re as _re_snap
        _current = str(self.prefs_vars["inference_blocks_to_swap"].get()).strip()
        _m = _re_snap.match(r'\d+', _current)
        if _m:
            _leading_int = _m.group()
            for _opt in inference_swap_options:
                if _opt.lstrip().startswith(_leading_int + " ") or _opt.lstrip().startswith(_leading_int + "  "):
                    self.prefs_vars["inference_blocks_to_swap"].set(_opt)
                    break

        # INT8 fast inference (on by default). Quantizes the workbench/preview DiT's block Linears to
        # int8 for a faster matmul. Same VRAM as fp8 (8-bit either way) — a speed knob, not a memory
        # one — stacks with block swap, and only affects previews (never the saved LoRA). Biggest win
        # on RTX 30-series (no fast fp8), modest on 40/50-series.
        ttk.Label(inf_card, text="INT8 fast inference:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        ttk.Checkbutton(
            inf_card, text="Quantize previews/workbench to int8 for a faster matmul (recommended)",
            variable=self.prefs_vars["inference_int8"], onvalue="1", offvalue="0",
        ).grid(row=1, column=1, sticky=tk.W, pady=4)
        tk.Label(inf_card, text="On by default. Same VRAM as fp8, near-identical quality, previews only. Turn off to use fp8.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_muted"], bg=COLORS["bg_surface"]).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=(0, 4))

        # Card 3: Output Directories
        out_card = self._start_section_card(
            outer, "Output Directories",
            "Paths stored as relative-to-repo when they live inside FizgigIndependent/ (portable across clones/moves), "
            "absolute otherwise. Dataset TOMLs always live in FizgigIndependent/dataset/ — not configurable.",
        )
        out_card.columnconfigure(1, weight=1)
        next_row = 0
        next_row = self._add_pref_row(out_card, next_row, "LoRA output:", "lora_output_dir", "Where trained LoRAs are saved", is_dir=True)
        next_row = self._add_pref_row(out_card, next_row, "Profiles:", "profiles_dir", "Where profiler HTML reports are saved", is_dir=True)
        next_row = self._add_pref_row(out_card, next_row, "Cache:", "cache_dir", "Cached latents and text encodings", is_dir=True)

        # Card 3b: Input Directories — default starting folders for the Browse
        # dialogs, so users don't re-hunt for their LoRAs / reference images.
        in_card = self._start_section_card(
            outer, "Input Directories",
            "Default folders the Browse dialogs open in. Saves hunting each session when loading "
            "LoRAs or reference images. Leave empty to use the last-used folder.",
        )
        in_card.columnconfigure(1, weight=1)
        in_row = 0
        in_row = self._add_pref_row(in_card, in_row, "LoRA loading:", "input_lora_dir",
                                    "Default folder for loading LoRAs — Repair Studio, LoRA the Explorer, and the Context LoRA picker", is_dir=True)
        in_row = self._add_pref_row(in_card, in_row, "Reference images:", "input_ref_dir",
                                    "Default folder for reference images — Repair Studio and LoRA the Explorer", is_dir=True)
        in_row = self._add_pref_row(in_card, in_row, "Training images:", "input_dataset_dir",
                                    "Default folder the Start tab's Browse opens in", is_dir=True)

        self._add_runpod_card(outer)


        # Card 4: Actions
        actions_card = self._start_section_card(outer, "Actions", None)
        action_row = tk.Frame(actions_card, bg=COLORS["bg_surface"])
        action_row.pack(anchor=tk.W)
        ttk.Button(action_row, text="Reset to Defaults", command=self._reset_prefs).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Button(action_row, text="Open prefs.json", command=self._open_prefs_file).pack(side=tk.LEFT)

        self._add_youtube_help_button(outer, "preferences")

    # Flip to True only when the RunPod template is public AND the URL below is the real one.
    # Until then the desktop card says "coming soon" instead of offering a button that goes
    # nowhere — so this can ship long before the template does. test_runpod_card.py refuses to
    # let LIVE be True while the URL is still the placeholder.
    RUNPOD_TEMPLATE_LIVE = True
    RUNPOD_GUIDE_URL = ("https://github.com/shootthesound/Fizgig/blob/master/docker/README.md")
    # Pre-selects an RTX 5090: it is the cheapest card that clears Fizgig's 32 GB
    # no-block-swap threshold for Krea 2, so the default lands people on the fastest
    # sensible option rather than the biggest or the cheapest.
    RUNPOD_DEPLOY_URL = ("https://console.runpod.io/deploy"
                         "?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um")
    RUNPOD_REFERRAL = "vkb387ep"

    def _runpod_deploy_url(self) -> str:
        u = self.RUNPOD_DEPLOY_URL
        return f"{u}&ref={self.RUNPOD_REFERRAL}" if self.RUNPOD_REFERRAL else u

    def _add_runpod_card(self, outer):
        """One card, two audiences.

        On a pod it is the control panel for things only a rented machine has — an hourly bill, a
        volume that fills up, and a browser tab people assume is holding the run up. On the desktop
        it is how someone finds out Fizgig runs on rented hardware at all."""
        if _running_on_pod():
            self._build_pod_controls(outer)
        else:
            self._build_pod_advert(outer)

    def _build_pod_controls(self, outer):
        card = self._start_section_card(
            outer, "RunPod",
            "Fizgig is running on a rented GPU. These settings only appear here.")
        self._runpod_card = card

        # The money one. A finished run on an idle rented GPU bills until someone notices.
        ttk.Checkbutton(
            card,
            text="Stop this pod when a training run finishes",
            variable=self.prefs_vars["runpod_stop_when_done"], onvalue="1", offvalue="0",
            style="Surface.TCheckbutton").pack(anchor=tk.W)
        tk.Label(card,
                 text="Only after a run completes on its own — never after a Pause, a Stop, or a "
                      "failure, since those are exactly the times you want the machine alive. You "
                      "get a two-minute countdown you can cancel. This stops the pod, it never "
                      "terminates it, so your files are still here when you start it again.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                 bg=COLORS["bg_surface"], wraplength=760,
                 justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 2))
        # The key field. Better here than as a template variable: a public template hands its
        # variables to everyone who deploys it, so nobody can safely ship a key in one.
        key_row = tk.Frame(card, bg=COLORS["bg_surface"])
        key_row.pack(anchor=tk.W, fill=tk.X, pady=(2, 0))
        tk.Label(key_row, text="RunPod API key:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=COLORS["bg_surface"]).pack(side=tk.LEFT,
                                                                            padx=(0, 8))
        self._pod_key_entry = ttk.Entry(key_row, textvariable=self.prefs_vars["runpod_api_key"],
                                        width=44, show="•")
        self._pod_key_entry.pack(side=tk.LEFT)
        ttk.Button(key_row, text="Clear", width=7,
                   command=lambda: self.prefs_vars["runpod_api_key"].set("")).pack(side=tk.LEFT,
                                                                                    padx=(6, 0))
        self._pod_key_status = tk.Label(key_row, text="", font=(FONT_FAMILY, 9),
                                        bg=COLORS["bg_surface"])
        self._pod_key_status.pack(side=tk.LEFT, padx=(10, 0))

        tk.Label(card,
                 text="Make one at RunPod → Settings → API Keys. The key RunPod gives a pod "
                      "automatically is pod-scoped and cannot stop pods, which is a RunPod "
                      "limitation rather than a Fizgig one. Saved to prefs.json on your volume, so "
                      "it persists across pods — and stays out of any shared template.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(4, 12))

        self.prefs_vars["runpod_api_key"].trace_add(
            "write", lambda *a: self._refresh_pod_key_status())
        self._refresh_pod_key_status()

        # Storage — the thing that silently ends a run at 3am.
        self._pod_storage_lbl = tk.Label(card, text="", font=(FONT_FAMILY, 10),
                                         fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                                         justify=tk.LEFT, anchor="w")
        self._pod_storage_lbl.pack(anchor=tk.W, fill=tk.X)
        self._refresh_pod_storage()

        tk.Label(card,
                 text="Your files: datasets in /workspace/datasets, models in /workspace/models, "
                      "finished LoRAs in /workspace/output_loras. Everything under /workspace "
                      "survives stopping and restarting the pod — anything outside it does not. "
                      "On the default Volume Disk it goes when you TERMINATE the pod, so stop "
                      "rather than terminate between sessions; a Network Volume survives that too. "
                      "Drag files in and out with the file manager on port 8080.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 0))

        # Asked constantly by anyone new to a remote desktop, and the answer is reassuring.
        tk.Label(card,
                 text="Closing this browser tab does not stop training. Fizgig runs on the pod, "
                      "not in your browser — shut the tab, come back later, and the run is still "
                      "going.",
                 font=(FONT_FAMILY, 10, "bold"), fg=COLORS["success"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 0))

        # Image version AND app commit: the template pins the image while the app updates itself
        # from git at every boot, so they diverge by design and "what are you running?" needs both.
        bits = []
        _img = (os.environ.get("FIZGIG_IMAGE_VERSION") or "").strip()
        if _img:
            bits.append(f"image {_img}")
        _sha = _app_commit()
        if _sha:
            bits.append(f"app {_sha}")
        _pid = _pod_id()
        if _pid:
            bits.append(f"pod {_pid}")
        if bits:
            _gpu = (os.environ.get("RUNPOD_GPU_NAME") or "").replace("+", " ").strip()
            line = "  ·  ".join(bits) + (f"  ·  {_gpu}" if _gpu else "")
            # Readable tier despite being a footer: this is the one line a user is asked to
            # transcribe into a bug report, so 8pt at 2.54:1 was exactly backwards.
            tk.Label(card, text=line, font=(FONT_FAMILY, 9), fg=COLORS["text_explain"],
                     bg=COLORS["bg_surface"], wraplength=760,
                     justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 0))

    def _build_pod_advert(self, outer):
        card = self._start_section_card(
            outer, "Run Fizgig on a rented GPU",
            "Train on whatever card you like, with as much VRAM as you want, billed by the hour. "
            "Nothing to install, and your own machine stays free while it trains.")
        self._runpod_card = card
        tk.Label(card,
                 text="Fizgig ships as a ready-made image: the full app in your browser, your "
                      "models and datasets on persistent storage, and drag-and-drop file transfer. "
                      "Rent a big card for an afternoon instead of buying one.",
                 font=(FONT_FAMILY, 10), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 10))

        if self.RUNPOD_TEMPLATE_LIVE:
            row = tk.Frame(card, bg=COLORS["bg_surface"])
            row.pack(anchor=tk.W)
            _btn = tk.Button(row, text="  Deploy on RunPod  ",
                             font=(FONT_FAMILY, 10, "bold"), fg="#FFFFFF", bg="#673AB7",
                             activebackground="#5E35B1", activeforeground="#FFFFFF",
                             relief="flat", bd=0, cursor="hand2", padx=16, pady=6,
                             command=lambda: self._open_url(self._runpod_deploy_url()))
            _btn.pack(side=tk.LEFT)

            tk.Label(card,
                     text="Deploying through this link supports Fizgig's development, at no extra "
                          "cost to you.",
                     font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                     bg=COLORS["bg_surface"]).pack(anchor=tk.W, pady=(6, 0))
        else:
            # Shipped before the template is public. A button that goes nowhere is worse than no
            # button, and saying so plainly is better than hiding the section and surprising
            # people with it later.
            tk.Label(card, text="Coming soon",
                     font=(FONT_FAMILY, 11, "bold"), fg=COLORS["warning"],
                     bg=COLORS["bg_surface"]).pack(anchor=tk.W)
            tk.Label(card,
                     text="The one-click image is built and being tested. This section will get a "
                          "Deploy button once it is published — nothing to do in the meantime.",
                     font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                     bg=COLORS["bg_surface"], wraplength=760,
                     justify=tk.LEFT).pack(anchor=tk.W, pady=(2, 6))

        # The guide, in both states: while it says Coming soon this is the only way to read about
        # it, and once it is live it is where the storage and cost decisions are explained.
        _guide = tk.Label(card, text="Read the guide: running Fizgig on a rented GPU",
                          font=(FONT_FAMILY, 10, "underline"), fg=COLORS["accent_hover"],
                          bg=COLORS["bg_surface"], cursor="hand2")
        _guide.pack(anchor=tk.W, pady=(4, 0))
        _guide.bind("<Button-1>", lambda e: self._open_url(self.RUNPOD_GUIDE_URL))
        tk.Label(card,
                 text="When Fizgig is running on a pod, this section turns into its controls — "
                      "stop the pod automatically when training finishes, check storage, and see "
                      "where your files live.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).pack(anchor=tk.W, pady=(10, 0))

    def _pod_stop_key(self) -> str:
        """The key to stop this pod with: the one saved in Preferences, else a template env var.

        Prefs first because that is the route that scales — a public template cannot carry anyone's
        key, since template variables are handed to every container deployed from it."""
        try:
            k = self.prefs_vars["runpod_api_key"].get().strip()
        except Exception:
            k = ""
        return k or _pod_stop_key_env()

    def _refresh_pod_key_status(self):
        lbl = getattr(self, "_pod_key_status", None)
        if lbl is None or not lbl.winfo_exists():
            return
        if self._pod_stop_key():
            lbl.config(text="auto-stop ready", fg=COLORS["success"])
        else:
            lbl.config(text="needed for auto-stop", fg=COLORS["warning"])

    def _refresh_pod_storage(self):
        """Free space on the volume, refreshed while the tab is open."""
        lbl = getattr(self, "_pod_storage_lbl", None)
        if lbl is None or not lbl.winfo_exists():
            return
        try:
            import shutil as _sh
            usage = _sh.disk_usage("/workspace")
            free_gb, total_gb = usage.free / 1024 ** 3, usage.total / 1024 ** 3
            colour = COLORS["text_explain"]
            if total_gb > 10000:
                # A network volume does not expose its quota to the container — the kernel reports
                # the host's whole backing pool, so this read said "431035 GB free of 1430281 GB"
                # for a 100 GB volume. Printing that verbatim looks broken and, worse, implies
                # there is room when your quota might be nearly full. Say what we actually know.
                txt = ("Storage: /workspace is a network volume. Its size is set in RunPod and "
                       "isn't visible from in here — check usage in the RunPod dashboard.")
            elif total_gb < 60:
                # This small means the volume never mounted and /workspace is container disk,
                # which RunPod wipes when the pod stops — a 32 GB model download would evaporate.
                txt = (f"Storage: only {free_gb:.0f} GB free of {total_gb:.0f} GB — that looks like "
                       f"container disk, not your volume. Check the template's volume mount path "
                       f"is /workspace, or your files will vanish when the pod stops.")
                colour = COLORS["warning"]
            else:
                txt = f"Storage: {free_gb:.0f} GB free of {total_gb:.0f} GB on /workspace"
                if free_gb < 40:
                    colour = COLORS["warning"]
            lbl.config(text=txt, fg=colour)
        except Exception:
            lbl.config(text="Storage: unavailable")
        lbl.after(30000, self._refresh_pod_storage)

    def _open_url(self, url):
        """Open a link. Central so the pod image's browser handling stays in one place."""
        webbrowser.open(url)

    def _add_fetch_models_row(self, frame, row, family, blurb, optional_label=None):
        """'Download them all for me' row at the foot of a model-paths card.

        The per-row Download links open a browser and leave you to save the file and paste the
        path back — five times over, ~32 GB, before Fizgig does anything at all. This does the
        same job unattended and writes the paths into prefs itself."""
        bar = tk.Frame(frame, bg=COLORS["bg_surface"])
        bar.grid(row=row, column=0, columnspan=3, sticky=tk.EW, pady=(14, 2))
        btn = ttk.Button(bar, text="⬇ Download models for me",
                         command=lambda f=family: self._start_fetch_models(f))
        btn.pack(side=tk.LEFT)
        setattr(self, f"_fetch_btn_{family}", btn)
        status = tk.Label(bar, text="", font=(FONT_FAMILY, 9),
                          fg=COLORS["text_secondary"], bg=COLORS["bg_surface"])
        status.pack(side=tk.LEFT, padx=(12, 0))
        setattr(self, f"_fetch_status_{family}", status)
        if optional_label:
            # Off by default: the only optional MiniMax weight is the 21 GB ref2va DiT, which a
            # first setup does not need and most runs never use. The point is that the fetcher
            # CAN get it - before this, its own Download link was the only route.
            var = tk.BooleanVar(value=False)
            setattr(self, f"_fetch_optional_{family}", var)
            ttk.Checkbutton(bar, text=optional_label, variable=var).pack(side=tk.LEFT, padx=(16, 0))
        tk.Label(frame, text=blurb, font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"],
                 bg=COLORS["bg_surface"], wraplength=760, justify=tk.LEFT
                 ).grid(row=row + 1, column=0, columnspan=3, sticky=tk.W, pady=(0, 2))
        return row + 2

    _HF_GATED_URLS = [
        ("Base DiT — accept the licence",
         "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-fp8"),
        ("Distilled DiT — accept the licence",
         "https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8"),
        ("VAE — accept the licence",
         "https://huggingface.co/black-forest-labs/FLUX.2-dev"),
        ("Then create a READ token here",
         "https://huggingface.co/settings/tokens"),
    ]

    def _ask_hf_token(self):
        """Modal token prompt. Returns the token, or '' if cancelled.

        Not simpledialog.askstring: its prompt is a Label, so the four URLs you have to visit
        would be un-selectable — readable but not copyable, which is useless when the whole
        point is to go to them. Each URL here is a readonly Entry (selectable, Ctrl+C works)
        with Copy and Open buttons."""
        result = {"token": ""}
        dlg = tk.Toplevel(self.master)
        dlg.title("HuggingFace token — Klein downloads")
        dlg.configure(bg=BG_COLOR)
        dlg.transient(self.master)
        dlg.resizable(True, False)

        # Buttons packed BOTTOM first so they can never be pushed off the edge (the v2.8.5
        # caption-dialog fix — same reasoning applies to any dialog with variable content).
        btn_row = ttk.Frame(dlg)
        btn_row.pack(side=tk.BOTTOM, pady=(6, 12))

        tk.Label(dlg, text="Black Forest Labs gate the Klein downloads",
                 font=(FONT_FAMILY, 11, "bold"), fg=COLORS["text_primary"], bg=BG_COLOR
                 ).pack(anchor=tk.W, padx=14, pady=(14, 2))
        tk.Label(dlg,
                 text="They require every user to accept the licence themselves — which is also why "
                      "Fizgig can't download these for you without a token of your own. Sign in to "
                      "HuggingFace, accept on all three pages, then paste a read token below. "
                      "Krea 2 needs none of this; those files aren't gated.",
                 font=(FONT_FAMILY, 10), fg=COLORS["text_explain"], bg=BG_COLOR,
                 wraplength=620, justify=tk.LEFT).pack(anchor=tk.W, padx=14, pady=(0, 10))

        rows = tk.Frame(dlg, bg=BG_COLOR)
        rows.pack(fill=tk.X, padx=14)
        rows.columnconfigure(0, weight=1)

        def copy(url, btn):
            self.master.clipboard_clear()
            self.master.clipboard_append(url)
            btn.config(text="copied")
            btn.after(1200, lambda: btn.config(text="Copy"))

        for i, (label, url) in enumerate(self._HF_GATED_URLS):
            tk.Label(rows, text=label, font=(FONT_FAMILY, 9), fg=COLORS["text_secondary"],
                     bg=BG_COLOR).grid(row=i * 2, column=0, columnspan=3, sticky=tk.W,
                                       pady=(6 if i else 0, 0))
            e = ttk.Entry(rows, width=62)
            e.insert(0, url)
            e.config(state="readonly")           # selectable and copyable; not editable
            e.grid(row=i * 2 + 1, column=0, sticky=tk.EW)
            cb = ttk.Button(rows, text="Copy", width=8)
            cb.config(command=lambda u=url, b=cb: copy(u, b))
            cb.grid(row=i * 2 + 1, column=1, padx=(6, 0))
            ttk.Button(rows, text="Open", width=8,
                       command=lambda u=url: webbrowser.open(u)
                       ).grid(row=i * 2 + 1, column=2, padx=(6, 0))

        ttk.Button(dlg, text="Open all three licence pages",
                   command=lambda: [webbrowser.open(u) for _, u in self._HF_GATED_URLS[:3]]
                   ).pack(anchor=tk.W, padx=14, pady=(12, 0))

        tk.Label(dlg, text="Paste your token:", font=(FONT_FAMILY, 10),
                 fg=COLORS["text_secondary"], bg=BG_COLOR).pack(anchor=tk.W, padx=14, pady=(14, 2))
        tok = ttk.Entry(dlg, width=62, show="*")
        tok.pack(anchor=tk.W, padx=14, fill=tk.X)
        tk.Label(dlg, text="Starts with hf_… Stored only for this download — never written to disk.",
                 font=(FONT_FAMILY, 8, "italic"), fg=COLORS["text_muted"], bg=BG_COLOR
                 ).pack(anchor=tk.W, padx=14, pady=(2, 0))

        def ok():
            result["token"] = tok.get().strip()
            dlg.destroy()

        ttk.Button(btn_row, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="Download", command=ok).pack(side=tk.LEFT, padx=5)
        tok.bind("<Return>", lambda e: ok())

        dlg.update_idletasks()
        tok.focus_set()
        dlg.grab_set()
        self.master.wait_window(dlg)
        return result["token"]

    def _start_fetch_models(self, family):
        """Run the fetcher in a worker thread, streaming progress into the status label."""
        if getattr(self, "_fetch_running", False):
            messagebox.showinfo("Already downloading", "A model download is already running.")
            return
        token = ""
        if family == "klein":
            # Klein's repos are gated: BFL require each user to accept the licence themselves,
            # which is exactly why these can't be bundled or pre-fetched on anyone's behalf.
            # An HF_TOKEN already in the environment (the container's documented env var for
            # exactly this) satisfies the gate with no prompt — only ask when there isn't one.
            token = os.environ.get("HF_TOKEN", "").strip()
            if not token:
                token = self._ask_hf_token()
                if not token:
                    return

        btn = getattr(self, f"_fetch_btn_{family}", None)
        status = getattr(self, f"_fetch_status_{family}", None)
        if btn:
            btn.config(state="disabled")
        self._fetch_running = True

        def worker():
            import subprocess
            # Helper models come first and with EVERY family: they're ~1.9 GB against tens of
            # GB of weights, and whichever button you press you'll hit Florence / the face model
            # / the translator / Gizmo's Whisper sooner or later. Fetching them up front means
            # the Captions tab, Look Filter and Transcribe work immediately (and offline), and
            # they survive abandoning the big download.
            # Re-running is cheap — everything here is a no-op once present.
            cmd = [sys.executable, "-m", "fizgig.scripts.fetch_models", "--progress",
                   "--family", "tools", "--family", family]
            _opt = getattr(self, f"_fetch_optional_{family}", None)
            if _opt is not None and _opt.get():
                cmd.append("--include-optional")
            env = self._cuda_env_for_subprocess(dict(os.environ))
            env["PYTHONPATH"] = os.path.join(_FIZGIG_DIR, "src")
            env["PYTHONUNBUFFERED"] = "1"
            if token.strip():
                env["HF_TOKEN"] = token.strip()
            ok = False
            try:
                p = subprocess.Popen(cmd, cwd=_FIZGIG_DIR, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, env=env,
                                     encoding="utf-8", errors="replace",
                                     creationflags=(subprocess.CREATE_NO_WINDOW
                                                    if os.name == "nt" else 0))
                self._fetch_proc = p
                for line in p.stdout:
                    line = line.rstrip()
                    if line:
                        self.master.after(0, lambda l=line: self._on_fetch_progress(family, l))
                ok = (p.wait() == 0)
            except Exception as e:
                self.master.after(0, lambda e=e: self._on_fetch_progress(
                    family, f"failed: {type(e).__name__}: {e}"))
            self.master.after(0, lambda: self._on_fetch_done(family, ok))

        self._open_fetch_progress_window(family)
        threading.Thread(target=worker, daemon=True).start()
        if status:
            status.config(text="downloading…", fg=COLORS["text_secondary"])

    def _open_fetch_progress_window(self, family):
        """Unmissable progress for a multi-GB download.

        A status label beside the button was the only feedback, and hf_hub_download's own
        progress never reached it (tqdm redraws with carriage returns, so a line-reader sees
        nothing until the transfer ends). The result was an app that looked frozen for ten
        minutes and then popped up 'done'."""
        win = tk.Toplevel(self.master)
        self._fetch_win = win
        win.title("Downloading models")
        win.configure(bg=BG_COLOR)
        win.transient(self.master)
        win.protocol("WM_DELETE_WINDOW", lambda: None)   # closing must go through Cancel
        win.resizable(False, False)

        tk.Label(win, text="Downloading models", font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=BG_COLOR).pack(anchor=tk.W, padx=18, pady=(16, 2))
        self._fetch_file_lbl = tk.Label(win, text="starting…", font=(FONT_FAMILY, 10),
                                        fg=COLORS["text_secondary"], bg=BG_COLOR,
                                        wraplength=460, justify=tk.LEFT, anchor="w")
        self._fetch_file_lbl.pack(anchor=tk.W, padx=18, pady=(0, 8), fill=tk.X)

        self._fetch_bar = ttk.Progressbar(win, length=460, mode="indeterminate")
        self._fetch_bar.pack(padx=18)
        self._fetch_bar.start(12)
        self._fetch_bar_mode = "indeterminate"

        self._fetch_bytes_lbl = tk.Label(win, text="", font=(FONT_FAMILY, 9),
                                         fg=COLORS["text_muted"], bg=BG_COLOR, anchor="w")
        self._fetch_bytes_lbl.pack(anchor=tk.W, padx=18, pady=(6, 0), fill=tk.X)
        tk.Label(win, text="You can leave this running and keep using other tabs. Interrupted "
                           "downloads resume where they left off.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=BG_COLOR,
                 wraplength=460, justify=tk.LEFT).pack(anchor=tk.W, padx=18, pady=(8, 0))

        row = ttk.Frame(win)
        row.pack(pady=(12, 14))
        ttk.Button(row, text="Cancel", command=self._cancel_fetch_models).pack()
        win.update_idletasks()

    def _cancel_fetch_models(self):
        """Stop the download. Safe: partial files are resumable, prefs are only written on success."""
        p = getattr(self, "_fetch_proc", None)
        if p is not None and p.poll() is None:
            try:
                p.terminate()
            except Exception:
                pass
        if getattr(self, "_fetch_file_lbl", None) is not None:
            try:
                self._fetch_file_lbl.config(text="cancelling…")
            except Exception:
                pass

    @staticmethod
    def _fmt_gb(n):
        return f"{n / 1024 ** 3:.2f} GB"

    def _on_fetch_progress(self, family, line):
        stripped = line.strip()
        if stripped.startswith("[progress]"):
            try:
                _tag, done, total, name = stripped.split(None, 3)
                done, total = int(done), int(total)
            except (ValueError, IndexError):
                return
            bar = getattr(self, "_fetch_bar", None)
            if bar is not None and bar.winfo_exists():
                if total > 0:
                    if self._fetch_bar_mode != "determinate":
                        bar.stop()
                        bar.config(mode="determinate", maximum=100)
                        self._fetch_bar_mode = "determinate"
                    bar["value"] = max(0, min(100, done * 100 / total))
                self._fetch_file_lbl.config(text=name)
                self._fetch_bytes_lbl.config(
                    text=f"{self._fmt_gb(done)} of {self._fmt_gb(total)}"
                         f"   ({done * 100 // total if total else 0}%)")
            return

        # Non-progress lines: the [get]/[ok]/[keep]/[gated] narration.
        status = getattr(self, f"_fetch_status_{family}", None)
        if status:
            status.config(text=stripped[:110])
        lbl = getattr(self, "_fetch_file_lbl", None)
        if lbl is not None and lbl.winfo_exists() and stripped:
            lbl.config(text=stripped)
            bar = getattr(self, "_fetch_bar", None)
            # A new item that isn't a download (a cache warm, a skip) — back to the pulsing bar
            # rather than leaving the previous file's percentage sitting there looking stalled.
            if bar is not None and bar.winfo_exists() and self._fetch_bar_mode == "determinate" \
                    and not stripped.startswith("[progress]"):
                bar.config(mode="indeterminate")
                bar.start(12)
                self._fetch_bar_mode = "indeterminate"
                self._fetch_bytes_lbl.config(text="")
        self.update_console(f"[models] {line}\n")

    def _on_fetch_done(self, family, ok):
        self._fetch_running = False
        self._fetch_proc = None
        win = getattr(self, "_fetch_win", None)
        if win is not None:
            try:
                win.destroy()
            except Exception:
                pass
            self._fetch_win = None
        btn = getattr(self, f"_fetch_btn_{family}", None)
        if btn:
            btn.config(state="normal")
        # The fetcher wrote prefs.json from another process, so re-read it — otherwise the
        # entry boxes keep showing the paths from before the download and saving from this
        # window would write the stale ones straight back over the new ones.
        try:
            fresh = load_prefs()
            for key, var in self.prefs_vars.items():
                if key in fresh and str(fresh[key]).strip():
                    var.set(fresh[key])
            self.prefs.update(fresh)
        except Exception:
            pass
        status = getattr(self, f"_fetch_status_{family}", None)
        if status:
            status.config(text="done — paths filled in" if ok else "finished with items missing (see console)",
                          fg=COLORS["success"] if ok else COLORS["warning"])
        if ok:
            messagebox.showinfo("Models ready",
                                "Downloaded and wired into Preferences.\n\n"
                                "The paths above are filled in — nothing else to set up.")

    def _add_pref_row(self, frame, row, label, pref_key, hint, is_dir=False, download_url=None, download_note=None,
                      download_label="Download", download_url2=None, download_label2="Download", download_note2=None):
        """Add a labeled pref entry with Browse button and hint text. Returns next row index.

        If download_url is set, a download link (text=download_label) is added next to Browse that opens
        the URL in the user's default browser. A second link (download_url2 / download_label2) can be
        added too — used to offer e.g. both fp8 and bf16 variants. download_note / download_note2 are
        appended to the hint line.
        """
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 10), pady=4)
        ttk.Entry(frame, textvariable=self.prefs_vars[pref_key], width=60).grid(row=row, column=1, sticky=tk.EW, pady=4)

        btn_frame = tk.Frame(frame, bg=COLORS["bg_surface"])
        btn_frame.grid(row=row, column=2, sticky=tk.W, padx=(8, 0))
        browse_cmd = (lambda: self._browse_pref_dir(pref_key)) if is_dir else (lambda: self._browse_pref_file(pref_key))
        ttk.Button(btn_frame, text="Browse", command=browse_cmd).pack(side=tk.LEFT)
        for _url, _lbl in ((download_url, download_label), (download_url2, download_label2)):
            if not _url:
                continue
            dl_link = tk.Label(btn_frame, text=_lbl,
                               fg=COLORS["accent_hover"], cursor="hand2",
                               font=(FONT_FAMILY, 9, "underline"),
                               bg=COLORS["bg_surface"])
            dl_link.pack(side=tk.LEFT, padx=(8, 0))
            dl_link.bind("<Button-1>", lambda e, url=_url: webbrowser.open(url))
            ToolTip(dl_link, f"Open download page in browser:\n{_url}")
        row += 1
        hint_text = hint
        _notes = [n for n in (download_note, download_note2) if n]
        if _notes:
            hint_text = hint + "  ·  " + "  ·  ".join(_notes)
        tk.Label(frame, text=hint_text,
                 font=(FONT_FAMILY, 10, "italic"),
                 fg=COLORS["text_explain"], bg=COLORS["bg_surface"],
                 wraplength=760, justify=tk.LEFT).grid(
            row=row, column=1, columnspan=2, sticky=tk.W, pady=(0, 6)
        )
        row += 1
        return row

    def _browse_pref_file(self, pref_key):
        filepath = filedialog.askopenfilename(
            title="Select file",
            filetypes=[("SafeTensors", "*.safetensors"), ("All files", "*.*")]
        )
        if filepath:
            self.prefs_vars[pref_key].set(filepath)

    def _browse_pref_dir(self, pref_key):
        dirpath = filedialog.askdirectory(title="Select directory")
        if dirpath:
            self.prefs_vars[pref_key].set(dirpath)

    def _reset_prefs(self):
        if messagebox.askyesno("Reset Preferences", "Restore all paths to defaults?"):
            for key, default in DEFAULT_PREFS.items():
                if key in self.prefs_vars:
                    self.prefs_vars[key].set(default)

    def _open_prefs_file(self):
        if os.path.exists(PREFS_FILE):
            self._open_in_file_manager(PREFS_FILE)
        else:
            messagebox.showinfo("Info", "prefs.json doesn't exist yet — change a path to create it.")
