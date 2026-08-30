import os
import re
import signal
import subprocess
import sys
import threading

import tkinter as tk
from tkinter import messagebox, ttk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY, BG_COLOR
from fizgig_gui.core.domain.architectures import ARCHITECTURES
from fizgig_gui.core.domain.minimax_math import minimax_block_spec, minimax_train_base, minimax_base_quant, \
    minimax_lownoise_to_shift, minimax_highnoise_lr, MINIMAX_LIKENESS_BLOCKS, MINIMAX_AUDIO_BLOCKS
from fizgig_gui.core.config.prefs import _running_on_pod, _pod_id
from fizgig_gui.core.config.presets import ft_checkpoint_continuation

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT, FIZGIG_DIR as _FIZGIG_DIR

class TrainingEngineMixin:
    def run_subprocess(self, cmd, name, callback=None):
        """Run a subprocess and handle its output with UTF-8 encoding"""
        env = self._cuda_env_for_subprocess(os.environ.copy())
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"  # flush stdout/stderr line-by-line so log output streams live

        if os.name == 'nt':
            # BELOW_NORMAL: Windows weights GPU scheduling by process priority class, and the
            # desktop compositor renders on the same card that training saturates. Below-normal
            # gives DWM the preemption slices it needs (fixes juddery mouse/desktop during a run)
            # and costs training ~1% — it only yields when something else actually wants time.
            creationflags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
                             | subprocess.BELOW_NORMAL_PRIORITY_CLASS)
            preexec_fn = None
        else:
            creationflags = 0
            preexec_fn = os.setsid

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',
            env=env,
            creationflags=creationflags,
            preexec_fn=preexec_fn
        )
        self.current_process = process
        if os.name == 'nt':
            self.process_group_id = process.pid

        def read_output(pipe, output_type):
            """Read subprocess output line by line"""
            while True:
                line = pipe.readline()
                if not line:
                    break
                self.master.after(0, self.update_console, line)
            pipe.close()

        threading.Thread(target=read_output, args=(process.stdout, "STDOUT"), daemon=True).start()
        threading.Thread(target=read_output, args=(process.stderr, "STDERR"), daemon=True).start()

        def check_process():
            """Check subprocess completion"""
            process.wait()
            self.master.after(0, self.update_console, f"{name} process completed.\n")
            self.current_process = None
            # Route pipeline exits through the pause/resume state machine (see the predicate
            # for which ones — a dead caching phase used to strand the app in "running").
            if self._pipeline_exit_routes_to_state_machine(name, process.returncode):
                self.master.after(0, self._on_training_subprocess_exited, process.returncode)
            if process.returncode != 0:
                self.master.after(0, self.update_console,
                    f"ERROR: {name} failed with exit code {process.returncode}. Pipeline stopped.\n")
                self.master.after(0, self.stop_samples_watcher)
                return
            if callback:
                # Marshal to the Tk main thread: pipeline-chain callbacks touch Tk widgets
                # (update_console etc.), and Tk calls off the main thread segfault on Linux.
                self.master.after(0, callback)

        threading.Thread(target=check_process, daemon=True).start()

    def start_training(self):
        """Start training with sequential cache process execution"""
        # Re-entrancy guard: the Start button stays enabled during a run, so a double-click
        # (or Start during caching) overwrote current_process and ORPHANED the first launch —
        # stop_training only ever kills the current one, and both runs wrote the same
        # checkpoints while the second's fresh-run wipe deleted the first's watch files.
        _proc = getattr(self, "current_process", None)
        try:
            if _proc is not None and _proc.poll() is None:
                # A run is active, so Start means QUEUE: capture the currently configured
                # run and append it. (The button already reads "Queue Train" in this state.)
                self._queue_current_run()
                return
        except Exception:
            pass

        # Validate inputs before starting
        if not self.validate_inputs():
            return

        # Resuming a state that's already at the final epoch trains nothing — the trainer's epoch
        # loop is empty and it just rewrites the final LoRA. A warning rather than a block,
        # because that fall-through is exactly how a run paused ON its last epoch gets completed.
        if not self._confirm_resume_has_epochs_left():
            return

        if not self._confirm_disk_headroom():
            return

        # Clear a stale pause sentinel from a previous session (window close / crash after
        # Pause left it on disk; the trainer would read it at epoch 1 and exit "cleanly").
        try:
            _stale_flag = self._pause_flag_path()
            if os.path.exists(_stale_flag) and getattr(self, "training_state", "idle") != "paused":
                os.remove(_stale_flag)
                self.update_console("[pause] removed stale .pause_requested from a previous session\n")
        except Exception:
            pass

        # Reset OOM warning flag for this run
        self._oom_warning_shown = False
        # Reset the VRAM/RAM peak markers so the status bar tracks THIS run.
        try:
            self.reset_status_peaks()
        except Exception:
            pass
        # Sync the sample-override sentinel to the current toggle (clears any
        # stale file from a previous session so it matches what the user sees).
        try:
            self._on_sample_override_changed()
        except Exception:
            pass

        # Auto-uncheck FP8 Base if the Base DiT file is already fp8-quantised (Klein only —
        # Krea 2 reads its own RAW DiT and dynamic-quantizes it, so this must not fire there).
        _is_krea2_run = ARCHITECTURES.get(self.architecture_var.get(), {}).get("is_krea2", False)
        base_dit_path = self.prefs_vars.get("base_dit", tk.StringVar()).get()
        if not _is_krea2_run and "fp8" in os.path.basename(base_dit_path).lower() and self.fp8_var.get():
            self.fp8_var.set(False)
            self.scaled_var.set(False)
            self.toggle_scaled()

        # Snapshot current settings for the "Load Last Train" button
        self._save_last_train_settings()
        # ...and as the queue window's pinned "training now" card: editing a queued job loads
        # its settings into this tab, so the window needs a way back to the run in progress.
        self._active_run_item = self._queue_snapshot()

        if getattr(self, "_training_start_pending", False):
            return
        if self._caption_worker_alive():
            self._training_start_pending = True
            self._caption_worker_released_for_training = True
            try:
                self._start_training_btn.configure(state=tk.DISABLED)
            except Exception:
                pass
            self._stop_caption_worker_async(self._start_training_launch, graceful=False)
            return

        self._start_training_launch()

    def _start_training_launch(self):
        """Launch training after validations and any caption-worker VRAM release."""
        self._training_start_pending = False
        try:
            self._start_training_btn.configure(state=tk.NORMAL)
        except Exception:
            pass
        # ...and the tool-tab engines (Repair Studio / Explorer / Royale, 10-20 GB each).
        # A manual Start implies a switch to the Training tab, which unloads them via
        # on_tab_changed — but a queue auto-advance or the queue window's "Start next now"
        # involves NO tab switch, and training would otherwise launch against a full card.
        # All three are idle-guarded internally, so this is safe and idempotent.
        for _unl in ("_unload_repair_studio_models", "_unload_explorer_models", "_royale_unload"):
            try:
                getattr(self, _unl)()
            except Exception:
                pass

        # Start samples watcher for live gallery updates
        if self.sample_enabled_var.get():
            self.start_samples_watcher()

        # Clear cache directory before training — but NOT when resuming: the cache is already
        # built and we skip re-caching, so wiping it would leave the resumed run with no latents
        # /text. Read the resume path from the entry (the live source of truth at this point).
        _resume_entry = self.entries.get("RESUME_TRAINING")
        # An armed FT continuation is a resume too — same cache, same frozen dataset.
        _is_resuming_clear = bool((_resume_entry and _resume_entry.get().strip())
                                  or self._ft_resume_active())
        cache_dir = self.dataset_cache_dir_var.get().strip()
        if cache_dir and os.path.isdir(cache_dir) and not _is_resuming_clear:
            try:
                import shutil
                for item in os.listdir(cache_dir):
                    item_path = os.path.join(cache_dir, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
            except Exception as e:
                self.update_console(f"Warning: Could not clear cache: {e}\n")


        # Get current architecture
        arch = self.architecture_var.get()
        config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])

        # Validate blocks swap
        try:
            is_auto = self.entries["BLOCKS_SWAP"].get().strip().lower().startswith("auto")
            if config.get("is_minimax") and is_auto:
                # MiniMax resolves "auto" in the TRAINER from real free VRAM at run time (correct
                # for queued runs too) — the Klein/Krea2 tier tables here don't fit its NF4 base.
                blocks_swap = "auto"
                self.update_console("Block Swap: Auto — the trainer plans swap + checkpointing "
                                    "from free VRAM at launch\n")
            else:
                blocks_swap = self._parse_blocks_swap()
                if is_auto:
                    self.update_console(f"Block Swap: Auto detected → {blocks_swap} (based on GPU VRAM)\n")
                if blocks_swap > config["blocks_swap_max"]:
                    messagebox.showwarning(
                        "Warning",
                        f"Blocks Swap value ({blocks_swap}) exceeds maximum for {arch} ({config['blocks_swap_max']}). Using maximum value."
                    )
                    blocks_swap = config["blocks_swap_max"]
                    self.entries["BLOCKS_SWAP"].delete(0, tk.END)
                    self.entries["BLOCKS_SWAP"].insert(0, str(blocks_swap))
        except ValueError:
            blocks_swap = config["blocks_swap_max"]

        # Update settings from entries
        # Path keys read via _get_path() (sourced from prefs_vars or hidden _dataset_config_var)
        # since the Model Paths section is no longer visible on the Training tab.
        # With Adaptive LR on the Learning Rate box is ignored (the trainer starts at the
        # geometric midpoint of Min/Max) — a stale value in the disabled box must not crash
        # collection, so fall back to a harmless placeholder the trainer will override.
        try:
            _lr_val = float(self.entries["LEARNING_RATE"].get())
        except ValueError:
            if hasattr(self, 'adaptive_lr_var') and self.adaptive_lr_var.get():
                _lr_val = 1e-4  # ignored by the trainer under adaptive
            else:
                raise
        # Captured BEFORE the update below overwrites DATASET_CONFIG with the live editor
        # path: on a resume this still holds the paused run's frozen snapshot (in-session
        # from the original launch, cross-restart from the startup sidecar restore), and
        # the freeze call needs it — reading settings AFTER the update made the resume
        # keep-rule dead code (three independent review agents, same finding).
        _prev_dataset_config = str(self.settings.get("DATASET_CONFIG", "") or "")
        self.settings.update({
            "ARCHITECTURE": arch,
            "MODEL_TYPE": self.entries["MODEL_TYPE"].get() if config["uses_model_type"] else "",
            "LEARNING_RATE": _lr_val,
            "LORA_LR_RATIO": int(self.entries["LORA_LR_RATIO"].get()),
            "NETWORK_DIM": int(self.entries["NETWORK_DIM"].get()),
            "NETWORK_ALPHA": float(self.entries["NETWORK_ALPHA"].get()),
            "NETWORK_TYPE": self.entries["NETWORK_TYPE"].get(),
            "LOKR_FACTOR": int(self.entries["LOKR_FACTOR"].get() or 8),
            "MAX_TRAIN_EPOCHS": int(self.entries["MAX_TRAIN_EPOCHS"].get()),
            "SAVE_EVERY_N_EPOCHS": int(self.entries["SAVE_EVERY_N_EPOCHS"].get()),
            "SEED": int(self.entries["SEED"].get()),
            "BLOCKS_SWAP": blocks_swap,
            # MiniMax-only; the widget exists (hidden) under every family, so read it unconditionally
            # and let the MiniMax command builder be the one that acts on it.
            # Stored as the percentage the user typed; the command builder converts it to the
            # trainer's shift. Keeping the percentage is what makes a saved preset mean the same
            # thing later, rather than a shift number nobody can interpret.
            "MINIMAX_LOWNOISE_PCT": str(self.entries["MINIMAX_LOWNOISE_PCT"].get() or "").strip(),
            "MINIMAX_HIGHNOISE_LR_PCT": str(
                self.entries["MINIMAX_HIGHNOISE_LR_PCT"].get() or "").strip(),
            # The hand-curated dict trap (5f20ba2): a control missing HERE silently never
            # reaches the trainer, however correct the widgets and the command builder are.
            "MIXED_STOP_CATEGORY": str(
                self.entries["MIXED_STOP_CATEGORY"].get() or "").strip(),
            "MIXED_STOP_EPOCH": str(
                self.entries["MIXED_STOP_EPOCH"].get() or "").strip(),
            "MIXED_STOP_MODE": str(
                self.entries["MIXED_STOP_MODE"].get() or "").strip(),
            # Likeness mode owns the block choice: the launch dict says "all" so the queue card,
            # snapshot and builder stay honest, while the combobox keeps the user's typed spec
            # for when they untick.
            "MINIMAX_BLOCKS": ("all" if self.entries["MINIMAX_LIKENESS_OPT"].get()
                               else minimax_block_spec(self.entries["MINIMAX_BLOCKS"].get())),
            "MINIMAX_LIKENESS_OPT": bool(self.entries["MINIMAX_LIKENESS_OPT"].get()),
            "MINIMAX_FT_CLIP_LIKENESS": bool(self.entries["MINIMAX_FT_CLIP_LIKENESS"].get())
            if "MINIMAX_FT_CLIP_LIKENESS" in self.entries else True,
            "MINIMAX_TRAIN_ADALN": bool(self.entries["MINIMAX_TRAIN_ADALN"].get()),
            "MINIMAX_DISTILL": bool(self.minimax_distill_var.get()),
            # Canonical key ("fl2va"/"ref2va"), never the display label. Preset-immune by
            # design — the var is outside self.entries and _collect_preset_values skips it.
            "MINIMAX_TRAIN_BASE": minimax_train_base(
                getattr(self, "minimax_train_base_var", None)
                and self.minimax_train_base_var.get()),
            "MINIMAX_MULTICONCEPT": bool(self.minimax_multiconcept_var.get()),
            "MINIMAX_CONCEPT_DIRS": [v.get().strip() for v in
                                     getattr(self, "_concept_folder_vars", [])],
            "MINIMAX_BASE_QUANT": self.entries["MINIMAX_BASE_QUANT"].get(),
            "MINIMAX_BLOCK_LIMIT": self.entries["MINIMAX_BLOCK_LIMIT"].get(),
            "MINIMAX_LR_WARMUP": self.entries["MINIMAX_LR_WARMUP"].get(),
            "MINIMAX_EMA": self.entries["MINIMAX_EMA"].get(),
            "MINIMAX_ADAPTER_RAMP": self.entries["MINIMAX_ADAPTER_RAMP"].get(),
            "MINIMAX_CAPTION_DROPOUT": self.entries["MINIMAX_CAPTION_DROPOUT"].get(),
            "MINIMAX_DISTILL_WEIGHT": str(self.entries["MINIMAX_DISTILL_WEIGHT"].get() or "0.8").strip(),
            "MINIMAX_DISTILL_REFS": str(self.entries["MINIMAX_DISTILL_REFS"].get() or "2").strip(),
            "MINIMAX_SLOW_BLOCKS": str(self.entries["MINIMAX_SLOW_BLOCKS"].get() or "").strip(),
            "MINIMAX_SLOW_LR_SCALE": str(self.entries["MINIMAX_SLOW_LR_SCALE"].get() or "1").strip(),
            "DATASET_CONFIG": self._get_path("DATASET_CONFIG"),
            "VAE_MODEL": self._get_path("VAE_MODEL"),
            "CLIP_MODEL": self._get_path("CLIP_MODEL"),
            "T5_MODEL": self._get_path("T5_MODEL"),
            "TEXT_ENCODER": self._get_path("TEXT_ENCODER"),
            "DIT_MODEL": self._get_path("DIT_MODEL"),
            "LORA_OUTPUT_DIR": self.entries["LORA_OUTPUT_DIR"].get(),
            "LORA_NAME": self.entries["LORA_NAME"].get(),
            "RESUME_TRAINING": self.entries["RESUME_TRAINING"].get(),
            "OPTIMIZER_TYPE": self.entries["OPTIMIZER_TYPE"].get(),
            "OPTIMIZER_ARGS": self.entries["OPTIMIZER_ARGS"].get(),
            "ATTENTION_MECHANISM": self.entries["ATTENTION_MECHANISM"].get(),
            "LOGGING_DIR": self.entries["LOGGING_DIR"].get(),
            "LOG_WITH": self.entries["LOG_WITH"].get(),
            "LOG_PREFIX": self.entries["LOG_PREFIX"].get(),
            "IMG_IN_TXT_IN_OFFLOADING": self.entries["IMG_IN_TXT_IN_OFFLOADING"].get(),
            "LR_SCHEDULER": self.entries["LR_SCHEDULER"].get(),
            "LR_WARMUP_STEPS": self.entries["LR_WARMUP_STEPS"].get(),
            "LR_DECAY_STEPS": self.entries["LR_DECAY_STEPS"].get(),
            "GRADIENT_ACCUMULATION": self.entries["GRADIENT_ACCUMULATION"].get(),
            "MAX_GRAD_NORM": self.entries["MAX_GRAD_NORM"].get(),
            "NETWORK_DROPOUT": self.entries["NETWORK_DROPOUT"].get(),
            "CONTEXT_LORA_PATH": self.entries["CONTEXT_LORA_PATH"].get(),
            "CONTEXT_LORA_STRENGTH": self.entries["CONTEXT_LORA_STRENGTH"].get(),
            "TIMESTEP_SAMPLING": self.ts_sampling_var.get(),
            "DISCRETE_FLOW_SHIFT": self.entries["DISCRETE_FLOW_SHIFT"].get(),
            "SIGMOID_SCALE": self.entries["SIGMOID_SCALE"].get(),
            "MIN_TIMESTEP": self.entries["MIN_TIMESTEP"].get(),
            "MAX_TIMESTEP": self.entries["MAX_TIMESTEP"].get(),
            "PRESERVE_DISTRIBUTION": self.preserve_dist_var.get(),
            "ADAPTIVE_LR": self.adaptive_lr_var.get(),
            "ADAPTIVE_LR_MIN": self.entries["ADAPTIVE_LR_MIN"].get(),
            "ADAPTIVE_LR_MAX": self.entries["ADAPTIVE_LR_MAX"].get(),
            "WEIGHTING_SCHEME": self.weighting_scheme_var.get(),
            "LOGIT_MEAN": self.entries["LOGIT_MEAN"].get(),
            "LOGIT_STD": self.entries["LOGIT_STD"].get(),
            "MODE_SCALE": self.entries["MODE_SCALE"].get(),
            "METADATA_TITLE": self.entries["METADATA_TITLE"].get(),
            "METADATA_AUTHOR": self.entries["METADATA_AUTHOR"].get(),
            "METADATA_DESCRIPTION": self.entries["METADATA_DESCRIPTION"].get(),
            "METADATA_LICENSE": self.entries["METADATA_LICENSE"].get(),
            "METADATA_TAGS": self.entries["METADATA_TAGS"].get(),
            "METADATA_TRIGGER_PHRASE": self.entries["METADATA_TRIGGER_PHRASE"].get(),
            "METADATA_THUMBNAIL": self.entries["METADATA_THUMBNAIL"].get(),
            "FP8": self.fp8_var.get(),
            "SCALED": self.scaled_var.get(),
            "QUANT_4BIT": self.quant_4bit_var.get(),
            "COMPILE_BLOCKS": self.compile_blocks_var.get(),
            "GRADIENT_CHECKPOINTING": self.grad_checkpoint_var.get(),
            "FP8_TEXT_ENCODER": self.fp8_text_encoder_var.get(),
            "SAVE_STATE": self.save_state_var.get(),
            "SAVE_STATE_ON_TRAIN_END": self.save_state_on_train_end_var.get(),
            "KEEP_LAST_N_STATES": self.entries["KEEP_LAST_N_STATES"].get(),
            "ENABLE_BUCKET": self.dataset_enable_bucket_var.get(),
            "BUCKET_NO_UPSCALE": self.dataset_no_upscale_var.get(),
        })

        # Freeze THIS run's dataset config (#98): the pipeline's stages each read the
        # dataset TOML at their own start, and the Start tab auto-saves that TOML on
        # every edit — so changing the dataset folder while run 1 initialised (e.g. to
        # queue run 2) retargeted run 1: dataset 2 trained under run 1's name and
        # settings. From here on, settings["DATASET_CONFIG"] is the run's immutable
        # snapshot; the live TOML belongs to the editor alone.
        self.settings["DATASET_CONFIG"] = self._snapshot_dataset_config_for_run(
            self.settings.get("DATASET_CONFIG", ""), resuming=_is_resuming_clear,
            prev_config=_prev_dataset_config)

        # The frozen config must describe the folders on the Start tab (#98 follow-up):
        # a dataset-field parse failure (e.g. Target Megapixels typed as "1,0") makes the
        # auto-saver skip its rewrite SILENTLY, so the launch would freeze a STALE toml —
        # and the previous dataset would train under this run's name. Never on a resume:
        # there the frozen config deliberately predates the Start tab.
        if not _is_resuming_clear:
            _missing = self._verify_frozen_dataset_config(self.settings["DATASET_CONFIG"])
            if _missing:
                self.stop_samples_watcher()
                _msg = ("The dataset config on disk does not include the training "
                        "folder(s) shown on the Start tab:\n\n"
                        + "\n".join(_missing)
                        + "\n\nThis usually means a dataset field failed to parse — check "
                        "Target Megapixels and Batch Size for typos — so the config was "
                        "never rewritten. Fix the value and press Start again.")
                self.update_console(f"[dataset] launch refused — {_msg}\n")
                messagebox.showerror("Dataset config out of date", _msg)
                return

        # Build training command based on architecture
        command = self.build_training_command(config)
        cache_latents_cmd = self.build_cache_latents_command(config)
        cache_text_cmd = self.build_cache_text_command(config)

        self.console_output.configure(state="normal")
        self.console_output.delete(1.0, tk.END)
        self.console_output.configure(state="disabled")

        if getattr(self, "_caption_worker_released_for_training", False):
            self._caption_worker_released_for_training = False
            self.update_console("Caption model released.\n")

        def on_training_complete():
            """Called when training finishes - cleanup watchers"""
            self.stop_samples_watcher()

        # On resume, skip cache preparation entirely — the cache is already built from the
        # original launch. An armed FT continuation counts: it is the same run continuing.
        is_resuming = bool(self.settings.get("RESUME_TRAINING", "").strip()
                           or self._ft_resume_active())
        if self.enable_cache_var.get() and not is_resuming:
            self.update_console(f"Starting cache preparation for {arch}...\n")

            def on_text_encoder_caching_complete():
                self.update_console("Text encoder caching completed.\nStarting training...\n")
                self.run_subprocess(command, "Training", on_training_complete)

            def on_cache_preparation_complete():
                self.update_console("Cache preparation completed.\nStarting text encoder caching...\n")
                self.run_subprocess(cache_text_cmd, "Text Encoder Caching", on_text_encoder_caching_complete)

            self.run_subprocess(cache_latents_cmd, "Cache Preparation", on_cache_preparation_complete)
        else:
            if is_resuming:
                self.update_console("Resuming from saved state — skipping cache preparation (cache already built).\n")
            else:
                self.update_console(f"Starting {arch} training without caching...\n")
            self.run_subprocess(command, "Training", on_training_complete)
        # Mark as running for the pause/resume state machine
        self.training_state = "running"
        self._refresh_training_buttons()

    DISK_WARN_GB = 15

    def _confirm_disk_headroom(self):
        """True to proceed. Warns when the output drive is nearly full.

        A threshold plus the REAL figure rather than a predicted requirement: what a run actually
        writes depends on rank, epochs, save cadence and keep-N, and a confidently wrong estimate
        is worse than showing someone the number and letting them judge. Running out of disk four
        hours into a run costs the whole run."""
        out_dir = (self.settings.get("LORA_OUTPUT_DIR") or "").strip()
        if not out_dir:
            return True
        probe = out_dir
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                return True
            probe = parent
        try:
            import shutil as _sh
            free_gb = _sh.disk_usage(probe).free / 1024 ** 3
        except Exception:
            return True                      # never block a run over a failed disk probe
        if free_gb >= self.DISK_WARN_GB:
            return True
        return messagebox.askyesno(
            "Low disk space",
            f"Only {free_gb:.1f} GB free where your LoRAs are saved:\n{probe}\n\n"
            f"A run writes a checkpoint every few epochs, plus resumable state dirs (a few hundred "
            f"MB each) and sample images. Running out part-way through loses the run.\n\n"
            f"Free some space, or lower Save Every N Epochs and Keep Last.\n\n"
            f"Start training anyway?")

    def _confirm_resume_has_epochs_left(self):
        """True to proceed. Warns when the resume state is already at/past Max Train Epochs.

        Deliberately a warning, not a validation error: pausing on the final epoch exits before
        the final LoRA is written, and resuming that state — with zero epochs left to run — is
        precisely what completes it. Blocking would break that recovery."""
        resume_path = (self.entries["RESUME_TRAINING"].get() or "").strip()
        if not resume_path:
            return True
        m = re.search(r"-(\d{6})-state$", os.path.basename(resume_path.rstrip("/\\")))
        if not m:
            return True
        state_epoch = int(m.group(1))
        try:
            max_epochs = int(self.entries["MAX_TRAIN_EPOCHS"].get())
        except (TypeError, ValueError):
            return True
        if state_epoch < max_epochs:
            return True
        return messagebox.askyesno(
            "Nothing left to train",
            f"That state is already at epoch {state_epoch}, and Max Train Epochs is {max_epochs}.\n\n"
            f"Resuming it will not train anything — it will just write the final LoRA from the "
            f"restored state. That is what you want if you paused on the last epoch and are "
            f"finishing the run.\n\n"
            f"To train further, cancel and raise Max Train Epochs above {state_epoch} first.\n\n"
            f"Continue anyway?")

    def _state_flags(self):
        """Save-state CLI flags, shared by both families (the flag names are identical).

        Keep-N is clamped to >= 1 here as well as in the trainer: a blank or zero box must never
        reach a prune that would take the state just written with it."""
        flags = []
        if self.settings.get("SAVE_STATE", True):
            flags.append("--save_state")
        if self.settings.get("SAVE_STATE_ON_TRAIN_END", True):
            flags.append("--save_state_on_train_end")
        if flags:
            try:
                keep_n = max(1, int(str(self.settings.get("KEEP_LAST_N_STATES", 2)).strip()))
            except (TypeError, ValueError):
                keep_n = 2
            flags += ["--keep_last_n_states", str(keep_n)]
        return flags

    def build_training_command(self, config):
        """Build the training command based on architecture configuration"""
        # Stamp whether THIS launch is a rotation fine-tune (and which family), for the
        # pause exit-handler: an FT pause leaves a full checkpoint rather than a state dir,
        # and the Tk checkbox can be flipped mid-run, so the truth is recorded at launch.
        self._launched_ft_family = None
        if config.get("is_krea2"):
            if bool(getattr(self, "krea2_finetune_var", None) and self.krea2_finetune_var.get()):
                self._launched_ft_family = "krea2"
            return self._build_krea2_train_command()
        if config.get("is_minimax"):
            if bool(getattr(self, "minimax_finetune_var", None) and self.minimax_finetune_var.get()):
                self._launched_ft_family = "minimax"
            return self._build_minimax_train_command()
        arch = self.settings["ARCHITECTURE"]
        # Same reasoning as _venv_python: fall back to whatever is on PATH when the bundled venv
        # is not a sibling of the repo, rather than pointing at a file that is not there.
        accelerate_path = (os.path.join(_FIZGIG_DIR, "venv", "Scripts", "accelerate.exe")
                           if os.name == 'nt'
                           else os.path.join(_FIZGIG_DIR, "venv", "bin", "accelerate"))
        if not os.path.isfile(accelerate_path):
            import shutil as _shutil
            accelerate_path = _shutil.which("accelerate") or accelerate_path
        train_script_path = self._resolve_script(config, "train_script")

        # Auto-detect mixed precision from DiT model filename
        # fp16 model files require fp16 mixed precision, bf16 requires bf16
        dit_path = self.settings["DIT_MODEL"]
        dit_filename = os.path.basename(dit_path).lower()
        if "fp16" in dit_filename:
            mixed_precision = "fp16"
        else:
            mixed_precision = "bf16"

        command = [
            accelerate_path, "launch",
            "--num_cpu_threads_per_process", "2",
            "--mixed_precision", mixed_precision,
            train_script_path,
        ]

        # Architecture-specific parameters
        if arch.startswith("Wan"):
            command.extend(["--task", self.settings["MODEL_TYPE"]])
        elif config["uses_model_version"]:
            command.extend(["--model_version", config["model_version"]])

        command.extend([
            "--dit", self.settings["DIT_MODEL"],
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--mixed_precision", mixed_precision,
        ])

        # VAE parameter (same flag for all architectures)
        command.extend(["--vae", self.settings["VAE_MODEL"]])

        # Text encoder parameters based on architecture
        if config["uses_text_encoder"]:
            command.extend(["--text_encoder", self.settings["TEXT_ENCODER"]])

        # Base weight optimization — 4-bit NF4 supersedes fp8 (mutually exclusive).
        if self.settings.get("QUANT_4BIT", False):
            command.append("--quant_4bit")
        elif self.settings["FP8"]:
            command.append("--fp8_base")
            if self.settings["SCALED"]:
                command.append("--fp8_scaled")

        # FP8 text encoder
        if self.settings["FP8_TEXT_ENCODER"] and config["fp8_text_encoder_flag"]:
            command.append(config["fp8_text_encoder_flag"])

        command.extend([
            "--blocks_to_swap", str(self.settings["BLOCKS_SWAP"]),
            "--optimizer_type", self.settings["OPTIMIZER_TYPE"],
            "--learning_rate", str(self.settings["LEARNING_RATE"]),
            "--max_data_loader_n_workers", "2",
            "--persistent_data_loader_workers",
            "--network_module", config["network_module"],
            "--network_dim", str(self.settings["NETWORK_DIM"]),
            "--network_alpha", str(self.settings["NETWORK_ALPHA"]),
            "--timestep_sampling", self.settings["TIMESTEP_SAMPLING"],
        ])
        # --network_args is nargs="*": a SECOND occurrence would REPLACE the first, so all
        # network args must be emitted as one occurrence (loraplus + include_patterns below).
        network_args_tokens = [f"loraplus_lr_ratio={self.settings['LORA_LR_RATIO']}"]

        # Gradient checkpointing — on by default (recomputes activations in backward
        # to fit a 9B LoRA on most cards). Off trades ~20-30% faster steps for much
        # higher VRAM; only sensible on big cards with no block swap.
        if self.settings.get("GRADIENT_CHECKPOINTING", True):
            command.append("--gradient_checkpointing")

        # Target layers (selective layer training)
        # Block assignments based on empirical testing on Klein 9B:
        #   single_blocks 0-1: composition (layout, structure)
        #   single_blocks 2-11: identity/face (the core face signal)
        #   single_blocks 12-23: style (aesthetic, color, lighting)
        #   double_blocks 0-7: cross-attention (included in All Layers only)
        preset = self.training_preset_var.get() if hasattr(self, 'training_preset_var') else "Full Model"
        STYLE_COMP_PATTERNS = [r".*double_blocks\..*", r".*single_blocks\.[01]\..*"]
        IDENTITY_PATTERNS = [r".*single_blocks\.(1[0-6]|[1-9])\..*"]
        DETAILS_PATTERNS = [r".*single_blocks\.(1[2-9]|2[0-3])\..*"]

        patterns = None
        if preset == "Identity":
            patterns = IDENTITY_PATTERNS
        elif preset in ("Style", "Style+Composition"):
            patterns = STYLE_COMP_PATTERNS
        elif preset == "Details":
            patterns = DETAILS_PATTERNS
        elif preset == "Custom":
            patterns = self._build_custom_training_patterns()
            if patterns is None:
                # Visible warning — print() went to stdout only, invisible under the
                # windowed launcher, and the run silently trained the full model.
                self.update_console("[Warning] Model Area is Custom but no blocks are "
                                    "selected — training the FULL model.\n")
                messagebox.showwarning(
                    "Custom blocks empty",
                    "Model Area to Train is set to Custom but no blocks are ticked.\n\n"
                    "This run will train the FULL model. Tick blocks (or pick a preset) "
                    "if you wanted block targeting.")
        # "Full Model" → patterns stays None (train everything)

        if patterns:
            # Escape backslashes for the shell-parsed network_args value
            quoted = ",".join(f'"{p.replace(chr(92), chr(92) * 2)}"' for p in patterns)
            network_args_tokens.append(f"include_patterns=[{quoted}]")
        # Single --network_args occurrence carrying every token (see note above).
        command.extend(["--network_args"] + network_args_tokens)

        # Discrete flow shift (not for Flux 2 which uses flux2_shift automatic)
        if config.get("supports_discrete_flow_shift", True):
            command.extend(["--discrete_flow_shift", str(self.settings["DISCRETE_FLOW_SHIFT"])])

        # Sigmoid scale (only meaningful for sigmoid/shift sampling)
        ts_sampling = self.settings["TIMESTEP_SAMPLING"]
        sigmoid_scale = self.settings.get("SIGMOID_SCALE", "1.0")
        if ts_sampling in ("sigmoid", "shift") and sigmoid_scale and sigmoid_scale != "1.0":
            command.extend(["--sigmoid_scale", str(sigmoid_scale)])

        # Timestep range (from user settings, not hardcoded config)
        min_ts = self.settings.get("MIN_TIMESTEP", "")
        max_ts = self.settings.get("MAX_TIMESTEP", "")
        if min_ts:
            command.extend(["--min_timestep", str(min_ts)])
        if max_ts:
            command.extend(["--max_timestep", str(max_ts)])
        if self.settings.get("PRESERVE_DISTRIBUTION", False):
            command.append("--preserve_distribution_shape")

        command.extend([
            "--max_train_epochs", str(self.settings["MAX_TRAIN_EPOCHS"]),
            "--save_every_n_epochs", str(self.settings["SAVE_EVERY_N_EPOCHS"]),
            "--seed", str(self.settings["SEED"]),
            "--output_dir", self.settings["LORA_OUTPUT_DIR"],
            "--output_name", self.settings["LORA_NAME"],
            "--pause_flag_path", os.path.join(self.settings["LORA_OUTPUT_DIR"], ".pause_requested"),
        ])

        # State saving. --save_state used to be passed unconditionally with no UI behind it, which
        # meant a 55-epoch run silently left 54 state dirs (hundreds of MB each) and never pruned.
        # Pause still saves state either way — the trainer forces it via --pause_flag_path.
        command.extend(self._state_flags())

        # Optional parameters
        if self.settings["OPTIMIZER_ARGS"]:
            # Klein's --optimizer_args is nargs='*' (one token per key=value). Passing the
            # whole box as ONE token made the trainer's key=value split fail with more than
            # one argument.
            command.extend(["--optimizer_args"] + self.settings["OPTIMIZER_ARGS"].split())

        # Gradient accumulation (effective batch = batch × this)
        gradient_accum = self.settings.get("GRADIENT_ACCUMULATION", 1)
        if isinstance(gradient_accum, str):
            gradient_accum = int(gradient_accum) if gradient_accum else 1
        if gradient_accum > 1:
            command.extend(["--gradient_accumulation_steps", str(gradient_accum)])

        # Max gradient norm (0 to disable clipping)
        max_grad_norm = self.settings.get("MAX_GRAD_NORM", 1.0)
        if isinstance(max_grad_norm, str):
            max_grad_norm = float(max_grad_norm) if max_grad_norm else 1.0
        if max_grad_norm > 0:
            command.extend(["--max_grad_norm", str(max_grad_norm)])

        # Network dropout (LoRA regularization)
        network_dropout = self.settings.get("NETWORK_DROPOUT", 0)
        if isinstance(network_dropout, str):
            network_dropout = float(network_dropout) if network_dropout else 0
        if network_dropout > 0:
            command.extend(["--network_dropout", str(network_dropout)])

        # Attention mechanism (user's choice, default is "sdpa")
        attention = self.settings["ATTENTION_MECHANISM"]
        if attention != "none":
            command.append(f"--{attention}")

        logging_dir = self.settings["LOGGING_DIR"]
        if logging_dir:
            command.extend(["--logging_dir", logging_dir])

        log_with = self.settings["LOG_WITH"]
        if log_with != "none":
            command.extend(["--log_with", log_with])

        log_prefix = self.settings["LOG_PREFIX"]
        if log_prefix:
            command.extend(["--log_prefix", log_prefix])

        if self.settings["IMG_IN_TXT_IN_OFFLOADING"]:
            command.append("--img_in_txt_in_offloading")

        # Adaptive LR overrides the step-based scheduler — force constant pre-phase.
        adaptive_on = bool(self.settings.get("ADAPTIVE_LR", False))
        lr_scheduler = "constant" if adaptive_on else self.settings["LR_SCHEDULER"]
        if lr_scheduler:
            command.extend(["--lr_scheduler", lr_scheduler])

        if not adaptive_on:
            lr_warmup_steps = self.settings["LR_WARMUP_STEPS"]
            if lr_warmup_steps:
                command.extend(["--lr_warmup_steps", lr_warmup_steps])

            lr_decay_steps = self.settings["LR_DECAY_STEPS"]
            if lr_decay_steps:
                command.extend(["--lr_decay_steps", lr_decay_steps])

        if adaptive_on:
            command.append("--adaptive_lr")
            min_lr = (self.settings.get("ADAPTIVE_LR_MIN", "1e-5") or "1e-5").split(" ")[0]
            max_lr = (self.settings.get("ADAPTIVE_LR_MAX", "4e-4") or "4e-4").split(" ")[0]
            command.extend(["--adaptive_lr_min", str(min_lr)])
            command.extend(["--adaptive_lr_max", str(max_lr)])

        # Context LoRA — train new LoRA with an existing one frozen + active
        ctx_path = self.settings.get("CONTEXT_LORA_PATH", "").strip()
        if ctx_path:
            command.extend(["--context_lora_path", ctx_path])
            ctx_strength = self.settings.get("CONTEXT_LORA_STRENGTH", "1.0") or "1.0"
            command.extend(["--context_lora_strength", str(ctx_strength)])

        weighting_scheme = self.settings["WEIGHTING_SCHEME"]
        if weighting_scheme != "none":
            command.extend(["--weighting_scheme", weighting_scheme])
            if weighting_scheme == "logit_normal":
                logit_mean = self.settings.get("LOGIT_MEAN", "0.0")
                logit_std = self.settings.get("LOGIT_STD", "1.0")
                if logit_mean and logit_mean != "0.0":
                    command.extend(["--logit_mean", str(logit_mean)])
                if logit_std and logit_std != "1.0":
                    command.extend(["--logit_std", str(logit_std)])
            elif weighting_scheme == "mode":
                mode_scale = self.settings.get("MODE_SCALE", "1.29")
                if mode_scale and mode_scale != "1.29":
                    command.extend(["--mode_scale", str(mode_scale)])

        # Metadata
        metadata_title = self.settings["METADATA_TITLE"]
        if metadata_title:
            command.extend(["--metadata_title", metadata_title])

        metadata_author = self.settings["METADATA_AUTHOR"]
        if metadata_author:
            command.extend(["--metadata_author", metadata_author])

        metadata_description = self.settings["METADATA_DESCRIPTION"]
        if metadata_description:
            command.extend(["--metadata_description", metadata_description])

        metadata_license = self.settings["METADATA_LICENSE"]
        if metadata_license:
            command.extend(["--metadata_license", metadata_license])

        metadata_tags = self.settings["METADATA_TAGS"]
        if metadata_tags:
            command.extend(["--metadata_tags", metadata_tags])

        metadata_trigger_phrase = self.settings["METADATA_TRIGGER_PHRASE"].strip() or \
            (self.caption_trigger_var.get().strip() if hasattr(self, "caption_trigger_var") else "")
        if metadata_trigger_phrase and metadata_trigger_phrase.lower() != "trigger_word":
            command.extend(["--metadata_trigger_phrase", metadata_trigger_phrase])

        metadata_thumbnail = self.settings["METADATA_THUMBNAIL"].strip()
        if metadata_thumbnail:
            command.extend(["--metadata_thumbnail", metadata_thumbnail])

        if self.settings["RESUME_TRAINING"].strip():
            command.append(f"--resume={self.settings['RESUME_TRAINING']}")

        # Sample generation (only if enabled and architecture supports it)
        if self.sample_enabled_var.get() and config.get("supports_samples", False):
            # Generate prompt file
            prompt_file = self.generate_sample_prompt_file()
            command.extend(["--sample_prompts", prompt_file])

            # Frequency settings. Non-numeric text used to raise a bare ValueError out of
            # the command builder — treat it as "not set" instead.
            every_n_epochs = self.sample_every_n_epochs_var.get().strip()
            if every_n_epochs.isdigit() and int(every_n_epochs) > 0:
                command.extend(["--sample_every_n_epochs", every_n_epochs])

            every_n_steps = self.sample_every_n_steps_var.get().strip()
            if every_n_steps.isdigit() and int(every_n_steps) > 0:
                command.extend(["--sample_every_n_steps", every_n_steps])

            if self.sample_at_first_var.get():
                command.append("--sample_at_first")

            # Reference image for samples (Klein edit conditioning) — auto-capped
            # to ~0.20 MP in the trainer, so any size is safe.
            ref_img = getattr(self, "sample_ref_image_var", None)
            ref_img = ref_img.get().strip() if ref_img else ""
            if ref_img and os.path.exists(ref_img):
                command.extend(["--sample_ref_image", ref_img])

            # Use Distilled model for sample generation
            if getattr(self, 'use_distilled_samples_var', None) and self.use_distilled_samples_var.get():
                distilled_path = self.prefs_vars.get("distilled_dit", tk.StringVar()).get()
                if distilled_path and os.path.exists(distilled_path):
                    command.extend(["--sample_dit", distilled_path])
                    cache_mode = getattr(self, "cache_sample_model_var", None)
                    cache_mode = cache_mode.get() if cache_mode else self.settings.get("CACHE_SAMPLE_MODEL", "auto")
                    command.extend(["--cache_sample_model", cache_mode])
                    # INT8 fast preview matmul — same app-wide 'INT8 fast inference' toggle as the
                    # workbench + Krea 2 previews; applies to the Distilled sample DiT.
                    if self._get_inference_int8():
                        command.append("--sample_int8")
                    # Note: we deliberately do NOT forward the Preferences "DiT Block
                    # Swap (inference)" pref here. That setting governs the in-app
                    # inference tools (Repair Studio / Profiler / Extract / Explorer).
                    # The trainer auto-picks the Distilled sample swap from VRAM
                    # (_auto_distilled_sample_swap), so training samples manage their
                    # own memory independently. A power user can still force it via
                    # the raw --sample_blocks_to_swap flag.

        return command

    def build_cache_latents_command(self, config):
        """Build the cache latents command based on architecture"""
        if config.get("is_krea2"):
            return self._build_krea2_cache_command("krea2_cache_latents.py",
                                                   "--vae", self._krea2_pref("krea2_vae"))
        if config.get("is_minimax"):
            # --skip_existing: re-launching the same dataset should not re-encode every image.
            # Safe on LATENTS specifically because the skip validates the cached latent against
            # the CURRENT bucket, not just the filename — change Target Megapixels and it
            # re-encodes anyway. Deliberately NOT passed to text caching, where the skip is
            # filename-only and would silently reuse the embedding of an edited caption.
            cmd = self._build_krea2_cache_command(
                "minimax_cache_latents.py", "--vae", self._krea2_pref("minimax_vae")) + \
                ["--skip_existing"]
            # Passed whenever it's set — the script itself decides whether to load it, and only
            # does when the dataset actually contains clips. A stills folder never pays the
            # 605 MB, so there's nothing to gate on here.
            _avae = self._krea2_pref("minimax_audio_vae")
            if _avae:
                cmd += ["--audio_vae", _avae]
            return cmd
        arch = self.settings["ARCHITECTURE"]
        python_path = self._venv_python()
        cache_script_path = self._resolve_script(config, "cache_latents_script")

        command = [
            python_path,
            cache_script_path,
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--vae", self.settings["VAE_MODEL"],
        ]

        # Wan needs CLIP for latent caching
        if config["uses_clip"]:
            command.extend(["--clip", self.settings["CLIP_MODEL"]])

        # Flux 2 needs model version
        if config["uses_model_version"]:
            command.extend(["--model_version", config["model_version"]])

        return command

    def build_cache_text_command(self, config):
        """Build the cache text encoder command based on architecture"""
        if config.get("is_krea2"):
            return self._build_krea2_cache_command("krea2_cache_text.py",
                                                   "--text_encoder", self._krea2_pref("krea2_text_encoder"))
        if config.get("is_minimax"):
            cmd = self._build_krea2_cache_command("minimax_cache_text.py",
                                                  "--text_encoder", self._krea2_pref("minimax_text_encoder"))
            # Reference distillation: the TEACHER's conditioning has to be built HERE, because
            # it needs the 15.7 GB vision-capable encoder and that can never be resident beside
            # the DiT at training time. Each image is paired with N others from this same
            # dataset — no picker, and no image is ever its own reference.
            if self.settings.get("MINIMAX_DISTILL"):
                cmd += ["--reference_count", str(self.settings.get("MINIMAX_DISTILL_REFS", "2"))]
            return cmd
        arch = self.settings["ARCHITECTURE"]
        python_path = self._venv_python()
        cache_script_path = self._resolve_script(config, "cache_text_script")

        command = [
            python_path,
            cache_script_path,
            "--dataset_config", self.settings["DATASET_CONFIG"],
        ]

        # Different text encoder parameters based on architecture
        if config["uses_t5"]:
            command.extend(["--t5", self.settings["T5_MODEL"]])
        elif config["uses_text_encoder"]:
            command.extend(["--text_encoder", self.settings["TEXT_ENCODER"]])

        command.extend(["--batch_size", "16"])

        # FP8 text encoder flag
        if self.settings["FP8_TEXT_ENCODER"] and config["fp8_text_encoder_flag"]:
            command.append(config["fp8_text_encoder_flag"])

        # Flux 2 needs model version
        if config["uses_model_version"]:
            command.extend(["--model_version", config["model_version"]])

        return command

    # === Krea 2 native command builders ===

    def _venv_python(self) -> str:
        """Python to launch training/caching subprocesses with.

        The bundled venv when it exists, otherwise whatever interpreter is running us. Without
        the fallback, any install where the venv is not a sibling of the repo — conda, a system
        install, the Docker image (venv lives at /opt/venv) — builds a command pointing at a
        file that is not there. The subprocess then fails to launch and the run dies silently
        right after "starting cache preparation", with nothing in the console to say why.
        """
        candidate = (os.path.join(_FIZGIG_DIR, "venv", "Scripts", "python.exe") if os.name == 'nt'
                     else os.path.join(_FIZGIG_DIR, "venv", "bin", "python"))
        return candidate if os.path.isfile(candidate) else sys.executable

    def _krea2_pref(self, key: str) -> str:
        """Read a Krea 2 model path from Preferences (krea2_raw_dit / krea2_turbo_dit / krea2_vae / krea2_text_encoder)."""
        var = self.prefs_vars.get(key)
        return var.get().strip() if var is not None else ""

    def _krea2_preview_engine(self) -> str:
        """Canonical Samples-tab preview engine for Krea 2: 'raw_lora' or 'turbo_model'.

        The combobox holds a display label; this maps it back. Unknown/missing -> 'raw_lora'
        (the default): renders previews on the resident training DiT with the Turbo LoRA @1.0
        instead of loading the Turbo checkpoint and parking the trainer to CPU."""
        var = getattr(self, "krea2_preview_engine_var", None)
        if var is not None:
            label = var.get()
            for key, text in self._KREA2_ENGINE_LABELS.items():
                if label == text:
                    return key
        return "raw_lora"

    def _krea2_script(self, name: str) -> str:
        return os.path.join(_FIZGIG_DIR, "src", "fizgig", "scripts", name)

    def _minimax_reference_canvas(self):
        """The generation size the reference is scaled against — the square at Target Megapixels.

        The trainer sizes the reference against the largest training bucket; matching that here
        keeps the cached teacher conditioning and the training-time reference latent describing
        the same picture."""
        try:
            mp = float(str(self.dataset_megapixels_var.get()).strip())
        except (TypeError, ValueError, AttributeError):
            mp = 0.5
        side = int(round((mp * 1_000_000) ** 0.5 / 32) * 32) or 512
        return side, side

    def _build_krea2_cache_command(self, script_name: str, model_flag: str, model_path: str):
        """Krea 2 caching: a plain venv-python call to krea2_cache_latents.py / krea2_cache_text.py."""
        return [
            self._venv_python(),
            self._krea2_script(script_name),
            "--dataset_config", self.settings["DATASET_CONFIG"],
            model_flag, model_path,
        ]

    def _write_krea2_sample_prompts(self, filename="krea2_prompts.txt"):
        """Write the Samples-tab prompts as clean lines (one prompt per line) for krea2_train.

        Klein's prompt file carries inline flags (`--w`/`--h`/`--s`/...); krea2_train takes
        resolution as CLI args and reads each line as a literal prompt, so we strip any
        trailing ` --flag ...` group. Returns the file path, or None if no prompts."""
        samples_dir = self.get_samples_dir()
        os.makedirs(samples_dir, exist_ok=True)
        lines = []
        for raw in self.sample_prompt_text.get("1.0", tk.END).splitlines():
            ln = raw.strip()
            if not ln or ln.startswith("#"):
                continue
            ln = ln.split(" --")[0].strip()  # drop Klein-style inline flags
            if ln:
                lines.append(ln)
        if not lines:
            return None
        path = os.path.join(samples_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return path

    def _build_krea2_train_command(self):
        """Build the native Krea 2 training command (RAW base, fp8 Turbo previews).

        Model paths come from Preferences (krea2_*); rank/alpha/lr/epochs/save/seed and the
        auto-resolved Blocks Swap come from the shared Training-tab knobs; sample resolution +
        frequency come from the Samples tab (same source Klein uses)."""
        # Armed fine-tune continuation (Resume after an FT pause): --dit becomes the pause
        # checkpoint — a one-run override, the preference is never touched — and the epoch
        # count is what's left of the original total. No --resume: FT has no state dirs.
        _fr = self._ft_resume_active()
        if getattr(self, "_ft_resume", None) and not _fr:
            self.update_console("[resume] armed fine-tune continuation IGNORED — the run being "
                                "launched is a different name or not a fine-tune.\n")
        _dit = _fr["checkpoint"] if _fr else self._krea2_pref("krea2_raw_dit")
        try:
            _epochs = int(str(self.settings["MAX_TRAIN_EPOCHS"]))
        except (KeyError, ValueError, TypeError):
            _epochs = 1
        if _fr:
            _epochs = max(1, _epochs - int(_fr.get("epochs_done", 0)))
            self.update_console(f"[resume] continuing fine-tune from "
                                f"{os.path.basename(_dit)} — {_epochs} epoch(s) to run\n")
        cmd = [
            self._venv_python(),
            self._krea2_script("krea2_train.py"),
            "--dit", _dit,
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--output_dir", self.settings["LORA_OUTPUT_DIR"],
            "--output_name", self.settings["LORA_NAME"],
            "--network_dim", str(self.settings["NETWORK_DIM"]),
            "--network_alpha", str(self.settings["NETWORK_ALPHA"]),
            "--learning_rate", str(self.settings["LEARNING_RATE"]),
            "--max_train_epochs", str(_epochs),
            "--save_every_n_epochs", str(self.settings["SAVE_EVERY_N_EPOCHS"]),
            "--blocks_to_swap", str(self.settings["BLOCKS_SWAP"]),
            "--seed", str(self.settings["SEED"]),
            "--discrete_flow_shift", "2.5",
        ]
        # LoKR (Kronecker) — dim/alpha still ride along above but the trainer ignores them;
        # the factor is the dial. Klein's builder never reads NETWORK_TYPE (standard only).
        # Never emitted under base-model fine-tuning: the adapter is inert there, so a LoKR
        # would only burn VRAM (the trainer coerces too — belt and braces). Tk var read
        # directly, same rule as the FT flags below (ab3cca2).
        if (str(self.settings.get("NETWORK_TYPE", "")).startswith("LoKR")
                and not bool(self.krea2_finetune_var.get())):
            cmd += ["--network_type", "lokr",
                    "--lokr_factor", str(self.settings.get("LOKR_FACTOR", 8))]
        # State saving. Krea 2 previously wrote state ONLY on Pause, so a crash or a run that
        # finished too early meant starting over. Pause still saves regardless of these flags.
        cmd += self._state_flags()
        # Resume from a saved <name>-NNNNNN-state dir (set by the Resume button / pause flow).
        resume_path = (self.settings.get("RESUME_TRAINING") or "").strip()
        if resume_path:
            cmd += ["--resume", resume_path]
        # Context LoRA — train with an existing LoRA frozen + active on the base (model-agnostic).
        ctx_path = (self.settings.get("CONTEXT_LORA_PATH") or "").strip()
        if ctx_path:
            ctx_strength = (self.settings.get("CONTEXT_LORA_STRENGTH") or "1.0").strip() or "1.0"
            cmd += ["--context_lora_path", ctx_path, "--context_lora_strength", ctx_strength]
        # Adaptive LR — bi-directional plateau tracker (model-agnostic). Min/Max combo values
        # can carry a trailing note (e.g. "2e-4 - rank 4/8 only"); take the leading token.
        if self.settings.get("ADAPTIVE_LR"):
            min_lr = str(self.settings.get("ADAPTIVE_LR_MIN", "1e-5")).split(" ")[0]
            max_lr = str(self.settings.get("ADAPTIVE_LR_MAX", "4e-4")).split(" ")[0]
            cmd += ["--adaptive_lr", "--adaptive_lr_min", min_lr, "--adaptive_lr_max", max_lr]
        else:
            # LR scheduler + warmup (Other Options). Only when adaptive is OFF — adaptive owns the
            # LR, and the trainer would ignore the schedule anyway. These fields were visible under
            # Krea 2 but silently unwired before this.
            sched = (self.settings.get("LR_SCHEDULER") or "constant").strip() or "constant"
            if sched != "constant":
                cmd += ["--lr_scheduler", sched]
            warmup = str(self.settings.get("LR_WARMUP_STEPS", "") or "").strip()
            if warmup:
                try:
                    if int(float(warmup)) > 0:
                        cmd += ["--lr_warmup_steps", str(int(float(warmup)))]
                except ValueError:
                    pass
        # Gradient accumulation + grad clipping (Optimizer section — both wired for krea2).
        try:
            _accum = int(str(self.settings.get("GRADIENT_ACCUMULATION", 1) or 1).strip() or 1)
        except ValueError:
            _accum = 1
        if _accum > 1:
            cmd += ["--gradient_accumulation_steps", str(_accum)]
        _mgn = str(self.settings.get("MAX_GRAD_NORM", "") or "").strip()
        if _mgn:
            try:
                if abs(float(_mgn) - 1.0) > 1e-9:   # 1.0 is the trainer default
                    cmd += ["--max_grad_norm", str(float(_mgn))]
            except ValueError:
                pass
        # Optimizer family + free-form kwargs. Sent whenever set: the trainer's own default is
        # adamw8bit, so passing it explicitly is harmless and keeps the launched command a full
        # record of what the run actually used.
        _opt = str(self.settings.get("OPTIMIZER_TYPE", "") or "").strip()
        if _opt:
            cmd += ["--optimizer_type", _opt]
        _opt_args = str(self.settings.get("OPTIMIZER_ARGS", "") or "").strip()
        if _opt_args:
            cmd += ["--optimizer_args", _opt_args]
        _cb = str(self.settings.get("COMPILE_BLOCKS", "auto") or "auto").lower()
        # "outside" is a hand-set power value (settings JSON only — the dropdown offers
        # Auto/On/Off): the high-res compile boundary (#99). Passing it through beats
        # silently downgrading a stated choice to auto.
        if _cb in ("auto", "on", "off", "outside"):
            cmd += ["--compile_blocks", _cb]
        # torch.compile and block swap are mutually exclusive — compiled graphs assume their
        # weights stay put, and swap moves them every step, so the trainer ignores compile
        # whenever swap is active. It says so in its own log, but a user who set compile to On
        # sees the GUI still reading "On" and reasonably believes it is running.
        try:
            _blocks_swap = int(self.settings.get("BLOCKS_SWAP", 0) or 0)
        except (TypeError, ValueError):
            _blocks_swap = 0
        if _cb == "on" and _blocks_swap > 0:
            self.update_console(
                f"[compile] ignored this run — block swap is active ({_blocks_swap} blocks), and "
                "compiled graphs can't tolerate weights moving between CPU and GPU each step. "
                "Use 4-bit (NF4) instead of swapping if you want compile as well.\n")
        # Output metadata (Other Options → Metadata) — previously visible but never wired
        # for Krea 2; now recorded in the saved LoRA.
        for _mkey, _mflag in (("METADATA_TITLE", "--metadata_title"),
                              ("METADATA_AUTHOR", "--metadata_author"),
                              ("METADATA_DESCRIPTION", "--metadata_description"),
                              ("METADATA_LICENSE", "--metadata_license"),
                              ("METADATA_TAGS", "--metadata_tags")):
            _mval = str(self.settings.get(_mkey, "") or "").strip()
            if _mval:
                cmd += [_mflag, _mval]
        # Trigger phrase falls back to the Captions tab's trigger word — independent of
        # --trigger_word above, which is only ever sent when auto-recaption is on.
        _mtrig = self.settings.get("METADATA_TRIGGER_PHRASE", "").strip() or \
            (self.caption_trigger_var.get().strip() if hasattr(self, "caption_trigger_var") else "")
        if _mtrig and _mtrig.lower() != "trigger_word":
            cmd += ["--metadata_trigger_phrase", _mtrig]
        _mthumb = self.settings.get("METADATA_THUMBNAIL", "").strip()
        if _mthumb:
            cmd += ["--metadata_thumbnail", _mthumb]
        # Base weight optimization. 4-bit NF4 supersedes fp8 (mutually exclusive): it quantizes the
        # frozen base to ~5.6 GB so a full LoRA trains on a 10-12 GB card with NO block swap (the
        # trainer forces blocks_to_swap=0 under 4-bit). Otherwise fp8 Base (the default) unless the
        # user unchecked it (bf16, 26 GB — big-card / heavy-swap only).
        # Explicit user choices FIRST — the auto branch used to be tested before them, so
        # unticking "FP8 Base" (an explicit bf16 request) did nothing when auto had chosen
        # INT8.
        # FP8 Base is hidden for Krea 2 and deliberately ignored here: --no_fp8 means a bf16
        # base (~28 GB) that no consumer card holds, the swap planner never accounted for it,
        # and this elif chain used to let it silently cancel the INT8 the planner had chosen.
        # A value persisted from Klein (or from before it was hidden) must not leak into a
        # Krea 2 run through a control the user can no longer see.
        _auto_i8 = getattr(self, "_auto_quant_int8", "")
        # An EXPLICIT INT8 pick must not depend on Blocks Swap being on Auto (#97): the auto
        # strategy is the only writer of _auto_quant_int8, and a manual swap value clears it
        # (the stale-leak guard in _parse_blocks_swap), so "Base Precision: INT8" plus a
        # manual swap silently fell back to the fp8 base — which Compile Blocks then dies on
        # for SM 8.6 cards (no fp8e4nv Triton support). At swap 0 the pick is honoured
        # directly. At swap N the fp8 fallback stays (INT8 weights don't ride the swap —
        # that pairing is the OOM the stale-leak guard exists for) but is now SAID, not
        # silent.
        try:
            _swap_now = int(str(self.settings.get("BLOCKS_SWAP", 0)).strip() or 0)
        except (TypeError, ValueError):
            _swap_now = 0
        try:
            _explicit_i8 = self._base_precision() == "int8"
        except Exception:
            _explicit_i8 = False
        if self.settings.get("QUANT_4BIT", False):
            cmd.append("--quantize_4bit")
        elif _explicit_i8 and _swap_now == 0:
            cmd += ["--quant_int8", "bf16"]
        elif _auto_i8:
            # Chosen by the auto strategy when there is VRAM for it: faster than NF4 and ~7x
            # more accurate, with exact gradients.
            cmd += ["--quant_int8", _auto_i8]
        elif _explicit_i8:
            self.update_console(
                f"[precision] INT8 needs Blocks Swap 0 — INT8 weights don't ride the swap. "
                f"Running the fp8 base with swap {_swap_now}; set Blocks Swap to 0 or Auto "
                f"to train on INT8.\n")

        # Per-image loss watch: detection logs/reports stuck images (Problem Images window);
        # per-image LR also throttles them (the trainer runs detection when either flag is on).
        # All four need Batch Size 1 (a batch-mean isn't a per-image signal) — the GUI greys
        # the toggles at batch > 1, and the flags are skipped here to match, with a note.
        try:
            _watch_bs = int(str(self.dataset_batch_size_var.get()).strip() or 1)
        except (ValueError, AttributeError):
            _watch_bs = 1
        _watch_ok = _watch_bs <= 1
        if not _watch_ok and (self.krea2_loss_watch_var.get() or self.krea2_per_image_lr_var.get()
                              or self.krea2_warmup_look_var.get() or self.krea2_auto_recaption_var.get()):
            self.update_console("[loss-watch] per-image features skipped — Batch Size is "
                                f"{_watch_bs}; they need Batch Size 1.\n")
        if _watch_ok and self.krea2_loss_watch_var.get():
            cmd.append("--log_per_image_loss")
        if _watch_ok and self.krea2_per_image_lr_var.get():
            cmd.append("--per_image_lr")
        if _watch_ok and self.krea2_warmup_look_var.get():
            cmd.append("--warmup_look_outliers")
        # Full base-model fine-tune (experimental): rotating trainable windows, full checkpoint out.
        # Read the Tk vars DIRECTLY, like every other krea2 toggle here — self.settings is only
        # refreshed when a preset is collected, so reading it made this silently never fire and
        # the run trained a LoRA instead.
        if bool(self.krea2_finetune_var.get()):
            mode = str(self.krea2_ft_mode_var.get() or "auto")
            if mode.lower().startswith("auto"):
                mode = "auto"   # the trainer resolves it from free VRAM at launch
            try:
                nblocks = int(str(self.krea2_ft_blocks_var.get()))
            except ValueError:
                nblocks = 14
            try:
                every = int(str(self.krea2_ft_every_var.get()))
            except ValueError:
                every = 1
            cmd += ["--finetune_rotation", str(max(1, nblocks)),
                    "--finetune_rotation_mode", mode,
                    "--finetune_rotate_every", str(max(1, every))]
            # Base precision under FINE-TUNE: NF4 is the trainer's default now, so an
            # explicit fp8 pick needs saying out loud. At the CLI an fp8 choice emits NO
            # flag (fp8_scaled = not --no_fp8, true by default), which is indistinguishable
            # from "Auto" — without this the dropdown's fp8 entry would silently produce
            # NF4 and the control would be lying. LoRA runs are untouched: this sits inside
            # the fine-tune branch.
            try:
                if self._base_precision() == "fp8":
                    cmd.append("--ft_base_fp8")
            except Exception:
                pass
            if _fr:
                # Continuation: pick the rotation cycle back up where the pause left it
                # (from the checkpoint's metadata) instead of restarting at window 0.
                cmd += ["--finetune_start_window", str(int(_fr.get("next_window", 0)))]
            if bool(self.krea2_ft_fused_var.get()):
                cmd.append("--finetune_fused_backward")
            if bool(self.krea2_fast_ft_var.get()):
                cmd.append("--fast_ft")
            _reg = self.krea2_reg_dir_var.get().strip()
            if _reg and os.path.isdir(_reg):
                try:
                    _rm = float(self.krea2_reg_mult_var.get())
                except ValueError:
                    _rm = 0.2
                cmd += ["--reg_lr_multiplier", str(max(0.0, _rm))]
        if _watch_ok and self.krea2_auto_recaption_var.get():
            cmd.append("--auto_recaption")
            # Trigger word from the Captions tab — appended (', <trigger>') to AI captions if
            # set. Reads the WIDGET-BOUND var (caption_text_var is an orphan that never
            # carried what the user typed). The placeholder guard stays in case an old
            # last_used.json seeded the literal "trigger_word".
            trig = (self.caption_trigger_var.get().strip()
                    if hasattr(self, "caption_trigger_var") else "")
            if trig and trig.lower() != "trigger_word":
                cmd += ["--trigger_word", trig]
            # Auto-recaption maps its two attempts onto two Captions-tab presets: attempt 1 uses
            # TRAINING CAPTION, attempt 2 uses EXHAUSTIVE DETAIL — your edited version of each
            # where you have one, the built-in otherwise. Deliberately not "whatever the tab is
            # set to": auto-recaption's job is fixed, so leaving the tab on "Short caption" must
            # not silently change what a training run writes mid-run.
            _ovr = self._caption_overrides()
            for _key, _flag in (("training", "--recaption_instruction"),
                                ("exhaustive", "--recaption_instruction_detailed")):
                _instr = str(_ovr.get(_key, "") or "").strip()
                if _instr:
                    cmd += [_flag, _instr]
        # Caption repair (manual edits from the Problem Images window AND auto-recaption)
        # re-encodes with the Qwen3-VL text encoder. --text_encoder used to be emitted only
        # inside the samples block, so with previews off the trainer had no TE path and every
        # caption fix bailed for the whole run, re-queueing forever. Emit it whenever any
        # watch toggle is on (a duplicate in the samples block is harmless — same value).
        if (self.krea2_loss_watch_var.get() or self.krea2_per_image_lr_var.get()
                or self.krea2_warmup_look_var.get() or self.krea2_auto_recaption_var.get()):
            _te = self._krea2_pref("krea2_text_encoder")
            if _te:
                cmd += ["--text_encoder", _te]

        # In-training previews: render the fp8 Turbo with the live LoRA. Resolution +
        # frequency come from the Samples tab; previews land in <output_dir>/sample, which
        # is exactly where the GUI samples watcher looks.
        if self.sample_enabled_var.get():
            prompt_file = self._write_krea2_sample_prompts()
            every = self.sample_every_n_epochs_var.get().strip()
            every_n = int(every) if every.isdigit() else 0
            # Krea 2 previews are per-EPOCH only. A steps-only config used to skip the
            # whole sample block in silence — say so instead.
            _steps_only = self.sample_every_n_steps_var.get().strip()
            if every_n <= 0 and _steps_only.isdigit() and int(_steps_only) > 0:
                self.update_console("[samples] Krea 2 previews are per-epoch — 'Every N Steps' "
                                    "has no effect. Set 'Every N Epochs' to enable previews.\n")
            ref_img = (getattr(self, "sample_ref_image_var", None).get().strip()
                       if getattr(self, "sample_ref_image_var", None) else "")
            ref_img = ref_img if (ref_img and os.path.exists(ref_img)) else ""
            _at_first = bool(getattr(self, "sample_at_first_var", None)
                             and self.sample_at_first_var.get())
            # Samples fire if there's a prompt OR a reference (ref-only = 'generate from this
            # picture' via the Qwen3-VL vision path). Sample-at-Start alone also counts.
            if (prompt_file or ref_img) and (every_n > 0 or _at_first):
                width = (self.sample_width_var.get().strip() or "1024")
                height = (self.sample_height_var.get().strip() or "1024")
                # Sample seed from the Samples tab (0 is a valid seed — don't let it fall through to
                # the trainer's default). Non-numeric/empty -> the SAMPLE_SEED default.
                try:
                    sample_seed = str(int(self.sample_seed_var.get().strip()))
                except (ValueError, AttributeError):
                    sample_seed = str(self.settings.get("SAMPLE_SEED", 1234))
                cmd += [
                    "--sample_every_n_epochs", str(every_n),
                    "--sample_width", width,
                    "--sample_height", height,
                    "--sample_seed", sample_seed,
                    "--turbo_dit", self._krea2_pref("krea2_turbo_dit"),
                    "--vae", self._krea2_pref("krea2_vae"),
                    "--text_encoder", self._krea2_pref("krea2_text_encoder"),
                    # Forward-only block swap on the preview Turbo, auto-detected for the Turbo's
                    # VRAM profile so previews fit the card — mirrors Klein's Distilled sample swap.
                    # (Ignored in raw_lora engine mode — that path uses the training placement.)
                    "--preview_blocks_to_swap", str(self._auto_krea2_inference_blocks_swap()),
                ]
                # Preview engine (Samples tab): raw_lora renders on the resident training DiT
                # with the Turbo LoRA @1.0 — no Turbo checkpoint load, no CPU parking. The
                # trainer prefers --turbo_lora over --turbo_dit when both are given, and falls
                # back to the Turbo checkpoint by itself if the LoRA file has gone missing.
                # Under a fine-tune the Turbo LoRA is REQUIRED for previews (the trained
                # weights live in the base, which the standalone Turbo can't show), so the
                # engine preference is overridden and the LoRA travels regardless.
                if (self._krea2_preview_engine() == "raw_lora"
                        or bool(self.krea2_finetune_var.get())):
                    _tlora = self._krea2_pref("krea2_turbo_lora")
                    if not _tlora or not os.path.isfile(_tlora):
                        # First use after an update: fetch it now (~470 MB, idempotent — the
                        # update script usually gets there first) and populate the pref.
                        try:
                            import sys as _sys
                            _sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
                            from fizgig.scripts.fetch_turbo_lora import ensure_turbo_lora
                            self.update_console("[preview] Turbo LoRA not set — downloading it "
                                                "now (one-time, ~470 MB)...\n")
                            _tlora = ensure_turbo_lora(
                                log=lambda m: self.update_console(f"[preview] {m}\n"),
                                require=True)
                            if _tlora and "krea2_turbo_lora" in self.prefs_vars:
                                self.prefs_vars["krea2_turbo_lora"].set(_tlora)
                        except Exception:
                            _tlora = None
                    if _tlora:
                        cmd += ["--turbo_lora", _tlora]
                    else:
                        self.update_console(
                            "[preview] Turbo LoRA unavailable (download failed?) — using the "
                            "classic Turbo model for previews this run. Set the path in "
                            "Preferences or re-run update_fizgig.bat.\n")
                # Steps / CFG / Negative / Sample-at-Start — previously visible on the Samples
                # tab but never wired into krea2_train.
                _st = self.sample_steps_var.get().strip()
                if _st.isdigit() and int(_st) > 0:
                    cmd += ["--sample_steps", _st]
                try:
                    _cfg = float(self.sample_cfg_scale_var.get().strip() or 1.0)
                except (ValueError, AttributeError):
                    _cfg = 1.0
                if _cfg > 0 and abs(_cfg - 1.0) > 1e-9:
                    cmd += ["--sample_cfg_scale", str(_cfg)]
                _negp = (self.sample_negative_var.get().strip()
                         if getattr(self, "sample_negative_var", None) else "")
                if _negp and _cfg > 1.0:
                    cmd += ["--sample_negative", _negp]
                if _at_first:
                    cmd.append("--sample_at_first")
                # INT8 fast preview matmul — same app-wide 'INT8 fast inference' toggle as the workbench.
                if self._get_inference_int8():
                    cmd.append("--preview_int8")
                if prompt_file:
                    cmd += ["--sample_prompts", prompt_file]
                if ref_img:
                    cmd += ["--sample_ref_image", ref_img]
        return cmd

    def _build_minimax_train_command(self):
        """Build the native MiniMax H3 training command — barebones image-only LoRA over an
        NF4-quantized frozen base. No samples, no block swap, no context LoRA, no LoKR, no
        per-image loss watch: just the core knobs (rank/alpha/lr/epochs/save/seed/optimizer) plus
        adaptive LR and output metadata. Model paths come from Preferences (minimax_*)."""
        # Armed fine-tune continuation (Resume after an FT pause): --dit becomes the pause
        # checkpoint — a one-run override that outranks the distill/ref2va choice too — and
        # the epoch count is what's left of the original total. No --resume under FT.
        _fr = self._ft_resume_active()
        if getattr(self, "_ft_resume", None) and not _fr:
            self.update_console("[resume] armed fine-tune continuation IGNORED — the run being "
                                "launched is a different name or not a fine-tune.\n")
        _dit = (_fr["checkpoint"] if _fr
                else (self._krea2_pref("minimax_ref_dit")
                      if ((self.settings.get("MINIMAX_DISTILL")
                           or self.settings.get("MINIMAX_TRAIN_BASE") == "ref2va")
                          and self._krea2_pref("minimax_ref_dit"))
                      else self._krea2_pref("minimax_dit")))
        try:
            _epochs = int(str(self.settings["MAX_TRAIN_EPOCHS"]))
        except (KeyError, ValueError, TypeError):
            _epochs = 1
        if _fr:
            _epochs = max(1, _epochs - int(_fr.get("epochs_done", 0)))
            self.update_console(f"[resume] continuing fine-tune from "
                                f"{os.path.basename(_dit)} — {_epochs} epoch(s) to run\n")
        cmd = [
            self._venv_python(),
            self._krea2_script("minimax_train.py"),
            # Distillation trains against ref2va — the teacher only exists on that model.
            # Otherwise the Training Base dropdown decides: ref2va when the user deploys on
            # the r2v workflow, the ordinary fl2va base by default.
            "--dit", _dit,
            "--dataset_config", self.settings["DATASET_CONFIG"],
            "--output_dir", self.settings["LORA_OUTPUT_DIR"],
            "--output_name", self.settings["LORA_NAME"],
            "--network_dim", str(self.settings["NETWORK_DIM"]),
            "--network_alpha", str(self.settings["NETWORK_ALPHA"]),
            "--learning_rate", str(self.settings["LEARNING_RATE"]),
            "--max_train_epochs", str(_epochs),
            "--save_every_n_epochs", str(self.settings["SAVE_EVERY_N_EPOCHS"]),
            "--seed", str(self.settings["SEED"]),
        ]
        # Blocks Swap: "Auto (detect from GPU)" resolves in the TRAINER (it reads real free VRAM
        # at run time — correct for queued runs too); an explicit number passes through.
        _bs = str(self.settings.get("BLOCKS_SWAP", "auto") or "auto").strip()
        cmd += ["--blocks_to_swap", "auto" if _bs.lower().startswith("auto") else _bs]
        # Base Precision. Always sent, including "auto", so the launched command records which
        # base a run used rather than leaving it implicit — these get A/B'd against each other.
        cmd += ["--base_quant", minimax_base_quant(self.settings.get("MINIMAX_BASE_QUANT"))]
        # Per-step movement clip: RETIRED (Peter, 10 Aug) — the Adapter-relative LR ramp removes
        # the overshoot at its root rather than capping it after the fact. Never emitted, so an
        # old preset or a saved config cannot revive it.
        # Gradient Accumulation (Optimizer section). The field was visible under MiniMax but
        # never emitted, so it silently did nothing on this family.
        try:
            _accum = int(str(self.settings.get("GRADIENT_ACCUMULATION", 1) or 1).strip() or 1)
        except ValueError:
            _accum = 1
        if _accum > 1:
            cmd += ["--gradient_accumulation_steps", str(_accum)]
        # LR warmup: RETIRED alongside the clip — the ramp eases the first epochs in by
        # construction, and does not need an epoch count guessed up front. Never emitted.
        # EMA stays: "0.99 (recommended)" -> 0.99.
        _em = str(self.settings.get("MINIMAX_EMA", "Off") or "Off").split(" ")[0]
        if _em.replace(".", "", 1).isdigit():
            cmd += ["--ema_decay", _em]
        _ar = str(self.settings.get("MINIMAX_ADAPTER_RAMP", "Off") or "Off").split(" ")[0]
        if _ar.replace(".", "", 1).isdigit():
            cmd += ["--adapter_ramp", _ar]
        # Caption dropout. ALWAYS sent, including 0 — the trainer's own default is 0.05, so
        # "Off" has to be stated explicitly or it silently keeps dropping captions.
        # Whatever the box says, including under Multi Concept — the builder used to force it to
        # 0 there, which quietly made every multi-concept run a dropout-off run and confounded
        # the very comparison it was meant to help.
        _cd = str(self.settings.get("MINIMAX_CAPTION_DROPOUT", "0.05") or "0.05").split(" ")[0]
        cmd += ["--caption_dropout", _cd if _cd.replace(".", "", 1).isdigit() else "0"]
        # Gradient Checkpointing. The flag used to not be sent at all here, so the checkbox was
        # decorative on this family. Ticked (the default) means AUTO — the planner decides from
        # free VRAM, exactly like Blocks Swap and Base Precision, and in practice that is "on"
        # for anything short of a 36 GB+ card. Unticked is the explicit override that forces it
        # off. Deliberately no "force ON": it only differs from auto where there is memory to
        # spare, and there it just costs ~0.1 s/step for nothing.
        cmd += ["--gradient_checkpointing",
                "auto" if self.settings.get("GRADIENT_CHECKPOINTING", True) else "off"]
        # Detail Focus -> --shift. Sent ALWAYS, including the reference 12, so the launched
        # command (and the console line recording it) states which density a run used instead of
        # leaving it implicit — these are meant to be A/B'd against each other, often queued
        # back to back, and "which one was this?" has to be answerable from the record alone.
        # The trainer stamps the same thing into the LoRA as ss_timestep_density.
        # Low-noise share -> shift. Always sent, including the default, so the launched command
        # records which density ran instead of leaving it implicit.
        # Always the plain uniform-base shift. A saved preset or queue row carrying the retired
        # MINIMAX_LOGNORM is deliberately ignored rather than migrated — mid-concentrated is the
        # thing being removed, so honouring it here would keep shipping the fault.
        _shift = minimax_lownoise_to_shift(self.settings.get("MINIMAX_LOWNOISE_PCT"))
        if _shift is not None:
            cmd += ["--shift", f"{_shift:g}"]
        _hl = minimax_highnoise_lr(self.settings.get("MINIMAX_HIGHNOISE_LR_PCT"))
        _ft_now = bool(getattr(self, "minimax_finetune_var", None)
                       and self.minimax_finetune_var.get())
        # Not under FT: the band multiplier rewrites optimizer param-group LRs, which the
        # fused fine-tune doesn't have (and rotation rebuilds discard the stash). The GUI
        # hides the control under FT; a stale saved value must not resurrect the flag.
        if _hl is not None and abs(_hl - 1.0) > 1e-9 and not _ft_now:
            cmd += ["--highnoise_lr_scale", f"{_hl:g}"]
        # Per-category retirement (mixed visual+voice datasets). One category, one epoch —
        # sent only when the epoch is set: the flag's presence means the run used it.
        try:
            _n = int(str(self.settings.get("MIXED_STOP_EPOCH", "") or "").strip() or 0)
        except ValueError:
            _n = 0
        if _n > 0:
            _flag = ("visual" if "photo" in
                     str(self.settings.get("MIXED_STOP_CATEGORY", "")).lower() else "audio")
            # Under FT only "stop" exists (the anchor rides param-group LR machinery FT
            # doesn't have) and the trainer snaps the epoch to a rotation-cycle boundary.
            _mode = ("stop" if (_ft_now or "stop" in
                     str(self.settings.get("MIXED_STOP_MODE", "")).lower()) else "anchor")
            cmd += [f"--{_flag}_stop_epoch", str(_n), f"--{_flag}_stop_mode", _mode]
        # Blocks to Train — only sent when it's a real range; "all" is the trainer's own default,
        # and not sending it keeps the flag's presence meaning "this run was a block experiment".
        _mft_cmd_on = bool(getattr(self, "minimax_finetune_var", None)
                           and self.minimax_finetune_var.get())
        _blocks = minimax_block_spec(self.settings.get("MINIMAX_BLOCKS", "all"))
        if _blocks.lower() != "all" and not _mft_cmd_on:
            cmd += ["--train_blocks", _blocks]
        # Optimised Likeness Learning — photo steps train the identity blocks only, clips train
        # everything. The launch dict already forced MINIMAX_BLOCKS to "all" when this is on, so
        # the two flags never fight. The flag TRAVELS under fine-tune too: the trainer honours
        # the same semantics there (cycle-tighten on photo-only data, per-parameter photo
        # freezing on mixed). --train_blocks stays adapter-only and is never emitted under FT.
        if self.settings.get("MINIMAX_LIKENESS_OPT"):
            cmd += ["--photo_blocks", MINIMAX_LIKENESS_BLOCKS]
            # Restrict video to likeness blocks (FT only, on by default with likeness):
            # a confined overnight video run trained perfectly well (field, 29 Aug).
            # Unticked, clips keep the original whole-model behaviour.
            if _mft_cmd_on and self.settings.get("MINIMAX_FT_CLIP_LIKENESS", True):
                cmd += ["--clip_blocks", MINIMAX_LIKENESS_BLOCKS]
        # Voice routing — audio steps train only the measured voice zone (34-49): outside it
        # they corrupt the visual blocks (A/B, 24 Aug). Under FT it always travels (the
        # trainer also tightens the cycle to the union of what the dataset trains); in LoRA
        # mode it is part of Optimised Likeness Learning. Harmless without audio files.
        if _ft_now or self.settings.get("MINIMAX_LIKENESS_OPT"):
            cmd += ["--audio_blocks", MINIMAX_AUDIO_BLOCKS]
        # Reference distillation. Both flags travel together; the trainer also needs --vae to
        # encode the reference, which the sample block may already have added.
        if self.settings.get("MINIMAX_DISTILL"):
            cmd += ["--distill",
                    "--distill_weight", str(self.settings.get("MINIMAX_DISTILL_WEIGHT", "0.8"))]
            # Identity-first phase length. "Auto" -> -1 (the trainer sizes it from the dataset),
            # "Off" -> 0 (blended throughout), otherwise the leading number of epochs.
            _p1 = str(self.settings.get("MINIMAX_DISTILL_PHASE1", "Auto") or "Auto")
            _p1n = "-1" if _p1.startswith("Auto") else ("0" if _p1.startswith("Off")
                                                        else _p1.split(" ")[0])
            cmd += ["--distill_phase1_epochs", _p1n if _p1n.lstrip("-").isdigit() else "-1"]
        # AdaLN LOCKED off (Peter, 9 Aug): the pruned builds everyone deploys on cannot load
        # AdaLN LoRA keys, so training it only wastes capacity. Checkbox hidden; always opt out.
        cmd.append("--no_train_adaln")
        # Depth-split LR is RETIRED (Peter, 9 Aug): it was the manual precursor of the limiter
        # + governor, which target whoever actually runs hot instead of a guessed range. The
        # controls are hidden and a stale saved range is deliberately not sent.
        # Rotation fine-tune. Read from the Tk vars, not self.settings — the Krea builder
        # learned the hard way that reading settings made the flags silently never fire.
        _mft_on = bool(getattr(self, "minimax_finetune_var", None)
                       and self.minimax_finetune_var.get())
        if _mft_on:
            # Component is the only mode (24 Aug — block/numeric windows removed).
            cmd += ["--finetune_rotation", "1", "--finetune_rotation_mode", "component"]
            _mfte = str(self.minimax_ft_every_var.get()).strip()
            cmd += ["--finetune_rotate_every", _mfte if _mfte.isdigit() else "1"]
            if _fr:
                # Continuation: pick the rotation cycle back up where the pause left it
                # (from the checkpoint's metadata) instead of restarting at window 0.
                cmd += ["--finetune_start_window", str(int(_fr.get("next_window", 0)))]
            if str(self.minimax_ft_scope_var.get()).startswith("Photos"):
                cmd += ["--finetune_scope", "photo"]
            _mftspec = str(self.minimax_ft_blockspec_var.get()).strip()
            if _mftspec:
                cmd += ["--finetune_blocks", _mftspec]
            if not bool(self.minimax_ft_fused_var.get()):
                cmd += ["--no_finetune_fused_backward"]
            # Regularisation LR multiplier — only meaningful when the TOML carries the
            # is_reg block (same gate as the block writer: FT on + a real folder).
            _mreg = (self.minimax_reg_dir_var.get().strip()
                     if hasattr(self, "minimax_reg_dir_var") else "")
            if _mreg and os.path.isdir(_mreg):
                try:
                    _mrm = float(self.minimax_reg_mult_var.get())
                except ValueError:
                    _mrm = 0.2
                # Floor 0.01: 0.0 would still pay a full forward/backward per reg step
                # for a near-zero update — clearing the folder is how you disable this.
                cmd += ["--reg_lr_multiplier", str(max(0.01, _mrm))]
        # LoKR (Kronecker) — dim/alpha still ride along above but the trainer ignores them;
        # the factor is the dial. Same flags as the Krea 2 builder. Suppressed under FT: the
        # trainer builds no adapter at all there.
        if str(self.settings.get("NETWORK_TYPE", "")).startswith("LoKR") and not _mft_on:
            cmd += ["--network_type", "lokr",
                    "--lokr_factor", str(self.settings.get("LOKR_FACTOR", 8))]
        # In-training previews. Prompts come from the Samples tab (same widgets every family
        # uses); the trainer pre-encodes them with the 32B TE before the DiT loads.
        if self.sample_enabled_var.get():
            # plain one-prompt-per-line file (same writer Krea 2 uses; own filename so a
            # MiniMax output dir doesn't sprout a "krea2_" artefact)
            prompt_file = self._write_krea2_sample_prompts("minimax_prompts.txt")
            _every = self.sample_every_n_epochs_var.get().strip()
            _every_n = int(_every) if _every.isdigit() else 0
            _at_first = bool(getattr(self, "sample_at_first_var", None)
                             and self.sample_at_first_var.get())
            # H3 previews are per-EPOCH only; a steps-only config would otherwise silently
            # produce nothing (same constraint, and same warning, as Krea 2).
            _steps_only = self.sample_every_n_steps_var.get().strip()
            if _every_n <= 0 and _steps_only.isdigit() and int(_steps_only) > 0:
                self.update_console("[samples] MiniMax H3 previews are per-epoch — 'Every N Steps' "
                                    "has no effect. Set 'Every N Epochs' to enable previews.\n")
            _te = self._krea2_pref("minimax_text_encoder")
            if prompt_file and (_every_n > 0 or _at_first) and _te:
                try:
                    _seed = str(int(self.sample_seed_var.get().strip()))
                except (ValueError, AttributeError):
                    _seed = str(self.settings.get("SAMPLE_SEED", 42))
                cmd += [
                    "--sample_prompts", prompt_file,
                    "--sample_every_n_epochs", str(_every_n),
                    "--sample_width", (self.sample_width_var.get().strip() or "512"),
                    "--sample_height", (self.sample_height_var.get().strip() or "512"),
                    "--sample_seed", _seed,
                    "--text_encoder", _te,
                    "--vae", self._krea2_pref("minimax_vae"),
                ]
                # Sample length: "124 frames (~5s — trained minimum)" -> 124. Always sent so
                # the launched command records whether a run previewed stills or clips.
                _sf_raw = str(getattr(self, "sample_frames_var", None)
                              and self.sample_frames_var.get() or "")
                _sf = _sf_raw.split(" ")[0]
                cmd += ["--sample_frames", _sf if _sf.isdigit() else "1"]
                # "with sound" variants: the samples also carry their generated audio,
                # decoded through the audio VAE — same file the caching pass uses.
                if "with sound" in _sf_raw.lower():
                    _avae = self._krea2_pref("minimax_audio_vae")
                    if _avae:
                        cmd += ["--sample_audio", "--audio_vae", _avae]
                    else:
                        self.update_console("[samples] 'with sound' needs the Audio VAE path "
                                            "— set it in Preferences. Samples render "
                                            "silent.\n")
                # The Turbo LoRA (Preferences) takes over the preview pace when set: its own
                # steps + strength from the Samples tab (6 @ 75% recommended), nothing else
                # changed. Without it, the ordinary Steps box applies as before.
                _turbo = self._krea2_pref("minimax_turbo_lora")
                if _turbo and os.path.isfile(_turbo):
                    # read the WIDGETS, like the other sample fields — settings can lag them
                    _ts = str(getattr(self, "turbo_steps_entry", None)
                              and self.turbo_steps_entry.get() or "").strip()
                    _ts = _ts if _ts.isdigit() and int(_ts) > 0 else "6"
                    try:
                        _tstr = float(str(getattr(self, "turbo_strength_entry", None)
                                          and self.turbo_strength_entry.get()
                                          or "").strip() or 75) / 100.0
                    except ValueError:
                        _tstr = 0.75
                    _tstr = min(2.0, max(0.0, _tstr))
                    cmd += ["--turbo_lora_path", _turbo,
                            "--turbo_lora_strength", f"{_tstr:.3f}",
                            "--sample_steps", _ts]
                else:
                    _st = self.sample_steps_var.get().strip()
                    if _st.isdigit() and int(_st) > 0:
                        cmd += ["--sample_steps", _st]
                try:
                    _cfg = float(self.sample_cfg_scale_var.get().strip())
                except (ValueError, AttributeError):
                    _cfg = 1.0
                if abs(_cfg - 1.0) > 1e-9:
                    cmd += ["--sample_cfg_scale", str(_cfg)]
                    _neg = (self.sample_negative_var.get().strip()
                            if getattr(self, "sample_negative_var", None) else "")
                    if _neg:
                        cmd += ["--sample_negative", _neg]
                if _at_first:
                    cmd.append("--sample_at_first")
            elif prompt_file and (_every_n > 0 or _at_first) and not _te:
                self.update_console("[samples] previews need the Qwen3-VL-32B text encoder path "
                                    "— set it in Preferences. Training continues without them.\n")
        # Resumable state saving + resume — identical flag names across all three families.
        cmd += self._state_flags()
        resume_path = (self.settings.get("RESUME_TRAINING") or "").strip()
        if resume_path:
            cmd += ["--resume", resume_path]
        # Adaptive LR is RETIRED for MiniMax (Peter, 9 Aug): ticking it silently disabled the
        # governor and warmup (both defer to it), quietly dismantling the stability stack. The
        # control is hidden under this family and a stale saved ADAPTIVE_LR=True is deliberately
        # ignored here — the governor owns the schedule.
        # Grad clipping (Optimizer section). 1.0 is the trainer default — send only when the
        # user changed it, keeping the launched command a faithful record otherwise.
        _mgn = str(self.settings.get("MAX_GRAD_NORM", "") or "").strip()
        if _mgn:
            try:
                if abs(float(_mgn) - 1.0) > 1e-9:
                    cmd += ["--max_grad_norm", str(float(_mgn))]
            except ValueError:
                pass
        # Optimizer LOCKED to adamw (Peter, 9 Aug): full-precision state was the single biggest
        # likeness change measured on H3 — 8-bit state costs fine detail for 1.9 GB. The dropdown
        # is hidden under this family; whatever the shared setting holds is overridden here.
        cmd += ["--optimizer_type", "adamw"]
        _opt_args = str(self.settings.get("OPTIMIZER_ARGS", "") or "").strip()
        if _opt_args:
            cmd += ["--optimizer_args", _opt_args]
        # Output metadata (Other Options → Metadata) — recorded in the saved LoRA.
        for _mkey, _mflag in (("METADATA_TITLE", "--metadata_title"),
                              ("METADATA_AUTHOR", "--metadata_author"),
                              ("METADATA_DESCRIPTION", "--metadata_description"),
                              ("METADATA_LICENSE", "--metadata_license"),
                              ("METADATA_TAGS", "--metadata_tags")):
            _mval = str(self.settings.get(_mkey, "") or "").strip()
            if _mval:
                cmd += [_mflag, _mval]
        _mtrig = self.settings.get("METADATA_TRIGGER_PHRASE", "").strip() or \
            (self.caption_trigger_var.get().strip() if hasattr(self, "caption_trigger_var") else "")
        if _mtrig and _mtrig.lower() != "trigger_word":
            cmd += ["--metadata_trigger_phrase", _mtrig]
        return cmd

    # === Pause / Resume support ===

    def _pause_flag_path(self) -> str:
        """Path to the pause sentinel file in the current output directory."""
        out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        return os.path.join(out_dir, ".pause_requested")

    def _paused_sidecar_path(self) -> str:
        """Path to the JSON sidecar that records paused-state metadata."""
        out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        return os.path.join(out_dir, ".fizgig_paused.json")

    def _refresh_training_buttons(self):
        """Show/hide Pause and Resume buttons based on self.training_state."""
        if not hasattr(self, "training_state"):
            self.training_state = "idle"
        # While a run is active the Start button queues instead of starting — say so on
        # the button itself rather than surprising the user with a popup.
        try:
            self._start_training_btn.config(
                text="Queue Train" if self.training_state in ("running", "pausing")
                else "Start Training")
        except Exception:
            pass
        # Every state transition passes through here, so it's the one hook that keeps an
        # OPEN queue window truthful (finish, advance, failure-hold, pause) — the render
        # no-ops when the window isn't up.
        try:
            self._render_queue_window()
        except Exception:
            pass
        # Pause: visible while running (Krea 2 now saves full state at the epoch boundary, so
        # graceful Pause/Resume works the same as Klein).
        if self.training_state == "running":
            try: self._pause_training_btn.pack(side=tk.LEFT, padx=(0, 12), after=self._start_training_btn)
            except Exception: pass
        else:
            try: self._pause_training_btn.pack_forget()
            except Exception: pass
        # Resume: visible while paused
        if self.training_state == "paused":
            try: self._resume_training_btn.pack(side=tk.LEFT, padx=(0, 12), after=self._start_training_btn)
            except Exception: pass
        else:
            try: self._resume_training_btn.pack_forget()
            except Exception: pass

    def _pause_training(self):
        """Request a graceful pause — trainer will save state at end of current epoch and exit."""
        if not getattr(self, "current_process", None) or self.current_process.poll() is not None:
            messagebox.showinfo("Not Running", "No active training to pause.")
            return
        if getattr(self, "training_state", "idle") != "running":
            return
        try:
            os.makedirs(os.path.dirname(self._pause_flag_path()) or ".", exist_ok=True)
            open(self._pause_flag_path(), "w").close()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write pause flag:\n{e}")
            return
        self.update_console(
            "\n=== PAUSE REQUESTED — trainer will save full state and exit cleanly at end of current epoch. "
            "GPU memory will be freed. Click Resume Training afterwards to continue. ===\n\n"
        )
        messagebox.showinfo(
            "Pause Requested",
            "Pause queued. The trainer will finish the CURRENT epoch, save full state, "
            "and exit cleanly to free GPU memory.\n\n"
            "Click Resume Training afterwards to continue.",
        )
        self.training_state = "pausing"
        self._refresh_training_buttons()

    def _detect_latest_state_dir(self):
        """Find the highest-numbered <output_name>-NNNNNN-state/ directory in the output dir."""
        import re as _re
        out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        out_name = self.settings.get("LORA_NAME", "") or ""
        if not out_name or not os.path.isdir(out_dir):
            return None
        pattern = _re.compile(rf"^{_re.escape(out_name)}-(\d{{6}})-state$")
        candidates = []
        try:
            for entry in os.listdir(out_dir):
                m = pattern.match(entry)
                # training_state.json is the save's commit marker (written last) — a dir without
                # it is a partial save from a crashed write, not a state. Skipping it here means
                # Resume lands on the previous GOOD state instead of a refusal.
                if (m and os.path.isdir(os.path.join(out_dir, entry))
                        and os.path.isfile(os.path.join(out_dir, entry, "training_state.json"))):
                    candidates.append((int(m.group(1)), entry))
        except Exception:
            return None
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return os.path.join(out_dir, candidates[0][1])

    def _detect_latest_ft_checkpoint(self):
        """The FT twin of _detect_latest_state_dir: highest-numbered
        <output_name>-NNNNNN.safetensors in the output dir. Fine-tunes leave full
        checkpoints, never state dirs — the checkpoint IS the continuation point."""
        import re as _re
        out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
        out_name = self.settings.get("LORA_NAME", "") or ""
        if not out_name or not os.path.isdir(out_dir):
            return None
        pattern = _re.compile(rf"^{_re.escape(out_name)}-(\d{{6}})\.safetensors$")
        candidates = []
        try:
            for entry in os.listdir(out_dir):
                m = pattern.match(entry)
                if m and os.path.isfile(os.path.join(out_dir, entry)):
                    candidates.append((int(m.group(1)), entry))
        except Exception:
            return None
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return os.path.join(out_dir, candidates[0][1])

    def _ft_resume_active(self):
        """The armed fine-tune continuation for THIS launch, or None.

        Guarded twice: the run being launched must still be a fine-tune of the same family
        (unticking the FT box means the user wants something else), and its output name must
        match the paused run's — a queued job launched via 'Start next now' while an FT
        continuation is armed must not consume another run's checkpoint."""
        _fr = getattr(self, "_ft_resume", None)
        if not _fr:
            return None
        if self._is_krea2_arch():
            _on = bool(getattr(self, "krea2_finetune_var", None) and self.krea2_finetune_var.get())
        elif self._is_minimax_arch():
            _on = bool(getattr(self, "minimax_finetune_var", None) and self.minimax_finetune_var.get())
        else:
            _on = False
        if not _on:
            return None
        if str(self.settings.get("LORA_NAME", "")) != str(_fr.get("output_name", "")):
            return None
        return _fr

    POD_STOP_COUNTDOWN = 120   # seconds

    def _maybe_stop_pod_after_training(self):
        """Offer to stop a rented pod once a run has finished on its own.

        Never silent and never immediate: the point is to stop billing on an UNATTENDED finish, so
        anyone actually sitting there must be able to stop it happening. A countdown they can
        cancel does both."""
        if not _running_on_pod():
            return
        if str(self.prefs_vars.get("runpod_stop_when_done", tk.StringVar()).get()).strip() != "1":
            return

        win = tk.Toplevel(self.master)
        win.title("Training finished — stopping pod")
        win.configure(bg=BG_COLOR)
        win.transient(self.master)
        win.protocol("WM_DELETE_WINDOW", lambda: None)
        win.resizable(False, False)

        # Cancel packed BOTTOM first so a long message can never push it off the edge (v2.8.5).
        row = ttk.Frame(win)
        row.pack(side=tk.BOTTOM, pady=(6, 14))

        tk.Label(win, text="Training finished", font=(FONT_FAMILY, 12, "bold"),
                 fg=COLORS["text_primary"], bg=BG_COLOR).pack(anchor=tk.W, padx=18, pady=(16, 2))
        msg = tk.Label(win, text="", font=(FONT_FAMILY, 10), fg=COLORS["text_explain"],
                       bg=BG_COLOR, wraplength=460, justify=tk.LEFT)
        msg.pack(anchor=tk.W, padx=18, pady=(0, 6))
        tk.Label(win,
                 text="Your LoRA and everything else under /workspace is on the persistent volume "
                      "and will still be there next time.",
                 font=(FONT_FAMILY, 9, "italic"), fg=COLORS["text_explain"], bg=BG_COLOR,
                 wraplength=460, justify=tk.LEFT).pack(anchor=tk.W, padx=18)

        state = {"left": self.POD_STOP_COUNTDOWN, "cancelled": False}

        def cancel():
            state["cancelled"] = True
            self.update_console("[pod] auto-stop cancelled — pod left running.\n")
            try:
                win.destroy()
            except Exception:
                pass

        ttk.Button(row, text="Keep the pod running", command=cancel).pack()

        def tick():
            if state["cancelled"]:
                return
            if state["left"] <= 0:
                try:
                    win.destroy()
                except Exception:
                    pass
                self._stop_this_pod()
                return
            msg.config(text=f"Stopping this pod in {state['left']}s to stop it billing.\n"
                            f"Cancel below if you want to keep working.")
            state["left"] -= 1
            win.after(1000, tick)

        tick()
        try:
            win.update_idletasks()
            win.grab_set()
        except Exception:
            pass

    def _stop_this_pod(self):
        """Stop this pod through RunPod's API. Reports what happened either way.

        Uses the GraphQL endpoint directly rather than runpodctl: runpodctl tries to sync SSH keys
        as a side effect of being configured, which fails noisily and has nothing to do with
        stopping a machine.

        The key RunPod injects as RUNPOD_API_KEY is POD-SCOPED and cannot manage pods — verified
        on a live pod, where `runpodctl pod list` returns 403 both before and after configuring
        with it. So this needs an account key the user supplies themselves."""
        key = self._pod_stop_key()
        pid = os.environ.get("RUNPOD_POD_ID", "").strip() or _pod_id()
        if not key:
            self.update_console(
                "[pod] auto-stop is on but no API key is set, so the pod is still running.\n"
                "[pod] Add RUNPOD_STOP_API_KEY to your template (RunPod > Settings > API Keys).\n"
                "[pod] The key RunPod provides automatically is pod-scoped and cannot stop pods.\n")
            return
        self.update_console(f"[pod] stopping pod {pid or '(unknown id)'}…\n")
        try:
            import json as _json
            import urllib.request as _u
            body = _json.dumps({
                "query": "mutation($id: String!) { podStop(input: {podId: $id}) "
                         "{ id desiredStatus } }",
                "variables": {"id": pid},
            }).encode()
            req = _u.Request(f"https://api.runpod.io/graphql?api_key={key}", data=body,
                             headers={"Content-Type": "application/json"})
            with _u.urlopen(req, timeout=45) as resp:
                payload = _json.loads(resp.read().decode())
            if payload.get("errors"):
                msg = payload["errors"][0].get("message", "unknown error")
                self.update_console(f"[pod] RunPod refused the stop: {msg}\n"
                                    f"[pod] Stop it from the dashboard to stop billing.\n")
            else:
                self.update_console("[pod] stop requested — this pod is shutting down.\n")
        except Exception as e:
            self.update_console(f"[pod] auto-stop failed ({type(e).__name__}: {e}). "
                                f"Stop it from the dashboard to stop billing.\n")

    def _on_training_subprocess_exited(self, return_code: int):
        """Called from check_process when the training subprocess ends. Routes to paused or idle."""
        # Captured BEFORE the branches below rewrite it — a pause is the one case that exits 0,
        # and telling it apart from a completed run is the whole basis of the auto-stop decision.
        was_state = getattr(self, "training_state", "idle")
        # Clean up pause flag if still present
        try:
            flag = self._pause_flag_path()
            if os.path.exists(flag):
                os.remove(flag)
        except Exception:
            pass
        # Only a run that finished on its own. Pause exits 0 too (state "pausing"); Stop and
        # crashes arrive non-zero. Each of the three conditions excludes a real case, and getting
        # it wrong shuts the machine down under someone who is still using it.
        # Every run exit invalidates whatever advance/retry timers were armed before it —
        # a stale tick must never fire into the state this exit is about to establish.
        self._cancel_pending_queue_advance()
        _queue_advancing = False
        if return_code == 0 and was_state == "running":
            if getattr(self, "training_queue", None):
                # Queue takes precedence over pod auto-stop: the pod must stay up until the
                # LAST queued run finishes — that final run's clean exit lands here with an
                # empty queue and fires the auto-stop as usual.
                _queue_advancing = True
                self._queue_busy_retries = 0
                self.update_console(f"\n[queue] run finished — next of "
                                    f"{len(self.training_queue)} queued run(s) starts in 5 s.\n")
                self._schedule_queue_advance(5000)
            else:
                self._maybe_stop_pod_after_training()
        elif getattr(self, "training_queue", None):
            # Failure, Stop, or Pause with runs still waiting: never cascade into the queue —
            # a crash loop through N queued runs would burn hours producing nothing. The queue
            # holds; the user restarts it from the queue window.
            if was_state == "pausing" and return_code == 0:
                self.update_console(f"[queue] run paused — {len(self.training_queue)} queued "
                                    f"run(s) are HELD. Resume the paused run from the Training "
                                    f"tab first; the queue continues after it finishes.\n")
            else:
                self.update_console(
                    f"[queue] run did not finish cleanly (exit {return_code}) — "
                    f"{len(self.training_queue)} queued run(s) are HELD. The FAILED run is not "
                    f"in the queue: its settings are still loaded in the Training tab (fix and "
                    f"Start Training to retry it), or open the queue (📋, bottom right) and "
                    f"'Start next now' to skip to the next job.\n")
        if getattr(self, "training_state", "idle") == "pausing" and return_code == 0:
            # Successful graceful exit — record paused state. A fine-tune leaves a full
            # checkpoint rather than a state dir (state dirs would hold only the inert
            # LoRA), so the paused artifact is detected per the launch's stamped mode.
            _ft_pause = bool(getattr(self, "_launched_ft_family", None))
            if _ft_pause:
                state_dir = self._detect_latest_ft_checkpoint()
            else:
                state_dir = self._detect_latest_state_dir()
            if state_dir is None:
                self.update_console(
                    "[pause] WARN: no fine-tune checkpoint found after pause exit. Treating as idle.\n"
                    if _ft_pause else
                    "[pause] WARN: no state directory found after pause exit. Treating as idle.\n")
                self.training_state = "idle"
            else:
                self.paused_state_path = state_dir
                self.paused_mode = "ft" if _ft_pause else "state"
                # Persist sidecar so paused state survives GUI restart
                try:
                    import json as _json
                    out_dir = self.settings.get("LORA_OUTPUT_DIR", "") or "."
                    sidecar = {
                        "mode": self.paused_mode,
                        "state_path": state_dir,
                        "output_name": self.settings.get("LORA_NAME", ""),
                        "dataset_config": self.settings.get("DATASET_CONFIG", ""),
                        "network_dim": str(self.settings.get("NETWORK_DIM", "")),
                        "network_alpha": str(self.settings.get("NETWORK_ALPHA", "")),
                        "max_train_epochs": str(self.settings.get("MAX_TRAIN_EPOCHS", "")),
                    }
                    with open(self._paused_sidecar_path(), "w") as f:
                        _json.dump(sidecar, f, indent=2)
                except Exception as e:
                    self.update_console(f"[pause] WARN: failed to write sidecar: {e}\n")
                self.training_state = "paused"
                self.update_console(
                    f"\n=== PAUSED — fine-tune checkpoint saved at {os.path.basename(state_dir)}. "
                    f"Click Resume Training to continue. ===\n\n"
                    if _ft_pause else
                    f"\n=== PAUSED — state saved at {state_dir}. Click Resume Training to continue. ===\n\n"
                )
        else:
            self.training_state = "idle"
        # A finished run must never leave its resume path armed: the next "fresh" Start
        # would silently continue the old LoRA from its saved state (restored optimizer/
        # RNG/scheduler) under a new output name, and skip re-caching a changed dataset.
        # The Resume button re-injects the right path itself when the user wants it.
        try:
            _entry = self.entries.get("RESUME_TRAINING")
            if _entry is not None and _entry.get().strip():
                _entry.delete(0, tk.END)
            self.settings["RESUME_TRAINING"] = ""
        except Exception:
            pass
        # Same hygiene for the FT twin: the armed continuation lives until its run ends
        # (it must survive the async caption-worker launch path), then dies here.
        self._ft_resume = None
        self._refresh_training_buttons()

    def _resume_training(self):
        """Re-launch training from the latest paused state (state dir, or FT checkpoint).

        A LoRA pause resumes via --resume <state dir>. A rotation fine-tune has no state
        dirs — its continuation is a fresh run whose --dit is the pause checkpoint, whose
        rotation cycle picks up at the window stamped in that checkpoint's metadata, and
        whose epoch count is what's left of the original total. The optimizer's second
        moments reset across that hop — the same reset every rotation boundary already
        performs, so it costs what one rotation costs."""
        if getattr(self, "training_state", "idle") != "paused":
            messagebox.showinfo("Not Paused", "No paused training to resume.")
            return
        state_path = getattr(self, "paused_state_path", None)
        if getattr(self, "paused_mode", "state") == "ft":
            if not state_path or not os.path.isfile(state_path):
                messagebox.showerror("Error", f"Paused fine-tune checkpoint not found:\n{state_path}")
                return
            next_window, epochs_done = ft_checkpoint_continuation(state_path)
            try:
                total = int(str(self.entries["MAX_TRAIN_EPOCHS"].get()).strip() or 0)
            except (KeyError, ValueError, TypeError):
                total = 0
            if total - epochs_done <= 0:
                messagebox.showinfo(
                    "Nothing left to train",
                    f"This fine-tune has already trained {epochs_done} epoch(s) — at or past "
                    f"Max Train Epochs ({total}).\n\n"
                    f"{os.path.basename(state_path)} IS the finished model — deploy it as-is.\n\n"
                    f"To train it further, raise Max Train Epochs above {epochs_done} and click "
                    "Resume Training again.")
                return
            # Arm the one-shot continuation. The command builders consume it (guarded by
            # family + output name); the exit handler clears it when the run ends.
            self._ft_resume = {"checkpoint": state_path, "next_window": next_window,
                               "epochs_done": epochs_done,
                               "output_name": self.settings.get("LORA_NAME", "")}
            self.update_console(
                f"\n=== RESUMING fine-tune from {os.path.basename(state_path)} — "
                f"{total - epochs_done} epoch(s) remaining, rotation window {next_window} ===\n\n")
        else:
            if not state_path or not os.path.isdir(state_path):
                messagebox.showerror("Error", f"Paused state directory not found:\n{state_path}")
                return
            # Inject resume path into settings + entry field, then reuse the standard start_training flow
            self.settings["RESUME_TRAINING"] = state_path
            try:
                entry = self.entries.get("RESUME_TRAINING")
                if entry is not None:
                    entry.delete(0, tk.END)
                    entry.insert(0, state_path)
            except Exception:
                pass
            self.update_console(f"\n=== RESUMING from {state_path} ===\n\n")
        self.start_training()
        # Only a start that actually LAUNCHED consumes the pause. start_training can decline
        # (validation, disk headroom, epochs-left) — destroying the sidecar and flipping the
        # state beforehand stranded a declinable resume with no way back to the paused run.
        _proc = getattr(self, "current_process", None)
        if getattr(self, "training_state", "idle") == "running" and _proc is not None and _proc.poll() is None:
            try:
                sidecar = self._paused_sidecar_path()
                if os.path.exists(sidecar):
                    os.remove(sidecar)
            except Exception:
                pass
        else:
            self.training_state = "paused"
            self._refresh_training_buttons()
            self.update_console("[resume] start declined — the run is still PAUSED and can be "
                                "resumed once the issue above is fixed.\n")

    def _check_for_paused_state_on_startup(self):
        """On GUI launch, detect a leftover paused state and restore the Resume button."""
        try:
            sidecar = self._paused_sidecar_path()
            if not os.path.exists(sidecar):
                return
            import json as _json
            with open(sidecar, "r") as f:
                meta = _json.load(f)
            state_path = meta.get("state_path", "")
            # "ft" pauses point at a full checkpoint FILE (fine-tunes have no state dirs);
            # LoRA pauses point at a state DIRECTORY. Validate whichever this one is.
            _mode = str(meta.get("mode", "state") or "state")
            _ok = bool(state_path) and (os.path.isfile(state_path) if _mode == "ft"
                                        else os.path.isdir(state_path))
            if _ok:
                self.paused_state_path = state_path
                self.paused_mode = _mode
                # Restore the paused run's frozen dataset config (#98) so a cross-restart
                # resume trains the dataset it started with — the resume launch keeps an
                # existing snapshot instead of re-freezing whatever the Start tab shows.
                _dc = str(meta.get("dataset_config", "") or "")
                if (os.path.basename(os.path.dirname(_dc)) == self._RUN_SNAPSHOT_DIRNAME
                        and os.path.isfile(_dc)):
                    self.settings["DATASET_CONFIG"] = _dc
                self.training_state = "paused"
                self._refresh_training_buttons()
                self.update_console(
                    f"=== Paused {'fine-tune' if _mode == 'ft' else 'training'} detected: "
                    f"{meta.get('output_name','?')} "
                    f"at {os.path.basename(state_path)}. Click Resume Training to continue. ===\n"
                )
        except Exception:
            pass

    def _on_app_close(self):
        """WM_DELETE_WINDOW: never orphan a live training subprocess on window close."""
        proc = getattr(self, "current_process", None)
        try:
            running = proc is not None and proc.poll() is None
        except Exception:
            running = False
        if running:
            if not messagebox.askyesno(
                "Training in progress",
                "A training run is active.\n\n"
                "Close Fizgig and STOP the training run?\n\n"
                "(To keep training, click No — or use Pause Training first for a clean, "
                "resumable exit.)"
            ):
                return
            try:
                self.stop_training()
            except Exception:
                pass
        # Final settings snapshot — some fields only persist via debounced traces or other
        # tabs' events, so closing mid-edit would otherwise drop the last change.
        try:
            self._save_last_used_paths()
        except Exception:
            pass
        try:
            self._stop_caption_worker(silent=True, wait=False)
        except Exception:
            pass
        try:
            self.master.destroy()
        except Exception:
            pass

    def stop_training(self):
        """Stop the current running process"""
        # Stop samples watcher
        self.stop_samples_watcher()
        # A user Stop invalidates any armed queue-advance/retry timer immediately — the
        # exit handler bumps too, but for non-pipeline kills this is the only bump.
        try:
            self._cancel_pending_queue_advance()
        except Exception:
            pass
        # Snapshot: check_process's worker nulls self.current_process the instant the kill
        # lands, racing the .wait(timeout=5) below into an AttributeError on None.
        _proc = self.current_process
        if _proc and _proc.poll() is None:
            try:
                if os.name == 'nt':
                    # CREATE_NO_WINDOW prevents CTRL_BREAK_EVENT from working,
                    # so terminate the process tree via taskkill instead.
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(_proc.pid)],
                        capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                else:
                    os.killpg(os.getpgid(_proc.pid), signal.SIGTERM)
            except Exception as e:
                self.update_console("Error stopping process: " + str(e) + "\n")
            try:
                _proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    _proc.kill()
                    _proc.wait()
                except Exception as e:
                    self.update_console("Error killing process: " + str(e) + "\n")
            self.current_process = None
            if self.training_thread:
                self.training_thread.join(timeout=1)
                self.training_thread = None
            self.update_console("Training stopped\n")
        else:
            self.update_console("No active process to stop\n")
