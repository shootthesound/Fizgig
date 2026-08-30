import os

import tkinter as tk
from tkinter import filedialog, messagebox

from fizgig_gui.core.domain.architectures import ARCHITECTURES
from fizgig_gui.core.domain.minimax_math import minimax_block_spec, MINIMAX_NUM_BLOCKS, minimax_lownoise_to_shift, minimax_train_base


class ConsoleValidationMixin:
    def show_context_menu(self, event):
        """Show context menu on right-click"""
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def copy_selected_text(self):
        """Copy selected text to clipboard"""
        if self.console_output.selection_get():
            self.master.clipboard_clear()
            self.master.clipboard_append(self.console_output.selection_get())

    def browse_directory(self, setting_name):
        path = filedialog.askdirectory()
        if path:
            self.entries[setting_name].delete(0, tk.END)
            self.entries[setting_name].insert(0, path)

    def on_mousewheel(self, event):
        """Handle scroll event"""
        if self.console_output.yview()[1] < 1.0:
            self.user_scrolled = True
        else:
            self.user_scrolled = False

    def update_console(self, line):
        """Update training console — only auto-scroll if user was already at the bottom.
        Uses the widget's own yview() position as the authoritative signal; the older
        self.user_scrolled flag sometimes got out of sync with actual widget state."""
        self._append_global_log(line)
        try:
            at_bottom = self.console_output.yview()[1] >= 0.999
        except Exception:
            at_bottom = True
        self.console_output.configure(state="normal")
        self.console_output.insert(tk.END, line)
        if at_bottom:
            self.console_output.see(tk.END)
        self.console_output.configure(state="disabled")

        # Detect CUDA OOM. A Krea 2 training-step OOM is best fixed by the 4-bit (NF4) base (it
        # frees far more than swap); otherwise suggest more block swap. (Preview OOMs are caught in
        # the trainer, auto-disable previews, and don't print this literal — so this only fires on a
        # genuine training-step OOM.)
        # The preview resolution ladder settled somewhere (trainer prints this per
        # downgrade): write it back into the Samples tab so the NEXT run starts at the size
        # that fit, instead of re-walking the ladder from the configured resolution. It
        # persists exactly like a hand edit of Width/Height.
        if "[preview] resolution settled:" in line:
            import re as _re_res
            _m = _re_res.search(r"resolution settled: (\d+)x(\d+)", line)
            if _m and hasattr(self, "sample_width_var"):
                self.sample_width_var.set(_m.group(1))
                self.sample_height_var.set(_m.group(2))
                self.update_console(f"[samples] preview resolution saved as the new "
                                    f"default: {_m.group(1)}x{_m.group(2)} — future runs "
                                    f"start there.\n")

        if "CUDA out of memory" in line or "OutOfMemoryError" in line:
            if not getattr(self, "_oom_warning_shown", False):
                self._oom_warning_shown = True
                # Read state WITHOUT side effects: _parse_blocks_swap() on "Auto" runs the
                # auto strategy (GPU probe + can flip the 4-bit toggle) — it mutated the
                # training config as a side effect of reporting an error, and did so BEFORE
                # reading the nf4 flag it then tested, hiding the Krea 2 advice. Mid-run the
                # probe is meaningless anyway (the trainer holds the VRAM).
                _raw_swap = self.entries["BLOCKS_SWAP"].get().strip()
                if _raw_swap.lower().startswith("auto"):
                    current_swap, _swap_disp = 0, "Auto"
                else:
                    import re as _re
                    _m = _re.match(r"\d+", _raw_swap)
                    current_swap = int(_m.group()) if _m else 0
                    _swap_disp = str(current_swap)
                nf4_on = getattr(self, "quant_4bit_var", None) and self.quant_4bit_var.get()
                if self._is_krea2_arch() and not nf4_on:
                    messagebox.showwarning("Out of Memory",
                        "CUDA ran out of memory during Krea 2 training.\n\n"
                        "The biggest win on a smaller card is the 4-bit (NF4) Base toggle in the "
                        "Memory & FP8 section: it shrinks the frozen base from ~14 GB to ~5.6 GB, so a "
                        "full LoRA trains on a 10-12 GB card with no block swap.\n\n"
                        "(Block swap helps too, but 4-bit frees far more. In-training previews on the "
                        "Turbo need a bigger card — if they can't fit they auto-disable and training "
                        "continues; evaluate the saved LoRA in ComfyUI.)")
                else:
                    messagebox.showwarning("Out of Memory",
                        f"CUDA ran out of memory during training.\n\n"
                        f"Current Block Swap: {_swap_disp}\n\n"
                        f"Try increasing Block Swap on the Training tab "
                        f"(Memory & FP8 section) to move more blocks to CPU. "
                        f"If set to Auto, switch to a manual value like "
                        f"{min(current_swap + 4, 16)}.")

    def _browse_context_lora(self):
        """File picker for the Context LoRA, filtered to .safetensors."""
        path = filedialog.askopenfilename(
            title="Select Context LoRA",
            filetypes=[("SafeTensors", "*.safetensors"), ("All files", "*.*")],
            initialdir=self._lora_initialdir(),
        )
        if path:
            self.entries["CONTEXT_LORA_PATH"].delete(0, tk.END)
            self.entries["CONTEXT_LORA_PATH"].insert(0, path)

    def browse_file(self, setting_name, input_type):
        # Resume Training points at a saved state dir, which lives under the LoRA
        # output folder — open the Browse there so users don't hunt for it.
        initial = self._pref_initialdir("lora_output_dir") if setting_name == "RESUME_TRAINING" else ""
        if input_type == "directory":
            path = filedialog.askdirectory(initialdir=initial)
        else:
            path = filedialog.askopenfilename(initialdir=initial)
        if path:
            self.settings[setting_name] = path
            self.entries[setting_name].delete(0, tk.END)
            self.entries[setting_name].insert(0, self.settings[setting_name])

    def _tidy_lora_name(self):
        """Clean the LoRA Name field in place. Returns (name, error or None).

        The name becomes a filename, but not until the FIRST CHECKPOINT SAVE — an epoch in. A
        stray character (a newline off a paste is the common one) trained for sixteen minutes
        and then died inside safetensors with a bare OS error that named neither the setting nor
        the character; and since the LoRA is written before the state dir, there was nothing left
        to resume from (#70).

        What has one obvious intent is fixed silently — surrounding whitespace, control
        characters, trailing dots Windows discards anyway — and WRITTEN BACK to the widget, so
        the field, the preset that gets persisted, the queue entry and --output_name cannot
        disagree about what this run is called. Everything else is refused by name.
        """
        entry = self.entries.get("LORA_NAME")
        raw = entry.get() if entry is not None else ""
        name = "".join(c for c in raw if c >= " ").strip().rstrip(".").strip()
        if entry is not None and name != raw:
            entry.delete(0, tk.END)
            entry.insert(0, name)
        if not name:
            return name, "LoRA name cannot be empty"
        bad = next((c for c in name if c in '<>:"|?*/\\'), None)
        if bad is not None:
            return name, (f"LoRA name cannot contain '{bad}' — file names can't include that "
                          f"character. Use letters, numbers, spaces, - _ or .")
        return name, None

    def validate_inputs(self):
        """Validate all inputs before starting training"""
        errors = []

        # Get current architecture config
        arch = self.architecture_var.get()
        config = ARCHITECTURES.get(arch, ARCHITECTURES["Flux 2 Klein Base 9B"])

        # Free-text numeric fields: a bare int()/float() further down the launch path used
        # to raise inside a Tk callback — swallowed to stderr, invisible under the windowed
        # launcher, and the Start button just "did nothing forever". Validate them HERE with
        # a message naming the field. Batch Size 0/blank was the sharpest: it reached
        # math.ceil(len(bucket)/batch_size) deep in the dataloader minutes after launch.
        def _check_num(label, raw, cast, minimum=None):
            raw = str(raw).strip()
            try:
                v = cast(raw)
            except (TypeError, ValueError):
                errors.append(f"{label} must be a number (got {raw!r})")
                return
            if minimum is not None and v < minimum:
                errors.append(f"{label} must be at least {minimum} (got {raw})")

        # Learning Rate box is ignored (and greyed) while Adaptive LR is on — don't let a
        # stale value in a disabled box block Start.
        if not (hasattr(self, 'adaptive_lr_var') and self.adaptive_lr_var.get()):
            _check_num("Learning Rate", self.entries["LEARNING_RATE"].get(), float, 0)
        _check_num("Network Dim (Rank)", self.entries["NETWORK_DIM"].get(), int, 1)
        _check_num("Network Alpha", self.entries["NETWORK_ALPHA"].get(), float, 0)
        if self._network_type_is_lokr():
            _check_num("LoKR Factor", self.entries["LOKR_FACTOR"].get(), int, 2)
        # Blocks to Train is free text, so a typo is caught HERE rather than after the 21 GB base
        # has streamed in — and a queued run must never fail an hour later on a bad spec.
        if self._is_minimax_arch():
            _spec = minimax_block_spec(self.entries["MINIMAX_BLOCKS"].get())
            # Likeness mode ignores (and disables) the box — a stale typo in it must not
            # block the launch.
            if self.entries["MINIMAX_LIKENESS_OPT"].get():
                _spec = "all"
            if _spec.lower() != "all":
                try:
                    from fizgig.minimax.trainer import parse_block_spec
                    parse_block_spec(_spec, MINIMAX_NUM_BLOCKS)
                except ValueError as e:
                    errors.append(f"Blocks to Train: {e}")
                except ImportError:
                    pass
            if minimax_lownoise_to_shift(self.entries["MINIMAX_LOWNOISE_PCT"].get()) is None:
                errors.append("Low-noise training must be a number above 0 and below 100 "
                              f"(got {self.entries['MINIMAX_LOWNOISE_PCT'].get()!r})")
            _slow_spec = str(self.entries["MINIMAX_SLOW_BLOCKS"].get() or "").strip()
            if _slow_spec:
                try:
                    from fizgig.minimax.trainer import parse_block_spec
                    parse_block_spec(_slow_spec, MINIMAX_NUM_BLOCKS)
                except ValueError as e:
                    errors.append(f"Slower LR for blocks: {e}")
                except ImportError:
                    pass
                _check_num("Slower LR multiplier",
                           self.entries["MINIMAX_SLOW_LR_SCALE"].get(), float, 0)
        _check_num("Max Train Epochs", self.entries["MAX_TRAIN_EPOCHS"].get(), int, 1)
        _check_num("Save Every N Epochs", self.entries["SAVE_EVERY_N_EPOCHS"].get(), int, 1)
        _check_num("Seed", self.entries["SEED"].get(), int)
        _check_num("LoRA+ LR Ratio", self.entries["LORA_LR_RATIO"].get(), int, 1)
        _check_num("Gradient Accumulation", self.entries["GRADIENT_ACCUMULATION"].get(), int, 1)
        _check_num("Max Grad Norm", self.entries["MAX_GRAD_NORM"].get(), float, 0)
        _check_num("Network Dropout", self.entries["NETWORK_DROPOUT"].get(), float, 0)
        _check_num("Batch Size (Dataset)", self.dataset_batch_size_var.get(), int, 1)
        # An unparseable megapixels value makes the TOML auto-saver skip its rewrite
        # SILENTLY (#98 follow-up) — catch it here with a named error instead of the
        # launch-time stale-config refusal.
        _check_num("Target Megapixels (Dataset)", self.dataset_megapixels_var.get(), float, 0)
        if "KEEP_LAST_N_STATES" in self.entries:
            _check_num("Keep Last (states)", self.entries["KEEP_LAST_N_STATES"].get(), int, 1)

        # Check required paths exist (sources: prefs_vars for model paths, hidden var for dataset)
        dataset_config = self._get_path("DATASET_CONFIG")
        if not dataset_config:
            errors.append("Dataset config file path is empty — set the training image folder on the Start tab")
        elif not os.path.exists(dataset_config):
            errors.append(f"Dataset config file does not exist: {dataset_config}")

        if config.get("is_minimax"):
            # MiniMax H3 reads its three model paths from Preferences (minimax_*). All are
            # required: the DiT to train (pruned int8 or bf16), the video VAE + Qwen3-VL-32B TE.
            for pref_key, label in (
                ("minimax_dit", "MiniMax H3 DiT"),
                ("minimax_vae", "MiniMax H3 Video VAE"),
                ("minimax_text_encoder", "Qwen3-VL-32B text encoder"),
            ):
                path = self._krea2_pref(pref_key)
                if not path:
                    errors.append(f"{label} path is empty (set it on the Preferences tab)")
                elif not os.path.exists(path):
                    errors.append(f"{label} file does not exist: {path}")
            # Training Base = ref2va needs the ref2va model actually set — without this check
            # the command builder would silently fall back to fl2va, the one thing the user
            # explicitly asked it not to train on.
            if (getattr(self, "minimax_train_base_var", None)
                    and minimax_train_base(self.minimax_train_base_var.get()) == "ref2va"
                    and not self._krea2_pref("minimax_ref_dit")):
                errors.append("Training Base is set to Reference (ref2va) but 'DiT (reference)' "
                              "is empty on the Preferences tab. Easiest fix: on the Preferences "
                              "tab tick 'Include the reference DiT (+21 GB)' and hit "
                              "'⬇ Download models for me' — it fetches the model and fills the "
                              "path in for you. Or switch Training Base back to First/last "
                              "frame.")
            # Reference distillation needs the ref2va model and a real reference photo.
            if getattr(self, "minimax_distill_var", None) and self.minimax_distill_var.get():
                if not self._krea2_pref("minimax_ref_dit"):
                    errors.append("Reference distillation needs the ref2va DiT — set "
                                  "'DiT (reference)' on the Preferences tab. It is a different "
                                  "model from the one above and the only H3 build that takes "
                                  "reference images.")
                _check_num("References each", self.entries["MINIMAX_DISTILL_REFS"].get(), int, 1)
            # Multi Concept: each extra folder becomes its own [[datasets]] block, so it has to
            # exist, be distinct, and carry its own captions. The dataset layer refuses two
            # blocks sharing a cache_directory, and the cache path hashes a case-folded,
            # slash-stripped path — so C:\A and c:/a/ are the SAME folder as far as it cares.
            if (getattr(self, "minimax_multiconcept_var", None)
                    and self.minimax_multiconcept_var.get()):
                _seen = {self.image_folder_var.get().strip().lower()
                         .replace("\\", "/").rstrip("/")}
                _extra = [v.get().strip() for v in getattr(self, "_concept_folder_vars", [])]
                if not any(_extra):
                    errors.append("Multi Concept is on but no second subject folder is set — "
                                  "pick one, or turn the mode off.")
                for _f in _extra:
                    if not _f:
                        continue
                    _norm = _f.lower().replace("\\", "/").rstrip("/")
                    if _norm in _seen:
                        errors.append(f"Multi Concept: {_f} is the same folder as another "
                                      f"subject — each needs its own folder.")
                        continue
                    _seen.add(_norm)
                    if not os.path.isdir(_f):
                        errors.append(f"Multi Concept: folder does not exist: {_f}")
                        continue
                    _ext = (self.dataset_caption_ext_var.get().strip() or ".txt")
                    if not any(fn.lower().endswith(_ext.lower()) for fn in os.listdir(_f)):
                        errors.append(
                            f"Multi Concept: no {_ext} captions in {_f}. Caption both folders "
                            f"before training — each subject needs its own trigger word in "
                            f"every caption, or they will blend.")
        elif config.get("is_krea2"):
            # Krea 2 reads its own four model paths from Preferences (krea2_*). The
            # Turbo DiT is only required when in-training previews are enabled.
            krea2_required = [
                ("krea2_raw_dit", "Krea 2 RAW DiT"),
                ("krea2_vae", "Qwen-Image VAE"),
                ("krea2_text_encoder", "Qwen3-VL-4B text encoder"),
            ]
            if self.sample_enabled_var.get():
                # raw_lora engine renders previews on the training DiT + Turbo LoRA — the LoRA
                # is auto-downloaded at training start if missing, so nothing is hard-required
                # here (a failed download degrades to no previews, with console messages, never
                # a blocked run). The classic engine still needs the Turbo checkpoint.
                if self._krea2_preview_engine() != "raw_lora":
                    krea2_required.append(("krea2_turbo_dit", "Krea 2 Turbo DiT (fp8) — needed for previews"))
            for pref_key, label in krea2_required:
                path = self._krea2_pref(pref_key)
                if not path:
                    errors.append(f"{label} path is empty (set it on the Preferences tab)")
                elif not os.path.exists(path):
                    errors.append(f"{label} file does not exist: {path}")
        else:
            vae_model = self._get_path("VAE_MODEL")
            if not vae_model:
                errors.append("VAE model file path is empty (set on the Preferences tab)")
            elif not os.path.exists(vae_model):
                errors.append(f"VAE model file does not exist: {vae_model}")

            dit_model = self._get_path("DIT_MODEL")
            if not dit_model:
                errors.append("DiT model file path is empty (set on the Preferences tab)")
            elif not os.path.exists(dit_model):
                errors.append(f"DiT model file does not exist: {dit_model}")

            # Architecture-specific validation (T5/CLIP are dead for Klein but kept for future flexibility)
            if config["uses_t5"]:
                t5_model = self._get_path("T5_MODEL")
                if not t5_model:
                    errors.append("T5 model file path is empty")
                elif not os.path.exists(t5_model):
                    errors.append(f"T5 model file does not exist: {t5_model}")

            if config["uses_text_encoder"]:
                text_encoder = self._get_path("TEXT_ENCODER")
                if not text_encoder:
                    errors.append("Text encoder file path is empty (set on the Preferences tab)")
                elif not os.path.exists(text_encoder):
                    errors.append(f"Text encoder file does not exist: {text_encoder}")

            if config["uses_clip"]:
                clip_model = self._get_path("CLIP_MODEL")
                if not clip_model:
                    errors.append("CLIP model file path is empty")
                elif not os.path.exists(clip_model):
                    errors.append(f"CLIP model file does not exist: {clip_model}")

        # Validate numeric fields. With Adaptive LR on, the Learning Rate box is IGNORED
        # (the run starts at the geometric midpoint of Min/Max), so only Min < Max matters
        # — the old "starting LR exceeds Max" check no longer applies.
        _adaptive_on = hasattr(self, 'adaptive_lr_var') and self.adaptive_lr_var.get()
        if _adaptive_on:
            try:
                max_lr_str = self.entries["ADAPTIVE_LR_MAX"].get().split(" ")[0]
                min_lr_str = self.entries["ADAPTIVE_LR_MIN"].get().split(" ")[0]
                if float(min_lr_str) >= float(max_lr_str):
                    errors.append(f"Adaptive Min LR ({min_lr_str}) must be lower than Max LR ({max_lr_str}).")
            except (ValueError, KeyError):
                errors.append("Adaptive Min/Max LR must be valid numbers.")
        else:
            try:
                lr = float(self.entries["LEARNING_RATE"].get())
                if lr <= 0:
                    errors.append("Learning rate must be positive")
            except ValueError:
                errors.append("Learning rate must be a valid number")

        # Context LoRA validation (supported by both Klein and Krea 2).
        ctx_path = self.entries.get("CONTEXT_LORA_PATH").get().strip() if "CONTEXT_LORA_PATH" in self.entries else ""
        if ctx_path:
            if not os.path.exists(ctx_path):
                errors.append(f"Context LoRA file does not exist: {ctx_path}")
            elif not ctx_path.lower().endswith(".safetensors"):
                errors.append(f"Context LoRA must be a .safetensors file: {ctx_path}")
            try:
                ctx_strength = float(self.entries["CONTEXT_LORA_STRENGTH"].get())
                if not (0.0 <= ctx_strength <= 2.0):
                    errors.append(f"Context LoRA Strength ({ctx_strength}) must be between 0.0 and 2.0")
            except (ValueError, KeyError):
                errors.append("Context LoRA Strength must be a valid number")

        try:
            network_dim = int(self.entries["NETWORK_DIM"].get())
            if network_dim <= 0:
                errors.append("Network dim must be a positive integer")
        except ValueError:
            errors.append("Network dim must be a valid integer")

        try:
            network_alpha = float(self.entries["NETWORK_ALPHA"].get())
            if network_alpha < 0:
                errors.append("Network alpha must be non-negative")
        except ValueError:
            errors.append("Network alpha must be a valid number")

        try:
            epochs = int(self.entries["MAX_TRAIN_EPOCHS"].get())
            if epochs <= 0:
                errors.append("Max train epochs must be a positive integer")
        except ValueError:
            errors.append("Max train epochs must be a valid integer")

        try:
            save_epochs = int(self.entries["SAVE_EVERY_N_EPOCHS"].get())
            if save_epochs <= 0:
                errors.append("Save every N epochs must be a positive integer")
            # Per-category retirement epoch: blank = never, else a positive whole number.
            _rk = self.entries.get("MIXED_STOP_EPOCH")
            _rv = str(_rk.get() if _rk else "").strip()
            if _rv and (not _rv.isdigit() or int(_rv) <= 0):
                errors.append(f"'Finish one category early: after epoch' must be blank or "
                              f"a positive whole number, not {_rv!r}")
        except ValueError:
            errors.append("Save every N epochs must be a valid integer")

        try:
            blocks_swap = self._parse_blocks_swap()
            if blocks_swap < 0:
                errors.append("Blocks swap must be non-negative")
            elif blocks_swap > config["blocks_swap_max"]:
                errors.append(f"Blocks swap ({blocks_swap}) exceeds maximum for {arch} ({config['blocks_swap_max']})")
        except ValueError:
            errors.append("Blocks swap must be a valid integer")

        _name, _name_error = self._tidy_lora_name()
        if _name_error:
            errors.append(_name_error)

        # Check output directory
        output_dir = self.entries["LORA_OUTPUT_DIR"].get()
        if not output_dir:
            errors.append("LoRA output directory is empty")

        # Check resume path if specified
        resume_path = self.entries["RESUME_TRAINING"].get()
        if resume_path and resume_path.strip() and not os.path.exists(resume_path):
            errors.append(f"Resume training path does not exist: {resume_path}")

        # Check caption files exist in the dataset folder
        image_dir = self.image_folder_var.get().strip()
        caption_ext = self.dataset_caption_ext_var.get().strip()
        # A SET folder that no longer exists is an error, not a skip: the TOML regenerator
        # early-returns on a missing folder, so proceeding trains whatever dataset the TOML
        # last pointed at — silently, under this run's name.
        if image_dir and not os.path.isdir(image_dir):
            errors.append(f"Training image folder does not exist: {image_dir}")
        if image_dir and os.path.isdir(image_dir) and caption_ext:
            import glob as _glob
            # glob.escape is load-bearing here: a folder like "[subject] photos" made this
            # find zero captions and block training with "No caption files found", while the
            # Captions tab (os.listdir) read and wrote that same folder perfectly happily.
            caption_files = _glob.glob(os.path.join(_glob.escape(image_dir), "*" + caption_ext))
            if not caption_files:
                errors.append(
                    f"No caption files (*{caption_ext}) found in {image_dir}. "
                    f"Use the Captions tab to generate them first."
                )

        if errors:
            error_message = "Please fix the following issues:\n\n" + "\n".join(f"• {e}" for e in errors)
            messagebox.showerror("Validation Error", error_message)
            return False

        return True

    @staticmethod
    def _pipeline_exit_routes_to_state_machine(name, returncode):
        """Which subprocess exits the pause/queue state machine must hear about.

        Training exits: always (clean finish, failure, stop, pause all carry state).
        Caching phases ("Cache Preparation" / "Text Encoder Caching"): only NONZERO exits —
        a clean cache exit continues the pipeline via its callback and the run is still
        alive. Before this predicate existed, a failed or stopped caching phase left
        training_state stranded at "running" with no process: the Start button read
        "Queue Train" but launched, Pause pointed at nothing, the queue window pinned a
        dead run, and a queued job that died in caching vanished without the HELD notice.
        """
        n = (name or "").lower()
        if "training" in n:
            return True
        return "cach" in n and returncode != 0
