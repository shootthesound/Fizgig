import json
import os

import tkinter as tk
from tkinter import messagebox, ttk, simpledialog

from PIL import Image, ImageTk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY, PRESETS_DIR, LAST_TRAIN_FILE, QUEUE_FILE
from fizgig_gui.core.domain.architectures import ARCHITECTURES, _canon_arch
from fizgig_gui.core.domain.minimax_math import minimax_train_base, MINIMAX_TRAIN_BASE_OPTIONS, minimax_block_spec
from fizgig_gui.core.config.prefs import _persist_disabled
from fizgig_gui.core.config.presets import MINIMAX_BUILT_IN_PRESETS, KREA2_BUILT_IN_PRESETS, BUILT_IN_PRESETS, _MM_DEFAULTS_KEY
from fizgig_gui.core.config.settings_map import PRESETS
from fizgig_gui.core.ui_base.widgets import ToolTip

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT, FIZGIG_DIR as _FIZGIG_DIR

class TrainingPresetMixin:
    def on_architecture_changed(self, event=None):
        """Handle architecture change"""
        self.update_ui_for_architecture()
        self.refresh_preset_combobox()
        self.load_default_preset(show_message=False)  # Auto-load defaults for new architecture
        # Update samples tab UI for new architecture
        if hasattr(self, 'sample_settings_frame'):
            self.update_samples_ui_for_architecture()

    def get_preset_dir_for_architecture(self, arch):
        """Get the per-architecture preset directory (under presets/), creating if needed.

        Presets live in <repo>/presets/<arch>/ — NOT at the repo root. An earlier build
        pointed this at the root, which dropped a stray "<arch>" folder into the project on
        every launch; the one-shot migration below rescues any presets saved there."""
        preset_dir = os.path.join(PRESETS_DIR, arch)
        # One-shot: rescue presets an older build wrote to <repo>/<arch>/ at the root.
        _legacy_root = os.path.join(_FIZGIG_DIR, arch)
        if os.path.isdir(_legacy_root):
            try:
                os.makedirs(preset_dir, exist_ok=True)
                for name in os.listdir(_legacy_root):
                    src = os.path.join(_legacy_root, name)
                    dst = os.path.join(preset_dir, name)
                    if os.path.isfile(src) and not os.path.exists(dst):
                        os.rename(src, dst)
                # Remove the stray root folder once it holds nothing more of ours.
                try:
                    os.rmdir(_legacy_root)
                except OSError:
                    pass  # not empty (user files) — leave it; presets were still rescued
            except Exception:
                pass
        # One-shot folder rename from the pre-2026-07-28 arch name, or user-saved Krea 2
        # presets would silently vanish from the dropdown.
        if arch == "Krea 2" and not os.path.isdir(preset_dir):
            _legacy = os.path.join(PRESETS_DIR, "Krea 2 (experimental)")
            if os.path.isdir(_legacy):
                try:
                    os.rename(_legacy, preset_dir)
                except OSError:
                    pass
        os.makedirs(preset_dir, exist_ok=True)
        return preset_dir

    def get_saved_presets(self, arch):
        """Get list of saved preset names for an architecture"""
        preset_dir = self.get_preset_dir_for_architecture(arch)
        presets = []
        for f in os.listdir(preset_dir):
            if f.endswith('.json'):
                presets.append(f[:-5])  # Remove .json extension
        return sorted(presets)

    def _builtins_for_arch(self, arch):
        """Return the built-in preset dict for an architecture. Krea 2 gets its single
        defaults entry (Klein's block/timestep/adaptive presets don't apply); everything
        else gets the full Klein built-in set."""
        cfg = ARCHITECTURES.get(arch, {})
        if cfg.get("is_minimax"):
            return MINIMAX_BUILT_IN_PRESETS
        return KREA2_BUILT_IN_PRESETS if cfg.get("is_krea2") else BUILT_IN_PRESETS

    def _update_preset_hint(self):
        """The bracketed note beside Load Preset: visible only while the MiniMax rank-16
        Defaults preset is the selection — with Fast as the shipped default, this label is
        what tells the user when the bigger recipe is the right reach."""
        lbl = getattr(self, "_preset_hint_label", None)
        if lbl is None:
            return
        try:
            if self._is_minimax_arch() and self.custom_preset_var.get() == _MM_DEFAULTS_KEY:
                lbl.pack(side=tk.LEFT, padx=(8, 0))
            else:
                lbl.pack_forget()
        except Exception:
            pass

    def refresh_preset_combobox(self):
        """Refresh the preset combobox: built-in presets first, then user-saved presets."""
        arch = self.architecture_var.get()
        user_presets = self.get_saved_presets(arch)
        builtins_map = self._builtins_for_arch(arch)
        builtins = list(builtins_map.keys())
        # Built-ins first; if a user saves a preset with same name as a built-in, it appears once (under user)
        combined = builtins + [p for p in user_presets if p not in builtins_map]
        self.custom_preset_combo['values'] = combined
        # Dynamic width: fit longest entry so names like "✨ Multi-Character (rank 16, noisy dataset)" don't truncate
        max_len = max((len(v) for v in combined), default=20)
        self.custom_preset_combo.config(width=max(20, min(max_len + 2, 60)))
        self.custom_preset_var.set('')  # Clear selection

    def load_default_preset(self, show_message=True):
        """Load recommended preset values for the current architecture"""
        arch = self.architecture_var.get()
        if arch not in PRESETS:
            if show_message:
                messagebox.showinfo("Info", f"No preset available for {arch}")
            return

        preset = PRESETS[arch]
        self._apply_preset_values(preset)
        if show_message:
            messagebox.showinfo("Preset Loaded", f"Loaded recommended preset for {arch}")

    # Comboboxes whose values feed directly into the launch command: a saved value the
    # current family doesn't offer (cross-family last-train leak, withdrawn LR floors,
    # removed optimizers) must NOT be .set() onto them — readonly Comboboxes accept any
    # value without complaint, and the bad name then dies (or misbehaves) at launch.
    _STRICT_COMBO_KEYS = {"OPTIMIZER_TYPE", "ADAPTIVE_LR_MIN", "ADAPTIVE_LR_MAX", "LR_SCHEDULER",
                          "NETWORK_TYPE"}

    def _apply_preset_values(self, preset):
        """Apply preset values to the UI (shared by load_default_preset and load_custom_preset)"""
        for key, value in preset.items():
            if key in self.entries:
                entry = self.entries[key]
                if isinstance(entry, ttk.Combobox):
                    value = str(value)
                    try:
                        # A saved plain value (e.g. "2e-4") should select its labeled combobox
                        # entry ("2e-4 - rank 4/8 only") so the warning suffix still shows.
                        opts = entry.cget("values") or ()
                        if value not in opts:
                            for opt in opts:
                                if str(opt).split(" ")[0] == value.split(" ")[0]:
                                    value = str(opt)
                                    break
                        if value not in opts and key in self._STRICT_COMBO_KEYS:
                            self.update_console(
                                f"[preset] {key}: saved value {value!r} isn't offered here — "
                                f"keeping {entry.get()!r}\n")
                            continue
                    except tk.TclError:
                        pass
                    entry.set(value)
                elif isinstance(entry, tk.BooleanVar):
                    # Some boolean settings (e.g. IMG_IN_TXT_IN_OFFLOADING, PRESERVE_DISTRIBUTION)
                    # are stored in self.entries as BooleanVars — they don't support .delete/.insert.
                    entry.set(bool(value))
                else:
                    try:
                        entry.delete(0, tk.END)
                        entry.insert(0, str(value))
                    except (AttributeError, tk.TclError):
                        # Unknown widget type — skip rather than crash
                        pass

        # Update timestep settings from preset. Validate against what the trainer accepts —
        # old presets can carry values a past version offered (e.g. "qwen_shift", which
        # argparse rejects at launch).
        if "TIMESTEP_SAMPLING" in preset:
            _ts_val = str(preset["TIMESTEP_SAMPLING"])
            _ts_ok = ("sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift",
                      "logsnr", "qinglong_flux")
            if _ts_val in _ts_ok:
                self.ts_sampling_var.set(_ts_val)
            else:
                self.update_console(f"[preset] TIMESTEP_SAMPLING {_ts_val!r} isn't supported — "
                                    f"keeping {self.ts_sampling_var.get()!r}\n")
        if "WEIGHTING_SCHEME" in preset:
            self.weighting_scheme_var.set(preset["WEIGHTING_SCHEME"])
        if "PRESERVE_DISTRIBUTION" in preset:
            self.preserve_dist_var.set(preset["PRESERVE_DISTRIBUTION"])
        # Refresh conditional field states after preset load
        if hasattr(self, 'ts_sampling_var'):
            self._on_timestep_sampling_changed()
            self._on_weighting_scheme_changed()
            self._update_noise_range_label()
        # Network Type drives a ROW SWAP (rank/alpha <-> LoKR factor), and setting a combobox
        # programmatically does NOT fire <<ComboboxSelected>> — so without this a preset that
        # changes the type left the old rows on screen: "LoRA (standard)" selected with the
        # LoKR Factor box still underneath it.
        if "NETWORK_TYPE" in preset and "LOKR_FACTOR" in getattr(self, "rows", {}):
            self._on_network_type_changed()

        # Update FP8/SCALED checkboxes from preset
        if "FP8" in preset:
            self.fp8_var.set(preset["FP8"])
        if "SCALED" in preset:
            self.scaled_var.set(preset["SCALED"])

        # Base precision — a dedicated var (not in self.entries), so the generic loop never
        # restores it. _normalize_base_precision takes the canonical key, the display label,
        # or a legacy Auto/On/Off alike, so old presets keep working.
        if hasattr(self, 'quant_4bit_mode_var'):
            _key = None
            if "QUANT_4BIT_MODE" in preset:
                _key = self._normalize_base_precision(preset["QUANT_4BIT_MODE"])
            elif "QUANT_4BIT" in preset:
                # Legacy boolean. False means "4-bit not requested", i.e. no opinion — NOT an
                # explicit demand for fp8, which would pin every old preset (including Klein's
                # defaults, which carry QUANT_4BIT: False) away from Auto.
                _key = "nf4" if bool(preset["QUANT_4BIT"]) else "auto"
            if _key:
                self.quant_4bit_mode_var.set(self._BASE_PRECISION_LABELS[_key])
                self._on_quant_4bit_mode_changed()

        # torch.compile mode — collected by _collect_preset_values but, until now, never
        # restored, so a preset's COMPILE_BLOCKS was silently dropped on load.
        if "COMPILE_BLOCKS" in preset and hasattr(self, 'compile_blocks_var'):
            _cb = str(preset["COMPILE_BLOCKS"]).capitalize()
            if _cb in ("Auto", "On", "Off"):
                self.compile_blocks_var.set(_cb)

        # Save-state toggles — dedicated vars, same not-in-self.entries situation. Presets saved
        # before these existed simply don't carry the keys, so they keep the current setting.
        if "SAVE_STATE" in preset and hasattr(self, 'save_state_var'):
            self.save_state_var.set(bool(preset["SAVE_STATE"]))
        if "SAVE_STATE_ON_TRAIN_END" in preset and hasattr(self, 'save_state_on_train_end_var'):
            self.save_state_on_train_end_var.set(bool(preset["SAVE_STATE_ON_TRAIN_END"]))

        # Per-image loss watch toggles (krea2) — dedicated vars, same not-in-self.entries situation.
        if "KREA2_LOSS_WATCH" in preset and hasattr(self, 'krea2_loss_watch_var'):
            self.krea2_loss_watch_var.set(bool(preset["KREA2_LOSS_WATCH"]))
        if "KREA2_PER_IMAGE_LR" in preset and hasattr(self, 'krea2_per_image_lr_var'):
            self.krea2_per_image_lr_var.set(bool(preset["KREA2_PER_IMAGE_LR"]))
        if "KREA2_AUTO_RECAPTION" in preset and hasattr(self, 'krea2_auto_recaption_var'):
            self.krea2_auto_recaption_var.set(bool(preset["KREA2_AUTO_RECAPTION"]))
        if "KREA2_WARMUP_LOOK" in preset and hasattr(self, 'krea2_warmup_look_var'):
            self.krea2_warmup_look_var.set(bool(preset["KREA2_WARMUP_LOOK"]))

        # Adaptive LR checkbox + sync enabled state of Min/Max LR dropdowns
        if "ADAPTIVE_LR" in preset and hasattr(self, 'adaptive_lr_var'):
            self.adaptive_lr_var.set(bool(preset["ADAPTIVE_LR"]))
            if hasattr(self, '_on_adaptive_lr_toggle'):
                self._on_adaptive_lr_toggle()
            # Re-apply the Learning Rate AFTER the toggle has settled. The entries loop above
            # runs while the LR box still reflects the PREVIOUS adaptive state — and a write to
            # a disabled ttk.Entry is silently dropped. Restoring a non-adaptive run while
            # adaptive happened to be on therefore kept the old rate, which on a fine-tune is
            # the difference between 1e-5 and a LoRA-grade 1e-4.
            _lr_entry = self.entries.get("LEARNING_RATE")
            if "LEARNING_RATE" in preset and _lr_entry is not None:
                try:
                    _was = _lr_entry.cget("state")
                    _lr_entry.config(state="normal")
                    _lr_entry.delete(0, tk.END)
                    _lr_entry.insert(0, str(preset["LEARNING_RATE"]))
                    _lr_entry.config(state=_was)
                except (AttributeError, tk.TclError):
                    pass

        # Krea 2 base-model fine-tune. Captured by _collect_preset_values but never applied
        # back, so "Load Settings From Last Train" silently dropped the entire fine-tune
        # config — mode, window size, fused backward, Fast FT and the regularisation set.
        # Order matters: the fine-tune flag goes first, because the regularisation block is
        # only written to the dataset TOML while fine-tune is on.
        _ft_map = [
            ("KREA2_FINETUNE", "krea2_finetune_var", bool),
            ("KREA2_FT_MODE", "krea2_ft_mode_var", str),
            ("KREA2_FT_BLOCKS", "krea2_ft_blocks_var", str),
            ("KREA2_FT_EVERY", "krea2_ft_every_var", str),
            ("KREA2_FT_FUSED", "krea2_ft_fused_var", bool),
            ("KREA2_FAST_FT", "krea2_fast_ft_var", bool),
            ("KREA2_REG_DIR", "krea2_reg_dir_var", str),
            ("KREA2_REG_MULT", "krea2_reg_mult_var", str),
            ("MINIMAX_FINETUNE", "minimax_finetune_var", bool),
            ("MINIMAX_FT_EVERY", "minimax_ft_every_var", str),
            ("MINIMAX_FT_SCOPE", "minimax_ft_scope_var", str),
            ("MINIMAX_FT_BLOCKSPEC", "minimax_ft_blockspec_var", str),
            ("MINIMAX_FT_FUSED", "minimax_ft_fused_var", bool),
            ("MINIMAX_REG_DIR", "minimax_reg_dir_var", str),
            ("MINIMAX_REG_MULT", "minimax_reg_mult_var", str),
        ]
        _ft_touched = False
        for _key, _attr, _cast in _ft_map:
            if _key in preset and hasattr(self, _attr):
                try:
                    getattr(self, _attr).set(_cast(preset[_key]))
                    _ft_touched = True
                except Exception:
                    pass
        if _ft_touched:
            # Show/hide the fine-tune panel to match, and rewrite the dataset TOML so the
            # regularisation block tracks the restored state rather than the previous run's.
            if hasattr(self, "_apply_krea2_ft_visibility"):
                self._apply_krea2_ft_visibility()
            if hasattr(self, "_apply_minimax_ft_visibility"):
                self._apply_minimax_ft_visibility()
            if hasattr(self, "auto_save_dataset_config_silent"):
                self.auto_save_dataset_config_silent()

        # LEARNING_RATE is state-gated: the adaptive checkbox greys the LR box, and a tk
        # Entry silently DROPS delete/insert while disabled — so the generic loop above
        # lost the preset's LR whenever adaptive was on at that moment (e.g. Old Reliable
        # active, then loading Identity kept 1e-4 instead of 4e-4). Re-apply after the
        # adaptive toggle has settled, forcing the widget writable for the write.
        if "LEARNING_RATE" in preset and "LEARNING_RATE" in self.entries:
            _lr_ent = self.entries["LEARNING_RATE"]
            try:
                _prev_state = str(_lr_ent.cget("state"))
                _lr_ent.config(state="normal")
                _lr_ent.delete(0, tk.END)
                _lr_ent.insert(0, str(preset["LEARNING_RATE"]))
                _lr_ent.config(state=_prev_state)
            except (AttributeError, tk.TclError):
                pass

        if "MINIMAX_DISTILL" in preset and hasattr(self, "minimax_distill_var"):
            self.minimax_distill_var.set(bool(preset["MINIMAX_DISTILL"]))

        # Multi Concept: a BooleanVar plus a LIST of folders, so neither is reachable by the
        # generic self.entries loop above. Restore the folders BEFORE the toggle so the handler
        # that rewrites the TOML and locks caption dropout sees the finished state.
        if "MINIMAX_CONCEPT_DIRS" in preset:
            _dirs = preset.get("MINIMAX_CONCEPT_DIRS") or []
            if isinstance(_dirs, str):                     # tolerate an older single-string save
                _dirs = [_dirs] if _dirs.strip() else []
            for _i, _v in enumerate(getattr(self, "_concept_folder_vars", [])):
                _v.set(str(_dirs[_i]).strip() if _i < len(_dirs) else "")
        if "MINIMAX_MULTICONCEPT" in preset and hasattr(self, "minimax_multiconcept_var"):
            self.minimax_multiconcept_var.set(bool(preset["MINIMAX_MULTICONCEPT"]))
        # Re-run unconditionally: a preset that carries MINIMAX_CAPTION_DROPOUT (the Defaults one
        # does) would otherwise leave the box showing 0.05 while Multi Concept is on. Training
        # was never at risk - the command builder locks it either way - but the UI would lie.
        if hasattr(self, "minimax_multiconcept_var"):
            try:
                self._on_minimax_multiconcept_toggle()
            except Exception:
                pass
        try:
            self._sync_distill_weight_state()
        except Exception:
            pass

        # Model Area to Train (training preset dropdown)
        if "TARGET_LAYERS" in preset and hasattr(self, 'training_preset_var'):
            legacy_map = {
                "All Layers": "Full Model",
                "Identity Blocks": "Identity",
                "Style+Composition Blocks": "Style+Composition",
                "Details Blocks": "Details",
            }
            raw = preset["TARGET_LAYERS"]
            mapped = legacy_map.get(raw, raw)
            valid = ("Full Model", "Identity", "Style", "Style+Composition", "Details", "Custom")
            self.training_preset_var.set(mapped if mapped in valid else "Full Model")
            if hasattr(self, '_on_training_preset_changed'):
                self._on_training_preset_changed()
                # _on_training_preset_changed auto-fills MIN/MAX_TIMESTEP from the Model
                # Area — which just overwrote the values the generic loop restored from
                # the preset. The preset's explicit values are the user's saved choice:
                # put them back.
                for _ts_key in ("MIN_TIMESTEP", "MAX_TIMESTEP"):
                    if _ts_key in preset and _ts_key in self.entries:
                        try:
                            self.entries[_ts_key].delete(0, tk.END)
                            self.entries[_ts_key].insert(0, str(preset[_ts_key]))
                        except Exception:
                            pass
        if "FP8_TEXT_ENCODER" in preset:
            self.fp8_text_encoder_var.set(preset["FP8_TEXT_ENCODER"])
        if "ENABLE_BUCKET" in preset:
            self.dataset_enable_bucket_var.set(preset["ENABLE_BUCKET"])
        if "BUCKET_NO_UPSCALE" in preset:
            self.dataset_no_upscale_var.set(preset["BUCKET_NO_UPSCALE"])
        # Dataset subsection (Training → Other Options → Dataset)
        if "DATASET_CAPTION_EXT" in preset and hasattr(self, "dataset_caption_ext_var"):
            self.dataset_caption_ext_var.set(preset["DATASET_CAPTION_EXT"])
        if "DATASET_MEGAPIXELS" in preset and hasattr(self, "dataset_megapixels_var"):
            self.dataset_megapixels_var.set(preset["DATASET_MEGAPIXELS"])
        if "DATASET_BATCH_SIZE" in preset and hasattr(self, "dataset_batch_size_var"):
            self.dataset_batch_size_var.set(preset["DATASET_BATCH_SIZE"])
        # Run card's Enable Cache checkbox
        if "ENABLE_CACHE" in preset and hasattr(self, "enable_cache_var"):
            self.enable_cache_var.set(bool(preset["ENABLE_CACHE"]))
        # Per-block custom training selection (only meaningful when TARGET_LAYERS=Custom)
        if "TRAINING_BLOCKS" in preset and hasattr(self, "training_block_vars"):
            for block_key, block_on in preset["TRAINING_BLOCKS"].items():
                if block_key in self.training_block_vars:
                    self.training_block_vars[block_key].set(bool(block_on))
        # Gradient mining
        if "GRADIENT_MINING" in preset and hasattr(self, "gradient_mining_var"):
            self.gradient_mining_var.set(bool(preset["GRADIENT_MINING"]))
        self.toggle_scaled()  # Update checkbox state

    def _save_last_train_settings(self):
        """Snapshot current settings just before launching training, so 'Load Last Train' can restore them."""
        if _persist_disabled():
            return   # headless tests must never overwrite the real .last_train_settings.json
        try:
            os.makedirs(PRESETS_DIR, exist_ok=True)
            snapshot = self._collect_preset_values()
            # Presets deliberately don't carry the family (a Krea 2 preset must not hijack your
            # model choice), but "restore my last launch" plainly includes WHICH model it was —
            # the same reasoning the training queue uses when it stores the architecture beside
            # its snapshot. Namespaced so _apply_preset_values ignores it as an unknown key.
            snapshot["__architecture__"] = self.architecture_var.get()
            # Training Base is preset-immune (outside self.entries, never collected) but
            # "restore my last launch" plainly includes which base it ran on — same reasoning
            # as the architecture above. Namespaced so _apply_preset_values ignores it.
            if hasattr(self, "minimax_train_base_var"):
                snapshot["__minimax_train_base__"] = minimax_train_base(
                    self.minimax_train_base_var.get())
            with open(LAST_TRAIN_FILE, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
        except Exception as e:
            print(f"[last_train] Failed to save snapshot: {e}")

    def _load_last_train_settings(self):
        """Restore settings from the most recent training launch."""
        if not os.path.exists(LAST_TRAIN_FILE):
            messagebox.showinfo(
                "No Last Train",
                "No previous training settings found.\n\n"
                "Launch a training run first; afterwards this button will restore those settings."
            )
            return
        try:
            with open(LAST_TRAIN_FILE, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
            # Switch family FIRST if the launch was on a different one: on_architecture_changed
            # loads that family's default preset, so doing it after would clobber everything the
            # snapshot just restored. Older snapshots have no architecture — they simply skip this.
            _arch = snapshot.pop("__architecture__", None)
            _switched = ""
            _arch = _canon_arch(_arch) if _arch else _arch
            if _arch and _arch in ARCHITECTURES and _arch != self.architecture_var.get():
                self.architecture_var.set(_arch)
                self.on_architecture_changed()
                _switched = f"\n\nSwitched the Base Model back to {_arch}."
            # Training Base rides beside the preset, not in it (preset-immune by design) —
            # pop before applying so _apply_preset_values never sees it even by accident.
            _base = snapshot.pop("__minimax_train_base__", None)
            self._apply_preset_values(snapshot)
            if _base and hasattr(self, "minimax_train_base_var"):
                self.minimax_train_base_var.set(MINIMAX_TRAIN_BASE_OPTIONS[
                    1 if minimax_train_base(_base) == "ref2va" else 0])
            messagebox.showinfo("Loaded",
                                f"Restored settings from your last training launch.{_switched}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load last train settings:\n{e}")

    # ------------------------------------------------------------
    # Training queue — settings snapshots that run back-to-back
    # ------------------------------------------------------------
    # A queue item is everything a run needs that the GUI would otherwise read live:
    # the preset snapshot (_collect_preset_values), the architecture (presets are
    # per-arch and deliberately don't carry it), the Start-tab dataset folder, and the
    # Samples-tab entries (presets deliberately skip those too). Restoring an item is
    # "load these into the GUI, then press Start" — the queue never bypasses
    # start_training, so validation, TOML regeneration, snapshotting and the pause
    # machinery all behave exactly as for a hand-started run.

    def _load_training_queue(self):
        if _persist_disabled():
            return []
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                items = json.load(f)
            if not isinstance(items, list):
                print(f"[queue] {QUEUE_FILE} does not hold a list — starting with an empty queue")
                return []
            good = [i for i in items if self._queue_item_valid(i)]
            if len(good) != len(items):
                print(f"[queue] dropped {len(items) - len(good)} unreadable entr(ies) from {QUEUE_FILE}")
            return good
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"[queue] failed to load {QUEUE_FILE}: {e}")
            return []

    def _save_training_queue(self):
        if _persist_disabled():
            return
        try:
            os.makedirs(PRESETS_DIR, exist_ok=True)
            tmp = QUEUE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.training_queue, f, indent=2, default=str)
            os.replace(tmp, QUEUE_FILE)
        except Exception as e:
            print(f"[queue] failed to save: {e}")

    @staticmethod
    def _queue_item_valid(item):
        """Deep-enough validation for anything about to flow into apply/summary/advance.
        Shallow checks (dict with a dict preset) let hand-edited variants through that then
        crashed AFTER the item was popped and saved away — losing it before the traceback."""
        return (isinstance(item, dict)
                and isinstance(item.get("preset"), dict)
                and isinstance(item.get("image_folder", ""), str)
                and isinstance(item.get("architecture", ""), str)
                and isinstance(item.get("samples", {}), dict))

    def _schedule_queue_advance(self, delay_ms):
        """The ONE way to arm a queue-advance timer. A generation counter makes every
        previously-armed timer a no-op: Stop, Pause, a failure-HOLD, or a manual start bumps
        the generation, so a stale after() callback from before the state change can never
        fire into a paused/held queue or double-launch across a pipeline phase gap."""
        gen = getattr(self, "_queue_advance_gen", 0)

        def _tick():
            if getattr(self, "_queue_advance_gen", 0) == gen:
                self._start_next_queued()
        self.master.after(delay_ms, _tick)

    def _cancel_pending_queue_advance(self):
        self._queue_advance_gen = getattr(self, "_queue_advance_gen", 0) + 1

    _QUEUE_SAMPLE_KEYS = ("SAMPLE_ENABLED", "SAMPLE_WIDTH", "SAMPLE_HEIGHT", "SAMPLE_STEPS",
                          "SAMPLE_SEED", "SAMPLE_EVERY_N_EPOCHS", "SAMPLE_EVERY_N_STEPS",
                          "SAMPLE_AT_FIRST", "SAMPLE_FLOW_SHIFT", "SAMPLE_NEGATIVE",
                          "SAMPLE_CFG_SCALE", "SAMPLE_FRAMES",
                          "MINIMAX_TURBO_STEPS", "MINIMAX_TURBO_STRENGTH")

    def _queue_snapshot(self):
        """Capture the currently configured run as a queue item."""
        import time as _time
        samples = {}
        for k in self._QUEUE_SAMPLE_KEYS:
            entry = self.entries.get(k)
            if entry is None:
                continue
            try:
                samples[k] = entry.get()
            except Exception:
                pass
        return {
            "id": f"q{int(_time.time() * 1000)}",
            "queued_at": _time.strftime("%Y-%m-%d %H:%M"),
            "architecture": self.architecture_var.get(),
            "image_folder": self.image_folder_var.get().strip(),
            # A queued Multi Concept run loses its second subject without this.
            "concept_folders": [v.get().strip() for v in
                                getattr(self, "_concept_folder_vars", [])],
            "preset": self._collect_preset_values(),
            "samples": samples,
            # Training Base rides beside the preset, not in it (preset-immune by design) — a
            # queued ref2va run would otherwise silently launch on fl2va.
            "minimax_train_base": minimax_train_base(
                getattr(self, "minimax_train_base_var", None)
                and self.minimax_train_base_var.get()),
        }

    def _apply_queue_item(self, item):
        """Load a queue item's settings back into the GUI (arch first — it swaps the UI)."""
        arch = _canon_arch(item.get("architecture", ""))
        if isinstance(arch, str) and arch and arch in ARCHITECTURES and self.architecture_var.get() != arch:
            self.architecture_var.set(arch)
            try:
                self.update_ui_for_architecture()
            except Exception as e:
                self.update_console(f"[queue] arch switch to {arch!r} failed: {e}\n")
        self._apply_preset_values(item.get("preset", {}))
        # Items queued before the Training Base dropdown existed carry no key — leave the
        # dropdown as it stands rather than forcing a default onto an old queue file.
        _base = item.get("minimax_train_base")
        if _base and hasattr(self, "minimax_train_base_var"):
            self.minimax_train_base_var.set(MINIMAX_TRAIN_BASE_OPTIONS[
                1 if minimax_train_base(_base) == "ref2va" else 0])
        folder = str(item.get("image_folder") or "").strip()
        if folder:
            self.image_folder_var.set(folder)   # traces regenerate Fizgig_train.toml
        # Multi Concept's extra folders ride separately: they are not the Start folder and must
        # not overwrite it. Restored before the toggle so the TOML rewrite sees them.
        _cf = item.get("concept_folders") or []
        for _i, _v in enumerate(getattr(self, "_concept_folder_vars", [])):
            _v.set(str(_cf[_i]).strip() if _i < len(_cf) else "")
        _samples = item.get("samples")
        for k, v in (_samples.items() if isinstance(_samples, dict) else ()):
            entry = self.entries.get(k)
            if entry is None:
                continue
            try:
                if isinstance(entry, (tk.BooleanVar, tk.StringVar, tk.IntVar, tk.DoubleVar)):
                    entry.set(v)
                elif isinstance(entry, ttk.Combobox):
                    entry.set(str(v))
                else:
                    entry.delete(0, tk.END)
                    entry.insert(0, str(v))
            except Exception:
                pass

    @staticmethod
    def _queue_signature(item):
        """What makes two queue entries THE SAME RUN: everything except id/queued_at."""
        try:
            return json.dumps({k: item.get(k) for k in
                               ("architecture", "image_folder", "preset", "samples",
                                "minimax_train_base")},
                              sort_keys=True, default=str)
        except Exception:
            return repr(item)

    @staticmethod
    def _queue_output_key(item):
        """(output dir, LoRA name) — two runs writing here overwrite each other's files."""
        p = item.get("preset", {}) if isinstance(item.get("preset"), dict) else {}
        return (str(p.get("LORA_OUTPUT_DIR", "")).strip().lower().replace("\\", "/").rstrip("/"),
                str(p.get("LORA_NAME", "")).strip().lower())

    def _queue_current_run(self):
        """Snapshot the current config to the end of the queue (Start pressed mid-run)."""
        # Queueing skips validate_inputs entirely (Start returns above), so without this a bad
        # name is written into the queue file, compared dirty by the clash check below, and only
        # rejected an hour later when the queue tries to launch it — modal, unattended, held.
        _name, _name_error = self._tidy_lora_name()
        if _name_error:
            messagebox.showwarning("Check the LoRA name", _name_error)
            return
        item = self._queue_snapshot()
        if not item["image_folder"]:
            messagebox.showwarning(
                "Nothing to queue",
                "Pick a training image folder on the Start tab first — a queued run "
                "needs to know its dataset.")
            return
        # An exact duplicate (same everything) is never useful — it would just train the
        # identical run twice. Point at the existing entry instead of adding another.
        sig = self._queue_signature(item)
        for pos, q in enumerate(self.training_queue):
            if self._queue_signature(q) == sig:
                messagebox.showinfo(
                    "Already queued",
                    f"This exact run is already in the queue (position {pos + 1}).\n\n"
                    "Change something — the dataset, the output name, any setting — "
                    "to queue a different run.")
                return
        # Same output dir + name as another queued job (or the run in progress) with
        # DIFFERENT settings: the later run would overwrite the earlier one's checkpoints,
        # state dirs and samples. Flag it; queueing anyway is a legitimate choice.
        okey = self._queue_output_key(item)
        if okey != ("", ""):
            clash = next((f"queued job {pos + 1}" for pos, q in enumerate(self.training_queue)
                          if self._queue_output_key(q) == okey), None)
            if clash is None:
                _active = getattr(self, "_active_run_item", None)
                if (_active is not None
                        and getattr(self, "training_state", "idle") in ("running", "pausing")
                        and self._queue_output_key(_active) == okey):
                    clash = "the run in progress"
            if clash is not None and not messagebox.askyesno(
                    "Same output name",
                    f"This run writes to the same output folder and LoRA name as {clash} — "
                    f"its checkpoints, state dirs and samples would be overwritten.\n\n"
                    f"Queue it anyway? (Change the Output Name to keep both.)"):
                return
        self.training_queue.append(item)
        self._save_training_queue()
        self._refresh_queue_button()
        self._render_queue_window()
        name = item["preset"].get("LORA_NAME") or os.path.basename(item["image_folder"])
        self.update_console(f"[queue] added '{name}' — position {len(self.training_queue)} in the "
                            f"queue. It starts automatically when the current run finishes.\n")

    def _start_next_queued(self):
        """Pop the head of the queue into the GUI and start it. Never called while busy."""
        _proc = getattr(self, "current_process", None)
        if _proc is not None and _proc.poll() is None:
            return
        # Process-gone is NOT idle: between pipeline phases current_process is briefly None
        # while training_state is still "running", and paused/pausing runs own the GPU's
        # future. A stale timer or an eager click must not launch into any of those.
        if getattr(self, "training_state", "idle") in ("running", "pausing", "paused"):
            return
        if not self.training_queue:
            return
        # A training subprocess isn't the only thing that owns the GPU: a Royale export, a
        # caption batch, an Extract or a live preview are all in-process threads the process
        # check can't see. Launching a run on top of them OOMs it (and a failed run HOLDS
        # the queue — the worst outcome for an unattended batch). Wait and retry — capped,
        # so a stuck busy flag can't spin forever: after ~10 minutes the queue HOLDs loudly.
        try:
            if self._is_any_busy():
                self._queue_busy_retries = getattr(self, "_queue_busy_retries", 0) + 1
                if self._queue_busy_retries > 40:
                    self._queue_busy_retries = 0
                    self.update_console("[queue] HELD — the app has reported other GPU work "
                                        "for 10+ minutes. Finish or cancel it, then use "
                                        "'Start next now' in the queue window.\n")
                    self._render_queue_window()
                    return
                self.update_console("[queue] GPU work in progress elsewhere in the app — "
                                    "next run retries in 15 s.\n")
                self._schedule_queue_advance(15000)
                return
        except Exception:
            pass
        self._queue_busy_retries = 0
        head = self.training_queue[0]
        # Malformed item (hand-edited/corrupted queue file): remove it LOUDLY, then move on
        # to the next — one bad entry must not wedge the whole queue or crash the advance.
        if not self._queue_item_valid(head):
            self.training_queue.pop(0)
            self._save_training_queue()
            self._refresh_queue_button()
            self._render_queue_window()
            self.update_console("[queue] removed an unreadable queue entry (corrupt or "
                                "hand-edited queue file) — continuing with the next.\n")
            self._schedule_queue_advance(100)
            return
        # Dataset gone (deleted/renamed/moved since queueing): without this check the stale
        # TOML silently trains the PREVIOUS job's dataset under this job's name. HOLD with
        # the item still queued so nothing is lost.
        folder = (head.get("image_folder") or "").strip()
        if not os.path.isdir(folder):
            self.update_console(f"[queue] HELD — the next run's image folder no longer exists:\n"
                                f"        {folder}\n"
                                f"        Restore the folder (or edit/delete the queued job in "
                                f"the queue window), then 'Start next now'.\n")
            self._render_queue_window()
            return
        item = self.training_queue.pop(0)
        self._save_training_queue()
        self._refresh_queue_button()
        name = item.get("preset", {}).get("LORA_NAME") or os.path.basename(item.get("image_folder", "?"))
        self.update_console(f"\n[queue] starting next run: '{name}' "
                            f"({len(self.training_queue)} still queued)\n")
        self._apply_queue_item(item)
        self.start_training()
        # start_training can decline (validation, disk warning declined). The item's settings
        # are in the GUI either way; put it back at the head so nothing is silently lost.
        # _training_start_pending counts as LAUNCHED: with a warm caption worker the real
        # launch is marshalled through after(0) and training_state is still "idle" here —
        # re-inserting then would run the same item twice (review agent, 25 Aug).
        if (getattr(self, "training_state", "idle") != "running"
                and not getattr(self, "_training_start_pending", False)):
            self.training_queue.insert(0, item)
            self._save_training_queue()
            self._refresh_queue_button()
            # Invalidate any timer armed before this decline — otherwise a pending advance
            # re-pops the same head and repeats the same modal validation error in a loop.
            self._cancel_pending_queue_advance()
            self.update_console("[queue] run did not start — it stays at the head of the queue. "
                                "Fix the issue and use the queue window's 'Start next' button.\n")
        self._render_queue_window()

    def _refresh_queue_button(self):
        btn = getattr(self, "_queue_btn", None)
        if btn is None:
            return
        n = len(getattr(self, "training_queue", []))
        try:
            # Dark text on the light blue in BOTH states — the old accent-blue-when-queued
            # would now be mid-blue on baby blue (~2:1, unreadable). The count carries the
            # signal instead.
            btn.config(text=f"📋 Queue ({n})" if n else "📋 Queue",
                       bg=COLORS["queue_blue"], fg=COLORS["bg_deep"])
        except Exception:
            pass

    def _queue_thumbnail(self, folder, size=56):
        """PhotoImage of the first image in `folder`, or None. Caller keeps the reference."""
        try:
            from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
            exts = {e.lower() for e in IMAGE_EXTENSIONS}
            first = next((f for f in sorted(os.listdir(folder))
                          if os.path.splitext(f)[1].lower() in exts), None)
            if first is None:
                return None
            img = Image.open(os.path.join(folder, first))
            img.thumbnail((size, size), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            return None

    def _open_queue_window(self):
        """The queue manager: one row per queued run — thumbnail, key settings, and the
        operations (reorder / edit in tab / update from tab / delete / start next)."""
        win = getattr(self, "_queue_win", None)
        if win is not None and win.winfo_exists():
            win.lift()
            self._render_queue_window()
            return
        win = tk.Toplevel(self.master)
        win.title("Training Queue")
        win.geometry("860x560")
        win.configure(bg=COLORS["bg_deep"])
        self._queue_win = win

        tk.Label(win, text="Training Queue", font=(FONT_FAMILY, 16, "bold"),
                 bg=COLORS["bg_deep"], fg=COLORS["text_primary"]).pack(anchor=tk.W, padx=16, pady=(14, 0))
        self._queue_win_status = tk.Label(win, text="", font=(FONT_FAMILY, 9),
                                          bg=COLORS["bg_deep"], fg=COLORS["text_muted"],
                                          justify=tk.LEFT)
        self._queue_win_status.pack(anchor=tk.W, padx=16, pady=(2, 8))

        holder = tk.Frame(win, bg=COLORS["bg_deep"])
        holder.pack(fill=tk.BOTH, expand=True, padx=16)
        canvas = tk.Canvas(holder, bg=COLORS["bg_deep"], highlightthickness=0)
        vsb = ttk.Scrollbar(holder, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rows = tk.Frame(canvas, bg=COLORS["bg_deep"])
        cw = canvas.create_window((0, 0), window=rows, anchor="nw")
        rows.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(cw, width=e.width))
        # Wheel: global router — scrolls this canvas when the pointer is over the window,
        # the main app everywhere else, with no bind_all steal in either direction.
        self._queue_rows_frame = rows

        foot = tk.Frame(win, bg=COLORS["bg_deep"])
        foot.pack(fill=tk.X, padx=16, pady=12)
        self._queue_start_next_btn = ttk.Button(foot, text="▶ Start next now",
                                                command=self._start_next_queued, style="Primary.TButton")
        self._queue_start_next_btn.pack(side=tk.LEFT)
        ttk.Button(foot, text="Clear queue", command=self._queue_clear_all).pack(side=tk.LEFT, padx=(12, 0))
        tk.Label(foot, text="Queued runs start automatically when the current run finishes cleanly. "
                            "After a failure, a Stop, or an app restart, the queue waits for you.",
                 font=(FONT_FAMILY, 8), bg=COLORS["bg_deep"], fg=COLORS["text_explain"],
                 wraplength=420, justify=tk.LEFT).pack(side=tk.RIGHT)
        self._render_queue_window()

    def _queue_clear_all(self):
        if self.training_queue and messagebox.askyesno(
                "Clear queue", f"Remove all {len(self.training_queue)} queued run(s)?"):
            self.training_queue.clear()
            self._save_training_queue()
            self._refresh_queue_button()
            self._render_queue_window()

    def _queue_row_summary(self, item):
        p = item.get("preset", {}) if isinstance(item.get("preset"), dict) else {}
        folder = str(item.get("image_folder") or "")
        try:
            from fizgig.dataset.image_dataset import IMAGE_EXTENSIONS
            _exts = {e.lower() for e in IMAGE_EXTENSIONS}
            # Clips and voice recordings are training items too — count them or a MiniMax
            # clip/audio folder reads "(0 images)" and looks like a queued mistake.
            if "MiniMax" in str(item.get("architecture", "")):
                _exts |= {".mp4"} | self.TRAINING_AUDIO_EXTENSIONS
            n_imgs = sum(1 for f in os.listdir(folder)
                         if os.path.splitext(f)[1].lower() in _exts) if os.path.isdir(folder) else 0
        except Exception:
            n_imgs = 0
        name = p.get("LORA_NAME") or os.path.basename(folder) or "(unnamed)"
        bits = [f"{item.get('architecture', '?')}",
                f"{os.path.basename(folder) or '?'} ({n_imgs} items)"]
        for label, key in (("LR", "LEARNING_RATE"), ("epochs", "MAX_TRAIN_EPOCHS"),
                           ("dim", "NETWORK_DIM"), ("type", "NETWORK_TYPE"),
                           ("area", "TARGET_LAYERS")):
            v = p.get(key)
            if v not in (None, ""):
                bits.append(f"{label} {v}")
        # Detail Focus only means anything for MiniMax, and it's the whole point of queueing a
        # shift sweep — without it two rows of an A/B look identical in the manager.
        if ARCHITECTURES.get(item.get("architecture", ""), {}).get("is_minimax"):
            _sh = str(p.get("MINIMAX_LOWNOISE_PCT") or "").strip()
            if _sh:
                bits.append(f"low-noise {_sh}%")
            _hl = str(p.get("MINIMAX_HIGHNOISE_LR_PCT") or "100").strip()
            if _hl and _hl != "100":
                bits.append(f"high-noise LR {_hl}%")
            if p.get("MINIMAX_LIKENESS_OPT"):
                bits.append("likeness-opt")
            else:
                _bl = minimax_block_spec(p.get("MINIMAX_BLOCKS"))
                if _bl.lower() != "all":
                    bits.append(f"blocks {_bl}")
            if p.get("MINIMAX_TRAIN_ADALN") is False:
                bits.append("no adaln")
            if p.get("MINIMAX_DISTILL"):
                bits.append(f"distill x{p.get('MINIMAX_DISTILL_WEIGHT', '0.8')}"
                            f" ({p.get('MINIMAX_DISTILL_REFS', '2')} refs)")
            _sl = str(p.get("MINIMAX_SLOW_BLOCKS") or "").strip()
            if _sl and str(p.get("MINIMAX_SLOW_LR_SCALE", "1")).strip() not in ("", "1", "1.0"):
                bits.append(f"slow {_sl} ×{p.get('MINIMAX_SLOW_LR_SCALE')}")
        return name, "  ·  ".join(str(b) for b in bits) + f"\nqueued {item.get('queued_at', '?')}"

    def _render_queue_window(self):
        rows = getattr(self, "_queue_rows_frame", None)
        if rows is None or not rows.winfo_exists():
            return
        for w in rows.winfo_children():
            w.destroy()
        self._queue_thumb_refs = []
        _busy = getattr(self, "current_process", None)
        _busy = _busy is not None and _busy.poll() is None
        _state = getattr(self, "training_state", "idle")
        _active = getattr(self, "_active_run_item", None)
        _show_active = _active is not None and (_busy or _state in ("running", "pausing", "paused"))
        try:
            n = len(self.training_queue)
            # Starting the next run while one is PAUSED would silently abandon the paused
            # run (its state dir resumes nothing once another run overwrites the GUI), so
            # paused disables the button just like busy does.
            _blocked = _busy or _state in ("pausing", "paused")
            self._queue_start_next_btn.config(
                state=(tk.DISABLED if (_blocked or not n) else tk.NORMAL))
            if _busy and _state == "pausing":
                txt = (f"{n} run(s) queued — the current run is pausing at the epoch end. "
                       f"A pause HOLDS the queue: Resume from the Training tab, or start the "
                       f"next run from here after it exits.") if n else \
                      "The current run is pausing at the epoch end."
            elif _busy:
                txt = (f"{n} run(s) queued — a run is active; the queue continues when it "
                       f"finishes cleanly." if n else
                       "A run is active and nothing is queued. The Start Training button reads "
                       "'Queue Train' — click it to add the currently configured run.")
            elif _state == "paused":
                txt = (f"{n} run(s) queued — a run is PAUSED. Resume it from the Training tab; "
                       f"'Start next now' is disabled because it would abandon the paused run."
                       if n else
                       "A run is paused — Resume it from the Training tab.")
            elif n:
                txt = (f"{n} run(s) queued — nothing is training. The queue HOLDS after a "
                       f"failure or Stop; use 'Start next now' to begin or continue.")
            else:
                txt = ("Queue is empty. While a run is active, the Start Training button "
                       "becomes 'Queue Train' — click it to add the currently configured run.")
            self._queue_win_status.config(text=txt)
        except Exception:
            pass

        # The run in progress, pinned on top — it isn't a queue item (never saved, can't be
        # reordered or deleted), but after editing a queued job in the Training tab, its ✎ is
        # the way BACK to the settings that are actually running.
        if _show_active:
            badge = ("⏸ paused" if _state == "paused" else
                     "⏸ pausing at epoch end" if _state == "pausing" else "▶ training now")
            card = tk.Frame(rows, bg=COLORS["bg_surface"],
                            highlightbackground=COLORS["accent"], highlightthickness=2)
            card.pack(fill=tk.X, pady=(0, 8))
            thumb = self._queue_thumbnail(_active.get("image_folder", ""))
            if thumb is not None:
                self._queue_thumb_refs.append(thumb)
                tk.Label(card, image=thumb, bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=10, pady=8)
            else:
                tk.Label(card, text="🖼", font=(FONT_FAMILY, 20), width=3,
                         bg=COLORS["bg_surface"], fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=10, pady=8)
            act = tk.Frame(card, bg=COLORS["bg_surface"])
            act.pack(side=tk.RIGHT, padx=10, pady=8)
            name, summary = self._queue_row_summary(_active)
            txt = tk.Frame(card, bg=COLORS["bg_surface"])
            txt.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=8)
            tk.Label(txt, text=f"{badge}  —  {name}", font=(FONT_FAMILY, 11, "bold"),
                     bg=COLORS["bg_surface"], fg=COLORS["accent"], anchor="w",
                     wraplength=520, justify=tk.LEFT).pack(anchor=tk.W)
            tk.Label(txt, text=summary.split("\n")[0], font=(FONT_FAMILY, 8),
                     bg=COLORS["bg_surface"], fg=COLORS["text_muted"], anchor="w",
                     wraplength=520, justify=tk.LEFT).pack(anchor=tk.W)
            abtn = tk.Button(act, text="✎", font=(FONT_FAMILY, 10), width=3,
                             bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                             activebackground=COLORS["border"], relief="flat", bd=0,
                             cursor="hand2", command=self._queue_restore_active)
            abtn.pack(side=tk.LEFT, padx=2)
            ToolTip(abtn, "Load this run's settings back into the Training tab — the way back "
                          "after editing a queued job")
            if _busy:
                cbtn = tk.Button(act, text="■", font=(FONT_FAMILY, 10), width=3,
                                 bg=COLORS["bg_surface"], fg=COLORS["error"],
                                 activebackground=COLORS["border"], relief="flat", bd=0,
                                 cursor="hand2", command=self._queue_cancel_active)
                cbtn.pack(side=tk.LEFT, padx=2)
                ToolTip(cbtn, "Stop this run (no save). Queued runs HOLD — they won't "
                              "auto-start after a cancel")

        if not self.training_queue:
            return
        for i, item in enumerate(list(self.training_queue)):
            # One corrupt entry (hand-edited file, interrupted write) must not take the
            # whole window down — render it as removable wreckage instead.
            if not isinstance(item, dict) or not isinstance(item.get("preset"), dict):
                bad = tk.Frame(rows, bg=COLORS["bg_surface"],
                               highlightbackground=COLORS["error"], highlightthickness=1)
                bad.pack(fill=tk.X, pady=(0, 8))
                tk.Label(bad, text=f"{i + 1}.  ⚠ unreadable queue entry (corrupt or hand-edited "
                                   f"queue file)", font=(FONT_FAMILY, 10),
                         bg=COLORS["bg_surface"], fg=COLORS["error"]).pack(side=tk.LEFT, padx=10, pady=10)
                tk.Button(bad, text="✕", font=(FONT_FAMILY, 10), width=3,
                          bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                          activebackground=COLORS["border"], relief="flat", bd=0, cursor="hand2",
                          command=lambda i=i: self._queue_delete(i)).pack(side=tk.RIGHT, padx=10)
                continue
            card = tk.Frame(rows, bg=COLORS["bg_surface"],
                            highlightbackground=COLORS["border"], highlightthickness=1)
            card.pack(fill=tk.X, pady=(0, 8))
            thumb = self._queue_thumbnail(item.get("image_folder", ""))
            if thumb is not None:
                self._queue_thumb_refs.append(thumb)
                tk.Label(card, image=thumb, bg=COLORS["bg_surface"]).pack(side=tk.LEFT, padx=10, pady=8)
            else:
                tk.Label(card, text="🖼", font=(FONT_FAMILY, 20), width=3,
                         bg=COLORS["bg_surface"], fg=COLORS["text_muted"]).pack(side=tk.LEFT, padx=10, pady=8)
            # Buttons pack FIRST (from the right): pack allocates space in order, so a long
            # unwrapped summary used to squeeze the ↑↓✎⤓✕ column clean off the card —
            # "my queued jobs have no delete button".
            btns = tk.Frame(card, bg=COLORS["bg_surface"])
            btns.pack(side=tk.RIGHT, padx=10, pady=8)
            name, summary = self._queue_row_summary(item)
            txt = tk.Frame(card, bg=COLORS["bg_surface"])
            txt.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=8)
            tk.Label(txt, text=f"{i + 1}.  {name}", font=(FONT_FAMILY, 11, "bold"),
                     bg=COLORS["bg_surface"], fg=COLORS["text_primary"], anchor="w",
                     wraplength=520, justify=tk.LEFT).pack(anchor=tk.W)
            tk.Label(txt, text=summary, font=(FONT_FAMILY, 8),
                     bg=COLORS["bg_surface"], fg=COLORS["text_muted"], anchor="w",
                     wraplength=520, justify=tk.LEFT).pack(anchor=tk.W)

            def _mk(parent, label, cmd, tip):
                b = tk.Button(parent, text=label, font=(FONT_FAMILY, 10), width=3,
                              bg=COLORS["bg_surface"], fg=COLORS["text_primary"],
                              activebackground=COLORS["border"], relief="flat", bd=0,
                              cursor="hand2", command=cmd)
                b.pack(side=tk.LEFT, padx=2)
                ToolTip(b, tip)
                return b
            _mk(btns, "↑", lambda i=i: self._queue_move(i, -1), "Move up")
            _mk(btns, "↓", lambda i=i: self._queue_move(i, +1), "Move down")
            _mk(btns, "✎", lambda i=i: self._queue_edit(i),
                "Load this run's settings into the Training tab to edit them")
            _mk(btns, "⤓", lambda i=i: self._queue_update_from_tab(i),
                "Overwrite this queued run with the Training tab's current settings")
            _mk(btns, "✕", lambda i=i: self._queue_delete(i), "Remove from queue")

    def _queue_move(self, i, delta):
        j = i + delta
        if 0 <= i < len(self.training_queue) and 0 <= j < len(self.training_queue):
            q = self.training_queue
            q[i], q[j] = q[j], q[i]
            self._save_training_queue()
            self._render_queue_window()

    def _queue_delete(self, i):
        if 0 <= i < len(self.training_queue):
            self.training_queue.pop(i)
            self._save_training_queue()
            self._refresh_queue_button()
            self._render_queue_window()

    def _queue_cancel_active(self):
        """Stop the run in progress from the queue window's pinned card. Confirmed first —
        the Training tab's own Stop button stays instant, but here a misclick between rows
        would kill hours of work. The existing hold policy applies: queued runs do NOT
        auto-start after a cancel."""
        _proc = getattr(self, "current_process", None)
        if _proc is None or _proc.poll() is not None:
            self._render_queue_window()
            return
        name = (getattr(self, "_active_run_item", None) or {}).get("preset", {}).get("LORA_NAME", "this run")
        if not messagebox.askyesno(
                "Stop training?",
                f"Stop '{name}' now? Progress since the last checkpoint is lost, and queued "
                f"runs will HOLD rather than auto-start.\n\n(To finish the epoch and save "
                f"first, use Pause Training on the Training tab instead.)"):
            return
        self.stop_training()
        self._render_queue_window()

    def _queue_restore_active(self):
        """Put the RUNNING job's settings back into the Training tab (the ✎ on the pinned
        'training now' card) — the undo for having edited a queued job in the tab."""
        item = getattr(self, "_active_run_item", None)
        if item is None:
            return
        self._apply_queue_item(item)
        self.update_console("[queue] Training tab restored to the run in progress.\n")

    def _queue_edit(self, i):
        """Load the item into the Training tab. The item stays queued — after editing,
        use ⤓ on the same row to write the changes back."""
        if not (0 <= i < len(self.training_queue)):
            return
        self._apply_queue_item(self.training_queue[i])
        self.update_console(f"[queue] loaded run {i + 1} into the Training tab — edit, then use "
                            f"the ⤓ button on its queue row to save the changes back.\n")

    def _queue_update_from_tab(self, i):
        if not (0 <= i < len(self.training_queue)):
            return
        old = self.training_queue[i]
        item = self._queue_snapshot()
        item["id"], item["queued_at"] = old.get("id", item["id"]), old.get("queued_at", item["queued_at"])
        self.training_queue[i] = item
        self._save_training_queue()
        self._render_queue_window()
        self.update_console(f"[queue] run {i + 1} updated from the Training tab's current settings.\n")

    # Keys in self.entries that belong to OTHER tabs — skipped when collecting
    # a training-tab preset. Everything else in self.entries is fair game.
    # RESUME_TRAINING is run-specific state, not a preset knob: capturing it baked an
    # absolute state-dir path into every saved preset (and Load Last Train), silently
    # turning future runs into resumes of an old checkpoint.
    _NON_TRAINING_ENTRY_KEYS = {
        "SAMPLE_ENABLED", "SAMPLE_WIDTH", "SAMPLE_HEIGHT", "SAMPLE_STEPS",
        "SAMPLE_SEED", "SAMPLE_EVERY_N_EPOCHS", "SAMPLE_EVERY_N_STEPS",
        "SAMPLE_AT_FIRST", "SAMPLE_FLOW_SHIFT",
        "SAMPLE_NEGATIVE", "SAMPLE_CFG_SCALE",
        "MINIMAX_TURBO_STEPS", "MINIMAX_TURBO_STRENGTH",
        "RESUME_TRAINING",
    }

    def _collect_preset_values(self):
        """Snapshot every user-editable value on the Training tab into a preset dict.

        Iterates all of self.entries (skipping keys that belong to other tabs or to
        system-level settings) plus every known Training-tab Boolean/StringVar — so
        saved presets capture anything the user touched in the Training UI, not just
        a hand-curated subset.
        """
        preset = {}

        # Everything in self.entries that's on the Training tab
        for key, entry in self.entries.items():
            if key in self._NON_TRAINING_ENTRY_KEYS:
                continue
            try:
                if isinstance(entry, (tk.BooleanVar, tk.StringVar, tk.IntVar, tk.DoubleVar)):
                    preset[key] = entry.get()
                else:
                    # ttk.Entry / ttk.Combobox / ttk.Spinbox all expose .get()
                    preset[key] = entry.get()
            except Exception:
                pass

        # Training-tab toggles that live on dedicated vars (not in self.entries)
        def _grab(attr, key):
            if hasattr(self, attr):
                try:
                    preset[key] = getattr(self, attr).get()
                except Exception:
                    pass

        # Multi Concept's folders are a LIST on a dedicated attribute, so neither the entries
        # loop nor _grab reaches them — and without this "Load Settings From Last Train" brings
        # the toggle back with no second subject behind it.
        if getattr(self, "_concept_folder_vars", None):
            preset["MINIMAX_CONCEPT_DIRS"] = [v.get().strip()
                                              for v in self._concept_folder_vars]

        _grab("preserve_dist_var", "PRESERVE_DISTRIBUTION")
        _grab("fp8_var", "FP8")
        _grab("scaled_var", "SCALED")
        _grab("quant_4bit_var", "QUANT_4BIT")
        # Store the canonical key, never the display label — the wording can change without
        # invalidating everyone's saved presets.
        if hasattr(self, "quant_4bit_mode_var"):
            preset["QUANT_4BIT_MODE"] = self._base_precision()
        _grab("compile_blocks_var", "COMPILE_BLOCKS")
        # BooleanVars aren't in self.entries, so they need grabbing explicitly — unlike
        # KEEP_LAST_N_STATES, which is an Entry and is captured by the generic sweep above.
        _grab("save_state_var", "SAVE_STATE")
        _grab("save_state_on_train_end_var", "SAVE_STATE_ON_TRAIN_END")
        _grab("krea2_loss_watch_var", "KREA2_LOSS_WATCH")
        _grab("krea2_per_image_lr_var", "KREA2_PER_IMAGE_LR")
        _grab("krea2_auto_recaption_var", "KREA2_AUTO_RECAPTION")
        _grab("krea2_warmup_look_var", "KREA2_WARMUP_LOOK")
        _grab("krea2_finetune_var", "KREA2_FINETUNE")
        _grab("krea2_fast_ft_var", "KREA2_FAST_FT")
        _grab("krea2_reg_dir_var", "KREA2_REG_DIR")
        _grab("krea2_reg_mult_var", "KREA2_REG_MULT")
        _grab("krea2_ft_mode_var", "KREA2_FT_MODE")
        _grab("krea2_ft_blocks_var", "KREA2_FT_BLOCKS")
        _grab("krea2_ft_every_var", "KREA2_FT_EVERY")
        _grab("krea2_ft_fused_var", "KREA2_FT_FUSED")
        _grab("minimax_finetune_var", "MINIMAX_FINETUNE")
        _grab("minimax_ft_every_var", "MINIMAX_FT_EVERY")
        _grab("minimax_ft_scope_var", "MINIMAX_FT_SCOPE")
        _grab("minimax_ft_blockspec_var", "MINIMAX_FT_BLOCKSPEC")
        _grab("minimax_ft_fused_var", "MINIMAX_FT_FUSED")
        _grab("minimax_reg_dir_var", "MINIMAX_REG_DIR")
        _grab("minimax_reg_mult_var", "MINIMAX_REG_MULT")
        # MiniMax reference distillation. A plain StringVar, so the generic self.entries sweep
        # above does NOT see it — without this a queued distillation run loses its reference
        # and silently becomes an ordinary run (tests/test_minimax_distill_gui.py).
        _grab("minimax_distill_var", "MINIMAX_DISTILL")
        _grab("minimax_multiconcept_var", "MINIMAX_MULTICONCEPT")
        _grab("grad_checkpoint_var", "GRADIENT_CHECKPOINTING")
        _grab("fp8_text_encoder_var", "FP8_TEXT_ENCODER")
        _grab("adaptive_lr_var", "ADAPTIVE_LR")
        _grab("training_preset_var", "TARGET_LAYERS")
        _grab("ts_sampling_var", "TIMESTEP_SAMPLING")
        _grab("weighting_scheme_var", "WEIGHTING_SCHEME")
        _grab("enable_cache_var", "ENABLE_CACHE")
        # Dataset subsection (now living in Training → Other Options)
        _grab("dataset_enable_bucket_var", "ENABLE_BUCKET")
        _grab("dataset_no_upscale_var", "BUCKET_NO_UPSCALE")
        _grab("dataset_caption_ext_var", "DATASET_CAPTION_EXT")
        _grab("dataset_megapixels_var", "DATASET_MEGAPIXELS")
        _grab("dataset_batch_size_var", "DATASET_BATCH_SIZE")
        # Gradient mining
        _grab("gradient_mining_var", "GRADIENT_MINING")
        # Per-block custom training selection (only meaningful when TARGET_LAYERS=Custom)
        if hasattr(self, "training_block_vars") and self.training_block_vars:
            preset["TRAINING_BLOCKS"] = {k: v.get() for k, v in self.training_block_vars.items()}

        return preset

    def save_custom_preset(self):
        """Save current settings as a custom preset for the current architecture"""
        arch = self.architecture_var.get()

        # Prompt for preset name
        preset_name = simpledialog.askstring(
            "Save Preset",
            f"Enter a name for your preset (for {arch}):",
            parent=self.master
        )

        if not preset_name:
            return  # User cancelled

        # Validate name (no special chars that could cause filesystem issues)
        invalid_chars = '<>:"/\\|?*'
        if any(c in preset_name for c in invalid_chars):
            messagebox.showerror("Invalid Name", f"Preset name cannot contain: {invalid_chars}")
            return

        preset_name = preset_name.strip()
        if not preset_name:
            messagebox.showerror("Invalid Name", "Preset name cannot be empty")
            return

        # Check if preset already exists
        preset_dir = self.get_preset_dir_for_architecture(arch)
        preset_path = os.path.join(preset_dir, f"{preset_name}.json")

        if os.path.exists(preset_path):
            overwrite = messagebox.askyesno(
                "Preset Exists",
                f"A preset named '{preset_name}' already exists.\nDo you want to overwrite it?"
            )
            if not overwrite:
                return

        # Collect and save preset
        preset = self._collect_preset_values()
        try:
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset, f, indent=4)
            self.refresh_preset_combobox()
            messagebox.showinfo("Preset Saved", f"Preset '{preset_name}' saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save preset: {str(e)}")

    def load_custom_preset(self, event=None):
        """Load a preset from the combobox selection — built-in or user-saved."""
        preset_name = self.custom_preset_var.get()
        if not preset_name:
            return

        # Check built-in presets first (architecture-specific)
        builtins = self._builtins_for_arch(self.architecture_var.get())
        if preset_name in builtins:
            self._apply_preset_values(builtins[preset_name])
            messagebox.showinfo("Preset Loaded", f"Loaded built-in preset '{preset_name}'")
            return

        # Otherwise, look for a user-saved preset on disk
        arch = self.architecture_var.get()
        preset_dir = self.get_preset_dir_for_architecture(arch)
        preset_path = os.path.join(preset_dir, f"{preset_name}.json")

        if not os.path.exists(preset_path):
            messagebox.showerror("Error", f"Preset file not found: {preset_name}")
            self.refresh_preset_combobox()
            return

        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                preset = json.load(f)
            self._apply_preset_values(preset)
            messagebox.showinfo("Preset Loaded", f"Loaded preset '{preset_name}'")
        except json.JSONDecodeError:
            messagebox.showerror("Error", f"Preset file is corrupted: {preset_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preset: {str(e)}")

    def delete_custom_preset(self):
        """Delete the currently selected custom preset"""
        preset_name = self.custom_preset_var.get()
        if not preset_name:
            messagebox.showinfo("Info", "Please select a preset to delete")
            return

        arch = self.architecture_var.get()

        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete the preset '{preset_name}'?\n\nThis action cannot be undone."
        )
        if not confirm:
            return

        preset_dir = self.get_preset_dir_for_architecture(arch)
        preset_path = os.path.join(preset_dir, f"{preset_name}.json")

        try:
            if os.path.exists(preset_path):
                os.remove(preset_path)
                self.refresh_preset_combobox()
                messagebox.showinfo("Preset Deleted", f"Preset '{preset_name}' deleted successfully")
            else:
                messagebox.showerror("Error", f"Preset file not found: {preset_name}")
                self.refresh_preset_combobox()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete preset: {str(e)}")