import os

import tkinter as tk
from tkinter import messagebox, ttk

from fizgig_gui.core.config.constants import COLORS, FONT_FAMILY, BG_COLOR
from fizgig_gui.core.config.prefs import FLORENCE_MODELS, QWEN_CAPTION_MODEL, FLORENCE_DEFAULT_MODEL, QWEN_CUSTOM_TASK, FLORENCE_TASKS, save_prefs

class CaptionModelMixin:
    def _qwen_captioner_path(self) -> str:
        """The Krea 2 text-encoder file, if it's set AND actually on disk — else ""."""
        p = self._krea2_pref("krea2_text_encoder") if hasattr(self, "prefs_vars") else ""
        return p if (p and os.path.isfile(p)) else ""

    def _qwen_captioner_hint(self) -> str:
        """Small nudge next to the Model dropdown when Qwen3-VL isn't offered -- someone who
        already has the weights (from ComfyUI, a pod image, wherever) has no other way to
        learn that this is an unset/broken Preferences path rather than a missing feature."""
        if self._qwen_captioner_path():
            return ""
        raw = self._krea2_pref("krea2_text_encoder") if hasattr(self, "prefs_vars") else ""
        if raw:
            return f"(Qwen3-VL path set but not found: {os.path.basename(raw)} — check Preferences)"
        return "(Already have the Qwen3-VL weights? Set the path in Preferences to caption with it)"

    def _caption_model_values(self):
        """Model dropdown contents. Qwen3-VL is offered whenever its file exists — it captions to
        .txt like Florence does, so it serves Klein datasets just as well as Krea 2 ones."""
        return FLORENCE_MODELS + ([QWEN_CAPTION_MODEL] if self._qwen_captioner_path() else [])

    def _default_caption_model(self) -> str:
        """What to select when there is no usable saved choice.

        Qwen3-VL wherever its file exists: it follows an editable instruction written to the
        captioning doctrine, so it produces better training data than Florence out of the box.
        Florence is the fallback for anyone without that file."""
        return QWEN_CAPTION_MODEL if self._qwen_captioner_path() else FLORENCE_DEFAULT_MODEL

    def _initial_caption_model(self) -> str:
        """Startup selection: the saved choice if it is still offered, else the default.

        Validating against the live values list is what stops a saved 'Qwen3-VL …' from being
        displayed after its file has been deleted — the combobox would otherwise show a value
        absent from its own options, since _refresh_caption_model_values only runs from the
        Preferences write-trace and never at startup."""
        saved = str(self.last_used.get("caption_model", "") or "").strip()
        return saved if saved in self._caption_model_values() else self._default_caption_model()

    def _refresh_caption_model_values(self):
        """Re-read the Preferences path and rebuild the dropdown (bound to that pref's trace)."""
        if not hasattr(self, "caption_model_combo"):
            return
        try:
            vals = self._caption_model_values()
            self.caption_model_combo.configure(values=vals)
            if self.caption_model_var.get() not in vals:
                # The path was cleared while Qwen3-VL was selected — fall back rather than leave a
                # selection that no longer resolves to anything loadable.
                self.caption_model_var.set(FLORENCE_DEFAULT_MODEL)
                self._on_caption_model_changed()
            if hasattr(self, "caption_model_hint_label"):
                self.caption_model_hint_label.configure(text=self._qwen_captioner_hint())
        except tk.TclError:
            pass

    def _restore_caption_selection(self):
        """Apply the saved caption model and its remembered task to the live vars.

        Each captioner remembers its OWN task: their task lists have nothing in common (Florence
        takes task tokens, Qwen an instruction preset), so a single shared value meant flipping
        models threw away whichever task you had picked. Same snapshot-on-leave /
        restore-on-return shape as the Training tab's per-family memory.

        Called at startup and by the tests, so both exercise the identical path."""
        self._caption_task_memory = dict(self.last_used.get("caption_tasks", {}) or {})
        model = self._initial_caption_model()
        # Upgrade path: the old flat caption_task belongs to whichever model is selected now.
        legacy = str(self.last_used.get("caption_task", "") or "").strip()
        if legacy and model not in self._caption_task_memory:
            self._caption_task_memory[model] = legacy
        self._caption_model_last = model
        self.caption_model_var.set(model)
        remembered = self._caption_task_memory.get(model)
        if remembered:
            self.caption_task_var.set(remembered)
        self._on_caption_model_changed()

    def _is_qwen_captioner(self) -> bool:
        return self.caption_model_var.get() == QWEN_CAPTION_MODEL

    def _qwen_task_labels(self):
        from fizgig.krea2.embedder import CAPTION_TASKS
        return [label for (label, _instr, _tok) in CAPTION_TASKS.values()] + [QWEN_CUSTOM_TASK]

    def _on_caption_model_changed(self):
        """Swap the Task dropdown between Florence's task tokens and Qwen3-VL's instruction menu,
        restore that model's remembered task, and show the instruction editor only for the model
        that has one."""
        try:
            new_model = self.caption_model_var.get()
            prev_model = getattr(self, "_caption_model_last", None)
            model_changed = bool(prev_model) and prev_model != new_model
            if model_changed:
                # Snapshot the model we're leaving before its task list is replaced.
                self._caption_task_memory[prev_model] = self.caption_task_var.get()
            self._caption_model_last = new_model

            if self._is_qwen_captioner():
                from fizgig.krea2.embedder import CAPTION_TASKS, DEFAULT_CAPTION_TASK
                labels = self._qwen_task_labels()
                default_task = CAPTION_TASKS[DEFAULT_CAPTION_TASK][0]
            else:
                labels = list(FLORENCE_TASKS)
                default_task = "<DETAILED_CAPTION>"
            self.caption_task_combo.configure(values=labels)
            remembered = self._caption_task_memory.get(new_model)
            # On a real model change ALWAYS prefer this model's remembered task. Only correcting
            # an out-of-range value would restore it merely by luck — it works today because the
            # two task lists share nothing, and would quietly stop working the moment they did.
            if model_changed or self.caption_task_var.get() not in labels:
                self.caption_task_var.set(remembered if remembered in labels else default_task)
            if self._is_qwen_captioner():
                self.caption_edit_instr_btn.pack(side=tk.LEFT, padx=(8, 0))
            else:
                self.caption_edit_instr_btn.pack_forget()
            self._on_caption_task_changed()
        except (tk.TclError, AttributeError):
            pass
        self._save_last_used_paths()

    def _on_caption_task_changed(self):
        """Seed Max Tokens from the task's suggested length, so each Qwen task gets a sensible
        budget without a second control. Florence keeps whatever is in the box."""
        # Record against the current model whether or not it's Qwen, so the memory is right even
        # when the task changes without a model switch.
        try:
            self._caption_task_memory[self.caption_model_var.get()] = self.caption_task_var.get()
        except (tk.TclError, AttributeError):
            pass
        if not self._is_qwen_captioner():
            self._save_last_used_paths()
            return
        try:
            from fizgig.krea2.embedder import CAPTION_TASKS
            label = self.caption_task_var.get()
            for _key, (lbl, _instr, tok) in CAPTION_TASKS.items():
                if lbl == label:
                    self.caption_max_tokens_var.set(str(tok))
                    break
        except Exception:
            pass
        self._save_last_used_paths()

    def _caption_task_key(self, label: str) -> str:
        """Canonical key for a task label — 'custom' for the free-slot entry."""
        from fizgig.krea2.embedder import CAPTION_TASKS
        for key, (lbl, _instr, _tok) in CAPTION_TASKS.items():
            if lbl == label:
                return key
        return "custom"

    def _caption_overrides(self) -> dict:
        """{task_key: instruction} — every preset the user has edited.

        Each preset is independently editable and keeps its own default, so a tuned Training
        caption and a tuned Exhaustive one can coexist. Migrates the older single-slot pref
        (one shared instruction) into the 'custom' entry."""
        raw = self.prefs.get("caption_qwen_instructions")
        out = dict(raw) if isinstance(raw, dict) else {}
        legacy = str(self.prefs.get("caption_qwen_instruction", "") or "").strip()
        if legacy and "custom" not in out:
            out["custom"] = legacy
        return out

    def _caption_builtin_for_key(self, key: str) -> str:
        """The shipped text for a task key. 'custom' has none, so it borrows the default task's."""
        from fizgig.krea2.embedder import CAPTION_TASKS, DEFAULT_CAPTION_TASK
        if key in CAPTION_TASKS:
            return CAPTION_TASKS[key][1]
        return CAPTION_TASKS[DEFAULT_CAPTION_TASK][1]

    def _caption_instruction_for_task(self, label: str, *, builtin_only: bool = False) -> str:
        """The instruction for a task label — the user's edited version if there is one."""
        key = self._caption_task_key(label)
        if not builtin_only:
            saved = str(self._caption_overrides().get(key, "") or "").strip()
            if saved:
                return saved
        return self._caption_builtin_for_key(key)

    def _caption_task_is_edited(self, label: str) -> bool:
        key = self._caption_task_key(label)
        return bool(str(self._caption_overrides().get(key, "") or "").strip())

    def _resolve_caption_instruction(self) -> str:
        """The instruction that will actually be sent for the current selection."""
        return self._caption_instruction_for_task(self.caption_task_var.get())

    def _open_caption_instruction_editor(self):
        """Show (and allow editing of) the instruction sent to Qwen3-VL for the current task.

        Saving stores it in prefs against THIS preset's key (each of the four keeps its own
        wording), so what you edited is what runs — here AND in the trainer's auto-recaption.
        Saving the shipped text back drops the override rather than storing a redundant copy."""
        opened_from = self.caption_task_var.get()
        edited = self._caption_task_is_edited(opened_from)
        dialog = tk.Toplevel(self.master)
        dialog.title(f"Captioning instruction — {opened_from}")
        dialog.configure(bg=BG_COLOR)
        dialog.minsize(660, 420)

        # Buttons packed BOTTOM first so a long instruction can never push them off the edge
        # (the v2.8.5 caption-dialog fix).
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(side=tk.BOTTOM, pady=10)

        tk.Label(dialog,
                 text=f"{opened_from}{'  (edited)' if edited else ''}",
                 font=(FONT_FAMILY, 11, "bold"), fg=COLORS["text_primary"],
                 bg=BG_COLOR).pack(anchor=tk.W, padx=14, pady=(14, 2))
        tk.Label(dialog,
                 text="This is the whole prompt the model is given, alongside the image. Saving "
                      "edits THIS preset — each one keeps its own wording, and Restore default "
                      "puts the shipped text back.",
                 font=(FONT_FAMILY, 10), fg=COLORS["text_explain"], bg=BG_COLOR,
                 wraplength=620, justify=tk.LEFT).pack(anchor=tk.W, padx=14, pady=(0, 8))

        instr_text = tk.Text(dialog, height=9, wrap="word", bg=COLORS["bg_surface"],
                             fg=COLORS["text_primary"], insertbackground=COLORS["text_primary"],
                             font=(FONT_FAMILY, 10), relief="flat", highlightthickness=1,
                             highlightbackground=COLORS["border"])
        instr_text.pack(fill=tk.BOTH, expand=True, padx=14)
        instr_text.insert("1.0", self._caption_instruction_for_task(opened_from))

        # The encoder's own system prompt, shown for context and deliberately NOT editable: it is
        # what conditions training and inference, and must stay byte-identical to ComfyUI's
        # Text-Encode-(Krea2) node. Editing it would silently change every cached embedding.
        # Read from the module constant so this can never drift from what actually runs.
        from fizgig.krea2.embedder import ENCODE_SYSTEM_DESCRIPTOR as sys_prompt
        tk.Label(dialog, text="For reference — the encoder's own system prompt (not editable):",
                 font=(FONT_FAMILY, 9, "bold"), fg=COLORS["text_secondary"],
                 bg=BG_COLOR).pack(anchor=tk.W, padx=14, pady=(10, 2))
        tk.Label(dialog, text=sys_prompt, font=(FONT_FAMILY, 9, "italic"),
                 fg=COLORS["text_explain"], bg=BG_COLOR, wraplength=620,
                 justify=tk.LEFT).pack(anchor=tk.W, padx=14)
        tk.Label(dialog,
                 text="That one conditions training and inference and must match ComfyUI exactly, "
                      "so Fizgig keeps it fixed. It is not the captioning instruction above.",
                 font=(FONT_FAMILY, 9), fg=COLORS["text_explain"], bg=BG_COLOR,
                 wraplength=620, justify=tk.LEFT).pack(anchor=tk.W, padx=14, pady=(2, 10))

        def restore_default():
            instr_text.delete("1.0", tk.END)
            instr_text.insert("1.0", self._caption_instruction_for_task(opened_from,
                                                                       builtin_only=True))

        def save_instruction():
            text = instr_text.get("1.0", tk.END).strip()
            if not text:
                messagebox.showerror("Empty instruction",
                                     "The instruction can't be empty — use Restore default.")
                return
            key = self._caption_task_key(opened_from)
            overrides = self._caption_overrides()
            if text == self._caption_builtin_for_key(key):
                # Saving the shipped text back = "no override" rather than a redundant copy, so
                # Restore default + Save genuinely returns the preset to stock.
                overrides.pop(key, None)
                msg = f"'{opened_from}' restored to its default instruction.\n"
            else:
                overrides[key] = text
                msg = f"'{opened_from}' saved — this preset now uses your instruction.\n"
            self.prefs["caption_qwen_instructions"] = overrides
            self.prefs.pop("caption_qwen_instruction", None)   # retire the old single slot
            save_prefs(self.prefs)
            self._save_last_used_paths()
            self.update_caption_log(msg)
            dialog.destroy()

        ttk.Button(btn_frame, text="Restore default", command=restore_default).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Save", command=save_instruction).pack(side=tk.LEFT, padx=5)
