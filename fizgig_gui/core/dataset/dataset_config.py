import math
import os
import re

from fizgig_gui.core.config.last_used import DATASET_DIR
from fizgig_gui.core.config.prefs import _persist_disabled


class DatasetConfigMixin:
    @staticmethod
    def _cache_dir_for(cache_root: str, image_dir: str) -> str:
        """`<cache_root>/<folder name>-<hash of full path>` — one cache dir per image folder.

        The trainer builds its item list by GLOBBING the cache directory, so two datasets sharing
        one folder would train on each other's leftovers; the dataset layer refuses outright
        (dataset/config.py: "cache_directory must be unique for each dataset"). The hash keeps it
        stable per folder and unique across same-named folders, and normalises case and trailing
        slash — which is also why the GUI must treat `C:\\A` and `c:/a/` as the SAME folder when
        validating Multi Concept."""
        import hashlib
        norm = image_dir.lower().replace("\\", "/").rstrip("/")
        h = hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]
        nm = "".join(c if (c.isalnum() or c in "-_") else "_"
                     for c in os.path.basename(image_dir.rstrip("/\\"))) or "dataset"
        return os.path.join(cache_root, f"{nm}-{h}")

    def _dataset_folders(self) -> list:
        """Every image folder that should become a `[[datasets]]` block, in order.

        Normally just the Start-tab folder. Multi Concept (MiniMax only) appends the extra
        concept folders, so each subject gets its own block — which is what makes reference
        distillation pair each image only with OTHERS OF ITS OWN SUBJECT (the rotation in
        scripts/minimax_cache_text.py runs per dataset block).

        The Start folder stays the single source of truth for Captions, Image Prep, the Look
        filter and the gallery; only the TOML writer and validation ever see this list."""
        folders = [self.image_folder_var.get().strip()]
        if (getattr(self, "minimax_multiconcept_var", None) is not None
                and self.minimax_multiconcept_var.get() and self._is_minimax_arch()):
            for var in getattr(self, "_concept_folder_vars", []):
                extra = var.get().strip()
                # Skip blanks and duplicates — the dataset layer hard-fails on a repeated
                # cache_directory, and two spellings of one path hash to the same place.
                if not extra:
                    continue
                norm = extra.lower().replace("\\", "/").rstrip("/")
                if norm in [f.lower().replace("\\", "/").rstrip("/") for f in folders]:
                    continue
                folders.append(extra)
        return [f for f in folders if f]

    def auto_save_dataset_config_silent(self):
        """Write the dataset TOML on startup and on every relevant edit (no Save button)."""
        if _persist_disabled():
            return
        try:
            built = self._build_dataset_toml_text()
            if built is None:
                return
            dataset_name, toml_content = built
            output_path = os.path.join(DATASET_DIR, f"{dataset_name}.toml")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(toml_content)
            self._dataset_config_var.set(output_path)
            # Deliberately NOT settings["DATASET_CONFIG"] (#98): during a run that key
            # points at the run's frozen snapshot, and this writer fires on every edit —
            # re-pointing it mid-pipeline is exactly the dataset-swap race. The launch
            # collects the live path from _dataset_config_var itself.
        except Exception:
            pass  # Silently fail - user can manually save if needed

    _RUN_SNAPSHOT_DIRNAME = "run_snapshots"

    def _snapshot_dataset_config_for_run(self, live_path, resuming=False, prev_config=None):
        """Copy the dataset TOML to an immutable per-run file and return its path (#98).

        Each launched run trains from its own frozen copy of the config, so edits made on
        the Start tab while a run initialises (or trains) can never retarget it. On
        resume, the run's EXISTING snapshot (prev_config — captured by the launch BEFORE
        the settings collection overwrites the key with the live path) is kept: a paused
        run must finish on the dataset it started with, not whatever the Start tab shows
        now. Any failure falls back to the live path — the pre-#98 behaviour, never
        worse. Deliberately NOT _persist_disabled-guarded: snapshots are ephemeral,
        pruned, gitignored copies, and the guard would make this untestable — headless
        tests patch DATASET_DIR instead."""
        import shutil as _shutil
        import time as _time
        try:
            if resuming:
                prev = str(prev_config or "")
                if (os.path.basename(os.path.dirname(prev)) == self._RUN_SNAPSHOT_DIRNAME
                        and os.path.isfile(prev)):
                    return prev
            if not live_path or not os.path.isfile(live_path):
                return live_path
            snap_dir = os.path.join(DATASET_DIR, self._RUN_SNAPSHOT_DIRNAME)
            os.makedirs(snap_dir, exist_ok=True)
            _name = re.sub(r"[^A-Za-z0-9._-]+", "_",
                           str(self.settings.get("LORA_NAME", "") or "run")) or "run"
            snap = os.path.join(snap_dir, f"{_name}-{int(_time.time() * 1000)}.toml")
            _shutil.copyfile(live_path, snap)
        except Exception:
            return live_path
        # Prune AFTER the snapshot is secured, in its own guard — a prune hiccup must
        # never un-freeze the run (the copy above already succeeded). Rule: the newest 12
        # always stay, and older files go only once they are ALSO older than 30 days —
        # so a paused run's frozen config outlives any burst of launches, whatever
        # output dir its sidecar lives in (no sidecar lookup: at this point the settings
        # already describe the LAUNCHING run, not the paused one).
        try:
            import glob as _glob
            _cutoff = _time.time() - 30 * 86400
            olds = sorted(_glob.glob(os.path.join(_glob.escape(snap_dir), "*.toml")),
                          key=lambda p: (os.path.getmtime(p) if os.path.exists(p) else 0),
                          reverse=True)
            for p in olds[12:]:
                try:
                    if (os.path.normpath(p) != os.path.normpath(snap)
                            and os.path.getmtime(p) < _cutoff):
                        os.remove(p)
                except OSError:
                    pass
        except Exception:
            pass
        return snap

    def _verify_frozen_dataset_config(self, path):
        """-> list of Start-tab folders MISSING from the frozen TOML, or None when all
        present (#98 follow-up). The auto-saver skips its rewrite silently when a dataset
        field fails to parse, so without this check a launch could freeze — and train —
        the PREVIOUS dataset under the new run's name. Unreadable/absent file returns
        None: existence is validate_inputs' job, and refusing here would double-report."""
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except Exception:
            return None

        def _norm(p):
            return str(p).strip().lower().replace("\\", "/").rstrip("/")

        listed = {_norm(m) for m in
                  re.findall(r'^\s*image_directory\s*=\s*"([^"]*)"', text, re.M)}
        if not listed:
            return None                    # not a TOML this writer produced — don't judge it
        missing = [f for f in self._dataset_folders() if f and _norm(f) not in listed]
        return missing or None

    def _build_dataset_toml_text(self):
        """-> (dataset_name, toml text), or None when the config is not writable yet.

        Split out of auto_save_dataset_config_silent so the CONTENT can be tested without
        touching the filesystem: the writer is guarded by _persist_disabled(), and defeating
        that guard in a test is how the real prefs got clobbered once already."""
        if True:
            dataset_name = self.dataset_name_var.get().strip()
            dataset_type = self.dataset_type_var.get()

            # Skip if no dataset name
            if not dataset_name:
                return

            # Check for invalid chars
            invalid_chars = '<>:"/\\|?*'
            if any(c in dataset_name for c in invalid_chars):
                return

            is_video = "Video" in dataset_type
            is_jsonl = "JSONL" in dataset_type

            # Check required fields exist
            if is_jsonl:
                jsonl_file = self.dataset_jsonl_file_var.get().strip()
                if not jsonl_file or not os.path.exists(jsonl_file):
                    return
            else:
                if is_video:
                    data_dir = self.dataset_video_dir_var.get().strip()
                else:
                    data_dir = self.image_folder_var.get().strip()
                if not data_dir or not os.path.exists(data_dir):
                    return

            # Validate numeric fields
            try:
                megapixels = float(self.dataset_megapixels_var.get())
                if megapixels <= 0:
                    return
                side = int(math.sqrt(megapixels * 1_000_000))
                side = (side // 16) * 16
                res_width = side
                res_height = side
                batch_size = int(self.dataset_batch_size_var.get())
                num_repeats = 1  # hardcoded — UI removed (Klein workflow always uses 1)
            except ValueError:
                return

            # Build TOML string
            toml_lines = ["[general]"]
            toml_lines.append(f"resolution = [{res_width}, {res_height}]")

            if not is_jsonl:
                caption_ext = self.dataset_caption_ext_var.get().strip()
                toml_lines.append(f'caption_extension = "{caption_ext}"')

            toml_lines.append(f"batch_size = {batch_size}")
            toml_lines.append(f"num_repeats = {num_repeats}")
            toml_lines.append(f"enable_bucket = {'true' if self.dataset_enable_bucket_var.get() else 'false'}")
            toml_lines.append(f"bucket_no_upscale = {'true' if self.dataset_no_upscale_var.get() else 'false'}")
            toml_lines.append("")
            toml_lines.append("[[datasets]]")

            # Cache directory is now sourced from Preferences (no longer a Dataset-tab field).
            # Each dataset gets its OWN subfolder: the trainer builds its item list by globbing
            # the cache directory, so two datasets sharing one folder would train on each other's
            # leftovers. <folder name>-<hash of full path> keeps it stable per dataset and unique
            # across same-named folders; switching datasets keeps both caches warm.
            cache_dir = self.prefs_vars["cache_dir"].get().strip() if "cache_dir" in self.prefs_vars else ""
            if cache_dir and not is_jsonl and not is_video:
                _cache_img_dir = self.image_folder_var.get().strip()
                if _cache_img_dir:
                    cache_dir = self._cache_dir_for(cache_dir, _cache_img_dir)

            # MiniMax uses ONE dataset at the Target Megapixels you set, exactly like Klein and
            # Krea 2. It briefly mirrored ai-toolkit's resolution: [512, 768, 1024], which copies
            # the dataset once per scale — every image trained three times per epoch. Dropped
            # (Peter, 4 Aug): with a bucketed dataset the scale variation is already there, and
            # tripling exposure to the same images per epoch is a much better way to overfit than
            # to teach scale invariance — most of all on tight face crops, where the extra copies
            # add no compositional diversity at all. It also silently tripled the work behind the
            # Epochs box, so "50 epochs" stopped meaning what it used to.
            if is_jsonl:
                jsonl_file = self.dataset_jsonl_file_var.get().strip().replace("\\", "/")
                if is_video:
                    toml_lines.append(f'video_jsonl_file = "{jsonl_file}"')
                else:
                    toml_lines.append(f'image_jsonl_file = "{jsonl_file}"')
            else:
                if is_video:
                    video_dir = self.dataset_video_dir_var.get().strip().replace("\\", "/")
                    toml_lines.append(f'video_directory = "{video_dir}"')
                else:
                    # One block per concept folder. Normally a single folder, so the output is
                    # byte-identical to the old single-block writer; Multi Concept adds a block
                    # per extra folder, each with its OWN cache directory (which is what keeps
                    # the reference rotation inside one subject).
                    _folders = self._dataset_folders()
                    _root = (self.prefs_vars["cache_dir"].get().strip()
                             if "cache_dir" in self.prefs_vars else "")
                    for _i, _folder in enumerate(_folders):
                        if _i:                       # the first block's header is already down
                            toml_lines.append("")
                            toml_lines.append("[[datasets]]")
                        toml_lines.append(
                            f'image_directory = "{_folder.replace(chr(92), "/")}"')
                        _cd = self._cache_dir_for(_root, _folder) if _root else ""
                        if _cd:
                            toml_lines.append(
                                f'cache_directory = "{_cd.replace(chr(92), "/")}"')
                    cache_dir = ""                   # emitted per block above

            if cache_dir:
                toml_lines.append(f'cache_directory = "{cache_dir.replace(chr(92), "/")}"')

            if is_video:
                try:
                    target_frames = [int(x.strip()) for x in self.dataset_target_frames_var.get().split(",")]
                    toml_lines.append(f"target_frames = [{', '.join(str(f) for f in target_frames)}]")
                    toml_lines.append(f'frame_extraction = "{self.dataset_frame_extraction_var.get()}"')
                    source_fps = float(self.dataset_source_fps_var.get())
                    toml_lines.append(f"source_fps = {source_fps}")
                except ValueError:
                    pass

            # Optional regularisation set (fine-tune only, per family): a second dataset
            # block marked is_reg, so the cache scripts pick it up for free and the trainer
            # can find its items. Only written when a folder is set — no folder, no block,
            # nothing changes. Fine-tune only: with FT off the block must not be written at
            # all, or the reg images would be cached and trained as ordinary subjects at
            # full LR. Arch-scoped: each family's reg row + FT toggle only speak for their
            # own family (a stale toggle from the other family must not leak a block in).
            if self._is_minimax_arch():
                reg_dir = (self.minimax_reg_dir_var.get().strip().replace("\\", "/")
                           if hasattr(self, "minimax_reg_dir_var") else "")
                reg_on = bool(getattr(self, "minimax_finetune_var", None)
                              and self.minimax_finetune_var.get())
            else:
                reg_dir = (self.krea2_reg_dir_var.get().strip().replace("\\", "/")
                           if hasattr(self, "krea2_reg_dir_var") else "")
                reg_on = bool(self._is_krea2_arch()
                              and getattr(self, "krea2_finetune_var", None)
                              and self.krea2_finetune_var.get())
            if reg_on and reg_dir and os.path.isdir(reg_dir) and not is_jsonl and not is_video:
                toml_lines.append("")
                toml_lines.append("[[datasets]]")
                toml_lines.append(f'image_directory = "{reg_dir}"')
                _reg_cache = self.prefs_vars["cache_dir"].get().strip() if "cache_dir" in self.prefs_vars else ""
                if _reg_cache:
                    # Its own subfolder for the same reason the subject set gets one: the
                    # trainer globs the cache dir, so a shared folder mixes the two sets.
                    _reg_cache = self._cache_dir_for(_reg_cache, reg_dir)
                    toml_lines.append(f'cache_directory = "{_reg_cache.replace(chr(92), "/")}"')
                toml_lines.append("is_reg = true")

            return dataset_name, "\n".join(toml_lines) + "\n"