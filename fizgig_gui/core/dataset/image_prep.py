import math
import os
import subprocess

import tkinter as tk
from tkinter import messagebox

from PIL import Image

from fizgig_gui.core.paths import REPO_ROOT as _REPO_ROOT, FIZGIG_DIR as _FIZGIG_DIR


class ImagePrepMixin:
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

    def _launch_gizmo(self):
        """Open Gizmo, the clip and voice prep tool, as its own process.

        On a pod this button is the ONLY route — there is no desktop icon and a .bat is useless
        on Linux. It works because the container runs openbox and DISPLAY=:1 is set in the image,
        both of which this process already inherited, so the child does too.

        Not a "close Fizgig and open Gizmo" flow, deliberately: on a pod Fizgig is PID 1's
        successor and closing it would kill the pod.
        """
        script = os.path.join(_FIZGIG_DIR, "gizmo.pyw" if os.name == "nt" else "gizmo.py")
        if not os.path.isfile(script):
            messagebox.showerror("Gizmo not found",
                                 f"{os.path.basename(script)} is missing from your Fizgig folder. "
                                 "Update Fizgig to get it.")
            return

        proc = getattr(self, "_gizmo_proc", None)
        if proc is not None and proc.poll() is None:
            messagebox.showinfo("Gizmo is already open",
                                "Gizmo is running — look for its window behind this one.")
            return

        exe = self._venv_python()
        if os.name == "nt":
            # pythonw, or the child inherits a console window Fizgig itself does not have.
            cand = os.path.join(_FIZGIG_DIR, "venv", "Scripts", "pythonw.exe")
            if os.path.isfile(cand):
                exe = cand
        try:
            self._gizmo_proc = subprocess.Popen([exe, script], cwd=_FIZGIG_DIR)
        except Exception as exc:
            messagebox.showerror("Gizmo could not start", f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _atomic_png_save(img, output_path):
        """Write the PNG to a temp file then os.replace into place. In-place mode saves
        straight over the original — a crash, full disk or End Task mid-write used to
        truncate the source photo beyond recovery."""
        tmp = output_path + ".fizgig-tmp"
        try:
            img.save(tmp, "PNG")
            os.replace(tmp, output_path)
        except Exception:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise

    def _safe_output_path(self, filepath, output_path):
        """Never overwrite a DIFFERENT existing file.

        Every prep mode writes `<stem>.png`, so a folder holding photo.jpg AND an
        unrelated photo.png would have the .jpg's output destroy the .png (and
        _handle_original then deletes the .jpg — one photo gone, silently).
        In-place re-save of the same file is fine; a genuine collision gets a
        `_2`/`_3`... suffix instead, with a log line."""
        if not os.path.exists(output_path):
            return output_path
        try:
            if os.path.samefile(filepath, output_path):
                return output_path  # in-place re-save of itself
        except OSError:
            pass
        stem, ext = os.path.splitext(output_path)
        n = 2
        candidate = f"{stem}_{n}{ext}"
        while os.path.exists(candidate):
            n += 1
            candidate = f"{stem}_{n}{ext}"
        self._log(f"Name collision: {os.path.basename(output_path)} already exists — "
                  f"writing {os.path.basename(candidate)} instead\n")
        return candidate

    def _stash_original_if_inplace(self, filepath, output_path, output_folder, replace_originals):
        """Call BEFORE saving. Keep-safe mode + in-place PNG output means the save OVERWRITES
        the original — by the time _handle_original ran, there was nothing left to move
        (issue #43: PNG originals silently destroyed while JPGs were kept). A COPY rather
        than a move, because PIL may still hold the source file open at this point and a
        move of an open file fails on Windows; the end state is identical — original
        content in originals/, processed image at the original name."""
        if replace_originals or filepath != output_path or not os.path.exists(filepath):
            return
        if not hasattr(self, '_originals_dir_cache'):
            self._originals_dir_cache = {}
        if output_folder not in self._originals_dir_cache:
            self._originals_dir_cache[output_folder] = self._get_originals_dir(output_folder)
        originals_dir = self._originals_dir_cache[output_folder]
        os.makedirs(originals_dir, exist_ok=True)
        import shutil
        shutil.copy2(filepath, os.path.join(originals_dir, os.path.basename(filepath)))

    def _handle_original(self, filepath, output_path, output_folder, replace_originals):
        """Handle the original file: delete if replacing, move to subfolder if preserving.
        The in-place keep-safe case is covered by _stash_original_if_inplace BEFORE the save
        — by this point the overwrite has already happened, correctly for replace mode and
        harmlessly for keep-safe (the original is already copied out)."""
        if filepath == output_path:
            return  # Output overwrote original, nothing to do
        if replace_originals:
            os.remove(filepath)
        else:
            if not hasattr(self, '_originals_dir_cache'):
                self._originals_dir_cache = {}
            if output_folder not in self._originals_dir_cache:
                self._originals_dir_cache[output_folder] = self._get_originals_dir(output_folder)
            originals_dir = self._originals_dir_cache[output_folder]
            os.makedirs(originals_dir, exist_ok=True)
            import shutil
            dest = os.path.join(originals_dir, os.path.basename(filepath))
            shutil.move(filepath, dest)

    def _get_originals_dir(self, output_folder):
        """Find the next available originals folder (originals, originals_2, originals_3, etc.)."""
        candidate = os.path.join(output_folder, "originals")
        if not os.path.isdir(candidate):
            return candidate
        # Check if it has any images
        has_images = any(
            os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS
            for f in os.listdir(candidate) if os.path.isfile(os.path.join(candidate, f))
        )
        if not has_images:
            return candidate
        # Find next numbered folder
        n = 2
        while True:
            candidate = os.path.join(output_folder, f"originals_{n}")
            if not os.path.isdir(candidate):
                return candidate
            has_images = any(
                os.path.splitext(f)[1].lower() in self.IMAGE_EXTENSIONS
                for f in os.listdir(candidate) if os.path.isfile(os.path.join(candidate, f))
            )
            if not has_images:
                return candidate
            n += 1

    def _get_image_files(self, folder):
        """Scan folder for image files, return sorted list of full paths."""
        files = []
        for filename in sorted(os.listdir(folder)):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath) and os.path.splitext(filename)[1].lower() in self.IMAGE_EXTENSIONS:
                files.append(filepath)
        return files

    def _load_image(self, filepath):
        """Load an image and convert to RGB/RGBA."""
        img = Image.open(filepath)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGBA')
        else:
            img = img.convert('RGB')
        return img

    @staticmethod
    def _bucket_step():
        """The resolution grid training buckets on (RESOLUTION_STEPS). Read from the dataset
        module so prep and bucketing can never drift apart; 16 if the import isn't available."""
        try:
            import sys as _sys
            _src = os.path.join(_REPO_ROOT, "src")
            if _src not in _sys.path:
                _sys.path.insert(0, _src)
            from fizgig.dataset.image_dataset import RESOLUTION_STEPS
            return int(RESOLUTION_STEPS)
        except Exception:
            return 16

    def _prep_target_area(self, mp):
        """Training's real target AREA for a megapixel setting.

        Derived exactly as the dataset TOML writer derives `resolution` — floor the square side
        to the bucket grid (1.0 MP -> 992x992 = 984 064 px, not 1 000 000). Matching it means
        prep lands at or just UNDER what training asks for, which keeps training's no-upscale
        path: the cache step then resamples and crops nothing at all."""
        step = self._bucket_step()
        side = max(step, int(math.sqrt(max(0.0, mp) * 1_000_000)) // step * step)
        return side * side

    def _prep_output_size(self, size, target_area):
        """The (w, h) `_resize_image` would produce for `size` — same maths, no pixels touched.
        Used by the summary card to show a worked example before anything is written."""
        width, height = size
        cur_area = width * height
        if cur_area <= target_area:
            return width, height
        step = self._bucket_step()
        scale = math.sqrt(target_area / cur_area)
        return (max(step, int(width * scale) // step * step),
                max(step, int(height * scale) // step * step))

    def _resize_image(self, img, target_area):
        """Resize to ~`target_area` PIXELS, preserving aspect ratio. Never upscales.
        Returns (img, resized_bool).

        Area, not longest edge: training chooses its resolution by area and — with No Upscale on,
        the default — leaves any image already at or under the target exactly as it is. A
        longest-edge cap therefore pushed every non-square image permanently below the training
        target (a 3:4 photo trained at 75% of the pixels it could have, 16:9 at 56%; issue #44).

        Both sides are floored to the bucket step (16). Training floors to that grid anyway, so
        doing it here makes the saved file exactly what trains, and lands just under the target
        area — which keeps training's no-upscale path and means the cache step resamples and
        crops nothing at all."""
        width, height = img.size
        cur_area = width * height
        if cur_area <= target_area:
            return img, False                      # never upscale — it would only invent detail
        step = self._bucket_step()
        scale = math.sqrt(target_area / cur_area)
        new_width = max(step, int(width * scale) // step * step)
        new_height = max(step, int(height * scale) // step * step)
        if (new_width, new_height) == (width, height):
            return img, False
        return img.resize((new_width, new_height), Image.LANCZOS), True

    def _select_face(self, faces, face_mode):
        """Select a face from detected faces based on face_mode. Returns (FaceInfo, note_str) or (None, note_str)."""
        if not faces:
            return None, ""
        if face_mode == "largest_male":
            selected = self.face_detector.get_largest_by_gender(faces, "male", fallback_to_any=True)
            note = "  Note: No male face, using largest face\n" if (selected and selected.gender != "male") else ""
        elif face_mode == "largest_female":
            selected = self.face_detector.get_largest_by_gender(faces, "female", fallback_to_any=True)
            note = "  Note: No female face, using largest face\n" if (selected and selected.gender != "female") else ""
        else:
            selected = self.face_detector.get_largest(faces)
            note = ""
        return selected, note

    def _get_next_facecrop_index(self, folder):
        """Find next available FaceCrop_NNN index in a folder."""
        import glob as glob_module
        existing = glob_module.glob(os.path.join(glob_module.escape(folder), "FaceCrop_*.png"))
        max_idx = 0
        for f in existing:
            basename = os.path.splitext(os.path.basename(f))[0]
            parts = basename.split("_")
            if len(parts) >= 2:
                try:
                    max_idx = max(max_idx, int(parts[1]))
                except ValueError:
                    pass
        return max_idx + 1

    def _log(self, text):
        """Append text to the convert log (preserves user scroll position). Marshals to the
        main thread — Tk widget writes from a worker are a hard crash, not an exception."""
        import threading
        if threading.current_thread() is not threading.main_thread():
            self.master.after(0, self._log, text)
            return
        self._append_global_log(text)
        try:
            at_bottom = self.convert_log.yview()[1] >= 0.999
        except Exception:
            at_bottom = True
        self.convert_log.insert(tk.END, text)
        if at_bottom:
            self.convert_log.see(tk.END)
        self.convert_log.see(tk.END)
        self.master.update_idletasks()