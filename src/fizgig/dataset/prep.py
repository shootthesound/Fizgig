"""Image Prep: resize / convert to PNG / face close-ups, and the Look Consistency Filter's maths - shared by the
desktop app's Image Prep tab and the web UI's, so the two can't drift apart.

Everything happens IN the training folder. Prep never upscales and sizes by AREA (megapixels), floored to the
training bucket grid, so a prepped image is exactly what trains (see resize_image). Originals are moved to an
'originals' subfolder unless the user chose to replace them.

The Look Filter scores every face against 3 baselines (the mean similarity - one photo would bake its own angle and
light into every score) with the face embedder Image Prep's face settings pick (ArcFace for people, DINOv2 for furry /
anime faces: face_utils), and saves <folder>/fizgig_look_scores.json, which training's "Warm up look outliers" reads:
keep its format.
"""
import glob
import json
import math
import os
import shutil

from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
LOOK_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
MODES = ("Auto Prep (Face Crops)", "Resize Only", "Face Crop Only")
MEGAPIXELS = ["0.25", "0.37", "0.5", "0.75", "1.0", "1.5", "2.0", "2.4", "3.0", "4.2"]
FACE_TARGETS = ["Largest Face", "Largest Male Face", "Largest Female Face"]


def bucket_step():
    """The resolution grid training buckets on (RESOLUTION_STEPS), so prep and bucketing can never drift apart."""
    try:
        from fizgig.dataset.image_dataset import RESOLUTION_STEPS
        return int(RESOLUTION_STEPS)
    except Exception:
        return 16


def target_area(mp):
    """Training's real target AREA for a megapixel setting, derived as the dataset TOML writer derives `resolution`:
    floor the square side to the bucket grid (1.0 MP -> 992x992). Prep then lands at or just under what training asks
    for, which keeps training's no-upscale path: the cache step resamples and crops nothing."""
    step = bucket_step()
    side = max(step, int(math.sqrt(max(0.0, mp) * 1_000_000)) // step * step)
    return side * side


def output_size(size, area):
    """The (w, h) resize_image would produce for `size` - same maths, no pixels touched."""
    width, height = size
    if width * height <= area:
        return width, height
    step = bucket_step()
    scale = math.sqrt(area / (width * height))
    return (max(step, int(width * scale) // step * step), max(step, int(height * scale) // step * step))


def resize_image(img, area):
    """Resize to ~`area` pixels, aspect kept, never upscaling; (img, resized). Area, not longest edge: training picks
    its resolution by area, and a longest-edge cap left non-square images below the target for good (issue #44).
    Both sides floored to the bucket step, so the saved file is exactly what trains."""
    width, height = img.size
    if width * height <= area:
        return img, False                      # never upscale - it would only invent detail
    new = output_size((width, height), area)
    if new == (width, height):
        return img, False
    return img.resize(new, Image.LANCZOS), True


def face_mode_of(label):
    """The face target dropdown's label as the detector's mode."""
    if "Male" in (label or ""):
        return "largest_male"
    if "Female" in (label or ""):
        return "largest_female"
    return "largest_face"


def image_files(folder):
    """The folder's images (top level), sorted."""
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]


def load_image(path):
    img = Image.open(path)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        return img.convert('RGBA')
    return img.convert('RGB')


def atomic_png_save(img, output_path):
    """Write to a temp file then os.replace: in-place mode saves over the original, and a crash mid-write used to
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


def source_stats(folder, max_sample=40):
    """(image count, median area, median size) of the folder's top-level images, from headers only (cheap)."""
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    try:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
    except OSError:
        return 0, None, None
    sizes = []
    for p in files[:max_sample]:
        try:
            with Image.open(p) as im:
                sizes.append(im.size)
        except Exception:
            pass
    if not sizes:
        return len(files), None, None
    sizes.sort(key=lambda wh: wh[0] * wh[1])
    med = sizes[len(sizes) // 2]
    return len(files), med[0] * med[1], med


def summary(folder, mode, replace, prep_mp, train_mp=None):
    """The 'What will happen' card: {lines, warn (prep below training's megapixels), count}."""
    area = target_area(prep_mp)
    n, median_area, median_size = source_stats(folder)
    example = ""
    if median_size:
        w, h = output_size(median_size, area)
        if (w, h) != tuple(median_size):
            example = f" (your typical {median_size[0]}×{median_size[1]} → {w}×{h})"
    count = f"your {n} images" if n else "your images"
    sized = f"resized to about {prep_mp:g} MP{example} and saved as PNG"
    if mode == "Auto Prep (Face Crops)":
        what = f"{count} → {sized}, plus one face close-up each{f' (≈{n * 2} files)' if n else ''}"
    elif mode == "Face Crop Only":
        what = f"{count} → replaced by just the cropped face from each photo, saved as PNG"
    else:
        what = f"{count} → {sized}"
    originals = ("Your original files are replaced ⚠ there is no undo." if replace else
                 "Your originals are moved to the 'originals' subfolder — nothing is deleted.")
    lines = [f"{what}. Everything lands in your training folder. {originals}"]
    if mode != "Resize Only" and median_area is not None and median_area <= area:
        lines.append(f"⚠ Your images are already at or below {prep_mp:g} MP — face close-ups cut from them will be "
                     f"soft. If you have higher-res versions, prep from those instead.")
    if mode != "Resize Only":
        lines.append("Next: eyeball the face close-ups on the Captions tab and Remove any blurry ones before "
                     "captioning.")
    warn = ""
    if train_mp is not None and prep_mp < train_mp:
        warn = (f"⚠ Training is set to {train_mp:g} MP but prep is set to {prep_mp:g} MP — your images would be "
                f"shrunk below what training asks for, and training cannot get that detail back. Match them, or "
                f"prep higher.")
    return {"lines": lines, "warn": warn, "count": n}


class Prep:
    """One prep run over a folder. log(text) receives the Output Log's lines; detector is a face_utils.FaceDetector
    (needed by the face modes)."""

    def __init__(self, log, detector=None):
        self.log = log
        self.detector = detector
        self._originals = {}

    # ---- originals ----
    def _originals_dir(self, folder):
        """originals, else originals_2, _3... - the first that holds no images."""
        n = 1
        while True:
            cand = os.path.join(folder, "originals" if n == 1 else f"originals_{n}")
            if not os.path.isdir(cand) or not any(
                    os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
                    for f in os.listdir(cand) if os.path.isfile(os.path.join(cand, f))):
                return cand
            n += 1

    def _originals_for(self, folder):
        if folder not in self._originals:
            self._originals[folder] = self._originals_dir(folder)
        return self._originals[folder]

    def safe_output_path(self, filepath, output_path):
        """Never overwrite a DIFFERENT file: photo.jpg's output must not destroy an unrelated photo.png."""
        if not os.path.exists(output_path):
            return output_path
        try:
            if os.path.samefile(filepath, output_path):
                return output_path            # in-place re-save of itself
        except OSError:
            pass
        stem, ext = os.path.splitext(output_path)
        n = 2
        cand = f"{stem}_{n}{ext}"
        while os.path.exists(cand):
            n += 1
            cand = f"{stem}_{n}{ext}"
        self.log(f"Name collision: {os.path.basename(output_path)} already exists — "
                 f"writing {os.path.basename(cand)} instead\n")
        return cand

    def stash_if_inplace(self, filepath, output_path, folder, replace):
        """Before saving: keep-safe + in-place PNG means the save overwrites the original (issue #43), so copy it to
        originals/ first (a copy: PIL may still hold the source open, and moving an open file fails on Windows)."""
        if replace or filepath != output_path or not os.path.exists(filepath):
            return
        d = self._originals_for(folder)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(filepath, os.path.join(d, os.path.basename(filepath)))

    def handle_original(self, filepath, output_path, folder, replace):
        """After saving: delete the original (replace) or move it to originals/ (keep safe)."""
        if filepath == output_path:
            return
        if replace:
            os.remove(filepath)
        else:
            d = self._originals_for(folder)
            os.makedirs(d, exist_ok=True)
            shutil.move(filepath, os.path.join(d, os.path.basename(filepath)))

    # ---- faces ----
    def select_face(self, faces, face_mode):
        """(FaceInfo or None, note) by the face target."""
        if not faces:
            return None, ""
        for want in ("male", "female"):
            if face_mode == f"largest_{want}":
                sel = self.detector.get_largest_by_gender(faces, want, fallback_to_any=True)
                if not sel or sel.gender == want:
                    return sel, ""
                if sel.gender == "unknown":      # a YOLO face (furry / anime): no gender to go by
                    return sel, "  Note: the face finder can't tell gender on this face, using largest face\n"
                return sel, f"  Note: No {want} face, using largest face\n"
        return self.detector.get_largest(faces), ""

    @staticmethod
    def next_facecrop_index(folder):
        best = 0
        for f in glob.glob(os.path.join(glob.escape(folder), "FaceCrop_*.png")):
            parts = os.path.splitext(os.path.basename(f))[0].split("_")
            if len(parts) >= 2:
                try:
                    best = max(best, int(parts[1]))
                except ValueError:
                    pass
        return best + 1

    # ---- the three modes ----
    def run(self, mode, folder, prep_mp, face_label="Largest Face", face_padding=20.0, replace=False):
        self._originals = {}
        area = target_area(prep_mp)
        face_mode = face_mode_of(face_label)
        if self.detector is not None and mode != "Resize Only" and hasattr(self.detector, "describe"):
            self.log(f"Face finder: {self.detector.describe()}\n")
        if mode == "Auto Prep (Face Crops)":
            self.auto_prep(folder, folder, area, face_mode, face_padding, replace)
        elif mode == "Face Crop Only":
            self.face_crop_only(folder, folder, area, face_mode, face_padding, replace)
        else:
            self.resize_only(folder, folder, area, replace)

    def resize_only(self, source, output, area, replace):
        self.log("Mode: Resize Only\n\n")
        converted = skipped = errors = 0
        for filepath in image_files(source):
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            try:
                img = load_image(filepath)
                original = img.size
                img, resized = resize_image(img, area)
                w, h = img.size
                output_path = os.path.join(output, os.path.splitext(filename)[0] + ".png")
                if filepath == output_path and ext == '.png' and not resized:
                    self.log(f"Skipped (no changes): {filename}\n")
                    skipped += 1
                    img.close()
                    continue
                output_path = self.safe_output_path(filepath, output_path)
                self.stash_if_inplace(filepath, output_path, output, replace)
                atomic_png_save(img, output_path)
                info = f"{original[0]}x{original[1]} -> {w}x{h}" if resized else f"{w}x{h}"
                self.log(f"Converted: {filename} [{info}]\n")
                converted += 1
                img.close()
                self.handle_original(filepath, output_path, output, replace)
            except Exception as e:
                self.log(f"Error ({filename}): {e}\n")
                errors += 1
        self.log(f"\n--- Summary ---\nConverted: {converted} | Skipped: {skipped} | Errors: {errors}\n")

    def face_crop_only(self, source, output, area, face_mode, face_padding, replace):
        from fizgig.face_utils import crop_to_face
        self.log(f"Mode: Face Crop Only ({face_mode}, padding {face_padding}%)\n\n")
        converted = skipped = errors = face_crops = no_face = 0
        for filepath in image_files(source):
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            try:
                img = load_image(filepath)
                original = img.size
                cropped, crop_info = False, ""
                try:
                    faces = self.detector.detect_from_pil(img)
                    if faces:
                        sel, note = self.select_face(faces, face_mode)
                        if note:
                            self.log(note)
                        if sel:
                            img = crop_to_face(img, sel, face_padding)
                            cropped = True
                            face_crops += 1
                            crop_info = f" [face: {sel.gender}]"
                    else:
                        self.log(f"  No face in {filename}, skipping crop\n")
                        no_face += 1
                except Exception as fe:
                    self.log(f"  Face error ({filename}): {fe}\n")
                img, resized = resize_image(img, area)
                w, h = img.size
                output_path = os.path.join(output, os.path.splitext(filename)[0] + ".png")
                if filepath == output_path and ext == '.png' and not resized and not cropped:
                    self.log(f"Skipped (no changes): {filename}\n")
                    skipped += 1
                    img.close()
                    continue
                output_path = self.safe_output_path(filepath, output_path)
                self.stash_if_inplace(filepath, output_path, output, replace)
                atomic_png_save(img, output_path)
                info = f"{original[0]}x{original[1]} -> {w}x{h}" if (resized or cropped) else f"{w}x{h}"
                self.log(f"Converted: {filename} [{info}]{crop_info}\n")
                converted += 1
                img.close()
                self.handle_original(filepath, output_path, output, replace)
            except Exception as e:
                self.log(f"Error ({filename}): {e}\n")
                errors += 1
        self.log(f"\n--- Summary ---\nConverted: {converted} | Skipped: {skipped} | Errors: {errors}\n")
        self.log(f"Face crops: {face_crops} | No face: {no_face}\n")

    def auto_prep(self, source, output, area, face_mode, face_padding, replace):
        """Resize the originals AND cut a face close-up from each HIGH-RES original (before it is overwritten or
        moved)."""
        from fizgig.face_utils import crop_to_face
        self.log("Mode: Auto Prep (Face Crops)\n")
        self.log(f"Face target: {face_mode}, padding: {face_padding}%\n")
        self.log(f"Output: {output}\n\n")
        converted = skipped = errors = crops_made = no_face = 0
        crop_index = self.next_facecrop_index(output)
        for filepath in image_files(source):
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            base = os.path.splitext(filename)[0]
            if base.startswith("FaceCrop_"):
                self.log(f"Skipped (derivative): {filename}\n")
                skipped += 1
                continue
            try:
                original_img = load_image(filepath)
                original = original_img.size
                try:
                    faces = self.detector.detect_from_pil(original_img)
                    if faces:
                        sel, note = self.select_face(faces, face_mode)
                        if note:
                            self.log(note)
                        if sel:
                            crop = crop_to_face(original_img, sel, face_padding)
                            crop, _ = resize_image(crop, area)
                            crop_name = f"FaceCrop_{crop_index:03d}.png"
                            crop.save(os.path.join(output, crop_name), "PNG")
                            cw, ch = crop.size
                            self.log(f"Face crop: {crop_name} ({cw}x{ch}) from {filename} "
                                     f"({original[0]}x{original[1]}) [{sel.gender}]\n")
                            crops_made += 1
                            crop_index += 1
                            crop.close()
                        else:
                            no_face += 1
                    else:
                        self.log(f"No face: {filename}\n")
                        no_face += 1
                except Exception as e:
                    self.log(f"Face crop error ({filename}): {e}\n")
                img, resized = resize_image(original_img, area)
                w, h = img.size
                output_path = os.path.join(output, base + ".png")
                if filepath == output_path and ext == '.png' and not resized:
                    self.log(f"OK (no changes): {filename}\n")
                    skipped += 1
                    img.close()
                    continue
                output_path = self.safe_output_path(filepath, output_path)
                self.stash_if_inplace(filepath, output_path, output, replace)
                atomic_png_save(img, output_path)
                info = f"{original[0]}x{original[1]} -> {w}x{h}" if resized else f"{w}x{h}"
                self.log(f"Converted: {filename} [{info}]\n")
                converted += 1
                img.close()
                self.handle_original(filepath, output_path, output, replace)
            except Exception as e:
                self.log(f"Error ({filename}): {e}\n")
                errors += 1
        self.log("\n--- Summary ---\n")
        self.log(f"Originals converted: {converted} | Skipped: {skipped} | Errors: {errors}\n")
        self.log(f"Face crops created: {crops_made} | No face: {no_face}\n")
        self.log(f"Total files in output: {len(image_files(output))}\n")


# ---- Look Consistency Filter ---------------------------------------------------------------------------------
ARCFACE = {"match": 0.45, "borderline": 0.30, "floor": 0.25}     # face_utils.IDENTITY_THRESHOLDS["arcface"]


def verdict(sim, thresholds=None):
    """(label, colour, blurb) for a similarity under the embedder's thresholds (face_utils.IDENTITY_THRESHOLDS; the
    default is ArcFace's - the same person across varied photos usually lands 0.30-0.70, a different person rarely
    clears 0.25). Scores from different embedders are on different scales."""
    t = thresholds or ARCFACE
    if sim is None:
        return ("no face", "#7F8C8D", "No face detected — can't be scored. Back shots are fine; judge by eye.")
    if sim >= t["match"]:
        return ("match", "#70AD47", "Solid match to the baseline look.")
    if sim >= t["borderline"]:
        return ("borderline", "#E67E22", "Same subject territory, but drifting — worth an eyeball.")
    return ("drift", "#E74C3C", "Weak match — likely off-look (synthetic drift or a different subject).")


def look_files(folder):
    """The folder's images to score, normpath'd (baselines are compared by path)."""
    return sorted(os.path.normpath(os.path.join(folder, f)) for f in os.listdir(folder)
                  if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in LOOK_EXTS)


def cutoff(scores, floor=0.25):
    """The auto-suggest fence: median - 1.5 x IQR of the scored faces, floored at the embedder's floor (ArcFace 0.25);
    None under 4 faces. (median, cutoff)."""
    ss = sorted(s for s in scores.values() if s is not None)
    if len(ss) < 4:
        return None, None
    n = len(ss)
    med, q1, q3 = ss[n // 2], ss[n // 4], ss[(3 * n) // 4]
    return med, max(med - 1.5 * (q3 - q1), floor)


def suggest(scores, baselines, floor=0.25):
    """The drifters auto-suggest marks: below the fence; never a baseline, never a no-face."""
    _, cut = cutoff(scores, floor)
    if cut is None:
        return set()
    return {p for p, s in scores.items() if s is not None and s < cut and p not in baselines}


def score(files, baselines, embed, progress=None):
    """{path: mean similarity to the 3 baselines, or None (no face)}; embed(path) -> normed embedding or None.
    Raises ValueError naming any baseline with no face."""
    import numpy as np
    base = [embed(b) for b in baselines]
    missing = [os.path.basename(b) for b, e in zip(baselines, base) if e is None]
    if missing:
        raise ValueError("No face found in baseline(s): " + ", ".join(missing) + " — pick images with a clear face.")
    out = {}
    for i, p in enumerate(files, 1):
        emb = embed(p)
        # the mean of the 3 similarities == similarity to the baselines' centroid: no one photo's framing dominates
        out[p] = None if emb is None else float(np.mean([np.dot(be, emb) for be in base]))
        if progress:
            progress(i, len(files))
    return out


def save_scores(folder, baselines, scores, embedder=None):
    """<folder>/fizgig_look_scores.json for training's "Warm up look outliers": baselines, the auto-suggest cutoff,
    each image's score by file stem, and the embedder that made them (face_utils.FaceEmbedder: its cache_id and
    label - a file without one is ArcFace). Keep the format."""
    floor = embedder.thresholds["floor"] if embedder is not None else ARCFACE["floor"]
    payload = {"baselines": [os.path.basename(b) for b in baselines], "cutoff": cutoff(scores, floor)[1],
               "scores": {os.path.splitext(os.path.basename(p))[0]: s for p, s in scores.items()}}
    if embedder is not None:
        payload["embedder"], payload["embedder_label"] = embedder.cache_id, embedder.label
    with open(os.path.join(folder, "fizgig_look_scores.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def move_out(folder, paths, caption_exts=(".txt",)):
    """Move the images (and their captions) to <folder>/excluded_by_look/ - never deletes. (moved paths, errors).
    A failed move that left a copy behind removes the copy, so the image isn't in and out of the dataset."""
    dest = os.path.join(folder, "excluded_by_look")
    os.makedirs(dest, exist_ok=True)
    moved, errors = [], []
    for p in paths:
        if not os.path.exists(p):
            continue
        target = None
        try:
            base = os.path.basename(p)
            target = os.path.join(dest, base)
            n = 2
            while os.path.exists(target):
                stem, ext = os.path.splitext(base)
                target = os.path.join(dest, f"{stem}_{n}{ext}")
                n += 1
            shutil.move(p, target)
            for cext in dict.fromkeys(caption_exts):
                cap = os.path.splitext(p)[0] + cext
                if os.path.exists(cap):
                    shutil.move(cap, os.path.splitext(target)[0] + cext)
            moved.append(p)
        except Exception as e:
            try:
                if target and os.path.exists(p) and os.path.exists(target):
                    os.remove(target)
            except Exception:
                pass
            errors.append(f"Could not move {os.path.basename(p)}: {e}")
    return moved, errors
