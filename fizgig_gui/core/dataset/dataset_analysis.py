import os

import tkinter as tk

class DatasetAnalysisMixin:
    def _analyze_dataset(self, folder):
        """Analyze a dataset folder: count images, detect face crops, check captions."""
        if not folder or not os.path.isdir(folder):
            return None

        files = self._get_image_files(folder)
        face_crops = 0
        full_shots = 0
        for f in files:
            basename = os.path.splitext(os.path.basename(f))[0]
            if basename.startswith("FaceCrop_"):
                face_crops += 1
            else:
                full_shots += 1

        # Count caption files
        caption_count = 0
        for f in os.listdir(folder):
            if f.endswith(".txt") and os.path.isfile(os.path.join(folder, f)):
                caption_count += 1

        return {
            "total_images": len(files),
            "face_crops": face_crops,
            "full_shots": full_shots,
            "has_captions": caption_count > 0,
            "caption_count": caption_count,
        }

    def _recommend_training_settings(self, analysis):
        """Recommend rank, LR, and epochs based on dataset analysis.
        Based on empirical findings from the Fizgig Expansion Vision document."""
        if analysis is None:
            return None

        total = analysis["total_images"]
        face_crops = analysis["face_crops"]

        if total >= 80 and face_crops >= 30:
            return {
                "rank": 4, "alpha": 4, "lr": 0.0004, "epochs": 12,
                "tier": "optimal",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Strong dataset for rank 4:4. Fast convergence expected.",
            }
        elif total >= 40 and face_crops >= 15:
            return {
                "rank": 4, "alpha": 4, "lr": 0.0003, "epochs": 16,
                "tier": "good",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Good dataset for rank 4:4. Slightly conservative LR recommended.",
            }
        elif total >= 20:
            warnings = []
            if face_crops < 15:
                warnings.append("Few face crops — use Auto Prep on the Image Prep tab to generate more.")
            return {
                "rank": 8, "alpha": 8, "lr": 0.0002, "epochs": 25,
                "tier": "caution",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Small dataset. Rank 8 recommended over rank 4.",
                "warnings": warnings,
            }
        else:
            return {
                "rank": 16, "alpha": 16, "lr": 0.0001, "epochs": 40,
                "tier": "limited",
                "summary": f"{total} images ({face_crops} face crops, {analysis['full_shots']} full shots)",
                "message": "Very small dataset. Higher rank needed to avoid underfitting.",
                "warnings": ["Very small dataset. Results may be inconsistent. Add more training images if possible."],
            }

    def _update_dataset_recommendation(self, *args):
        """Analyze dataset and update the recommendation panel on the Training tab."""
        if not hasattr(self, '_rec_summary_var'):
            return  # UI not built yet

        # Authoritative training folder lives on the Start tab (self.image_folder_var).
        folder = self.image_folder_var.get()
        analysis = self._analyze_dataset(folder)
        rec = self._recommend_training_settings(analysis)

        if rec is None:
            self._rec_summary_var.set("")
            self._rec_detail_var.set("")
            self._rec_warning_var.set("")
            self._last_recommendation = None
            return

        self._last_recommendation = rec

        # Tier colors
        tier_prefix = {"optimal": "Optimal", "good": "Good", "caution": "Caution", "limited": "Limited"}
        self._rec_summary_var.set(f"Dataset: {rec['summary']}")
        self._rec_detail_var.set(
            f"Recommended: rank {rec['rank']}:{rec['alpha']}, LR {rec['lr']}, ~{rec['epochs']} epochs  [{tier_prefix[rec['tier']]}]"
        )

        # Warnings
        warnings = rec.get("warnings", [])
        # Also check current rank vs recommendation
        try:
            current_rank = int(self.entries.get("NETWORK_DIM", tk.Entry()).get())
        except (ValueError, AttributeError):
            current_rank = 0
        if current_rank > 0 and current_rank <= 4 and rec["rank"] > 4:
            warnings.append(f"Current rank {current_rank} may be too low for this dataset size. Recommended: {rec['rank']}.")
        if analysis and not analysis["has_captions"]:
            warnings.append("No caption files (.txt) found — captions are required for training.")

        self._rec_warning_var.set("\n".join(warnings) if warnings else "")

    def _apply_recommendation(self):
        """Apply recommended settings to the training fields."""
        rec = getattr(self, '_last_recommendation', None)
        if rec is None:
            return

        if "NETWORK_DIM" in self.entries:
            self.entries["NETWORK_DIM"].delete(0, tk.END)
            self.entries["NETWORK_DIM"].insert(0, str(rec["rank"]))
        if "NETWORK_ALPHA" in self.entries:
            self.entries["NETWORK_ALPHA"].delete(0, tk.END)
            self.entries["NETWORK_ALPHA"].insert(0, str(rec["alpha"]))
        if "LEARNING_RATE" in self.entries:
            self.entries["LEARNING_RATE"].delete(0, tk.END)
            self.entries["LEARNING_RATE"].insert(0, str(rec["lr"]))
        if "MAX_TRAIN_EPOCHS" in self.entries:
            self.entries["MAX_TRAIN_EPOCHS"].delete(0, tk.END)
            self.entries["MAX_TRAIN_EPOCHS"].insert(0, str(rec["epochs"]))

        self._update_dataset_recommendation()  # Refresh warnings