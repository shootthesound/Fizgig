"""SmilingWolf's WD taggers (booru tags, for anime and illustration datasets: Anima, anime SDXL). The timm build
(model.safetensors) on the GPU in half precision, else the CPU; its tags with their scores, Fizgig picks which to keep.

Preprocessing as the model was trained: padded to a square on white, resized to its size (bicubic), scaled to -1..1,
channels in BGR order. Rating tags (general / sensitive / ...) are left out; a character tag only when it scores 0.85
or more (SmilingWolf's own default), since a wrong character name is worse in a caption than a missing one."""
import csv

import torch

from fizgig.captioners import Captioner, CaptionRequest, CaptionResult

RATING, CHARACTER = 9, 4
CHARACTER_MIN = 0.85


def _fetch(files, name):
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(files["repo"], name, revision=files["revision"], local_files_only=True)
    except Exception:
        return hf_hub_download(files["repo"], name, revision=files["revision"])


class WDTagger(Captioner):
    def load(self):
        import json

        import timm
        from safetensors.torch import load_file
        with open(_fetch(self.files, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        with open(_fetch(self.files, "selected_tags.csv"), encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.names = [r["name"] for r in rows]
        self.category = [int(r["category"]) for r in rows]
        model = timm.create_model(cfg["architecture"], pretrained=False, num_classes=cfg["num_classes"],
                                  **(cfg.get("model_args") or {}))
        model.load_state_dict(load_file(_fetch(self.files, "model.safetensors")))
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = model.to(self.device, self.dtype).eval()
        self.size = int(cfg["pretrained_cfg"]["input_size"][-1])

    @torch.no_grad()
    def caption(self, image, request: CaptionRequest) -> CaptionResult:
        from PIL import Image
        side = max(image.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        square = square.resize((self.size, self.size), Image.BICUBIC)
        x = torch.frombuffer(bytearray(square.tobytes()), dtype=torch.uint8).view(self.size, self.size, 3)
        x = x.permute(2, 0, 1)[[2, 1, 0]].float().div(127.5).sub(1.0)          # BGR, -1..1
        probs = torch.sigmoid(self.model(x[None].to(self.device, self.dtype)).float())[0].cpu().tolist()
        tags = [(n, p) for n, c, p in zip(self.names, self.category, probs)
                if c != RATING and (c != CHARACTER or p >= CHARACTER_MIN)]
        return CaptionResult(tags=tags)

    def unload(self):
        self.model = None
