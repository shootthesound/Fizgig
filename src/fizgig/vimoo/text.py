"""Vimoo 1.0 Lite's three encoders, as its pipeline (Vimoo's diffusers fork, modular_pipelines/vimoo) runs them.

- Qwen2.5-VL-7B's text tower (transformers' Qwen2_5_VLTextModel) in FP16, a fixed system prompt, the hidden state
  two layers from the end, cropped after the chat template's user marker. The reference pads to 1400 tokens; the
  tower is causal and the transformer drops padded keys, so the valid tokens alone are the same conditioning.
- Glyph-ByT5 (a T5 encoder with Glyph-SDXL-v2's colour and font vocabulary) in FP32 on the CPU, only for text in
  "double quotes" in the prompt; a prompt without quotes has no glyph tokens at all.
- SigLIP so400m for image-to-video: the reference picture cover-resized and centre-cropped to the output size, then
  SigLIP's own processor; its 729 tokens.

The weights load from the single files Vimoo ships for ComfyUI; configs and tokenizers come from their upstream
repos, cache first (fizgig.utils.hf_cache).
"""
from __future__ import annotations

import re

import torch

QWEN_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
VIMOO_REPO = "Vimoo-AI/Vimoo-1.0-Lite"
SIGLIP_REPO = "google/siglip-so400m-patch14-384"
QWEN_MAX = 1400
QWEN_SKIP = 2
BYT5_MAX = 256
SYSTEM = (
    "You are a helpful assistant. Describe the video by detailing the following aspects:         "
    "1. The main content and theme of the video.         "
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects.         "
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects.         "
    "4. background environment, light, style and atmosphere.         "
    "5. camera angles, movements, and transitions used in the video.")
NEGATIVE = ("oversaturated colors, overexposed, static, blurry details, subtitles, style, artwork, painting, still "
            "image, grayish overall tone, worst quality, low quality, JPEG artifacts, ugly, mutilated, extra fingers, "
            "poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, motionless "
            "scene, cluttered background, three legs, many people in the background, walking backwards")


def _cached(cls, repo, **kw):
    from fizgig.utils.hf_cache import from_pretrained_cache_first
    return from_pretrained_cache_first(cls, repo, **kw)


def _load_file(path):
    from safetensors.torch import load_file
    return load_file(str(path), device="cpu")


# ---- Qwen2.5-VL ---------------------------------------------------------------------------------------------------
class QwenText:
    """The text tower and its tokenizer; encode(prompts) -> list of ([L, 3584] fp16 states, L)."""

    def __init__(self, path, device):
        from transformers import AutoConfig, AutoTokenizer, Qwen2_5_VLTextModel
        cfg = _cached(AutoConfig, QWEN_REPO)
        tcfg = getattr(cfg, "text_config", cfg)
        tcfg._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = Qwen2_5_VLTextModel(tcfg)
        sd = {k[len("model."):] if k.startswith("model.") else k: v.to(torch.float16)
              for k, v in _load_file(path).items() if not k.startswith("lm_head")}
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        if [m for m in missing if "rotary" not in m] or unexpected:
            raise ValueError(f"{path} isn't Qwen2.5-VL-7B's text tower: missing {missing[:3]}, "
                             f"unexpected {unexpected[:3]}")
        # the rotary tables are buffers made at init (on meta here; the model and each attention layer keep one):
        # rebuilt on the device, FP32 as from_pretrained leaves them for an FP16 load
        for name, mod in list(model.named_modules()):
            if type(mod).__name__.endswith("RotaryEmbedding"):
                parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
                setattr(parent, name.rsplit(".", 1)[-1], type(mod)(config=tcfg, device=device))
        self.model = model.eval().requires_grad_(False).to(device)
        self.tok = _cached(AutoTokenizer, QWEN_REPO)
        self.device = device
        conv = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": " "}]
        ids = self.tok.apply_chat_template([conv], tokenize=True, add_generation_prompt=True)
        ids = ids[0] if ids and isinstance(ids[0], list) else ids
        marker = self.tok("<|im_start|>user\n", add_special_tokens=False)["input_ids"]
        self.crop = next(i + len(marker) for i in range(len(ids)) if ids[i:i + len(marker)] == marker)

    @torch.no_grad()
    def encode(self, prompts):
        out = []
        for p in prompts:
            conv = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p or " "}]
            ids = self.tok.apply_chat_template([conv], tokenize=True, add_generation_prompt=True, truncation=True,
                                               max_length=QWEN_MAX + self.crop)
            ids = ids[0] if ids and isinstance(ids[0], list) else ids
            x = torch.tensor([ids], device=self.device)
            hs = self.model(input_ids=x, attention_mask=torch.ones_like(x), output_hidden_states=True).hidden_states
            out.append(hs[-(QWEN_SKIP + 1)][0, self.crop:].to(torch.float16).cpu())
        return out


# ---- Glyph-ByT5 ---------------------------------------------------------------------------------------------------
def glyph_texts(prompt):
    """The quoted texts in a prompt, in order, each once (straight or curly quotes)."""
    found = re.findall(r'"(.*?)"|“(.*?)”', prompt or "")
    return list(dict.fromkeys(a or b for a, b in found))


def byt5_tokenizer():
    """ByT5's byte tokenizer plus Glyph-SDXL-v2's 1251 added tokens (125 extra ids, colours, fonts) at ids 259 on,
    as Vimoo's tokenizer_2 lists them. Built here: its tokenizer_config is in transformers 5's form, which the pinned
    transformers can't read."""
    import json
    from huggingface_hub import hf_hub_download
    from transformers import ByT5Tokenizer
    try:
        f = hf_hub_download(VIMOO_REPO, "tokenizer_2/added_tokens.json", local_files_only=True)
    except Exception:
        f = hf_hub_download(VIMOO_REPO, "tokenizer_2/added_tokens.json")
    with open(f, encoding="utf-8") as fh:
        added = sorted(json.load(fh).items(), key=lambda kv: kv[1])
    tok = ByT5Tokenizer(extra_ids=0)
    tok.add_tokens([k for k, _ in added], special_tokens=True)
    bad = [(k, i) for k, i in added if tok.convert_tokens_to_ids(k) != i]
    if bad:
        raise ValueError(f"Glyph-ByT5 tokenizer ids don't match Vimoo's: {bad[:3]}")
    return tok


class GlyphByT5:
    """encode(prompt) -> ([Lg, 1472] fp32 states, Lg) or None for a prompt with no quoted text."""

    def __init__(self, path):
        from transformers import AutoConfig, T5EncoderModel
        cfg = _cached(AutoConfig, VIMOO_REPO, subfolder="text_encoder_2")
        with torch.device("meta"):
            model = T5EncoderModel(cfg)
        missing, unexpected = model.load_state_dict(_load_file(path), strict=False, assign=True)
        if [m for m in missing if m not in ("encoder.embed_tokens.weight",)] or unexpected:
            raise ValueError(f"{path} isn't Vimoo's Glyph-ByT5: missing {missing[:3]}, unexpected {unexpected[:3]}")
        model.tie_weights()
        self.model = model.eval().requires_grad_(False).float()
        self.tok = byt5_tokenizer()

    @torch.no_grad()
    def encode(self, prompt):
        texts = glyph_texts(prompt)
        if not texts:
            return None
        enc = self.tok("".join(f'Text "{t}". ' for t in texts), padding="max_length", max_length=BYT5_MAX,
                       truncation=True, add_special_tokens=True, return_tensors="pt")
        states = self.model(input_ids=enc.input_ids, attention_mask=enc.attention_mask.float())[0]
        n = int(enc.attention_mask.sum())
        return states[0, :n].float()


# ---- SigLIP -------------------------------------------------------------------------------------------------------
def crop_reference(image, width, height):
    """Cover-resize (Lanczos) and centre-crop to the output size, offsets rounded (the Lite profile)."""
    from PIL import Image
    rgb = image.convert("RGB")
    if rgb.size == (width, height):
        return rgb
    s = max(width / rgb.width, height / rgb.height)
    r = rgb.resize((int(round(rgb.width * s)), int(round(rgb.height * s))), Image.LANCZOS)
    left, top = int(round((r.width - width) / 2)), int(round((r.height - height) / 2))
    return r.crop((left, top, left + width, top + height))


class SigLIP:
    """encode(image, width, height) -> [729, 1152] fp32 states of the cropped reference picture."""

    def __init__(self, path, device):
        from transformers import AutoConfig, SiglipVisionModel
        cfg = _cached(AutoConfig, SIGLIP_REPO)
        vcfg = getattr(cfg, "vision_config", cfg)
        model = SiglipVisionModel(vcfg)              # built for real (0.4B): its position-id buffer comes with it
        missing, unexpected = model.load_state_dict(_load_file(path), strict=False)
        if missing or unexpected:
            raise ValueError(f"{path} isn't SigLIP so400m: missing {missing[:3]}, unexpected {unexpected[:3]}")
        self.model = model.eval().requires_grad_(False).to(device)
        try:
            from transformers import SiglipImageProcessorFast as Proc
        except ImportError:
            from transformers import SiglipImageProcessor as Proc
        self.proc = _cached(Proc, SIGLIP_REPO)
        self.device = device

    @torch.no_grad()
    def encode(self, image, width, height):
        import numpy as np
        px = self.proc.preprocess(images=np.asarray(crop_reference(image, width, height)), return_tensors="pt")
        dt = next(self.model.parameters()).dtype
        px = {k: v.to(device=self.device, dtype=dt) for k, v in px.items()}
        return self.model(**px).last_hidden_state[0].float().cpu()
