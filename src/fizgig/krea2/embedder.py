"""Krea 2 (K2) text encoder: Qwen3-VL-4B conditioner.

Returns the stacked selected hidden states (b, seq, num_select_layers, dim) plus the
attention mask; the layerwise fusion lives inside the DiT (TextFusionTransformer), so
the raw stack is what gets cached during training.

Loading follows musubi conventions (cf. qwen_image's load_qwen2_5_vl): the model config
is vendored here so it is built without fetching config.json from the Hub, weights are
loaded directly from a local safetensors file (ComfyUI-style `model.`/`visual.` keys are
accepted as well as the official HF layout), and only the tokenizer is still pulled by
repo id. This lets K2 share the same Qwen3-VL-4B weights a user already has for ComfyUI,
instead of requiring a separate transformers/Diffusers checkpoint.

The tokenizer/processor/chat-template can't be vendored as a Python dict the way the model
config is — transformers wants them as actual files (tokenizer.json, chat_template.json,
etc). So the same "naked safetensors" convention the checkpoint itself follows applies here
too: if those files are sitting in a `qwen3vl_tokenizer/` folder next to the checkpoint (see
_local_tokenizer_dir), they're used directly with no Hub involved at all — a fully offline
machine can have them sneakernet'd in by hand, no HF cache archaeology required. Only falls
back to `tokenizer_repo`'s normal cache-first Hub resolution if that folder isn't there —
which is what fetch_models.py's "tools" family (hf-config: entries) already keeps warm for
anyone who's run it online at least once; this is a second, folder-based way in.
"""

import logging
import os
import re
from dataclasses import dataclass

import torch
from accelerate import init_empty_weights
from torch import Tensor
from transformers import (
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLConfig,
    Qwen3VLForConditionalGeneration,
)

from fizgig.krea2.safetensors_utils import load_split_weights

logger = logging.getLogger(__name__)


# Only the tokenizer is still fetched by repo id (small, HF-cached after first use) —
# unless a local qwen3vl_tokenizer/ folder is found first; see _local_tokenizer_dir.
QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"

# The non-weight files this repo carries — everything apply_chat_template / the image
# processor need, none of the multi-GB weight shards (those come from the local checkpoint).
# Named explicitly so a fully offline install has a precise shopping list: download these from
# https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/main and drop them into the folder
# _local_tokenizer_dir names.
QWEN3_VL_TOKENIZER_FILES = (
    "chat_template.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def _local_tokenizer_dir(model_path: str) -> str:
    """Where a sneakernet'd copy of QWEN3_VL_TOKENIZER_FILES lives: a `qwen3vl_tokenizer/`
    folder next to the checkpoint itself, so it travels with the checkpoint regardless of
    where the user's models directory is."""
    return os.path.join(os.path.dirname(os.path.abspath(model_path)), "qwen3vl_tokenizer")


def _bundled_tokenizer_dir() -> str:
    """The copy Fizgig SHIPS (src/fizgig/assets/qwen3vl_tokenizer — Apache 2.0, see its
    ATTRIBUTION.md). ~11 MB of config/vocab, no weights: with these in the repo, text
    encoding and captioning need the Hub for nothing at all."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "assets", "qwen3vl_tokenizer")


def _resolve_tokenizer_source(model_path: str, tokenizer_repo: str) -> str:
    """Pick where tokenizer/processor files come from, most-specific first — but only when
    the caller didn't explicitly ask for something other than the default repo:

    1. a `qwen3vl_tokenizer/` folder next to the checkpoint (user override / sneakernet)
    2. the copy bundled with Fizgig (the normal case — fully offline, nothing to fetch)
    3. the Hub repo id, cache-first (fallback if the bundled files are ever missing)
    """
    if tokenizer_repo == QWEN3_VL_4B_INSTRUCT_REPO_ID:
        for cand in (_local_tokenizer_dir(model_path), _bundled_tokenizer_dir()):
            if os.path.isfile(os.path.join(cand, "tokenizer_config.json")):
                return os.path.normpath(cand)
    return tokenizer_repo


def _offline_tokenizer_error(local_dir: str, tokenizer_repo: str, err: Exception) -> RuntimeError:
    """Turn a bare ConnectionError (etc.) from the Hub into the sneakernet instructions: which
    files, from where, and exactly which local folder to drop them into."""
    files = "\n".join(f"  - {name}" for name in QWEN3_VL_TOKENIZER_FILES)
    local_dir = local_dir or "a qwen3vl_tokenizer/ folder next to the text-encoder file"
    return RuntimeError(
        f"Couldn't load the Qwen3-VL tokenizer/chat-template ({tokenizer_repo}): "
        f"{type(err).__name__}: {err}\n"
        "No internet connection, and no local copy was found. To caption fully offline, "
        f"download these files from https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/main "
        f"on any machine with internet:\n{files}\n"
        f"...and place them in:\n  {local_dir}"
    )

# Vendored copy of the Qwen3-VL-4B-Instruct config.json so the text encoder is built
# without fetching the config from the Hugging Face Hub. Qwen3-VL is natively supported by
# transformers (no auto_map / remote code), so Qwen3VLConfig.from_dict reproduces
# AutoConfig.from_pretrained exactly. Mirror upstream config.json if Qwen ever revises it.
QWEN3_VL_4B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "transformers_version": "4.57.0.dev0",
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}


@dataclass
class TextEncoderConfig:
    max_length: int = 512
    select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID


def _convert_comfyui_qwen3vl_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map a ComfyUI-style (bare ``model.`` / ``visual.``) Qwen3-VL state dict onto the HF
    ``Qwen3VLForConditionalGeneration`` layout. Official HF checkpoints already use the
    ``model.language_model.`` / ``model.visual.`` layout and pass through unchanged.
    """
    converted: dict[str, Tensor] = {}
    for key, value in sd.items():
        if key.startswith("model.language_model.") or key.startswith("model.visual."):
            new_key = key
        elif key.startswith("visual."):
            new_key = "model.visual." + key[len("visual.") :]
        elif key.startswith("language_model."):
            new_key = "model." + key
        elif key.startswith("model."):
            new_key = "model.language_model." + key[len("model.") :]
        else:
            new_key = key
        converted[new_key] = value
    return converted


def qwen3_vl_size(model_path: str) -> str:
    """"8b" for a Qwen3-VL-8B file (Qwen Image 2.1's text encoder, the bigger captioner: hidden size 4096, its own LM
    head), else "4b" - read from the file's header, nothing loaded."""
    from safetensors import safe_open
    try:
        with safe_open(model_path, framework="pt") as f:
            for k in f.keys():
                if k.endswith("embed_tokens.weight"):
                    return "8b" if f.get_slice(k).get_shape()[-1] == 4096 else "4b"
    except Exception:
        pass
    return "4b"


def _qwen3_vl_config(model_path: str) -> dict:
    if qwen3_vl_size(model_path) == "8b":
        import json
        with open(os.path.join(os.path.dirname(__file__), "..", "qwen_image21", "qwen3vl_8b_config.json"),
                  encoding="utf-8") as f:
            return json.load(f)
    return QWEN3_VL_4B_INSTRUCT_CONFIG


def _gpu_room_gb(device) -> float:
    try:
        d = torch.device(device)
        if d.type != "cuda" or not torch.cuda.is_available():
            return float("inf")
        return torch.cuda.mem_get_info(d)[0] / 2 ** 30
    except Exception:
        return float("inf")


# what a caption needs on the card beside the weights: a 1 MP picture's vision tokens, the KV cache, the logits
_CAPTION_WORK_GB = 2.5


def _load_qwen3_vl_model(
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    disable_mmap: bool = True,
) -> Qwen3VLForConditionalGeneration:
    """Build Qwen3-VL (4B, or the 8B Qwen Image 2.1 ships) from its vendored config and load weights from a local
    safetensors. A model bigger than the card's free memory keeps what fits on the card and streams the rest of its
    language layers from system RAM for each step (slower, but it runs)."""
    config = Qwen3VLConfig.from_dict(_qwen3_vl_config(model_path))
    with init_empty_weights():
        model = Qwen3VLForConditionalGeneration._from_config(config)
    weights_gb = os.path.getsize(model_path) / 2 ** 30
    room = _gpu_room_gb(device)
    stream = weights_gb + _CAPTION_WORK_GB > room
    if stream:
        logger.info(f"Qwen3-VL text encoder: {weights_gb:.1f} GB of weights, {room:.1f} GB free on the card - loading "
                    "to system RAM and streaming the layers that don't fit")

    logger.info(f"Loading Krea 2 text encoder (Qwen3-VL) weights from {model_path}")
    # ComfyUI's fp8_scaled variant quantises ONLY the language Linears — its 315 vision-tower
    # tensors are bf16, so reference images and captioning work exactly as on the bf16 file.
    # (An earlier comment here claimed the vision tower "can't run in fp8"; that was wrong, and
    # the real blocker was simply that this loader had no fp8 path.) Keeping the weights fp8 and
    # dequantising per matmul is what actually saves the ~3.6 GB — converting to bf16 at load
    # would load fine and save nothing.
    from fizgig.krea2.utils import (is_prequantized_fp8, _FP8_SCALE_SUFFIX,
                                    _COMFY_FP8_MARKER_SUFFIX, _reshape_prequant_fp8_scale)
    prequant = is_prequantized_fp8(model_path)

    # dtype=None on the fp8 path: passing a dtype converts the fp8 weights on load and there is
    # nothing left to be resident about.
    sd = load_split_weights(model_path, device="cpu" if stream else str(device), disable_mmap=disable_mmap,
                            dtype=(None if prequant else dtype))
    sd = _convert_comfyui_qwen3vl_state_dict(sd)

    if prequant:
        # Comfy stores `.weight_scale`; the monkey patch looks for `.scale_weight` (and wants it
        # broadcastable against [out, in]). Same normalisation the DiT does — reusing its helpers
        # so the two can't drift. Dropping these keys WITHOUT applying them is the bug that once
        # left the Klein DiT with weights ~1230x too large, so they are renamed, never discarded.
        fixed = {}
        for k, v in sd.items():
            if k.endswith(_COMFY_FP8_MARKER_SUFFIX):
                continue
            if k.endswith(_FP8_SCALE_SUFFIX):
                fixed[k[: -len(_FP8_SCALE_SUFFIX)] + ".scale_weight"] = \
                    _reshape_prequant_fp8_scale(v).to(dtype)
            else:
                fixed[k] = v
        sd = fixed
        # Registers a scale_weight buffer on each quantised Linear and swaps in the dequantising
        # forward. Must run BEFORE load_state_dict, or the scale keys have nowhere to land and
        # come back as "unexpected". Only the language Linears carry scales, so the bf16 vision
        # tower is untouched by design.
        from fizgig.krea2.fp8_optimization_utils import apply_fp8_monkey_patch
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

    info = model.load_state_dict(sd, strict=False, assign=True)
    # Qwen3-VL-4B ties the LM head to the input embeddings (tie_word_embeddings=true), so the
    # checkpoint omits lm_head.weight; re-tie after loading to materialize it.
    model.tie_weights()

    unexpected = list(info.unexpected_keys)
    missing = [k for k in info.missing_keys if k != "lm_head.weight"]
    if unexpected or missing:
        raise RuntimeError(
            f"Qwen3-VL text encoder checkpoint did not match the model: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    # Never cast an fp8 model to dtype — that would convert the weights straight back to bf16 and
    # throw away the whole point. The unquantised parts (vision tower, norms, embeddings) are
    # already stored in the target dtype anyway. Cast before placing: a streamed model can't be
    # moved or cast once its layers are offloaded.
    if dtype is not None and not prequant:
        model.to(dtype)
    if stream:
        _stream_layers(model, device, room)
    else:
        model.to(device)
    return model.eval().requires_grad_(False)


def _stream_layers(model, device, room_gb):
    """Everything but the language layers on the card (the vision tower, embeddings, final norm, LM head), then as
    many language layers as fit beside a caption's working memory; the rest stay in system RAM and are moved to the
    card for their forward only (accelerate's offload hooks), so a card smaller than the model still captions."""
    from accelerate import dispatch_model
    lm = model.model.language_model
    nbytes = lambda m: sum(t.numel() * t.element_size() for t in list(m.parameters()) + list(m.buffers()))
    layers = list(lm.layers)
    fixed = (nbytes(model) - sum(nbytes(l) for l in layers)) / 2 ** 30
    per = (sum(nbytes(l) for l in layers) / len(layers)) / 2 ** 30
    fit = max(0, min(len(layers), int((room_gb - _CAPTION_WORK_GB - fixed) // per)))
    dev = str(torch.device(device))
    device_map = {"model.visual": dev, "model.language_model.embed_tokens": dev, "model.language_model.norm": dev,
                  "model.language_model.rotary_emb": dev, "lm_head": dev}
    for i in range(len(layers)):
        device_map[f"model.language_model.layers.{i}"] = dev if i < fit else "cpu"
    logger.info(f"Qwen3-VL text encoder: {fit} of {len(layers)} language layers on the card, {len(layers) - fit} "
                f"streamed from system RAM ({per:.2f} GB each)")
    dispatch_model(model, device_map=device_map, main_device=dev)


def load_qwen3_vl_conditioner(
    model_path: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cpu",
    max_length: int = TextEncoderConfig.max_length,
    select_layers: tuple[int, ...] = TextEncoderConfig.select_layers,
    tokenizer_repo: str = QWEN3_VL_4B_INSTRUCT_REPO_ID,
    disable_mmap: bool = True,
) -> "Qwen3VLConditioner":
    """Load the Qwen3-VL-4B conditioner used by K2: weights from ``model_path`` (safetensors),
    tokenizer from ``tokenizer_repo`` (Hub id or local dir)."""
    from fizgig.utils.hf_cache import from_pretrained_cache_first
    qwen = _load_qwen3_vl_model(model_path, dtype=dtype, device=device, disable_mmap=disable_mmap)
    local_dir = _local_tokenizer_dir(model_path)
    tokenizer_repo = _resolve_tokenizer_source(model_path, tokenizer_repo)
    try:
        tokenizer = from_pretrained_cache_first(AutoTokenizer, tokenizer_repo, max_length=max_length)
        processor = from_pretrained_cache_first(Qwen2TokenizerFast, tokenizer_repo, max_length=max_length)
    except Exception as e:
        raise _offline_tokenizer_error(local_dir, tokenizer_repo, e) from e
    conditioner = Qwen3VLConditioner(qwen, tokenizer, processor, max_length=max_length,
                                     select_layers=select_layers, tokenizer_repo=tokenizer_repo,
                                     local_tokenizer_dir=local_dir)
    return conditioner.eval().requires_grad_(False)


from fizgig.krea2.caption_text import (  # noqa: E402,F401  (the instructions: text only, kept apart)
    SUBJECT_RULE, NO_PREAMBLE_RULE, CAPTION_INSTRUCTION, DETAILED_CAPTION_INSTRUCTION, ENCODE_SYSTEM_DESCRIPTOR, SHORT_CAPTION_INSTRUCTION, DETAILED_DESCRIPTION_INSTRUCTION, STYLE_CAPTION_INSTRUCTION, CAPTION_TASKS, DEFAULT_CAPTION_TASK,
)


_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"(?:in|within)\s+th(?:is|e)\s+(?:image|photo|photograph|picture|shot)\s*,?\s*|"
    r"th(?:is|e)\s+(?:image|photo|photograph|picture|shot)\s+"
    r"(?:shows?|depicts?|features?|captures?|presents?|displays?|is\s+of)\s+|"
    r"here\s+(?:is|we\s+see)\s+|"
    r"we\s+see\s+|"
    r"the\s+(?:image|photo)\s+is\s+a\s+"
    r")",
    re.IGNORECASE,
)


def _strip_caption_preamble(text: str) -> str:
    """Remove a leading 'This image shows…' style preamble.

    Instructing the model not to produce one is unreliable at 4B and temperature 0.5, and the
    preamble is pure noise in a training caption — worse once the trigger word is prepended, where
    it reads "<trigger>, this image shows a woman…". Applied repeatedly because the model
    occasionally stacks them ("In this image, we see …"). Never returns empty: if stripping would
    consume everything, the original is kept."""
    out = text
    for _ in range(3):
        stripped = _PREAMBLE_RE.sub("", out, count=1)
        if stripped == out:
            break
        out = stripped
    out = out.strip()
    return out if out else text.strip()


def generate_caption(conditioner: "Qwen3VLConditioner", image_path, *,
                     max_new_tokens: int = 120, megapixels: float = 1.0,
                     detailed: bool = False, seed: int = None,
                     instruction: str = None) -> str:
    """Caption an image with the SAME Qwen3-VL the trainer conditions on (its LM head is
    legitimately tied to the embeddings — unlike Klein's stripped Qwen3-8B — so generation is
    real). Used by auto-recaption to rewrite a stuck image's caption from what's actually in it,
    with an instruction tuned to Peter's captioning doctrine: name the viewpoint / visibility.

    Decoding is SAMPLED with a random seed (seed=None) so repeated attempts on the same image get
    fresh phrasings instead of the identical greedy caption — attempt 2 varies by wording as well
    as by instruction. (A seed does NOTHING under greedy decode — sampling is what makes it
    matter; temperature is kept LOW at 0.5 so the variation stays in phrasing, not in factual
    confidence.) Sampling uses the global torch RNG, so the state is saved and restored around
    the call: caption generation must never perturb the training noise stream."""
    import random as _random
    from PIL import Image

    proc = conditioner._get_image_processor()
    # image_path is a path OR an already-opened PIL image. The second form exists because a
    # MiniMax H3 training item can be a video clip, and what gets captioned is a frame decoded
    # from it — there is no file on disk to hand over.
    from fizgig.utils.upright import open_upright
    src = open_upright(image_path) if isinstance(image_path, (str, os.PathLike)) else image_path
    im = conditioner._cap_image(src, megapixels)
    # An explicit instruction wins (the Captions tab's task menu, or a user-edited one); otherwise
    # the historical detailed/standard pair, so existing callers are unaffected.
    if instruction is None:
        instruction = DETAILED_CAPTION_INSTRUCTION if detailed else CAPTION_INSTRUCTION
    if detailed:
        max_new_tokens = max(max_new_tokens, 240)
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": instruction}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=[prompt], images=[im], return_tensors="pt").to(conditioner.qwen.device)

    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(seed if seed is not None else _random.randint(1, 2**31 - 1))
        with torch.no_grad():
            out = conditioner.qwen.generate(**inputs, max_new_tokens=max_new_tokens,
                                            do_sample=True, temperature=0.5, top_p=0.9)
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return _strip_caption_preamble(" ".join(text.split()).strip())


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        qwen: Qwen3VLForConditionalGeneration,
        tokenizer,
        processor,
        max_length: int = 512,
        select_layers: tuple[int, ...] = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35),
        tokenizer_repo: str | None = None,
        local_tokenizer_dir: str | None = None,
    ):
        super().__init__()
        self.qwen = qwen.eval().requires_grad_(False)
        self.tokenizer = tokenizer
        self.processor = processor
        self.tokenizer_repo = tokenizer_repo
        self.local_tokenizer_dir = local_tokenizer_dir
        self._image_processor = None  # lazily-loaded full Qwen3-VL processor (for image refs)
        self.max_length = max_length
        self.select_layers = select_layers
        self.system_descriptor = ENCODE_SYSTEM_DESCRIPTOR
        self.prompt_template_encode_prefix = "<|im_start|>system\n" + self.system_descriptor + "<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5

    def forward(self, text: list[str], images: list | None = None,
                vision_megapixels: float = 1.0) -> tuple[Tensor, Tensor]:
        """Encode prompts to the K2 multi-layer hidden stack + mask.

        `images`, when given, is a per-prompt list (same length as `text`); each entry is a
        list of PIL.Image references (or None). When any prompt has references they are fed
        through Qwen3-VL's *vision* path under the same descriptor template, so the conditioning
        becomes "visually aware" of the image (a prompt-from-a-picture effect — Krea 2's DiT has
        no reference-latent slot, so this is the only reference mechanism).

        Works with either checkpoint. An earlier version of this docstring claimed the vision
        tower "can't run in fp8" — that was wrong: ComfyUI's fp8_scaled file quantises only the
        language Linears and ships all 315 vision-tower tensors in bf16. The real constraint was
        that this loader had no fp8 dequantisation path; see _load_qwen3_vl_model.
        """
        has_imgs = bool(images) and any(images[i] for i in range(min(len(images), len(text))))
        if has_imgs:
            return self._forward_with_images(text, images, vision_megapixels)
        return self._forward_text(text)

    def _forward_text(self, text: list[str]) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        suffix_inputs = self.processor(text=suffix_text, return_tensors="pt").to(self.qwen.device, non_blocking=True)
        suffix_ids, suffix_mask = (
            suffix_inputs["input_ids"],
            suffix_inputs["attention_mask"].bool(),
        )

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                max_length=self.max_length + prefix_idx - self.prompt_template_encode_suffix_start_idx,
                return_tensors="pt",
            ).to(self.qwen.device, non_blocking=True)
            input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
            mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
            states = self.qwen(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)

            hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]

            return hiddens, mask

    def _get_image_processor(self):
        """Lazily load the full Qwen3-VL processor (text + image). Only needed for image refs,
        so text-only training never pays for it."""
        if self._image_processor is None:
            from transformers import AutoProcessor
            from fizgig.utils.hf_cache import from_pretrained_cache_first
            repo = self.tokenizer_repo or QWEN3_VL_4B_INSTRUCT_REPO_ID

            def _has_template(p):
                return bool(getattr(p, "chat_template", None)
                            or getattr(getattr(p, "tokenizer", None), "chat_template", None))
            try:
                proc = from_pretrained_cache_first(AutoProcessor, repo)
                if not _has_template(proc) and not os.path.isdir(repo):
                    # local_files_only can succeed on a cache that only has tokenizer files
                    # (warmed by earlier text-encoding runs) but not chat_template.json —
                    # transformers treats a missing chat template as optional, so the
                    # cache-first helper's exception-triggered network fallback never fires
                    # and hands back a processor apply_chat_template can't use (issue #37).
                    # Force a real fetch to complete the cache instead of caching that broken
                    # processor for every caption. (A local qwen3vl_tokenizer/ dir can't
                    # self-heal this way — there's nowhere else on disk to look — so that
                    # case falls straight through to the error below.)
                    proc = AutoProcessor.from_pretrained(repo)
                if not _has_template(proc):
                    raise ValueError("processor has no chat_template")
            except Exception as e:
                raise _offline_tokenizer_error(self.local_tokenizer_dir, repo, e) from e
            self._image_processor = proc
        return self._image_processor

    @staticmethod
    def _cap_image(im, megapixels: float):
        """RGB + downscale an image so its pixel area is <= megapixels (never upscale)."""
        from PIL import Image
        im = im.convert("RGB")
        cap = int(megapixels * 1024 * 1024)
        w, h = im.size
        if w * h > cap and w > 0 and h > 0:
            scale = (cap / (w * h)) ** 0.5
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
        return im

    def _forward_with_images(self, text, images, vision_megapixels) -> tuple[Tensor, Tensor]:
        """Vision-aware encode: build the descriptor template with vision placeholders, run the
        Qwen3-VL processor (text + pixel_values), and extract the same select-layer stack.

        Mirrors the ComfyUI Text-Encode-(Krea2) node: forced Krea 2 descriptor system prompt,
        image tokens in the user turn, vision_megapixels as a downscale cap. The system prefix
        (start_idx tokens) is trimmed exactly as in the text path.
        """
        proc = self._get_image_processor()
        prefix_idx = self.prompt_template_encode_start_idx
        full_texts, flat_images = [], []
        for i, prompt in enumerate(text):
            imgs = (images[i] if images and i < len(images) and images[i] else [])
            imgs = [self._cap_image(im, vision_megapixels) for im in imgs]
            vis = "".join("<|vision_start|><|image_pad|><|vision_end|>" for _ in imgs)
            full_texts.append(self.prompt_template_encode_prefix + vis + prompt
                              + self.prompt_template_encode_suffix)
            flat_images.extend(imgs)

        with torch.no_grad():
            inputs = proc(text=full_texts, images=flat_images or None,
                          padding=True, return_tensors="pt").to(self.qwen.device)
            states = self.qwen(**inputs, output_hidden_states=True)
            hiddens = torch.stack([states.hidden_states[i] for i in self.select_layers], dim=2)
            mask = inputs["attention_mask"].bool()
            # Trim the system descriptor prefix (same fixed prefix as the text path; the image
            # tokens live in the user turn, after it).
            hiddens = hiddens[:, prefix_idx:]
            mask = mask[:, prefix_idx:]
            return hiddens, mask
