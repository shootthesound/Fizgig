"""Batch AI-caption images/clips in an isolated process (Florence-2 or Qwen3-VL).

The Captions tab runs this so GPU memory stays out of the GUI process.  In --serve
mode the model stays loaded between jobs until the worker receives QUIT (e.g. when
training starts).

Stdout protocol (flushed):
  READY
  INFO: ...
  PROGRESS: <current> <total>
  OK: <basename>
  FAIL: <basename> (<reason>)
  STOPPED
  DONE
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4"}


def _quiet_worker_logs() -> None:
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    for name in (
        "transformers",
        "transformers.generation",
        "transformers.generation.utils",
        "fizgig.krea2.fp8_optimization_utils",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def _open_training_frame(path: str):
    from PIL import Image

    ext = os.path.splitext(path)[1].lower()
    if ext not in VIDEO_EXTENSIONS:
        from fizgig.utils.upright import open_upright
        return open_upright(path)
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("could not open video")
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if n > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
            ok, frame = cap.read()
            if not ok and n > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("could not read frame")
            return Image.fromarray(frame[:, :, ::-1])
        finally:
            cap.release()
    except Exception:
        from fizgig.minimax.clip import read_frames

        frames = read_frames(path)
        return Image.fromarray(frames[len(frames) // 2])


def _load_image_list(list_file: str) -> list[str]:
    out: list[str] = []
    with open(list_file, encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                out.append(p)
    return out


def _write_caption(path: str, caption: str, trigger: str, ext: str = ".txt") -> None:
    cap_path = os.path.splitext(path)[0] + (ext or ".txt")
    text = f"{trigger}, {caption}" if trigger else caption
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(text)


def _load_florence(model_name: str, revision: str | None, code_revision: str | None, device: str):
    from transformers import AutoModelForCausalLM, AutoProcessor

    from fizgig.utils.hf_cache import from_pretrained_cache_first

    kwargs = {"trust_remote_code": True}
    if revision:
        kwargs["revision"] = revision
    if code_revision:
        kwargs["code_revision"] = code_revision
    processor = from_pretrained_cache_first(AutoProcessor, model_name, **kwargs)
    dtype = torch.float16 if device == "cuda" else torch.float32
    # torch_dtype, not dtype: the pinned transformers reads torch_dtype; an unknown kwarg
    # falls through silently and Florence loads fp32 (double VRAM), which nothing reports.
    model = from_pretrained_cache_first(
        AutoModelForCausalLM,
        model_name,
        torch_dtype=dtype,
        attn_implementation="eager",
        **kwargs,
    ).to(device)
    return model.eval(), processor


def _caption_florence(model, processor, device, path: str, task: str, max_new_tokens: int) -> str:
    from fizgig.utils.image import flatten
    image = flatten(_open_training_frame(path))
    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=3,
        use_cache=False,
    )
    caption = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(caption, task=task, image_size=(image.width, image.height))
    return parsed.get(task, caption)


def _caption_qwen(encoder, path: str, instruction: str, max_new_tokens: int) -> str:
    from fizgig.krea2.embedder import generate_caption

    frame = _open_training_frame(path)
    return generate_caption(
        encoder,
        frame,
        max_new_tokens=max_new_tokens,
        instruction=instruction,
    )


class CaptionState:
    backend: str
    device: str
    model: object
    include_video: bool

    def __init__(self, backend: str, device: str, model: object, include_video: bool):
        self.backend = backend
        self.device = device
        self.model = model
        self.include_video = include_video


def _load_state_from_config(config: dict, device: str) -> CaptionState:
    backend = config["backend"]
    include_video = bool(config.get("include_video"))

    if backend == "florence":
        model_name = config.get("florence_model") or ""
        if not model_name:
            raise ValueError("florence_model required")
        rev = config.get("florence_revision") or None
        code_rev = config.get("florence_code_revision") or None
        print(f"INFO: Loading {model_name}...", flush=True)
        model, processor = _load_florence(model_name, rev, code_rev, device)
        print("INFO: Model ready.", flush=True)
        return CaptionState(backend, device, (model, processor), include_video)

    text_encoder = config.get("text_encoder") or ""
    if not text_encoder:
        raise ValueError("text_encoder required")
    from fizgig.krea2.utils import load_krea2_text_encoder

    te_name = os.path.basename(text_encoder)
    print(f"INFO: Loading Qwen3-VL from {te_name}...", flush=True)
    model = load_krea2_text_encoder(text_encoder, dtype=torch.bfloat16, device=device)
    print("INFO: Qwen3-VL ready.", flush=True)
    return CaptionState(backend, device, model, include_video)


def _run_job(state: CaptionState, job: dict) -> bool:
    """Caption images listed in job. Returns True if stopped early."""
    list_file = job["list_file"]
    trigger = (job.get("trigger") or "").strip()
    max_new_tokens = int(job.get("max_new_tokens") or 120)
    stop_file = (job.get("stop_file") or "").strip()
    include_video = state.include_video

    images = _load_image_list(list_file)
    allowed = IMAGE_EXTENSIONS | (VIDEO_EXTENSIONS if include_video else set())
    from fizgig.dataset.masks import is_mask_file
    images = [p for p in images if os.path.isfile(p) and os.path.splitext(p)[1].lower() in allowed
              and not is_mask_file(p)]
    total = len(images)
    if not total:
        print("FAIL: no images in list", flush=True)
        return False

    if state.backend == "florence":
        model, processor = state.model
        task = job.get("florence_task") or "<DETAILED_CAPTION>"

        def caption_one(path: str) -> str:
            return _caption_florence(model, processor, state.device, path, task, max_new_tokens)
    else:
        instruction_file = job.get("instruction_file") or ""
        if not instruction_file:
            print("FAIL: instruction_file required", flush=True)
            return False
        with open(instruction_file, encoding="utf-8") as f:
            instruction = f.read().strip()
        if not instruction:
            print("FAIL: empty instruction", flush=True)
            return False
        encoder = state.model

        def caption_one(path: str) -> str:
            return _caption_qwen(encoder, path, instruction, max_new_tokens)

    stopped = False
    for i, path in enumerate(images, 1):
        if stop_file and os.path.exists(stop_file):
            stopped = True
            break
        print(f"PROGRESS: {i} {total}", flush=True)
        base = os.path.basename(path)
        try:
            caption = caption_one(path)
            if not caption:
                print(f"FAIL: {base} (empty caption)", flush=True)
                continue
            _write_caption(path, caption, trigger, job.get("caption_ext") or ".txt")
            print(f"OK: {base}", flush=True)
        except Exception as exc:
            print(f"FAIL: {base} ({exc})", flush=True)

    if stopped:
        print("STOPPED", flush=True)
    else:
        print("DONE", flush=True)
    return stopped


def _serve(config_path: str) -> int:
    _quiet_worker_logs()
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        state = _load_state_from_config(config, device)
    except Exception as exc:
        print(f"FAIL: {exc}", flush=True)
        return 1

    print("READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "QUIT":
            break
        if line.startswith("RUN "):
            job_path = line[4:].strip()
            try:
                with open(job_path, encoding="utf-8") as f:
                    job = json.load(f)
                _run_job(state, job)
            except Exception as exc:
                print(f"FAIL: job ({exc})", flush=True)
                print("DONE", flush=True)
            continue
        print(f"FAIL: unknown command {line!r}", flush=True)

    del state.model
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return 0


def setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch AI-caption (isolated process)")
    p.add_argument("--serve", action="store_true", help="Keep model loaded; read RUN/QUIT on stdin")
    p.add_argument("--config", default="", help="Worker config JSON (--serve)")
    p.add_argument("--backend", choices=("florence", "qwen"))
    p.add_argument("--list_file")
    p.add_argument("--trigger", default="")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--stop_file", default="")
    p.add_argument("--include_video", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--florence_model", default="")
    p.add_argument("--florence_task", default="<DETAILED_CAPTION>")
    p.add_argument("--florence_revision", default="")
    p.add_argument("--florence_code_revision", default="")
    p.add_argument("--text_encoder", default="")
    p.add_argument("--instruction_file", default="")
    return p


def main() -> int:
    args = setup_parser().parse_args()

    if args.serve:
        if not args.config:
            print("FAIL: --config required with --serve", flush=True)
            return 1
        return _serve(args.config)

    if not args.backend or not args.list_file:
        print("FAIL: --backend and --list_file required", flush=True)
        return 1

    _quiet_worker_logs()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "backend": args.backend,
        "include_video": args.include_video,
        "device": device,
        "florence_model": args.florence_model,
        "florence_task": args.florence_task,
        "florence_revision": args.florence_revision,
        "florence_code_revision": args.florence_code_revision,
        "text_encoder": args.text_encoder,
    }
    try:
        state = _load_state_from_config(config, device)
    except Exception as exc:
        print(f"FAIL: {exc}", flush=True)
        return 1

    job = {
        "list_file": args.list_file,
        "trigger": args.trigger,
        "max_new_tokens": args.max_new_tokens,
        "stop_file": args.stop_file,
        "florence_task": args.florence_task,
        "instruction_file": args.instruction_file,
    }
    _run_job(state, job)

    del state.model
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
