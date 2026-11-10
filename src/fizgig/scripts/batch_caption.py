"""Batch AI-caption images/clips in an isolated process, with any captioner in fizgig/captioners/.

The Captions tab runs this so GPU memory stays out of the GUI process - and out of every training run: a captioner
loads nowhere else (the memory rule in fizgig/captioners/__init__.py). In --serve mode the model stays loaded between
jobs until the worker receives QUIT (e.g. when training starts).

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


def picture(path: str):
    """What every captioner is handed: the file upright (a clip: its middle frame), see-through laid on white, RGB."""
    from fizgig.utils.image import flatten
    return flatten(_open_training_frame(path))


def _load_image_list(list_file: str) -> list[str]:
    out: list[str] = []
    with open(list_file, encoding="utf-8") as f:
        for line in f:
            p = line.strip()
            if p:
                out.append(p)
    return out


def _write_caption(path: str, caption: str, ext: str = ".txt") -> None:
    with open(os.path.splitext(path)[0] + (ext or ".txt"), "w", encoding="utf-8") as f:
        f.write(caption)


class CaptionState:
    def __init__(self, desc, captioner, include_video: bool):
        self.desc, self.captioner, self.include_video = desc, captioner, include_video


def _load_state_from_config(config: dict, device: str) -> CaptionState:
    """config: {"captioner": a registry key, "files": {"path"} for one whose file Preferences holds, "include_video"}."""
    from fizgig import captioners
    desc = captioners.get(config.get("captioner") or "")
    if desc is None:
        raise ValueError(f"unknown captioner {config.get('captioner')!r}")
    if isinstance(desc.files, captioners.HFRepo):
        files = {"repo": desc.files.repo, "revision": desc.files.revision, "code_revision": desc.files.code_revision}
    else:
        files = {"path": (config.get("files") or {}).get("path") or ""}
        if not files["path"]:
            raise ValueError(f"{desc.label} needs its model file")
    free = None
    if device == "cuda" and desc.precisions:
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
    precision = captioners.pick_precision(desc, free)
    print(f"INFO: Loading {desc.label}" + (f" ({precision}, {free:.0f} GB free)" if precision and free is not None
                                            else f" ({precision})" if precision else "") + "...", flush=True)
    cap = captioners.make(desc, device, files, precision)
    cap.load()
    print("INFO: Model ready.", flush=True)
    return CaptionState(desc, cap, bool(config.get("include_video")))


def request_for(desc, job: dict):
    """The job's CaptionRequest, or (None, why) when the job doesn't give the captioner what it needs."""
    from fizgig.captioners import CaptionRequest
    instruction = ""
    if desc.instruction:
        instruction = str(job.get("instruction") or "").strip()
        if not instruction and job.get("instruction_file"):
            with open(job["instruction_file"], encoding="utf-8") as f:
                instruction = f.read().strip()
        if not instruction:
            return None, "no instruction"
    return CaptionRequest(task=job.get("task") or desc.default_task, instruction=instruction,
                          max_new_tokens=int(job.get("max_new_tokens") or 120)), ""


def caption_file(state: CaptionState, path: str, request, job: dict) -> str:
    """One file's caption as Fizgig writes it (the captioner's result finished, the trigger word first), or ""."""
    from fizgig import captioners
    text = captioners.finish(state.captioner.caption(picture(path), request),
                             threshold=float(job.get("threshold") or 0.35), leave_out=job.get("leave_out") or (),
                             underscores=bool(job.get("underscores")))
    return captioners.with_trigger(text, job.get("trigger") or "")


def _run_job(state: CaptionState, job: dict) -> bool:
    """Caption images listed in job. Returns True if stopped early."""
    stop_file = (job.get("stop_file") or "").strip()
    images = _load_image_list(job["list_file"])
    allowed = IMAGE_EXTENSIONS | (VIDEO_EXTENSIONS if state.include_video else set())
    from fizgig.dataset.masks import is_mask_file
    images = [p for p in images if os.path.isfile(p) and os.path.splitext(p)[1].lower() in allowed
              and not is_mask_file(p)]
    total = len(images)
    if not total:
        print("FAIL: no images in list", flush=True)
        return False
    request, why = request_for(state.desc, job)
    if request is None:
        print(f"FAIL: {why}", flush=True)
        return False

    stopped = False
    for i, path in enumerate(images, 1):
        if stop_file and os.path.exists(stop_file):
            stopped = True
            break
        print(f"PROGRESS: {i} {total}", flush=True)
        base = os.path.basename(path)
        try:
            caption = caption_file(state, path, request, job)
            if not caption:
                print(f"FAIL: {base} (empty caption)", flush=True)
                continue
            _write_caption(path, caption, job.get("caption_ext") or ".txt")
            print(f"OK: {base}", flush=True)
        except Exception as exc:
            print(f"FAIL: {base} ({exc})", flush=True)

    print("STOPPED" if stopped else "DONE", flush=True)
    return stopped


def free(state: CaptionState) -> None:
    state.captioner.unload()
    state.captioner = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


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

    free(state)
    return 0


def setup_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Batch AI-caption (isolated process)")
    p.add_argument("--serve", action="store_true", help="Keep model loaded; read RUN/QUIT on stdin")
    p.add_argument("--config", default="", help="Worker config JSON (--serve)")
    p.add_argument("--captioner", default="", help="A captioner's key (fizgig/captioners/registry.py), e.g. qwen3vl")
    p.add_argument("--model_file", default="", help="Its model file, for one that doesn't download itself (qwen3vl)")
    p.add_argument("--list_file")
    p.add_argument("--trigger", default="")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--stop_file", default="")
    p.add_argument("--include_video", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--task", default="", help="One of the captioner's tasks (Florence: <DETAILED_CAPTION>...)")
    p.add_argument("--instruction_file", default="", help="The instruction, for a captioner that follows one")
    p.add_argument("--caption_ext", default=".txt")
    p.add_argument("--fetch", default="", help="Download a captioner's files now (its key), so it works offline")
    return p


def main() -> int:
    args = setup_parser().parse_args()

    if args.fetch:                               # Preferences' Download: the captioner's own fetch(), in this process
        from fizgig import captioners
        desc = captioners.get(args.fetch)
        if desc is None or not isinstance(desc.files, captioners.HFRepo):
            print(f"FAIL: {args.fetch} doesn't download itself", flush=True)
            return 1
        try:
            captioners.captioner_class(desc).fetch(desc, {"repo": desc.files.repo, "revision": desc.files.revision,
                                                          "code_revision": desc.files.code_revision})
        except Exception as exc:
            print(f"FAIL: {type(exc).__name__}: {exc}", flush=True)
            return 1
        print("DONE", flush=True)
        return 0

    if args.serve:
        if not args.config:
            print("FAIL: --config required with --serve", flush=True)
            return 1
        return _serve(args.config)

    if not args.captioner or not args.list_file:
        print("FAIL: --captioner and --list_file required", flush=True)
        return 1

    _quiet_worker_logs()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state = _load_state_from_config({"captioner": args.captioner, "include_video": args.include_video,
                                         "files": {"path": args.model_file}}, device)
    except Exception as exc:
        print(f"FAIL: {exc}", flush=True)
        return 1
    _run_job(state, {"list_file": args.list_file, "trigger": args.trigger, "max_new_tokens": args.max_new_tokens,
                     "stop_file": args.stop_file, "task": args.task, "instruction_file": args.instruction_file,
                     "caption_ext": args.caption_ext})
    free(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
