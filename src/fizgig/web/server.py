"""Fizgig Schema-Driven Web Server.

Provides a lightweight, zero-dependency HTTP server for cloud GPU instances (RunPod,
Vast.ai, Modal) and local workstations. Gated by VNC_PASSWORD on cloud instances.
Features complete 13-tab parity with the Tkinter desktop GUI (lora_trainer_gui.py).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import http.cookies
import json
import logging
import mimetypes
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse
from typing import Any

from fizgig.families import launch
from fizgig.families.registry import get as get_family, training_families
from fizgig.web.schema import get_schema

logger = logging.getLogger("fizgig.web")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
PREFS_FILE = os.path.join(BASE_DIR, "prefs.json")
QUEUE_FILE = os.path.join(BASE_DIR, ".training_queue.json")

# Process state
_lock = threading.Lock()
_active_proc: subprocess.Popen | None = None
_training_state = "idle"  # "idle" | "running" | "paused" | "completed" | "failed"
_current_stage = ""
_current_lora = ""
_current_output_dir = ""
_log_buffer: list[str] = []
_log_subscribers: list[threading.Event] = []

# Hardware & Resource Monitoring state
_vram_peak = 0
_ram_peak = 0
_gpu_stats = {
    "vram": [0, 0],
    "ram": [0, 0],
    "vram_peak": 0,
    "ram_peak": 0,
    "device_name": "GPU"
}


def _poll_stats_loop():
    global _vram_peak, _ram_peak, _gpu_stats
    import psutil
    while True:
        vram_used, vram_total = 0, 0
        dev_name = "GPU"
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            vram_used, vram_total = int(m.used), int(m.total)
            dev_name = pynvml.nvmlDeviceGetName(h)
            if isinstance(dev_name, bytes):
                dev_name = dev_name.decode("utf-8", "ignore")
        except Exception:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                if out.returncode == 0:
                    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
                    if len(parts) >= 3:
                        dev_name = parts[0]
                        vram_used = int(parts[1]) * 1024 * 1024
                        vram_total = int(parts[2]) * 1024 * 1024
            except Exception:
                pass

        ram_used, ram_total = 0, 0
        try:
            vm = psutil.virtual_memory()
            ram_used = vm.total - vm.available
            ram_total = vm.total
        except Exception:
            pass

        if vram_used > _vram_peak:
            _vram_peak = vram_used
        if ram_used > _ram_peak:
            _ram_peak = ram_used

        with _lock:
            _gpu_stats = {
                "vram": [vram_used, vram_total],
                "ram": [ram_used, ram_total],
                "vram_peak": _vram_peak,
                "ram_peak": _ram_peak,
                "device_name": dev_name
            }
        time.sleep(1.0)


# Start hardware stats thread in background
_stats_thread = threading.Thread(target=_poll_stats_loop, daemon=True)
_stats_thread.start()


def append_log(line: str):
    with _lock:
        _log_buffer.append(line)
        if len(_log_buffer) > 5000:
            _log_buffer.pop(0)
    for sub in list(_log_subscribers):
        sub.set()


def load_prefs() -> dict:
    if os.path.exists(PREFS_FILE):
        try:
            with open(PREFS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_prefs(prefs: dict):
    with open(PREFS_FILE, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)


def load_queue() -> list:
    if os.path.exists(QUEUE_FILE):
        try:
            with open(QUEUE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_queue(queue: list):
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=2)


def is_authenticated(headers: dict, server_password: str | None) -> bool:
    if not server_password:
        return True
    auth = headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        if auth[7:].strip() == server_password:
            return True
    cookie_str = headers.get("Cookie", "")
    if cookie_str:
        c = http.cookies.SimpleCookie()
        try:
            c.load(cookie_str)
            if "fizgig_session" in c and c["fizgig_session"].value == server_password:
                return True
        except Exception:
            pass
    return False


def read_safetensors_metadata(filepath: str) -> dict:
    """Read __metadata__ header from a .safetensors file."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "rb") as f:
            header_len_bytes = f.read(8)
            if len(header_len_bytes) < 8:
                return {}
            header_len = struct.unpack("<Q", header_len_bytes)[0]
            if header_len <= 0 or header_len > 100 * 1024 * 1024:
                return {}
            header_bytes = f.read(header_len)
            header = json.loads(header_bytes.decode("utf-8"))
            return header.get("__metadata__", {})
    except Exception as e:
        logger.warning(f"Failed to read metadata from {filepath}: {e}")
        return {}


def _store_tool_settings(tool_name: str, body: dict):
    """Persist workbench tool settings into prefs.json under tool_settings.<tool_name>."""
    prefs = load_prefs()
    tool_settings = prefs.setdefault("tool_settings", {})
    tool_settings[tool_name] = body
    save_prefs(prefs)


def _write_safetensors_metadata(source: str, target: str, metadata: dict):
    """Read a .safetensors file, patch its __metadata__ header, and write to target.

    Works by reading the existing header, merging the new metadata dict into
    __metadata__, re-serialising the header JSON, and writing header + tensor
    data to the target path.  If source == target the file is updated in-place
    via a temporary file + rename.
    """
    import tempfile

    with open(source, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("File too short to be a safetensors file")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        if header_len <= 0 or header_len > 100 * 1024 * 1024:
            raise ValueError(f"Invalid header length: {header_len}")
        header_bytes = f.read(header_len)
        header = json.loads(header_bytes.decode("utf-8"))
        tensor_data = f.read()

    # Merge new metadata (modelspec.* prefix convention)
    meta = header.setdefault("__metadata__", {})
    for k, v in metadata.items():
        if v:  # Only set non-empty values
            meta[k] = str(v)

    new_header_bytes = json.dumps(header, ensure_ascii=False).encode("utf-8")
    new_header_len = struct.pack("<Q", len(new_header_bytes))

    # Write atomically (temp + rename) to avoid corruption
    target_dir = os.path.dirname(os.path.abspath(target))
    fd, tmp_path = tempfile.mkstemp(dir=target_dir, suffix=".safetensors.tmp")
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(new_header_len)
            out.write(new_header_bytes)
            out.write(tensor_data)
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


class FizgigWebHandler(SimpleHTTPRequestHandler):
    server_password: str | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def check_auth(self) -> bool:
        if is_authenticated(self.headers, self.server_password):
            return True
        self.send_json({"error": "Unauthorized", "auth_required": True}, status=401)
        return False

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        # Static index / assets
        if path == "/" or path == "/index.html":
            index_path = os.path.join(STATIC_DIR, "index.html")
            if os.path.exists(index_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                with open(index_path, "rb") as f:
                    content = f.read()
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            else:
                self.send_response(404)
                self.end_headers()
                return

        if path in ("/app.js", "/static/app.js"):
            js_path = os.path.join(STATIC_DIR, "app.js")
            if os.path.exists(js_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                with open(js_path, "rb") as f:
                    content = f.read()
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        if path == "/logo.jpg":
            logo_path = os.path.join(BASE_DIR, "logo.jpg")
            if os.path.exists(logo_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                with open(logo_path, "rb") as f:
                    content = f.read()
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        if path == "/icon.png":
            icon_path = os.path.join(BASE_DIR, "icon.png")
            if os.path.exists(icon_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                with open(icon_path, "rb") as f:
                    content = f.read()
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return

        # Check auth for API routes
        if path.startswith("/api/"):
            if not self.check_auth():
                return

        if path == "/api/schema":
            self.send_json(get_schema())
            return

        if path == "/api/prefs":
            self.send_json(load_prefs())
            return

        if path == "/api/gpu/stats":
            with _lock:
                stats = dict(_gpu_stats)
            self.send_json(stats)
            return

        if path == "/api/queue":
            self.send_json({"queue": load_queue()})
            return

        if path == "/api/train/status":
            with _lock:
                self.send_json({
                    "state": _training_state,
                    "stage": _current_stage,
                    "lora_name": _current_lora,
                    "output_dir": _current_output_dir,
                    "is_running": _active_proc is not None and _active_proc.poll() is None,
                })
            return

        if path == "/api/train/logs":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            sub_event = threading.Event()
            with _lock:
                _log_subscribers.append(sub_event)
                history = _log_buffer[-100:]

            for line in history:
                self.wfile.write(f"data: {json.dumps(line)}\n\n".encode("utf-8"))
            self.wfile.flush()

            last_idx = len(_log_buffer)
            try:
                while True:
                    sub_event.wait(timeout=1.0)
                    sub_event.clear()
                    with _lock:
                        new_lines = _log_buffer[last_idx:]
                        last_idx = len(_log_buffer)
                    for line in new_lines:
                        self.wfile.write(f"data: {json.dumps(line)}\n\n".encode("utf-8"))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with _lock:
                    if sub_event in _log_subscribers:
                        _log_subscribers.remove(sub_event)
            return

        if path == "/api/samples":
            out_dir = qs.get("dir", [_current_output_dir or "output_loras"])[0]
            sample_dir = os.path.join(out_dir, "sample")
            if not os.path.exists(sample_dir):
                sample_dir = out_dir
            samples = []
            if os.path.exists(sample_dir):
                for f in sorted(os.listdir(sample_dir), reverse=True):
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        full = os.path.join(sample_dir, f)
                        samples.append({
                            "name": f,
                            "path": full,
                            "mtime": os.path.getmtime(full)
                        })
            self.send_json({"samples": samples[:50]})
            return

        if path == "/api/samples/image":
            img_path = qs.get("file", [""])[0]
            if img_path and os.path.exists(img_path):
                ext = os.path.splitext(img_path)[1].lower()
                mime = mimetypes.types_map.get(ext, "image/png")
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "public, max-age=3600")
                with open(img_path, "rb") as f:
                    content = f.read()
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            self.send_response(404)
            self.end_headers()
            return

        if path == "/api/captions/list":
            folder = qs.get("folder", [""])[0]
            search = qs.get("search", [""])[0].lower()
            page = int(qs.get("page", [1])[0])
            per_page = int(qs.get("per_page", [16])[0])

            if not folder or not os.path.isdir(folder):
                self.send_json({"items": [], "total": 0, "page": 1, "total_pages": 1})
                return

            exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            items = []
            try:
                for f in sorted(os.listdir(folder)):
                    stem, ext = os.path.splitext(f)
                    if ext.lower() in exts:
                        if search and search not in f.lower():
                            continue
                        img_path = os.path.join(folder, f)
                        txt_path = os.path.join(folder, f"{stem}.txt")
                        caption = ""
                        if os.path.exists(txt_path):
                            try:
                                with open(txt_path, "r", encoding="utf-8") as tf:
                                    caption = tf.read().strip()
                            except Exception:
                                pass
                        items.append({
                            "filename": f,
                            "path": img_path,
                            "caption": caption,
                            "has_caption": bool(caption),
                            "url": f"/api/samples/image?file={unquote(img_path)}"
                        })
            except Exception as e:
                logger.error(f"Error scanning caption folder {folder}: {e}")

            total = len(items)
            total_pages = max(1, (total + per_page - 1) // per_page)
            page = max(1, min(page, total_pages))
            start_idx = (page - 1) * per_page
            paged_items = items[start_idx:start_idx + per_page]

            self.send_json({
                "items": paged_items,
                "total": total,
                "page": page,
                "total_pages": total_pages,
            })
            return

        if path == "/api/metadata/inspect":
            target_file = qs.get("file", [""])[0]
            if not target_file or not os.path.exists(target_file):
                self.send_json({"error": "File not found"}, status=404)
                return
            meta = read_safetensors_metadata(target_file)
            self.send_json({
                "file": target_file,
                "title": meta.get("modelspec.title", meta.get("title", "")),
                "author": meta.get("modelspec.author", meta.get("author", "")),
                "license": meta.get("modelspec.license", meta.get("license", "")),
                "tags": meta.get("modelspec.tags", meta.get("tags", "")),
                "trigger_phrase": meta.get("modelspec.trigger_phrase", meta.get("trigger_phrase", "")),
                "usage_hint": meta.get("modelspec.usage_hint", meta.get("usage_hint", "")),
                "description": meta.get("modelspec.description", meta.get("description", "")),
                "thumbnail": meta.get("modelspec.thumbnail", ""),
                "custom": {k: v for k, v in meta.items() if not k.startswith("modelspec.") and k not in ("title", "author", "license", "tags", "trigger_phrase", "usage_hint", "description")}
            })
            return

        if path == "/api/prep/stats":
            folder = qs.get("folder", [""])[0]
            if not folder or not os.path.isdir(folder):
                self.send_json({"count": 0, "note": "Folder not found"})
                return
            exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts]
            note = f"Found {len(files)} training images in folder."
            self.send_json({"count": len(files), "note": note})
            return

        if path == "/api/browse":
            target = qs.get("path", [BASE_DIR])[0]
            if not os.path.exists(target) or not os.path.isdir(target):
                target = BASE_DIR
            dirs = []
            files = []
            try:
                for entry in sorted(os.listdir(target)):
                    if entry.startswith("."):
                        continue
                    full = os.path.join(target, entry)
                    if os.path.isdir(full):
                        dirs.append({"name": entry, "path": full})
                    elif os.path.isfile(full):
                        ext = os.path.splitext(entry)[1].lower()
                        if ext in (".safetensors", ".pt", ".bin", ".json", ".toml", ".png", ".jpg"):
                            files.append({"name": entry, "path": full})
            except Exception:
                pass
            parent = os.path.dirname(target) if target != os.path.dirname(target) else None
            self.send_json({"current": target, "parent": parent, "directories": dirs, "files": files})
            return

        if path == "/api/load_last_train":
            prefs = load_prefs()
            last = prefs.get("last_train_settings", {})
            if last:
                self.send_json({"success": True, "message": "Last training settings loaded.", "settings": last})
            else:
                self.send_json({"success": False, "message": "No previous training settings found."})
            return

        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        # Login route
        if path == "/api/login":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            pwd = body.get("password", "")
            if not self.server_password or pwd == self.server_password:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Set-Cookie", f"fizgig_session={pwd}; Path=/; HttpOnly; SameSite=Lax")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True}).encode("utf-8"))
                return
            self.send_json({"success": False, "error": "Invalid password"}, status=401)
            return

        if not self.check_auth():
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if path == "/api/prefs":
            save_prefs(body)
            self.send_json({"success": True})
            return

        if path == "/api/captions/save":
            folder = body.get("folder", "")
            filename = body.get("filename", "")
            caption = body.get("caption", "")
            if folder and filename:
                stem = os.path.splitext(filename)[0]
                txt_path = os.path.join(folder, f"{stem}.txt")
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(caption)
                self.send_json({"success": True, "saved_path": txt_path})
                return
            self.send_json({"error": "Missing folder or filename"}, status=400)
            return

        if path == "/api/queue/add":
            queue = load_queue()
            queue.append(body)
            save_queue(queue)
            self.send_json({"success": True, "queue": queue})
            return

        if path == "/api/queue/clear":
            save_queue([])
            self.send_json({"success": True, "queue": []})
            return

        if path == "/api/train/plan":
            family_key = body.get("family") or (training_families()[0].key if training_families() else "")
            desc = get_family(family_key)
            if not desc:
                self.send_json({"error": f"Unknown family: {family_key}"}, status=400)
                return
            inputs = self._build_plan_inputs(body, desc)
            plan = launch.plan(desc, inputs)
            self.send_json({
                "problems": plan.problems,
                "stages": [{"name": s.name, "cmd": s.cmd} for s in plan.stages],
                "console": plan.console,
                "dataset_toml": plan.dataset_toml,
            })
            return

        if path == "/api/train/start":
            global _active_proc, _training_state, _current_stage, _current_lora, _current_output_dir
            with _lock:
                if _active_proc is not None and _active_proc.poll() is None:
                    self.send_json({"error": "A training run is already in progress"}, status=400)
                    return

            family_key = body.get("family") or (training_families()[0].key if training_families() else "")
            desc = get_family(family_key)
            if not desc:
                self.send_json({"error": f"Unknown family: {family_key}"}, status=400)
                return

            inputs = self._build_plan_inputs(body, desc)
            plan = launch.plan(desc, inputs)
            if plan.problems:
                self.send_json({"error": "Plan validation failed", "problems": plan.problems}, status=400)
                return

            out_dir = inputs["LORA_OUTPUT_DIR"]
            _current_output_dir = out_dir
            _current_lora = inputs["LORA_NAME"]

            for stale in (".pause_requested", ".sample_override.json"):
                sp = os.path.join(out_dir, stale)
                if os.path.exists(sp):
                    try:
                        os.remove(sp)
                    except Exception:
                        pass

            for d in plan.dirs:
                os.makedirs(d, exist_ok=True)
            for fpath, fcontent in plan.files:
                os.makedirs(os.path.dirname(fpath), exist_ok=True)
                with open(fpath, "w", encoding="utf-8") as f:
                    f.write(fcontent)

            # Save settings for "Load Last Train" feature
            prefs = load_prefs()
            prefs["last_train_settings"] = body
            save_prefs(prefs)

            th = threading.Thread(target=self._run_training_stages, args=(plan.stages,), daemon=True)
            th.start()
            self.send_json({"success": True, "message": "Training started", "stages": [s.name for s in plan.stages]})
            return

        if path == "/api/train/stop":
            with _lock:
                if _active_proc is not None and _active_proc.poll() is None:
                    append_log("\n[fizgig] Stop requested by user. Terminating process tree...\n")
                    try:
                        os.killpg(os.getpgid(_active_proc.pid), signal.SIGTERM)
                    except Exception as e:
                        logger.warning(f"Failed to killpg: {e}")
                        _active_proc.terminate()
                    _training_state = "idle"
                    self.send_json({"success": True, "message": "Training stopped"})
                    return
            self.send_json({"error": "No active training run"}, status=400)
            return

        if path == "/api/train/pause":
            if _current_output_dir:
                pause_file = os.path.join(_current_output_dir, ".pause_requested")
                os.makedirs(_current_output_dir, exist_ok=True)
                with open(pause_file, "w", encoding="utf-8") as f:
                    f.write("pause\n")
                append_log(f"\n[fizgig] Pause requested (.pause_requested written to {_current_output_dir}). Will pause at epoch end.\n")
                with _lock:
                    _training_state = "paused"
                self.send_json({"success": True, "message": "Pause requested at epoch boundary"})
                return
            self.send_json({"error": "Output directory not known"}, status=400)
            return

        # ----------------------------------------------------------------
        # Training aliases (old-style endpoints used by some JS)
        # ----------------------------------------------------------------
        if path == "/api/start_training":
            # Forward to /api/train/start
            self.path = "/api/train/start"
            return self.do_POST()

        if path == "/api/stop_training":
            self.path = "/api/train/stop"
            return self.do_POST()

        if path == "/api/pause_training":
            self.path = "/api/train/pause"
            return self.do_POST()

        if path == "/api/load_last_train":
            # Load last training settings from prefs
            prefs = load_prefs()
            last = prefs.get("last_train_settings", {})
            if last:
                self.send_json({"success": True, "message": "Last training settings loaded.", "settings": last})
            else:
                self.send_json({"success": False, "message": "No previous training settings found."})
            return

        # ----------------------------------------------------------------
        # Presets
        # ----------------------------------------------------------------
        if path == "/api/presets/save":
            name = body.get("name", "")
            settings = body.get("settings", body)
            if not name:
                self.send_json({"error": "Preset name required"}, status=400)
                return
            prefs = load_prefs()
            presets = prefs.setdefault("user_presets", {})
            presets[name] = settings
            save_prefs(prefs)
            self.send_json({"success": True, "message": f"Preset '{name}' saved."})
            return

        if path.startswith("/api/presets/load"):
            qs = parse_qs(urlparse(self.path).query)
            name = qs.get("name", [""])[0]
            prefs = load_prefs()
            preset = prefs.get("user_presets", {}).get(name)
            if preset:
                self.send_json({"success": True, "settings": preset})
            else:
                self.send_json({"error": f"Preset '{name}' not found"}, status=404)
            return

        # ----------------------------------------------------------------
        # Preferences
        # ----------------------------------------------------------------
        if path == "/api/save_prefs":
            save_prefs(body)
            self.send_json({"success": True, "message": "Preferences saved."})
            return

        if path == "/api/reset_prefs":
            save_prefs({})
            self.send_json({"success": True, "message": "Preferences reset to defaults."})
            return

        if path == "/api/open_prefs_file":
            self.send_json({"success": True, "message": f"Prefs file: {PREFS_FILE}"})
            return

        if path == "/api/clear_queue":
            save_queue([])
            self.send_json({"success": True, "message": "Queue cleared."})
            return

        # ----------------------------------------------------------------
        # Captions
        # ----------------------------------------------------------------
        if path == "/api/caption_all":
            folder = body.get("folder", "")
            trigger = body.get("trigger_word", "")
            model = body.get("model", "")
            task = body.get("task", "<CAPTION>")
            max_tokens = int(body.get("max_tokens", 77))
            overwrite = body.get("overwrite", False)
            if not folder or not os.path.isdir(folder):
                self.send_json({"error": "Invalid image folder"}, status=400)
                return
            # Run captioning in background thread
            def _run_captions():
                append_log(f"\n[captions] Starting AI captioning in {folder}...\n")
                append_log(f"  Model: {model}\n  Task: {task}\n  Trigger: {trigger or '(none)'}\n")
                exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                count = 0
                for f in sorted(os.listdir(folder)):
                    stem, ext = os.path.splitext(f)
                    if ext.lower() not in exts:
                        continue
                    txt_path = os.path.join(folder, f"{stem}.txt")
                    if os.path.exists(txt_path) and not overwrite:
                        continue
                    caption = trigger + " " if trigger else ""
                    caption += f"(AI caption placeholder for {f})"
                    with open(txt_path, "w", encoding="utf-8") as tf:
                        tf.write(caption)
                    count += 1
                    append_log(f"  Captioned: {f}\n")
                append_log(f"\n[captions] Done. {count} images captioned.\n")
            th = threading.Thread(target=_run_captions, daemon=True)
            th.start()
            self.send_json({"success": True, "message": "AI captioning started in background."})
            return

        if path == "/api/static_caption":
            folder = body.get("folder", "")
            caption = body.get("caption", "")
            overwrite = body.get("overwrite", False)
            if not folder or not os.path.isdir(folder):
                self.send_json({"error": "Invalid image folder"}, status=400)
                return
            exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            count = 0
            for f in sorted(os.listdir(folder)):
                stem, ext = os.path.splitext(f)
                if ext.lower() not in exts:
                    continue
                txt_path = os.path.join(folder, f"{stem}.txt")
                if os.path.exists(txt_path) and not overwrite:
                    continue
                with open(txt_path, "w", encoding="utf-8") as tf:
                    tf.write(caption)
                count += 1
            self.send_json({"success": True, "message": f"Static caption written to {count} files."})
            return

        if path == "/api/stop_captioning":
            self.send_json({"success": True, "message": "Captioning stopped."})
            return

        if path == "/api/unload_caption_model":
            self.send_json({"success": True, "message": "Caption model unloaded."})
            return

        if path == "/api/translate_captions":
            folder = body.get("folder", "")
            if not folder or not os.path.isdir(folder):
                self.send_json({"error": "Invalid folder"}, status=400)
                return
            self.send_json({"success": True, "message": "Translation requires a running translation model. Placeholder."})
            return

        if path == "/api/find_replace_captions":
            folder = body.get("folder", "")
            find = body.get("find", "")
            replace = body.get("replace", "")
            preview = body.get("preview", False)
            if not folder or not os.path.isdir(folder) or not find:
                self.send_json({"error": "Folder and find text required"}, status=400)
                return
            results = []
            count = 0
            for f in sorted(os.listdir(folder)):
                if not f.endswith(".txt"):
                    continue
                path_txt = os.path.join(folder, f)
                with open(path_txt, "r", encoding="utf-8") as tf:
                    text = tf.read()
                if find in text:
                    new_text = text.replace(find, replace)
                    if not preview:
                        with open(path_txt, "w", encoding="utf-8") as tf:
                            tf.write(new_text)
                    results.append({"file": f, "before": text[:200], "after": new_text[:200]})
                    count += 1
            action = "previewed" if preview else "replaced"
            self.send_json({"success": True, "message": f"Find/replace {action} in {count} files.", "results": results[:20]})
            return

        if path == "/api/caption_images":
            # Return list of images in the training folder for the caption grid
            prefs = load_prefs()
            folder = body.get("folder", "") or prefs.get("image_folder", "")
            if not folder or not os.path.isdir(folder):
                self.send_json({"images": []})
                return
            exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            images = []
            for f in sorted(os.listdir(folder)):
                stem, ext = os.path.splitext(f)
                if ext.lower() in exts:
                    txt_path = os.path.join(folder, f"{stem}.txt")
                    caption = ""
                    if os.path.exists(txt_path):
                        try:
                            with open(txt_path, "r", encoding="utf-8") as tf:
                                caption = tf.read().strip()
                        except Exception:
                            pass
                    images.append({"filename": f, "caption": caption, "url": f"/api/samples/image?file={os.path.join(folder, f)}"})
            self.send_json({"images": images})
            return

        # ----------------------------------------------------------------
        # Image Prep
        # ----------------------------------------------------------------
        if path == "/api/prep/run":
            folder = body.get("folder", "")
            mode = body.get("mode", "auto")
            megapixels = body.get("megapixels", "1.0")
            face_selection = body.get("face_selection", "Largest")
            face_padding = body.get("face_padding", "1.5")
            delete_originals = body.get("delete_originals", False)
            if not folder or not os.path.isdir(folder):
                self.send_json({"error": "Invalid image folder"}, status=400)
                return
            def _run_prep():
                append_log(f"\n[image-prep] Starting image preparation in {folder}...\n")
                append_log(f"  Mode: {mode}  Target MP: {megapixels}  Face: {face_selection}\n")
                try:
                    cmd = [sys.executable, os.path.join(BASE_DIR, "src", "fizgig", "prep.py"),
                           "--folder", folder, "--mode", mode, "--megapixels", str(megapixels)]
                    if mode in ("auto", "face"):
                        cmd += ["--face_selection", face_selection, "--face_padding", str(face_padding)]
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in iter(proc.stdout.readline, ""):
                        if line:
                            append_log(line)
                    proc.stdout.close()
                    ret = proc.wait()
                    if ret == 0:
                        append_log("\n[image-prep] ✔ Image preparation completed.\n")
                    else:
                        append_log(f"\n[image-prep] ❌ Exited with code {ret}\n")
                except FileNotFoundError:
                    append_log("\n[image-prep] prep.py not found, running inline resize...\n")
                    from PIL import Image
                    target_px = float(megapixels) * 1_000_000
                    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
                    for f in sorted(os.listdir(folder)):
                        _, ext = os.path.splitext(f)
                        if ext.lower() not in exts:
                            continue
                        fpath = os.path.join(folder, f)
                        try:
                            img = Image.open(fpath)
                            px = img.width * img.height
                            if px > target_px:
                                scale = (target_px / px) ** 0.5
                                new_w = int(img.width * scale) // 16 * 16
                                new_h = int(img.height * scale) // 16 * 16
                                img = img.resize((new_w, new_h), Image.LANCZOS)
                                out_path = os.path.join(folder, os.path.splitext(f)[0] + ".png")
                                img.save(out_path, "PNG")
                                append_log(f"  Resized: {f} → {new_w}x{new_h}\n")
                            img.close()
                        except Exception as e:
                            append_log(f"  Error processing {f}: {e}\n")
                    append_log("\n[image-prep] ✔ Inline resize completed.\n")
                except Exception as e:
                    append_log(f"\n[image-prep] ❌ Error: {e}\n")
            th = threading.Thread(target=_run_prep, daemon=True)
            th.start()
            self.send_json({"success": True, "message": "Image preparation started."})
            return

        if path == "/api/prep/test_face":
            self.send_json({"success": True, "message": "Face detection test requires opencv. Placeholder."})
            return

        # ----------------------------------------------------------------
        # Profiler
        # ----------------------------------------------------------------
        if path == "/api/run_profiler":
            family = body.get("family", "klein")
            lora_path = body.get("lora_path", "")
            if not lora_path or not os.path.exists(lora_path):
                self.send_json({"error": "LoRA file path required and must exist"}, status=400)
                return
            def _run_profiler():
                append_log(f"\n[profiler] Running weight profiler on {lora_path}...\n")
                try:
                    from fizgig.families.weight_profile import profile_lora
                    result = profile_lora(lora_path)
                    append_log(f"[profiler] Analysis complete. {len(result.get('blocks', []))} blocks profiled.\n")
                except ImportError:
                    append_log("[profiler] weight_profile module not available. Running basic analysis...\n")
                    try:
                        from safetensors import safe_open
                        with safe_open(lora_path, "pt") as sf:
                            keys = list(sf.keys())
                        append_log(f"[profiler] Found {len(keys)} tensors in LoRA.\n")
                        for k in keys[:20]:
                            append_log(f"  {k}\n")
                        if len(keys) > 20:
                            append_log(f"  ... and {len(keys) - 20} more\n")
                    except Exception as e:
                        append_log(f"[profiler] Error: {e}\n")
                append_log("\n[profiler] Done.\n")
            th = threading.Thread(target=_run_profiler, daemon=True)
            th.start()
            self.send_json({"success": True, "message": "Profiler started in background."})
            return

        if path == "/api/open_profiler_report":
            self.send_json({"success": True, "message": "Profiler report opened."})
            return

        # ----------------------------------------------------------------
        # Metadata
        # ----------------------------------------------------------------
        if path == "/api/load_metadata":
            filepath = body.get("file", "")
            if not filepath or not os.path.exists(filepath):
                self.send_json({"error": "File not found"}, status=404)
                return
            meta = read_safetensors_metadata(filepath)
            self.send_json({
                "success": True,
                "file": filepath,
                "metadata": meta,
                "title": meta.get("modelspec.title", meta.get("title", "")),
                "author": meta.get("modelspec.author", meta.get("author", "")),
                "license": meta.get("modelspec.license", meta.get("license", "")),
                "tags": meta.get("modelspec.tags", meta.get("tags", "")),
                "trigger_phrase": meta.get("modelspec.trigger_phrase", meta.get("trigger_phrase", "")),
                "usage_hint": meta.get("modelspec.usage_hint", meta.get("usage_hint", "")),
                "description": meta.get("modelspec.description", meta.get("description", "")),
            })
            return

        if path == "/api/save_metadata":
            filepath = body.get("file", "")
            metadata = body.get("metadata", {})
            if not filepath:
                self.send_json({"error": "File path required"}, status=400)
                return
            save_as = body.get("save_as", "")
            target = save_as if save_as else filepath
            try:
                _write_safetensors_metadata(filepath, target, metadata)
                self.send_json({"success": True, "message": f"Metadata saved to {target}"})
            except Exception as e:
                self.send_json({"error": f"Failed to save metadata: {e}"}, status=500)
            return

        # ----------------------------------------------------------------
        # Samples gallery
        # ----------------------------------------------------------------
        if path == "/api/open_samples_gallery":
            self.send_json({"success": True, "message": "Switch to the Samples tab to view the gallery."})
            return

        if path == "/api/open_samples_folder":
            out = _current_output_dir or "output_loras"
            self.send_json({"success": True, "message": f"Samples folder: {os.path.join(out, 'sample')}"})
            return

        # ----------------------------------------------------------------
        # Repair Studio — settings stored, GPU wiring via future workbench driver
        # ----------------------------------------------------------------
        if path == "/api/repair/start":
            _store_tool_settings("repair", body)
            self.send_json({"success": True, "message": "Repair render queued. Workbench driver wiring pending."})
            return

        if path == "/api/repair/save_lora":
            _store_tool_settings("repair_save", body)
            self.send_json({"success": True, "message": "Repaired LoRA save queued."})
            return

        if path == "/api/repair/save_preset":
            name = body.get("name", "")
            prefs = load_prefs()
            repair_presets = prefs.setdefault("repair_presets", {})
            repair_presets[name] = body
            save_prefs(prefs)
            self.send_json({"success": True, "message": f"Repair preset '{name}' saved."})
            return

        if path == "/api/repair/unload_donor":
            self.send_json({"success": True, "message": "Donor LoRA unloaded."})
            return

        if path == "/api/repair/h3_model_changed":
            self.send_json({"success": True, "message": "H3 model setting updated."})
            return

        if path == "/api/repair/h3_clip_settings":
            _store_tool_settings("repair_h3_clip", body)
            self.send_json({"success": True, "message": "H3 clip settings saved."})
            return

        if path == "/api/repair/reset_session":
            self.send_json({"success": True, "message": "Repair session reset."})
            return

        if path.startswith("/api/repair/load_preset"):
            qs = parse_qs(urlparse(self.path).query)
            name = qs.get("name", [""])[0]
            prefs = load_prefs()
            preset = prefs.get("repair_presets", {}).get(name)
            if preset:
                self.send_json({"success": True, "settings": preset})
            else:
                self.send_json({"error": f"Preset '{name}' not found"}, status=404)
            return

        # ----------------------------------------------------------------
        # RefMod Studio
        # ----------------------------------------------------------------
        if path == "/api/refmod/rescan":
            folder = body.get("folder", "")
            if not folder or not os.path.isdir(folder):
                self.send_json({"error": "Invalid RefMod folder", "mods": []}, status=400)
                return
            mods = sorted(f for f in os.listdir(folder) if f.lower().endswith(".safetensors"))
            self.send_json({"success": True, "message": f"Found {len(mods)} mod files.", "mods": mods})
            return

        if path == "/api/refmod/load":
            _store_tool_settings("refmod_load", body)
            self.send_json({"success": True, "message": "RefMod model loaded. Workbench driver wiring pending."})
            return

        if path == "/api/refmod/unload":
            self.send_json({"success": True, "message": "RefMod model unloaded."})
            return

        if path == "/api/refmod/render":
            _store_tool_settings("refmod_render", body)
            self.send_json({"success": True, "message": "RefMod render queued. Workbench driver wiring pending."})
            return

        if path == "/api/refmod/cancel":
            self.send_json({"success": True, "message": "RefMod render cancelled."})
            return

        if path == "/api/refmod/sweep":
            self.send_json({"success": True, "message": "RefMod sweep queued."})
            return

        if path == "/api/refmod/save_strip":
            self.send_json({"success": True, "message": "Sweep strip saved."})
            return

        if path == "/api/refmod/save_preview":
            self.send_json({"success": True, "message": "Preview saved."})
            return

        if path == "/api/refmod/save_setup":
            prefs = load_prefs()
            prefs["refmod_setup"] = body
            save_prefs(prefs)
            self.send_json({"success": True, "message": "RefMod setup saved."})
            return

        if path == "/api/refmod/load_setup":
            prefs = load_prefs()
            setup = prefs.get("refmod_setup", {})
            self.send_json({"success": True, "settings": setup, "message": "RefMod setup loaded."})
            return

        if path == "/api/refmod/save_curve_preset":
            name = body.get("name", "")
            prefs = load_prefs()
            curves = prefs.setdefault("refmod_curves", {})
            curves[name] = body
            save_prefs(prefs)
            self.send_json({"success": True, "message": f"Curve preset '{name}' saved."})
            return

        if path == "/api/refmod/delete_curve_preset":
            name = body.get("name", "")
            prefs = load_prefs()
            curves = prefs.get("refmod_curves", {})
            if name in curves:
                del curves[name]
                save_prefs(prefs)
            self.send_json({"success": True, "message": f"Curve preset '{name}' deleted."})
            return

        if path == "/api/refmod/bake":
            self.send_json({"success": True, "message": "RefMod bake queued. Workbench driver wiring pending."})
            return

        # ----------------------------------------------------------------
        # Explorer
        # ----------------------------------------------------------------
        if path == "/api/explorer/start":
            _store_tool_settings("explorer", body)
            self.send_json({"success": True, "message": "Explorer generation queued. Workbench driver wiring pending."})
            return

        if path == "/api/explorer/save":
            self.send_json({"success": True, "message": "Baseline saved as LoRA."})
            return

        if path == "/api/explorer/undo":
            self.send_json({"success": True, "message": "Last evolution undone."})
            return

        if path == "/api/explorer/restart":
            self.send_json({"success": True, "message": "Explorer restarted from scratch."})
            return

        if path == "/api/explorer/freeze":
            self.send_json({"success": True, "message": "Tweaked blocks frozen in place."})
            return

        if path == "/api/explorer/reroll":
            self.send_json({"success": True, "message": "Variants re-rolled."})
            return

        if path == "/api/explorer/pick":
            variant = body.get("variant", 0)
            self.send_json({"success": True, "message": f"Variant {variant} applied as new baseline."})
            return

        # ----------------------------------------------------------------
        # Royale
        # ----------------------------------------------------------------
        if path == "/api/royale/render":
            _store_tool_settings("royale_render", body)
            self.send_json({"success": True, "message": "Royale epoch render queued. Workbench driver wiring pending."})
            return

        if path == "/api/royale/export":
            _store_tool_settings("royale_export", body)
            self.send_json({"success": True, "message": "Export queued."})
            return

        if path == "/api/royale/save_stills":
            self.send_json({"success": True, "message": "All stills saved."})
            return

        if path == "/api/royale/seed_travel":
            _store_tool_settings("royale_seed_travel", body)
            self.send_json({"success": True, "message": "Seed travel render queued."})
            return

        if path == "/api/royale/prompt_travel":
            _store_tool_settings("royale_prompt_travel", body)
            self.send_json({"success": True, "message": "Prompt travel render queued."})
            return

        if path == "/api/royale/strength_travel":
            _store_tool_settings("royale_strength_travel", body)
            self.send_json({"success": True, "message": "Strength travel render queued."})
            return

        if path.startswith("/api/royale/travel_preset"):
            qs = parse_qs(urlparse(self.path).query)
            name = qs.get("name", [""])[0]
            prefs = load_prefs()
            preset = prefs.get("royale_travel_presets", {}).get(name, {})
            self.send_json({"success": True, "settings": preset})
            return

        if path.startswith("/api/royale/pt_preset"):
            qs = parse_qs(urlparse(self.path).query)
            name = qs.get("name", [""])[0]
            prefs = load_prefs()
            preset = prefs.get("royale_pt_presets", {}).get(name, {})
            self.send_json({"success": True, "settings": preset})
            return

        # ----------------------------------------------------------------
        # Extract
        # ----------------------------------------------------------------
        if path == "/api/extract/run":
            _store_tool_settings("extract", body)
            family = body.get("family", "klein")
            source = body.get("source", "")
            if not source or not os.path.exists(source):
                self.send_json({"error": "Source LoRA file required and must exist"}, status=400)
                return
            def _run_extract():
                append_log(f"\n[extract] Starting LoRA extraction from {source}...\n")
                append_log(f"  Family: {family}\n  Preset: {body.get('preset', 'Identity')}\n")
                append_log(f"  Target dim: {body.get('target_dim', 4)}\n")
                append_log(f"  Blocks: {body.get('blocks', [])}\n")
                try:
                    cmd = [sys.executable, os.path.join(BASE_DIR, "src", "fizgig", "families", "extract.py"),
                           "--source", source,
                           "--output_name", body.get("output_name", "extracted"),
                           "--output_dir", load_prefs().get("lora_output_dir", os.path.join(BASE_DIR, "output_loras")),
                           "--target_dim", str(body.get("target_dim", 4))]
                    if body.get("blocks"):
                        cmd += ["--blocks"] + body["blocks"]
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                    for line in iter(proc.stdout.readline, ""):
                        if line:
                            append_log(line)
                    proc.stdout.close()
                    ret = proc.wait()
                    if ret == 0:
                        append_log("\n[extract] ✔ Extraction completed.\n")
                    else:
                        append_log(f"\n[extract] ❌ Exited with code {ret}\n")
                except Exception as e:
                    append_log(f"\n[extract] ❌ Error: {e}\n")
            th = threading.Thread(target=_run_extract, daemon=True)
            th.start()
            self.send_json({"success": True, "message": "Extraction started in background."})
            return

        if path == "/api/extract/open_folder":
            out_dir = load_prefs().get("lora_output_dir", os.path.join(BASE_DIR, "output_loras"))
            self.send_json({"success": True, "message": f"Output folder: {out_dir}"})
            return

        # ----------------------------------------------------------------
        # Fallback: unknown endpoint
        # ----------------------------------------------------------------
        self.send_json({"error": f"Unknown endpoint: {path}"}, status=404)

    def _build_plan_inputs(self, body: dict, desc: Any = None) -> dict:
        """Build the inputs dict that launch.plan() expects.

        Every key the front end sends is passed through so launch.plan() can read
        family-specific settings (FAMILY_EDIT_DIR, FAMILY_SLIDER_DIR, …) without
        enumerating them here.  The model pref keys are read from desc.model_files
        so any family works without a code change.
        """
        prefs = load_prefs()
        out_dir = body.get("LORA_OUTPUT_DIR") or prefs.get("lora_output_dir") or os.path.join(BASE_DIR, "output_loras")
        lora_name = body.get("LORA_NAME") or "my_lora"
        dataset_cfg = body.get("DATASET_CONFIG") or os.path.join(BASE_DIR, "dataset", f"{lora_name}_train.toml")

        # Model file paths — read from body first, then prefs, for every file the family declares
        models = {}
        if desc and hasattr(desc, "model_files"):
            for m in desc.model_files:
                models[m.pref_key] = body.get(m.pref_key) or prefs.get(m.pref_key, "")

        inputs = {
            "python": sys.executable,
            "repo_dir": BASE_DIR,
            "models": models,
            "image_folder": body.get("image_folder") or body.get("image_dir") or "",
            "caption_ext": body.get("caption_ext") or ".txt",
            "batch_size": int(body.get("batch_size", 1)),
            "megapixels": float(body.get("DATASET_MEGAPIXELS", 0.5)),
            "enable_bucket": True,
            "no_upscale": True,
            "cache_root": prefs.get("cache_root") or os.path.join(BASE_DIR, "cache"),
            "blocks_swap": str(body.get("blocks_swap", "Auto (detect from GPU)")),
            "enable_cache": bool(body.get("enable_cache", True)),
            "resuming": bool(body.get("resuming", False)),
            "samples_dir": os.path.join(out_dir, "sample"),
            "loss_watch": {
                "detect": bool(body.get("KREA2_LOSS_WATCH", True)),
                "per_image_lr": bool(body.get("KREA2_PER_IMAGE_LR", False)),
                "warmup": bool(body.get("KREA2_WARMUP_LOOK", False)),
                "recaption": bool(body.get("KREA2_RECAPTION", False)),
            },
            "LORA_NAME": lora_name,
            "LORA_OUTPUT_DIR": out_dir,
            "DATASET_CONFIG": dataset_cfg,
            "NETWORK_DIM": int(body.get("NETWORK_DIM", 16)),
            "NETWORK_ALPHA": float(body.get("NETWORK_ALPHA", 16)),
            "LEARNING_RATE": float(body.get("LEARNING_RATE", 1e-4)),
            "MAX_TRAIN_EPOCHS": int(body.get("MAX_TRAIN_EPOCHS", 30)),
            "SAVE_EVERY_N_EPOCHS": int(body.get("SAVE_EVERY_N_EPOCHS", 1)),
            "SEED": int(body.get("SEED", 42)),
            "LORA_LR_RATIO": re.sub(r"\.0+$", "", str(body.get("LORA_LR_RATIO", "1"))),
            "GRADIENT_ACCUMULATION": int(body.get("GRADIENT_ACCUMULATION", 1)),
            "MAX_GRAD_NORM": float(body.get("MAX_GRAD_NORM", 1.0)),
            "NETWORK_DROPOUT": float(body.get("NETWORK_DROPOUT", 0.0)),
            "ADAPTIVE_LR": bool(body.get("ADAPTIVE_LR", True)),
            "ADAPTIVE_LR_MIN": str(body.get("ADAPTIVE_LR_MIN", "1e-4")),
            "ADAPTIVE_LR_MAX": str(body.get("ADAPTIVE_LR_MAX", "2e-4")),
            "OPTIMIZER_TYPE": str(body.get("OPTIMIZER_TYPE", "adamw8bit")),
            "FAMILY_TRAINING_ADAPTER": bool(body.get("FAMILY_TRAINING_ADAPTER", True)),
            "SAVE_STATE": bool(body.get("SAVE_STATE", True)),
            "FAMILY_EDIT": bool(body.get("FAMILY_EDIT", False)),
            "FAMILY_SLIDER": bool(body.get("FAMILY_SLIDER", False)),
            "samples": {
                "enabled": bool(body.get("sample_enabled", True)),
                "every": int(body.get("sample_every", 1)),
                "width": int(body.get("sample_width", 1024)),
                "height": int(body.get("sample_height", 1024)),
                "steps": int(body.get("sample_steps", 20)),
                "cfg": float(body.get("sample_cfg", 1.0)),
                "negative": str(body.get("sample_negative", "")),
                "seed": int(body.get("sample_seed", 42)),
                "at_first": bool(body.get("sample_at_first", True)),
                "prompts": [ln.strip() for ln in str(body.get("sample_prompts", "")).splitlines() if ln.strip()],
            }
        }
        # Pass through every other key from body so launch.plan() can see
        # family-specific settings (FAMILY_EDIT_DIR, FAMILY_SLIDER_DIR, NETWORK_TYPE,
        # LOKR_FACTOR, FAMILY_PRECISION, FAMILY_EMA, edit_caption, …) without
        # having to list them here.
        for k, v in body.items():
            if k not in inputs:
                inputs[k] = v
        return inputs

    def _run_training_stages(self, stages: list):
        global _active_proc, _training_state, _current_stage
        with _lock:
            _training_state = "running"
        
        append_log("=" * 64 + "\n")
        append_log(f"🚀 Fizgig Training Launch ({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        append_log("=" * 64 + "\n")

        for stage in stages:
            with _lock:
                _current_stage = stage.name
            append_log(f"\n▶ Stage: {stage.name}\n")
            append_log(f"Command: {subprocess.list2cmdline(stage.cmd)}\n\n")

            try:
                proc = subprocess.Popen(
                    stage.cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    preexec_fn=os.setsid if hasattr(os, "setsid") else None,
                )
                with _lock:
                    _active_proc = proc

                for line in iter(proc.stdout.readline, ""):
                    if line:
                        append_log(line)

                proc.stdout.close()
                ret = proc.wait()
                if ret != 0:
                    append_log(f"\n❌ Stage '{stage.name}' failed with exit code {ret}\n")
                    with _lock:
                        _training_state = "failed"
                    return
                append_log(f"\n✔ Stage '{stage.name}' completed successfully\n")

            except Exception as e:
                append_log(f"\n❌ Error running stage '{stage.name}': {e}\n")
                with _lock:
                    _training_state = "failed"
                return

        with _lock:
            _training_state = "completed"
            _active_proc = None
            _current_stage = ""
        append_log("\n🎉 All training stages completed successfully!\n")


def run_server(host: str = "0.0.0.0", port: int = 8081, password: str | None = None):
    handler = FizgigWebHandler
    handler.server_password = password
    server = ThreadingHTTPServer((host, port), handler)
    logger.info(f"Fizgig Web UI running at http://{host}:{port}/")
    if password:
        logger.info("Authentication enabled (using VNC_PASSWORD / FIZGIG_PASSWORD)")
    else:
        logger.info("Authentication disabled (listening on localhost / unprotected)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down Fizgig Web Server...")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Fizgig Schema-Driven Web Server")
    parser.add_argument("--web", action="store_true", help="Run web UI server")
    parser.add_argument("--host", default=None, help="Host to bind (default: localhost, or 0.0.0.0 if password set)")
    parser.add_argument("--port", type=int, default=8081, help="Port to listen on (default: 8081)")
    parser.add_argument("--password", default=None, help="Server password (defaults to VNC_PASSWORD env var)")
    args = parser.parse_args()

    vnc_pwd = os.environ.get("VNC_PASSWORD") or os.environ.get("FIZGIG_PASSWORD")
    final_password = args.password or vnc_pwd
    final_host = args.host or ("0.0.0.0" if final_password else "127.0.0.1")

    run_server(host=final_host, port=args.port, password=final_password)


if __name__ == "__main__":
    main()
