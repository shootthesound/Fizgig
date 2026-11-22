"""FIZGIG_MIOPEN=0 turns MIOpen off for convolutions on AMD launches (#389: ComfyUI's default on RDNA3+); unset, or on
NVIDIA launches, nothing changes."""
import os
import subprocess
import sys

import torch

from fizgig.utils import gpu_backend


def test_switch_turns_miopen_off(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setenv("FIZGIG_MIOPEN", "0")
    gpu_backend.miopen_switch()
    assert torch.backends.cudnn.enabled is False


def test_unset_changes_nothing(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.delenv("FIZGIG_MIOPEN", raising=False)
    gpu_backend.miopen_switch()
    assert torch.backends.cudnn.enabled is True


def _cudnn_after_import(backend, cwd):
    src = os.path.dirname(os.path.dirname(os.path.abspath(gpu_backend.__file__)))
    env = dict(os.environ, FIZGIG_MIOPEN="0", CUDA_VISIBLE_DEVICES="", PYTHONPATH=os.path.dirname(src))
    env.pop("FIZGIG_GPU_BACKEND", None)
    if backend:
        env["FIZGIG_GPU_BACKEND"] = backend
    out = subprocess.run([sys.executable, "-c", "import fizgig, torch; print(torch.backends.cudnn.enabled)"],
                         capture_output=True, text=True, env=env, cwd=cwd, timeout=120)
    assert out.stdout.strip(), out.stderr
    return out.stdout.strip().splitlines()[-1]


def test_every_amd_process_applies_it(tmp_path):
    # importing fizgig is what every stage does first (cache, train, Studio worker)
    assert _cudnn_after_import("rocm", tmp_path) == "False"
    assert _cudnn_after_import("", tmp_path) == "True"
