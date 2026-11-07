"""Importing fizgig turns off onnxruntime's and Hugging Face's telemetry before either loads; a value the user set wins."""
import os
import subprocess
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _env_after_import(extra):
    env = {k: v for k, v in os.environ.items() if k not in ("ORT_DISABLE_TELEMETRY", "HF_HUB_DISABLE_TELEMETRY")}
    env.update(extra, PYTHONPATH=SRC)
    out = subprocess.run([sys.executable, "-c", "import fizgig, os; print(os.environ['ORT_DISABLE_TELEMETRY'], "
                          "os.environ['HF_HUB_DISABLE_TELEMETRY'])"], env=env, cwd=SRC, capture_output=True, text=True)
    return out.stdout.split()


def test_off_by_default():
    assert _env_after_import({}) == ["1", "1"]


def test_user_value_wins():
    assert _env_after_import({"ORT_DISABLE_TELEMETRY": "0"}) == ["0", "1"]


def test_windows_switch_never_raises():
    from fizgig.utils import ort_telemetry
    ort_telemetry.off()
