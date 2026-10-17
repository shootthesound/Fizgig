"""The MOSS-Image text encoder's own small Python environment (`envs/moss_te`).

Qwen3.5 (model_type qwen3_5, Gated DeltaNet layers) needs transformers >= 5.3, and 5.x needs huggingface-hub 1.x;
Fizgig pins transformers 4.57 / hub 0.34 for every other family. So the text encoder runs in a side venv that holds
ONLY the newer transformers / hub / safetensors / tokenizers and borrows everything else - torch with its CUDA
wheels above all - from Fizgig's own venv through a .pth file (appended to sys.path, so the side venv's own packages
win). ~80 MB, no second torch. Fizgig's venv is never modified.

    venv\\Scripts\\python.exe src\\fizgig\\moss_image\\te_env.py            # create / repair, print the status
    venv\\Scripts\\python.exe src\\fizgig\\moss_image\\te_env.py --status   # (python src/... on Linux)

FIZGIG_MOSS_TE_ENV overrides the folder; FIZGIG_MOSS_TE_PYTHON points at any interpreter that already has
transformers >= 5.3 (the fallback if the .pth overlay ever breaks: a full venv with its own torch).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import sysconfig

logger = logging.getLogger(__name__)

REQUIREMENTS = ("transformers==5.12.1", "huggingface-hub>=1.5,<2", "safetensors>=0.6", "tokenizers>=0.22,<0.23")
MIN_TRANSFORMERS = (5, 3)
PTH_NAME = "zz_fizgig_main.pth"


def fizgig_root() -> str:
    """The checkout (src/fizgig/moss_image/ -> three levels up)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def env_dir() -> str:
    return os.environ.get("FIZGIG_MOSS_TE_ENV") or os.path.join(fizgig_root(), "envs", "moss_te")


def python_path() -> str:
    override = os.environ.get("FIZGIG_MOSS_TE_PYTHON")
    if override:
        return override
    d = env_dir()
    return os.path.join(d, "Scripts", "python.exe") if os.name == "nt" else os.path.join(d, "bin", "python")


def _main_site_dirs() -> list:
    paths = sysconfig.get_paths()
    out = []
    for k in ("purelib", "platlib"):
        p = paths.get(k)
        if p and p not in out and os.path.isdir(p):
            out.append(p)
    return out


def _side_site_dir(py: str) -> str:
    out = subprocess.run([py, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def status(py: str = None) -> dict:
    """{"ok", "python", "transformers", "huggingface_hub", "torch", "cuda", "error"} of the side interpreter."""
    py = py or python_path()
    info = {"ok": False, "python": py}
    if not os.path.isfile(py):
        info["error"] = "not created yet"
        return info
    code = ("import json, torch, transformers, huggingface_hub, safetensors\n"
            "print(json.dumps({'transformers': transformers.__version__, 'huggingface_hub': huggingface_hub.__version__,"
            " 'safetensors': safetensors.__version__, 'torch': torch.__version__, 'cuda': torch.cuda.is_available()}))")
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=300)
    except Exception as e:                       # noqa: BLE001
        info["error"] = str(e)
        return info
    if out.returncode:
        info["error"] = (out.stderr.strip().splitlines() or ["failed"])[-1]
        return info
    info.update(json.loads(out.stdout.strip().splitlines()[-1]))
    major, minor = (int(x) for x in info["transformers"].split(".")[:2])
    info["ok"] = (major, minor) >= MIN_TRANSFORMERS
    if not info["ok"]:
        info["error"] = f"transformers {info['transformers']} < 5.3"
    return info


def _uv_cmd():
    """uv as Fizgig's installer leaves it: a module in Fizgig's venv (install_fizgig.py pip-installs it), else on
    PATH, else None."""
    import importlib.util
    import shutil
    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv"]
    uv = shutil.which("uv")
    return [uv] if uv else None


def _install(py: str, site: str) -> None:
    uv = _uv_cmd()
    if uv:
        cmd = [*uv, "pip", "install", "--python", py, *REQUIREMENTS]
    else:
        # the side venv has no pip of its own: Fizgig's pip installs into its site-packages (deps included)
        cmd = [sys.executable, "-m", "pip", "install", "--target", site, "--upgrade", *REQUIREMENTS]
    logger.info(f"[moss te env] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure(force: bool = False) -> str:
    """Create or repair the side venv; returns its python. Idempotent: a healthy env is left as it is."""
    if os.environ.get("FIZGIG_MOSS_TE_PYTHON"):
        st = status()
        if not st["ok"]:
            raise RuntimeError(f"FIZGIG_MOSS_TE_PYTHON={st['python']}: {st.get('error')}")
        return st["python"]
    py = python_path()
    if not force and status(py)["ok"]:
        return py
    d = env_dir()
    logger.info(f"[moss te env] creating the MOSS-Image text-encoder environment at {d} (transformers 5.x on "
                f"Fizgig's own torch; one-off, ~80 MB)")
    if not os.path.isfile(py) or force:
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", "--clear", d], check=True)
    site = _side_site_dir(py)
    os.makedirs(site, exist_ok=True)
    with open(os.path.join(site, PTH_NAME), "w", encoding="utf-8") as f:
        f.write("\n".join(_main_site_dirs()) + "\n")
    _install(py, site)
    st = status(py)
    if not st["ok"]:
        raise RuntimeError(f"the MOSS-Image text-encoder environment at {d} did not come up: {st.get('error')}")
    logger.info(f"[moss te env] ready: transformers {st['transformers']}, hub {st['huggingface_hub']}, "
                f"torch {st['torch']} (Fizgig's), CUDA {st['cuda']}")
    return py


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--status" not in sys.argv:
        ensure(force="--force" in sys.argv)
    print(json.dumps(status(), indent=2))
