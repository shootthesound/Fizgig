"""Fizgig: the web UI in its own app window (the default, --app), or served for a web browser (--web; --open
opens the browser on it). Options for --web: python fizgig.py --web --help."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

# Set before anything imports torch, and inherited by every caching and training run started from here:
# - CUDA's allocator grows and shrinks segments instead of fragmenting under the workbench's load / preview / swap
#   churn (not on Windows, whose allocator rejects the option). An existing value wins; FIZGIG_NO_EXPANDABLE=1 opts
#   out.
# - Intel OpenMP's thread pool sleeps instead of spinning 200 ms after every parallel region: training's small
#   per-step CPU work kept every core at 100% doing nothing (#18).
#   ROCm reads its own name, PYTORCH_HIP_ALLOC_CONF, and ignores the CUDA one (#341). On AMD only for datacenter
#   cards: on consumer Radeon cards they corrupt memory silently (src/fizgig/utils/rocm_alloc.py, #363).
if sys.platform != "win32" and os.environ.get("FIZGIG_NO_EXPANDABLE") != "1":
    if os.environ.get("FIZGIG_GPU_BACKEND") != "rocm":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    else:
        from fizgig.utils.rocm_alloc import expandable_ok
        if expandable_ok():
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("KMP_BLOCKTIME", "0")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from fizgig import user_data  # noqa: E402
user_data.migrate()                       # Fizgig 7's loose files into user_data (once; nothing to do after)


def main():
    if "--web" in sys.argv[1:]:
        from fizgig.web.server import main as web
        return web()
    from fizgig.web.app_window import main as app
    return app()


if __name__ == "__main__":
    sys.exit(main())
