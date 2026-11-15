"""AMD diagnostic (#342): follows MOSS-Image's text-encoder caching step by step, in the environment Fizgig gets.

Run it with:  Launch Fizgig AMD.bat check   (from the Fizgig folder)

After each step it runs the same RoPE cos the text encoder runs first (a (1, 50, 64) float32 tensor on "cuda"), says
whether it worked, and lists the HIP runtime libraries loaded so far - so the step where it starts failing, and any
second runtime that came in with it, show. It reads Preferences for the text encoder's path and changes nothing.
Paste everything it prints into the issue."""
import json
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

_seen = set()


def runtimes():
    """HIP / ROCm libraries in this process, new ones marked."""
    try:
        import psutil
        paths = sorted({m.path for m in psutil.Process().memory_maps()
                        if any(k in os.path.basename(m.path).lower() for k in ("amdhip", "hiprtc", "rocblas",
                                                                               "miopen", "bitsandbytes", "rocm"))})
    except Exception as e:
        return [f"(couldn't list: {e})"]
    out = [("NEW  " if p not in _seen else "     ") + p for p in paths]
    _seen.update(paths)
    return out


def step(name, fn=None):
    print(f"\n== {name}", flush=True)
    if fn is not None:
        try:
            r = fn()
            if r:
                print(f"   {r}", flush=True)
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    try:
        import torch
        x = (torch.arange(50, dtype=torch.float32)[:, None] * torch.ones(32)[None])
        emb = torch.cat((x, x), dim=-1)[None].to("cuda")
        emb.cos().to(torch.bfloat16)
        emb.sin().to(torch.bfloat16)
        torch.cuda.synchronize()
        print(f"   cos OK on device {torch.cuda.current_device()}, {torch.cuda.get_device_name()}", flush=True)
    except Exception as e:
        print(f"   cos FAILED: {type(e).__name__}: {e}", flush=True)
    for line in runtimes():
        print(f"   {line}", flush=True)


def main():
    print("Fizgig AMD check (#342)")
    print("  " + " ".join(f"{k}={os.environ.get(k) or '-'}" for k in (
        "FIZGIG_GPU_BACKEND", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "ROCM_PATH")))
    print("  PATH (first 6): " + " ; ".join(os.environ.get("PATH", "").split(os.pathsep)[:6]), flush=True)

    def torch_info():
        import torch
        n = torch.cuda.device_count()
        return (f"torch {torch.__version__}, HIP {torch.version.hip}, {n} devices: "
                + ", ".join(f"{i}: {torch.cuda.get_device_name(i)}" for i in range(n)))
    step("1. torch", torch_info)
    step("2. import fizgig (the device pin, the attention fix)", lambda: __import__("fizgig") and None)
    step("3. the caching code (fizgig.families.cache)", lambda: __import__("fizgig.families.cache") and None)

    holder = {}

    def driver():
        from fizgig.families.registry import get
        holder["desc"] = get("moss_image")
        holder["driver"] = holder["desc"].load_driver()
    step("4. MOSS-Image's driver", driver)
    step("5. transformers", lambda: __import__("transformers") and None)

    def te():
        from fizgig.user_data import user_file
        prefs = json.load(open(user_file("prefs.json"), encoding="utf-8"))

        def look(k):
            v = str(prefs.get(k) or "").strip()
            return v if not v or os.path.isabs(v) else os.path.normpath(os.path.join(ROOT, v))
        path = holder["desc"].model_path("text_encoder", look)
        if not path or not os.path.exists(path):
            return f"no text encoder file set in Preferences ({path or 'empty'}): skipping steps 6-7"
        import torch
        holder["te"] = holder["driver"].load_text_encoder(path, torch.device("cuda"))
        return f"loaded {os.path.basename(path)} (streamed: {getattr(holder['te'], 'stream', '?')})"
    step("6. load the text encoder as the caching stage does", te)

    def encode():
        if "te" not in holder:
            return "skipped (no text encoder)"
        out = holder["driver"].encode_text(holder["te"], ["a photo of a cat sitting on a wooden table"])
        return f"encoded one caption: {tuple(out[0]['hidden_states'].shape) if out else '?'}"
    step("7. encode one caption (where caching fails)", encode)
    print("\nDone. Please paste everything above into the issue.", flush=True)


if __name__ == "__main__":
    main()
