"""Set up everything the demos need, once: Fizgig's models, a local language model with its llama.cpp server,
Affinity by Canva with its MCP server, your photos and name. Run by '0 - Set up everything.bat' (Fizgig itself is
installed first). Each step is skipped when it's done already, so it's safe to run again."""
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

import kit

LLM_REPO = "unsloth/Qwen3.8-27B-GGUF"
MMPROJ = "mmproj-F16.gguf"
GH = f"https://github.com/ggml-org/llama.cpp/releases/download/{kit.LLAMA_BUILD}/"


def step(n, text):
    print(f"\n=== {n}. {text}")


def ask(q, default=""):
    a = input(f"{q}{f' [{default}]' if default else ''}: ").strip()
    return a or default


def gpu():
    """(vendor, GB of graphics memory) of the first card."""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20).stdout.split()
            return "nvidia", int(out[0]) / 1024
        except Exception:
            return "nvidia", 0
    return "other", 0


def llm_file(vendor, gb):
    """The language model's size for this card: it has to fit beside nothing else (Fizgig takes it off the GPU to
    render and train), with room for its context."""
    if vendor != "nvidia":
        gb = float(os.environ.get("DEMO_LLM_GB", "0") or 0) or 32      # unified memory: set DEMO_LLM_GB to override
    if gb >= 60:
        return "Qwen3.8-27B-Q8_0.gguf", 131072
    if gb >= 30:
        return "Qwen3.8-27B-UD-Q6_K.gguf", 65536
    if gb >= 22:
        return "Qwen3.8-27B-UD-Q4_K_XL.gguf", 32768
    return "Qwen3.8-27B-UD-IQ3_XXS.gguf", 16384


def download(url, dest):
    if os.path.isfile(dest):
        return dest
    print(f"  {os.path.basename(dest)} ...", flush=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(tmp, dest)
    return dest


def setup_llama(vendor):
    if kit.llama_server():
        print("  llama.cpp is there.")
        return
    flavour = "cuda-12.4-x64" if vendor == "nvidia" else "vulkan-x64"
    zips = [f"llama-{kit.LLAMA_BUILD}-bin-win-{flavour}.zip"] + (
        ["cudart-llama-bin-win-cuda-12.4-x64.zip"] if vendor == "nvidia" else [])
    dest = os.path.join(kit.LLM_DIR, "llama.cpp")
    os.makedirs(dest, exist_ok=True)
    for z in zips:
        p = download(GH + z, os.path.join(kit.LLM_DIR, z))
        with zipfile.ZipFile(p) as zf:
            zf.extractall(dest)
        os.remove(p)
    if not kit.llama_server():
        raise SystemExit("llama.cpp didn't unpack a llama-server.")


def setup_llm_model(name, ctx):
    from huggingface_hub import hf_hub_download
    models = os.path.join(kit.LLM_DIR, "models")
    paths = []
    for f in (name, MMPROJ):
        print(f"  {f} (from {LLM_REPO}) ...", flush=True)
        paths.append(hf_hub_download(LLM_REPO, f, local_dir=models))
    preset = f"""; Written by the demo setup. The language model the Assistant tab talks to.
[*]
parallel = 1
temp = 1.0
top-p = 0.95
top-k = 20
min-p = 0.0

[{kit.LLM_NAME}]
model = {paths[0]}
mmproj = {paths[1]}
n-gpu-layers = all
flash-attn = on
ctx-size = {ctx}
cache-type-k = q8_0
cache-type-v = q8_0
"""
    with open(os.path.join(kit.LLM_DIR, "presets.ini"), "w", encoding="utf-8") as f:
        f.write(preset)


def affinity_installed():
    if os.name != "nt":
        return False
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                         "Get-AppxPackage Canva.Affinity | Select-Object -ExpandProperty Version"],
                         capture_output=True, text=True).stdout.strip()
    if out:
        return True
    w = subprocess.run(["winget", "list", "--id", "Canva.Affinity", "-e", "--accept-source-agreements"],
                       capture_output=True, text=True)
    return "Canva.Affinity" in w.stdout


def affinity_mcp():
    from fizgig.bridge import affinity
    try:
        affinity.status()
        return True
    except Exception:
        return False


def main():
    c = kit.config()
    vendor, gb = gpu()

    step(1, "Fizgig's models (Qwen Image 2.1 and the captioner, about 40 GB)")
    r = subprocess.run([kit.PY, "-m", "fizgig.scripts.fetch_models", "--family", "tools", "--family", "qwen_image21"],
                       cwd=kit.REPO, env={**os.environ, "PYTHONPATH": os.path.join(kit.REPO, "src")})
    if r.returncode:
        raise SystemExit("The model download failed; run this again to carry on where it stopped.")

    step(2, "Fizgig's settings for the demos")
    kit.prefs_update(mcp_server="1", assistant_llm_url=kit.LLM + "/v1", assistant_free_gpu="1", assistant_vision="1")
    print("  Fizgig's MCP server on (this computer only); the Assistant tab talks to the local language model.")

    step(3, f"The local language model (llama.cpp {kit.LLAMA_BUILD} + Qwen3.8-27B)")
    name, ctx = llm_file(vendor, gb)
    print(f"  {'NVIDIA, %.0f GB' % gb if vendor == 'nvidia' else 'Not NVIDIA: the Vulkan build'} -> {name}")
    setup_llama(vendor)
    setup_llm_model(name, ctx)

    step(4, "Affinity by Canva (free; version 3.2 or later has the MCP server)")
    if affinity_installed():
        print("  Affinity is installed.")
    elif ask("  Affinity isn't installed. Install it now with winget? (y/n)", "y").lower().startswith("y"):
        subprocess.run(["winget", "install", "-e", "--id", "Canva.Affinity", "--accept-source-agreements",
                        "--accept-package-agreements"])
    else:
        print("  Get it from https://www.affinity.studio and run this again.")
    while not affinity_mcp():
        print("  Open Affinity and sign in. Then Edit > Settings > Model Context Protocol: turn on 'Enable Affinity MCP'\n"
              "  and 'Access files on your Desktop', and quit and reopen Affinity. (Leave it open for the demos.)")
        if ask("  Press Enter once it's on (or type s to skip for now)").lower() == "s":
            break
    else:
        print("  Affinity's MCP server answers.")

    step(5, "You: the photos the demos train on")
    os.makedirs(kit.PHOTOS, exist_ok=True)
    pics = lambda: [f for f in os.listdir(kit.PHOTOS) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    while len(pics()) < 12:
        print(f"  Put 15-25 photos of yourself in:\n    {kit.PHOTOS}\n"
              "  Your face clear, different days and angles, head-and-shoulders to half-length. "
              f"({len(pics())} there now.)")
        os.startfile(kit.PHOTOS) if os.name == "nt" else None
        if ask("  Press Enter when they're in (or s to skip)").lower() == "s":
            break
    c["name"] = ask("  Your name, as it goes on the posters", c.get("name", ""))
    c["role"] = ask("  Your role, under your name", c.get("role", "Creator"))
    c["who"] = ask("  Say who you are in a prompt: 'a man' or 'a woman'", c.get("who", "a man"))
    default_t = c.get("trigger") or ("zq" + "".join(w[0] for w in c["name"].lower().split() if w)[:3] + "x")
    c["trigger"] = ask("  A trigger word for you (a made-up word)", default_t)
    kit.save_config(c)

    step(6, "Demo 5 only: a RunPod API key (training on a rented cloud GPU)")
    from fizgig.web import pod_stop
    from fizgig.user_data import user_file
    try:
        prefs = json.load(open(user_file("prefs.json"), encoding="utf-8"))
    except (OSError, ValueError):
        prefs = {}
    if pod_stop.api_key(prefs):
        print("  A RunPod key is saved in Fizgig's Preferences.")
    else:
        print("  Optional. Paste it into Fizgig's Preferences > Pods (RunPod API key) to run demo 5.")

    print("\nAll set. Run the demos in order: '1 - ...' trains the LoRA the others use.")


if __name__ == "__main__":
    main()
