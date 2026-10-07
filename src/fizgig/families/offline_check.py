"""Does a family run offline once its files are on disk? Loads the text encoder (and encodes a prompt) and the VAE with
the Hugging Face Hub switched off, in a fresh process, and reports what still needs the network.

    python src/fizgig/families/offline_check.py --family zimage                # model paths from prefs.json
    python src/fizgig/families/offline_check.py --family zimage --text_encoder TE --vae VAE

First it checks every `helper_files` repo has a cached copy (the model downloader fetches them); then the child process
runs with HF_HUB_OFFLINE=1, so a tokenizer, processor or config loaded with plain from_pretrained (which asks the Hub on
every call) fails here instead of on a user's machine. Use FamilyDriver.from_pretrained / helper_dir in the driver.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", ".."))
REPO = os.path.abspath(os.path.join(SRC, ".."))
sys.path.insert(0, SRC)


def _child(family, te_path, vae_path):
    import torch
    from fizgig.families import registry
    d = registry.get(family)
    drv = d.load_driver()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    te = drv.load_text_encoder(te_path, dev)
    cond = drv.encode_text(te, ["a photo of a cat on a windowsill"])
    drv.unload_text_encoder(te)
    print(f"text encoder: loaded and encoded offline ({', '.join(cond[0])})", flush=True)
    vae = drv.load_vae(vae_path, dev)
    print(f"VAE: loaded offline ({type(vae).__name__})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--text_encoder", default="")
    ap.add_argument("--vae", default="")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._child:
        _child(a.family, a.text_encoder, a.vae)
        return
    from fizgig.families import registry
    from fizgig.utils.hf_cache import cached_snapshot_dir
    d = registry.get(a.family)
    if d is None:
        sys.exit(f"no family {a.family!r}")
    prefs = {}
    pp = os.path.join(REPO, "prefs.json")
    if os.path.exists(pp):
        prefs = json.load(open(pp, encoding="utf-8"))
    te = a.text_encoder or prefs.get(d.pref_for("text_encoder") or "", "")
    vae = a.vae or prefs.get(d.pref_for("vae") or "", "")
    ok = True
    for repo, pats in d.helper_files:
        hit = cached_snapshot_dir(repo)
        print(f"helper {repo} ({', '.join(pats)}): {'cached' if hit else 'NOT CACHED - run fetch_models --family tools'}")
        ok &= bool(hit)
    if not (te and vae):
        sys.exit("set the text encoder and VAE paths (--text_encoder / --vae, or Preferences)")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--family", a.family, "--text_encoder", te,
                        "--vae", vae, "--_child"], env=env, capture_output=True, text=True)
    print(r.stdout.strip())
    if r.returncode != 0:
        ok = False
        tail = [ln for ln in r.stderr.strip().splitlines() if ln.strip()][-6:]
        print("FAILED offline:\n  " + "\n  ".join(tail))
    print(f"OFFLINE CHECK {'PASSED' if ok else 'FAILED'}: {d.display_name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
