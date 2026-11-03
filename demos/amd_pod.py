"""The demos on a rented AMD GPU: `up` rents an AMD Instinct MI300X (192 GB) on RunPod with AMD's ROCm image and the
kit's AMD bootstrap, waits until Fizgig and the language model answer there, and points this computer's Assistant at
it - Affinity stays here, everything else runs on the AMD card. `down` brings every LoRA the pod made home, deletes the
pod and points the Assistant back at this computer. Needs a RunPod API key in Fizgig's Preferences (Pods)."""
import json
import os
import secrets
import sys
import time
import urllib.parse
import urllib.request

import kit

GPU = "AMD Instinct MI300X OAM"
IMAGE = "rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.10.0"
BOOT = "https://raw.githubusercontent.com/shootthesound/Fizgig/{branch}/demos/amd/pod_bootstrap.sh"
STATE = os.path.join(kit.HERE, "amd_pod.json")
HOME = os.path.join(kit.HERE, "AMD LoRAs")


def _key():
    from fizgig.user_data import user_file
    from fizgig.web import pod_stop
    try:
        prefs = json.load(open(user_file("prefs.json"), encoding="utf-8"))
    except (OSError, ValueError):
        prefs = {}
    k = pod_stop.api_key(prefs)
    if not k:
        raise SystemExit("Put your RunPod API key in Fizgig's Preferences > Pods first.")
    return k


def _answers(url, headers=None):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Fizgig-kit/1", **(headers or {})})
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def _stock(key):
    from fizgig.web import pods
    g = next((x for x in pods._rest(key, "GET", "/catalog/gpus", query={
        "include": "AVAILABILITY", "product": "POD", "cloud": "SECURE"}).get("gpus") or [] if x.get("id") == GPU), {})
    return g.get("availability") or "NONE", (g.get("price") or {}).get("secure")


def up(hours=6.0):
    from fizgig.web import pods
    key = _key()
    st = {}
    if os.path.isfile(STATE):
        st = json.load(open(STATE))
        print(f"An AMD pod is already set up ({st['pod']}); 'down' first to rent another.")
    else:
        end = time.time() + hours * 3600
        while True:
            avail, price = _stock(key)
            print(time.strftime("%H:%M"), f"MI300X: {avail.lower()} stock, ${price}/hr", flush=True)
            if avail != "NONE":
                st = {"FIZGIG_PASSWORD": secrets.token_urlsafe(14), "LLM_KEY": secrets.token_urlsafe(20)}
                body = {"name": "fizgig-demos-amd", "image": IMAGE, "cloud": "SECURE", "gpu": {"id": GPU, "count": 1},
                        "disk": 400, "ports": ["8081/http", "38100/http"], "env": dict(st),
                        "entrypoint": ["/bin/bash", "-c"],
                        "cmd": [f"curl -fsSL {BOOT.format(branch=kit.config().get('branch', 'wip-tools'))} | bash"]}
                try:
                    st["pod"] = pods._rest(key, "POST", "/pods", body, timeout=60)["id"]
                except Exception as e:                                # noqa: BLE001 - sold out since: wait
                    print("  RunPod refused:", str(e)[:160])
                else:
                    st["price"] = price
                    json.dump(st, open(STATE, "w"))
                    print(f"Rented {st['pod']} at ${price}/hr. It installs Fizgig, its models and the language model "
                          "(about 10 minutes).", flush=True)
                    break
            if time.time() > end:
                raise SystemExit(f"No MI300X came free in {hours} hours.")
            time.sleep(180)
    fiz = f"https://{st['pod']}-8081.proxy.runpod.net"
    llm = f"https://{st['pod']}-38100.proxy.runpod.net"
    t0 = time.time()
    while not (_answers(fiz + "/") and _answers(llm + "/v1/models", {"Authorization": "Bearer " + st["LLM_KEY"]})):
        if time.time() - t0 > 45 * 60:
            raise SystemExit("The AMD pod didn't come up in 45 minutes: run 'down' to delete it.")
        time.sleep(20)
    kit.prefs_update(assistant_fizgig_url=fiz, assistant_fizgig_password=st["FIZGIG_PASSWORD"],
                     assistant_llm_url=llm + "/v1", assistant_llm_key=st["LLM_KEY"])
    print(f"Ready after {(time.time() - t0) / 60:.0f} min. The Assistant now trains, renders and thinks on the AMD GPU; "
          "Affinity stays on this computer. Run the demos as usual, and 'down' when you're done (the pod costs "
          f"${st.get('price')}/hr until then).")


def down():
    from fizgig.web import pods
    from fizgig.web.helpers import Client
    if not os.path.isfile(STATE):
        print("No AMD pod is set up.")
    else:
        st = json.load(open(STATE))
        key = _key()
        try:                                               # every LoRA the pod made, home first
            c = Client(f"https://{st['pod']}-8081.proxy.runpod.net")
            c.login(st["FIZGIG_PASSWORD"])
            runs = c.call("/api/runs")
            for run in (runs if isinstance(runs, list) else runs.get("runs", [])):
                d = run.get("output_dir") or ""
                for it in (c.call("/api/checkpoints?dir=" + urllib.parse.quote(d)).get("items") or []) if d else []:
                    if it.get("name", "").endswith(".safetensors"):
                        dest = os.path.join(HOME, f"{time.strftime('%Y-%m-%d')} pod {st['pod']}",   # never another
                                            run.get("lora") or "run", it["name"])                    # session's files
                        os.makedirs(os.path.dirname(dest), exist_ok=True)
                        if not os.path.isfile(dest):
                            c.download("/api/download?path=" + urllib.parse.quote(it["path"]), dest, lambda *a: None)
                            print("  brought home", os.path.relpath(dest, kit.HERE))
        except Exception as e:                             # noqa: BLE001
            print("Couldn't bring the LoRAs home:", e)
            if "--force" not in sys.argv:
                raise SystemExit("The pod is still there. Run 'down --force' to delete it anyway.")
        pods.terminate(key, st["pod"])
        for _ in range(30):                                # gone for sure: RunPod no longer knows it
            try:
                pods._rest(key, "GET", f"/pods/{st['pod']}")
            except Exception:
                break
            time.sleep(5)
        os.remove(STATE)
        print(f"Deleted the AMD pod {st['pod']}.")
    kit.prefs_update(assistant_fizgig_url="", assistant_fizgig_password="", assistant_llm_url=kit.LLM + "/v1",
                     assistant_llm_key="")
    print("The Assistant is back on this computer.")


def active():
    return os.path.isfile(STATE)


if __name__ == "__main__":
    {"up": up, "down": down}[sys.argv[1]]()
