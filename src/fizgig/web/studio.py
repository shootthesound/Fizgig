"""The web UI's Repair Studio: the desktop tab's engine (fizgig/families/workbench.py), kept in a worker process
(scripts/studio_worker.py) so the model stays loaded between renders, driven from here.

One render runs at a time. A slider moved while it renders cancels it, and only the newest state renders next - the
desktop's cancel-and-refire. The page polls status() for progress and the newest pictures. The load plan (which model
files, preview checkpoint or speed LoRA, the preview settings) follows the desktop's (_repair_engine_plan_family).
"""
import json
import os
import subprocess
import sys
import threading
import time

from fizgig.user_data import user_file  # noqa: E402  (Fizgig's own files: user_data/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT_DIR = user_file("studio")
RM_DIR = os.path.join(OUT_DIR, "refmod")      # RefMod Toolkit's renders: its history outlives a worker restart
EV = "@@EV "

_lock = threading.Lock()
_S = {"proc": None, "sock": None, "port": 0, "state": "idle", "family": "", "primary": "", "donor": "", "primary_blocks": [],
      "donor_blocks": [], "hash": "", "status": "", "progress": None, "gen": 0, "shown": 0, "baseline": "",
      "tweaked": "", "pending": None, "rendering": False, "error": "", "saved": None, "profile": None,
      "used": 0.0, "log": []}


_send_lock = threading.Lock()


def _send(cmd):
    """A command to the worker, over its local socket (not stdin: see the worker's note on the C runtime's lock)."""
    p, sock = _S["proc"], _S["sock"]
    if p is None or p.poll() is not None or sock is None:
        raise RuntimeError("The Studio worker isn't running.")
    with _send_lock:
        sock.sendall((json.dumps(cmd) + "\n").encode("utf-8"))


# LoRA Royale's travel modes and comparison sheet: each mode's last render (frames, or the sheet)
_T = {m: {"state": "idle", "gen": 0, "frames": [], "status": "", "error": "", "sheet": "", "size": None}
      for m in ("seed", "strength", "prompt", "sheet")}
_TG = [0]                                    # one generation counter across the modes

# LoRA Royale's run on the same worker (one model on the card, as the desktop): its renders, by generation
_R = {"state": "idle", "gen": 0, "items": [], "total": 0, "status": "", "error": "", "progress": None, "meta": {}}


# RefMod Toolkit on the same worker: the With-mods render, the comparison (kept while its settings hold), the sweep's
# chips and the last renders (history)
_M = {"state": "idle", "gen": 0, "kind": "", "status": "", "error": "", "progress": None, "early": "", "tweaked": None,
      "baseline": None, "baseline_key": None, "pending_key": None, "chips": [], "history": [], "label": "", "sweep": ""}
RM_HISTORY = 12


def _reader(proc):
    for line in proc.stdout:
        if not line.startswith(EV):
            with _lock:
                _S["log"] = (_S["log"] + [line.rstrip()])[-200:]
            continue
        try:
            ev = json.loads(line[len(EV):])
        except ValueError:
            continue
        with _lock:
            current = _S["proc"] is proc
        if current:                              # a worker being stopped (a model switch) has its say no more
            _event(ev)
    with _lock:
        if _S["proc"] is proc:
            _S.update(proc=None, state="idle", rendering=False, status="The Studio worker stopped.", progress=None)
            if _R["state"] == "rendering":
                _R.update(state="failed", error="The GPU worker stopped.", status="The GPU worker stopped.")
            for t in _T.values():
                if t["state"] == "rendering":
                    t.update(state="failed", error="The GPU worker stopped.", status="The GPU worker stopped.")
            if _M["state"] == "rendering":
                _M.update(state="failed", error="The GPU worker stopped.", status="The GPU worker stopped.",
                          progress=None)


def _event(ev):
    kind = ev.get("ev")
    nxt = None
    with _lock:
        if kind == "ready":
            _S["port"] = int(ev.get("port") or 0)
        elif kind == "status":
            _S["status"] = ev.get("text", "")
        elif kind == "step":
            _S["progress"] = [ev.get("done", 0), ev.get("total", 0)]
            _R["progress"] = _M["progress"] = _S["progress"]
        elif kind == "rm_status":
            _M["status"] = ev.get("text", "")
        elif kind == "rm_early" and ev.get("gen") == _M["gen"]:
            _M.update(early=ev.get("png", ""), status=f"Early look after pass {ev.get('step')} of {ev.get('n')} - finishing…")
        elif kind == "rm_item" and ev.get("gen") == _M["gen"]:
            item = {k: ev.get(k) for k in ("gen", "side", "i", "png", "mp4", "label", "describe", "steps", "turbo",
                                           "frames")}
            item["label"] = item["label"] or _M["label"]
            if ev.get("side") == "chip":
                _M["chips"].append(item)
                _M["status"] = f"Sweep: {len(_M['chips'])} done…"
            elif ev.get("side") == "baseline":
                _M.update(baseline=item, baseline_key=_M["pending_key"])
            else:
                _M.update(tweaked=item, early="")
                _M["history"] = ([item] + _M["history"])[:RM_HISTORY]
        elif kind == "rm_done" and ev.get("gen") == _M["gen"]:
            _M.update(progress=None, early="",
                      state="failed" if ev.get("error") else "done",
                      error=ev.get("error") or "",
                      status=("Cancelled." if ev.get("cancelled") else
                              f"Render failed: {ev['error']}" if ev.get("error") else
                              f"Sweep done: {_M['label']}. Click a chip to put it on the left." if _M["kind"] == "sweep"
                              else _rm_done_line()))
        elif kind in ("travel_status", "travel_frame", "travel_done") and ev.get("mode") in _T \
                and ev.get("gen") == _T[ev["mode"]]["gen"]:
            t = _T[ev["mode"]]
            if kind == "travel_status":
                t["status"] = ev.get("text", "")
            elif kind == "travel_frame":
                t["frames"].append({"i": ev.get("i"), "png": ev.get("png"), "label": ev.get("label", "")})
            else:
                n = len(t["frames"]) or (1 if ev.get("sheet") else 0)
                t.update(state="done" if n and not ev.get("error") else "failed", sheet=ev.get("sheet", ""),
                         size=ev.get("size"), error=ev.get("error") or ("" if n else "Nothing rendered."),
                         status=("Stopped." if ev.get("cancelled") else
                                 f"Sheet ready ({ev['size'][0]}x{ev['size'][1]}) - review, then save." if ev.get("sheet")
                                 else f"Rendered {len(t['frames'])} frames - scrub to review, then save." if n
                                 else f"Failed: {ev.get('error') or 'nothing rendered'}"))
        elif kind == "royale_status":
            _R["status"] = ev.get("text", "")
        elif kind == "royale_item" and ev.get("gen") == _R["gen"]:
            _R["items"].append({k: ev.get(k) for k in ("i", "label", "path", "png", "error")})
        elif kind == "royale_done" and ev.get("gen") == _R["gen"]:
            ok = [x for x in _R["items"] if x.get("png")]
            _R.update(state="done" if ok else "failed", progress=None,
                      error=ev.get("error") or ("" if ok else "No renders produced."),
                      status=("Stopped." if ev.get("cancelled") else
                              f"Rendered {len(ok)} of {_R['total']}." if ok else "No renders produced."))
        elif kind in ("loaded", "donor"):
            _S.update({k: ev.get(k) for k in ("primary_blocks", "donor_blocks", "hash", "eq")})
            if kind == "loaded":
                _S.update(nolora="", nolora_media="", nolora_gen=0, nolora_busy=False)
                _S.update(state="ready", family=ev.get("family", ""), primary=ev.get("primary", ""),
                          donor=ev.get("donor", ""), status="Loaded.", error="")
                _S["profile"] = _profile(ev.get("hash", ""))
            else:
                _S.update(donor=ev.get("donor", ""), status="Donor loaded, fully on in its blocks: turn a block down or off to use less of it."
                          if ev.get("donor") else "Donor unloaded.")
        elif kind == "early" and _S.get("rendering"):     # H3's Show early: the tweaked pane's unfinished look
            _S.update(early=ev.get("png", ""), early_gen=ev.get("gen", 0),
                      status=f"Tweaked: pass {ev.get('step')} of {ev.get('n')}, unfinished - rendering the rest…")
        elif kind == "rendered":
            _S.update(rendering=False, progress=None, baseline=ev.get("baseline", ""), tweaked=ev.get("tweaked", ""),
                      early="",
                      baseline_media=ev.get("baseline_media", ""), tweaked_media=ev.get("tweaked_media", ""),
                      baseline_act=ev.get("baseline_act", ""), tweaked_act=ev.get("tweaked_act", ""),
                      shown=ev.get("gen", 0), status="Ready.", error="", nolora_busy=bool(ev.get("nolora_pending")),
                      notes=ev.get("notes"))
            nxt, _S["pending"] = _S["pending"], None
        elif kind == "nolora":                   # the No-LoRA pane's render landed (or a newer edit cut it short)
            _S["nolora_busy"] = False
            if ev.get("nolora") and ev["nolora"] != _S.get("nolora"):   # the same file again: the pane plays on
                _S.update(nolora=ev["nolora"], nolora_media=ev.get("nolora_media", ""), nolora_gen=ev.get("gen", 0))
            elif ev.get("msg"):
                _S["status"] = "The No-LoRA render failed: " + ev["msg"]
        elif kind == "aborted":
            _S.update(rendering=False, progress=None)
            nxt, _S["pending"] = _S["pending"], None
        elif kind == "error":
            where = ev.get("where")
            _S.update(error=ev.get("msg", ""), progress=None,
                      status=("Preview error: " if where == "render" else "Couldn't load: " if where == "load" else "")
                      + ev.get("msg", ""))
            if where == "render":
                _S["rendering"] = False
                nxt, _S["pending"] = _S["pending"], None
            elif where == "load":
                _S["state"] = "idle"
        elif kind == "saved":
            _S["saved"] = {"out": ev.get("out"), "summary": ev.get("summary")}
            _S["status"] = "Saved."
        elif kind == "reset":
            _S.update(state="idle", primary="", donor="", primary_blocks=[], donor_blocks=[], hash="", profile=None,
                      rendering=False, pending=None, progress=None, status="Models unloaded.")
    if nxt is not None:
        render(*nxt)


def _profile(h):
    """The Profiler's sidecar for this LoRA (same file hash), for the pills panel; None without one."""
    try:
        from fizgig.utils.lora_files import find_profile_for_hash
        from fizgig.web.server import load_prefs, pref_path
        d = pref_path(load_prefs(), "profiles_dir") or user_file("profiles")
        prof = find_profile_for_hash(d, h)
        if not prof:
            return None
        rep = os.path.join(os.path.dirname(prof.get("_sidecar_path", "")), prof.get("html_report", "") or "")
        return {"lora_name": prof.get("lora_name", ""), "created": str(prof.get("created", ""))[:10],
                "blocks": [{"name": b.get("name"), "category": b.get("category"), "pct": b.get("pct")}
                           for b in (prof.get("top_active_blocks") or [])[:8]],
                "report": rep if os.path.isfile(rep) else ""}
    except Exception:
        return None


def _ensure_worker():
    with _lock:
        if _S["proc"] is not None and _S["proc"].poll() is None:
            return
    try:                                         # a fresh worker: an earlier one's renders are nobody's now
        for f in os.listdir(OUT_DIR):
            if os.path.splitext(f)[1].lower() in (".png", ".mp4", ".wav", ".json"):
                os.remove(os.path.join(OUT_DIR, f))
    except OSError:
        pass
    from fizgig.web.server import gpu_env
    env = gpu_env(dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))     # the Preferences GPU
    kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    proc = subprocess.Popen([sys.executable, "-m", "fizgig.scripts.studio_worker"], cwd=BASE_DIR,
                            env={**env, "PYTHONPATH": os.path.join(BASE_DIR, "src") + os.pathsep + BASE_DIR},
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, **kw)
    with _lock:
        _S.update(proc=proc, sock=None, port=0, status="Starting the Studio worker…")
    threading.Thread(target=_reader, args=(proc,), daemon=True).start()
    import socket
    for _ in range(600):                         # its first start imports torch: up to a couple of minutes
        if _S["port"] or proc.poll() is not None:
            break
        time.sleep(0.2)
    if not _S["port"]:
        raise RuntimeError("The Studio worker didn't start.")
    sock = socket.create_connection(("127.0.0.1", _S["port"]), timeout=10)
    sock.settimeout(None)
    with _lock:
        _S["sock"] = sock


# ---- the load plan, as the desktop builds it ---------------------------------------------------------------------
def plan(desc, prefs, pref_path, fast=True, samples=None, negative=None):
    """(load kwargs for the worker, preview settings or None, status line, problems)."""
    look = lambda k: pref_path(prefs, k)
    paths = {r: desc.model_path(r, look) for r in ("dit", "vae", "text_encoder")}
    labels = {f.role: f.label for f in desc.model_files}
    problems = [f"{labels.get(r, r)} isn't set (or isn't where Preferences says). Set it on the Preferences tab."
                for r, p in paths.items() if not p or not os.path.exists(p)]
    follow = desc.workbench_follows_samples
    fast = fast or follow
    dit, speed, use_ck, ck = paths["dit"], "", False, None
    have = lambda k: bool(k and look(k) and os.path.exists(look(k)))
    if fast:
        c = desc.preview_checkpoint()
        if c is not None and have(c[0].pref_key):
            dit, use_ck, ck = look(c[0].pref_key), True, c[1]
        else:
            sp = desc.preview_speed()
            if sp is not None and have(sp.pref_key):
                speed = look(sp.pref_key)
    sp = desc.preview_speed()
    steps = ck.steps if ck else (sp.settings.steps if speed else desc.default_sampling().steps)
    settings = None
    if follow:
        smp = samples or {}
        settings = {"steps": str(smp.get("steps") or desc.preview_steps),
                    "cfg": str(smp.get("cfg") or f"{desc.preview_cfg:g}"),
                    "turbo": str(smp.get("turbo") if smp.get("turbo") not in (None, "") else
                                 (desc.preview_speed_defaults() or (0, 0))[1]),
                    "negative": negative if negative is not None else str(smp.get("negative") or "")}
        steps = settings["steps"]
    swap = str(prefs.get("inference_blocks_to_swap") or "Auto").strip()
    kw = {"dit_path": dit, "vae_path": paths["vae"], "text_encoder_path": paths["text_encoder"],
          "speed_lora_path": speed, "use_checkpoint_sampling": use_ck, "device": "cuda",
          "precision": "int8" if str(prefs.get("inference_int8", "")).strip() in ("1", "True", "true") else "auto",
          "blocks_to_swap": "auto" if swap.lower().startswith("auto") else int(
              "".join(ch for ch in swap.split(" ")[0] if ch.isdigit()) or 0)}
    status = f"Loading {desc.display_name} ({ck.name + ', ' if ck else ''}{steps}-step previews)…"
    return kw, settings, status, problems


def load(desc, primary, donor, kw, settings, status, turbo, canvas=None):
    if not primary or not os.path.isfile(primary):
        raise ValueError("Choose a primary LoRA first (that file isn't there).")
    if donor and not os.path.isfile(donor):
        raise ValueError("That donor LoRA isn't there.")
    with _lock:
        other = _S["family"] and _S["family"] != desc.key
        _S.update(state="loading", status="Starting the Studio worker…", error="", progress=None, pending=None,
                  family=desc.key, used=time.time())

    def go():                                    # the worker's first start imports torch: not in the request
        try:
            if other:                            # another model's worker: a fresh process (an engine's reset can
                stop_worker()                    # leave enough on the card to crawl the next model - ACE then H3)
            _ensure_worker()
            with _lock:
                _S["status"] = status
            _send({"cmd": "load", "family": desc.key, "plan": kw, "primary": primary, "donor": donor or "",
                   "turbo": bool(turbo), "preview_settings": settings, "status": status, "canvas": canvas})
        except Exception as e:
            with _lock:
                _S.update(state="idle", error=str(e), status=f"Couldn't start: {e}")
    threading.Thread(target=go, daemon=True).start()


def donor(path):
    if path and not os.path.isfile(path):
        raise ValueError("That donor LoRA isn't there.")
    _send({"cmd": "donor", "path": path or ""})


def settings(turbo=None, preview_settings=None):
    c = {"cmd": "settings"}
    if turbo is not None:
        c["turbo"] = bool(turbo)
    if preview_settings is not None:
        c["preview_settings"] = preview_settings
    _send(c)


def render(state, extra=None):
    """Render this slider state: now if nothing renders, else it waits (replacing any older wait) and the render
    going is cancelled - the newest state is what renders next."""
    with _lock:
        if _S["state"] != "ready":
            return {"error": "Load a LoRA first."}
        _S["used"] = time.time()
        if _S["rendering"]:
            _S["pending"] = (state, extra)
            busy, cut = True, False
        else:
            _S["gen"] += 1
            gen = _S["gen"]
            _S.update(rendering=True, progress=None, status="Generating preview…")
            busy = False
            cut = _S.get("nolora_busy")
    if busy or cut:                              # the newest edit takes the engine (from a No-LoRA render too)
        _send({"cmd": "cancel"})
    if busy:
        return {"ok": True, "queued": True}
    _send({"cmd": "render", "gen": gen, "state": state, "out_dir": OUT_DIR, **(extra or {})})
    return {"ok": True, "gen": gen}


def metrics(ref=""):
    """The desktop's metrics strip for the pair on screen: likeness to a reference photo (the Look Filter's face
    embedder; with Auto look match decided from the reference - ArcFace for a person, DINOv2 for a furry / anime
    face) and repair_studio.metrics' paired numbers - patch grid, face detail, clipping, saturation. CPU only."""
    import numpy as np
    from PIL import Image
    from fizgig.repair_studio.metrics import PATCH_PITCH, compare
    from fizgig.web.prep_jobs import _embed, embed_face_box, session_embedder
    with _lock:
        base_p, tweak_p, fam = _S.get("baseline") or "", _S.get("tweaked") or "", _S.get("family") or ""
    if not (base_p and tweak_p and os.path.isfile(base_p) and os.path.isfile(tweak_p)):
        return {"error": "Nothing rendered yet."}
    from fizgig.families.registry import get as get_family
    d = get_family(fam)
    pitch = PATCH_PITCH.get((d.shares_prefs_with or d.key) if d is not None else fam, 16)
    try:
        has_ref = bool(ref and os.path.isfile(ref))
        emb, _ = session_embedder([ref] if has_ref else [base_p], log=None)    # one embedder for all three
        b_emb, b_box = embed_face_box(base_p, emb)
        t_emb, t_box = embed_face_box(tweak_p, emb)
        ref_emb = _embed(ref, emb) if has_ref else None
        m = compare(np.array(Image.open(base_p).convert("RGB")), np.array(Image.open(tweak_p).convert("RGB")),
                    pitch, b_box, t_box)
        m["ref_set"], m["ref_face"] = bool(ref), ref_emb is not None
        m["embedder"], m["like_match"] = emb.label, emb.thresholds["match"]
        m["like_base"] = float(np.dot(ref_emb, b_emb)) if ref_emb is not None and b_emb is not None else None
        m["like_tweak"] = float(np.dot(ref_emb, t_emb)) if ref_emb is not None and t_emb is not None else None
        return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in m.items()}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def keep_tweaked():
    """ACE-Step's "Use as source": the Tweaked song copied where it stays (renders are tidied away), its path back."""
    import shutil
    with _lock:
        src = _S.get("tweaked_media") or ""
    if not src or not os.path.isfile(src) or not src.lower().endswith(".wav"):
        raise ValueError("Nothing to use yet: the Tweaked song renders after the first preview.")
    folder = os.path.join(OUT_DIR, "sources")
    os.makedirs(folder, exist_ok=True)
    out = os.path.join(folder, f"take_{int(time.time() * 1000)}.wav")
    shutil.copyfile(src, out)
    return out


def save(out, state, include_donor):
    with _lock:
        _S["saved"] = None
    _send({"cmd": "save", "out": out, "state": state, "include_donor": bool(include_donor)})


def reset():
    """Unload: the worker quits, so the card and the RAM it held come back whole (the request waits for it - a
    Start pressed straight after finds no half-stopped worker)."""
    stop_worker()
    with _lock:
        _S.update(state="idle", primary="", donor="", primary_blocks=[], donor_blocks=[], hash="", profile=None,
                  rendering=False, pending=None, progress=None, family="", status="Models unloaded.")


def stop_worker():
    """Quit the worker (the GPU is wanted for a training run, or the server is stopping)."""
    with _lock:
        p = _S["proc"]
    if p is None or p.poll() is not None:
        return
    try:
        _send({"cmd": "cancel"})
        _send({"cmd": "quit"})
    except Exception:
        pass
    with _lock:                                  # ours to stop: its reader says nothing when it ends
        if _S["proc"] is p:
            _S.update(proc=None, sock=None, port=0)
            if _R["state"] == "rendering":
                _R.update(state="failed", error="Stopped (the GPU was wanted elsewhere).", status="Stopped.")
            for t in _T.values():
                if t["state"] == "rendering":
                    t.update(state="failed", error="Stopped (the GPU was wanted elsewhere).", status="Stopped.")
            if _M["state"] == "rendering":
                _M.update(state="failed", error="Stopped (the GPU was wanted elsewhere).", status="Stopped.",
                          progress=None)
    try:
        p.wait(timeout=20)
    except Exception:
        p.kill()


def holding_gpu() -> bool:
    with _lock:
        return _S["proc"] is not None and _S["proc"].poll() is None and (
            _S["state"] in ("loading", "ready") or _R["state"] == "rendering" or _M["state"] == "rendering"
            or any(t["state"] == "rendering" for t in _T.values()))


def royale_render(desc, plan, settings, items, prompt, seed, width, height, ref="", ref_strength=1.0, meta=None):
    """Render every chosen epoch on the worker (LoRA Royale). The worker's model is Royale's from here: a Studio
    session there ends (Start loads it again)."""
    with _lock:
        if _R["state"] == "rendering":
            raise ValueError("Royale is rendering - wait, or Stop it.")
        if _S["state"] == "loading" or _S["rendering"]:
            raise ValueError("Studio is using the GPU - wait for it, or Unload it.")
        _R["gen"] += 1
        gen = _R["gen"]
        _R.update(state="rendering", items=[], total=len(items), status="Starting the GPU worker…", error="",
                  progress=None, meta=dict(meta or {}, prompt=prompt, seed=seed, width=width, height=height))
        if _S["state"] == "ready":
            _S.update(state="idle", primary="", donor="", rendering=False, pending=None, progress=None,
                      status="LoRA Royale took the GPU worker - press Start to load Studio again.")
        family = _S.get("family")

    def go():
        try:
            if family and family != desc.key:
                stop_worker()                     # another model's worker: a fresh one (Studio's rule)
                with _lock:
                    _R.update(state="rendering", error="", status="Starting the GPU worker…")   # (that stop was ours)
            _ensure_worker()
            with _lock:
                _S["family"] = desc.key
            _send({"cmd": "royale", "gen": gen, "family": desc.key, "plan": plan, "settings": settings,
                   "items": items, "prompt": prompt, "seed": int(seed), "width": int(width), "height": int(height),
                   "ref": ref or "", "ref_strength": float(ref_strength), "out_dir": OUT_DIR})
        except Exception as e:
            with _lock:
                _R.update(state="failed", error=str(e), status=f"Couldn't start: {e}")
    threading.Thread(target=go, daemon=True).start()
    return gen


def travel_render(mode, desc, plan, settings, params):
    """A Royale travel mode (or the comparison sheet) on the worker; Studio's session there ends, as for a render."""
    with _lock:
        if any(t["state"] == "rendering" for t in _T.values()) or _R["state"] == "rendering":
            raise ValueError("Royale is rendering - wait, or Stop it.")
        if _S["state"] == "loading" or _S["rendering"]:
            raise ValueError("Studio is using the GPU - wait for it, or Unload it.")
        _TG[0] += 1
        gen = _TG[0]
        _T[mode].update(state="rendering", gen=gen, frames=[], status="Starting the GPU worker…", error="", sheet="",
                        size=None)
        if _S["state"] == "ready":
            _S.update(state="idle", primary="", donor="", rendering=False, pending=None, progress=None,
                      status="LoRA Royale took the GPU worker - press Start to load Studio again.")
        family = _S.get("family")

    def go():
        try:
            if family and family != desc.key:
                stop_worker()
                with _lock:
                    _T[mode].update(state="rendering", error="", status="Starting the GPU worker…")
            _ensure_worker()
            with _lock:
                _S["family"] = desc.key
            _send({"cmd": "royale_travel", "gen": gen, "mode": mode, "family": desc.key, "plan": plan,
                   "settings": settings, "out_dir": OUT_DIR, **params})
        except Exception as e:
            with _lock:
                _T[mode].update(state="failed", error=str(e), status=f"Couldn't start: {e}")
    threading.Thread(target=go, daemon=True).start()
    return gen


def refmod_render(desc, plan, canvas, kind, jobs, lora, compare, base_key, label=""):
    """RefMod Toolkit's render (kind "render": With mods, then the comparison unless the kept one still matches
    base_key) or sweep, on the worker; a Studio session there ends, as for Royale."""
    with _lock:
        if _M["state"] == "rendering" or _R["state"] == "rendering" or any(t["state"] == "rendering" for t in _T.values()):
            raise ValueError("The GPU worker is rendering - wait, or Cancel it.")
        if _S["state"] == "loading" or _S["rendering"]:
            raise ValueError("Studio is using the GPU - wait for it, or Unload it.")
        b = _M["baseline"]
        need_base = kind == "render" and not (b and _M["baseline_key"] == base_key and os.path.isfile(b["png"]))
        _M["gen"] += 1
        gen = _M["gen"]
        _M.update(state="rendering", kind=kind, status="Starting the GPU worker…", error="", progress=None, early="",
                  pending_key=base_key, label=label)
        if kind == "sweep":
            _M.update(chips=[], sweep=label)
        if _S["state"] == "ready":
            _S.update(state="idle", primary="", donor="", rendering=False, pending=None, progress=None,
                      status="RefMod Toolkit took the GPU worker - press Start to load Studio again.")
        family = _S.get("family")
        keep = {_M["baseline"]["gen"]} if _M["baseline"] and not need_base else set()
        keep |= {h["gen"] for h in _M["history"]} | ({_M["chips"][0]["gen"]} if _M["chips"] else set())
        if _M["tweaked"]:
            keep.add(_M["tweaked"]["gen"])
    try:                                         # renders nothing shows any more (a new server: all of them)
        import re
        for f in os.listdir(RM_DIR):
            m = re.match(r"rm_(\d+)_", f)
            if m and (gen == 1 or (int(m.group(1)) not in keep and int(m.group(1)) != gen)):
                os.remove(os.path.join(RM_DIR, f))
    except OSError:
        pass

    def go():
        try:
            if family and family != desc.key:
                stop_worker()
                with _lock:
                    _M.update(state="rendering", error="", status="Starting the GPU worker…")
            _ensure_worker()
            with _lock:
                _S["family"] = desc.key
            _send({"cmd": "refmod", "gen": gen, "kind": kind, "family": desc.key, "plan": plan, "canvas": canvas,
                   "jobs": jobs, "lora": lora or "", "compare": compare, "need_base": need_base, "out_dir": RM_DIR})
        except Exception as e:
            with _lock:
                _M.update(state="failed", error=str(e), status=f"Couldn't start: {e}")
    threading.Thread(target=go, daemon=True).start()
    return {"gen": gen, "need_base": need_base}


def _rm_done_line():
    t = _M["tweaked"] or {}
    tu = t.get("turbo")
    tut = "no Turbo LoRA" if tu is None else "Turbo off" if abs(float(tu)) < 1e-9 else f"Turbo {float(tu):g}"
    return f"Done: {t.get('steps')} steps · {tut}." if t.get("steps") else "Done."


def refmod_cancel():
    with _lock:
        busy = _M["state"] == "rendering"
    if busy:
        _send({"cmd": "cancel"})
        with _lock:
            _M["status"] = "Cancelling…"
    return busy


def refmod_status():
    with _lock:
        d = {k: _M[k] for k in ("state", "gen", "kind", "status", "error", "progress", "label", "sweep")}
        d["early"] = bool(_M["early"])
        pub = lambda it: None if not it else {k: it.get(k) for k in ("gen", "side", "i", "label", "describe", "steps",
                                                                     "turbo", "frames")} | {"media": bool(it.get("mp4"))}
        d["tweaked"], d["baseline"] = pub(_M["tweaked"]), pub(_M["baseline"])
        d["chips"] = [pub(c) for c in _M["chips"]]
        d["history"] = [pub(h) for h in _M["history"]]
        d["baseline_key"] = _M["baseline_key"]
    return d


def refmod_item(gen, side, i):
    """The item a page asks for by (gen, side, i): its record, or None."""
    with _lock:
        if side == "early":
            return {"png": _M["early"]} if _M["early"] and os.path.isfile(_M["early"]) else None
        pool = ([_M["tweaked"], _M["baseline"]] + _M["history"] + _M["chips"])
        return next((dict(x) for x in pool if x and x["gen"] == int(gen) and x["side"] == side and x["i"] == int(i)),
                    None)


def refmod_chips():
    with _lock:
        return [dict(c) for c in _M["chips"]]


def travel_status():
    with _lock:
        return {m: {k: (list(t[k]) if k == "frames" else t[k]) for k in t} for m, t in _T.items()}


def travel_frames(mode):
    with _lock:
        t = _T.get(mode) or {}
        return [(f["png"], f["label"]) for f in sorted(t.get("frames", []), key=lambda f: f["i"])
                if f.get("png") and os.path.isfile(f["png"])]


def travel_file(mode, i=None):
    """A travel frame's PNG (i), or the sheet."""
    with _lock:
        t = _T.get(mode) or {}
        if i is None:
            return t.get("sheet") if t.get("sheet") and os.path.isfile(t["sheet"]) else ""
        f = next((x for x in t.get("frames", []) if x.get("i") == int(i)), None)
    return f["png"] if f and os.path.isfile(f["png"]) else ""


def royale_cancel():
    with _lock:
        busy = _R["state"] == "rendering" or any(t["state"] == "rendering" for t in _T.values())
    if busy:
        _send({"cmd": "cancel"})
    return busy


def royale_status():
    with _lock:
        return {k: (list(_R[k]) if k == "items" else _R[k]) for k in ("state", "gen", "items", "total", "status",
                                                                     "error", "progress", "meta")}


def royale_png(i):
    with _lock:
        it = next((x for x in _R["items"] if x.get("i") == int(i) and x.get("png")), None)
    return it["png"] if it and os.path.isfile(it["png"]) else ""


def royale_rendered():
    """[(label, path, png)] of the last run's renders, in order."""
    with _lock:
        return [(x["label"], x["path"], x["png"]) for x in sorted(_R["items"], key=lambda x: x["i"]) if x.get("png")]


def status():
    with _lock:
        d = {k: _S[k] for k in ("state", "family", "primary", "donor", "primary_blocks", "donor_blocks", "status",
                                "progress", "shown", "rendering", "error", "saved", "profile")}
        d["notes"] = _S.get("notes")
        d["eq"] = _S.get("eq") or {}
        d["media"] = {k: bool(_S.get(k + "_media")) for k in ("baseline", "tweaked", "nolora")}
        d["act"] = {k: bool(_S.get(k + "_act")) for k in ("baseline", "tweaked")}
        d["nolora_gen"], d["nolora_busy"] = _S.get("nolora_gen", 0), bool(_S.get("nolora_busy"))
        d["pending"] = _S["pending"] is not None
        d["early"] = _S.get("early_gen", 0) if _S.get("early") and _S.get("rendering") else 0
        d["log"] = _S["log"][-12:]                 # the worker's own lines (the engine's log), for when it goes wrong
    return d


def image(which, gen):
    with _lock:
        p = _S.get(which) if which in ("baseline", "tweaked", "nolora", "baseline_media", "tweaked_media",
                                       "nolora_media", "baseline_act", "tweaked_act", "early") else ""
    return p if p and os.path.isfile(p) else ""


_peaks = {}


def song_peaks(path, bins=2000):
    """{dur, pk, rms}: a 16-bit WAV's channels averaged, each bin's peak and loudness against the song's loudest
    (the desktop strip's measure), for the song player to draw at any width."""
    import wave
    import numpy as np
    key = (path, os.path.getmtime(path))
    if key in _peaks:
        return _peaks[key]
    with wave.open(path, "rb") as f:
        ch, rate, n = f.getnchannels(), f.getframerate(), f.getnframes()
        a = np.frombuffer(f.readframes(n), dtype=np.int16).astype(np.float32)
    a = np.abs(a.reshape(-1, ch).mean(1)) if ch > 1 else np.abs(a)
    k = max(1, len(a) // bins)
    cols = a[:k * bins].reshape(-1, k) if len(a) >= bins else a.reshape(-1, 1)
    peak = max(1e-6, float(cols.max()))
    out = {"dur": n / float(rate), "pk": [round(float(v), 3) for v in cols.max(1) / peak],
           "rms": [round(float(v), 3) for v in np.sqrt((cols ** 2).mean(1)) / peak]}
    if len(_peaks) > 12:
        _peaks.clear()
    _peaks[key] = out
    return out


# ---- what the page draws for a model (the desktop tab's layout, from the family's description) ----------------
CAT_COLOR = {"style_composition": "#5B9BD5", "style_ident_overlap": "#5BB3A6", "identity": "#70AD47",
             "ident_details_overlap": "#B8A547", "details": "#ED7D31", "look": "#5B9BD5"}
CAT_SHORT = {"style_composition": "Style+Comp", "style_ident_overlap": "Style/ID", "identity": "Identity",
             "ident_details_overlap": "ID/Detail", "details": "Details", "look": "Look"}
# Klein's (category_masters) built-in presets: which categories stay ticked
KLEIN_PRESETS = {"✨Reset All": None, "✨Identity Only": ["identity", "style_ident_overlap", "ident_details_overlap"],
                 "✨Style+Composition Only": ["style_composition", "style_ident_overlap"],
                 "✨Details Only": ["details", "ident_details_overlap"]}
STILLS_SIZES = ["256", "384", "512", "640", "768", "1024", "1280", "1536"]   # the Samples tab's + two quick ones


def families():
    """The models Studio offers: every family whose description lists the repair workbench (the desktop's list) -
    stills, H3's clips and ACE-Step's songs."""
    from fizgig.families.registry import training_families
    return [d for d in training_families() if "repair" in (d.workbench or ()) and d.training_ready]


SONG_LENGTHS = ["10", "20", "30", "60", "120"]


def plan_video(desc, prefs, pref_path, checkpoint="", base="", refmod=False):
    """A video family's load plan, as the desktop's _repair_engine_plan_video: the DiT the checkpoint picker names and
    the base its Base picker names (the description's video_controls), the Turbo LoRA at the family's preview
    strength, the prompt cache folder, the audio VAE when it is there."""
    look = lambda k: pref_path(prefs, k)
    path = lambda role: desc.model_path(role, look)
    labels = {f.role: f.label for f in desc.model_files}
    vc = desc.video_controls
    roles = (vc.checkpoint(checkpoint, refmod).role if vc and vc.checkpoints else "dit", "vae", "text_encoder")
    base_mode = vc.base_mode(base) if vc else "auto"
    problems = [f"{labels.get(r, r)} isn't set (or isn't where Preferences says). Set it on the Preferences tab."
                for r in roles if not path(r) or not os.path.exists(path(r))]
    sp = desc.preview_speed()
    turbo = path("speed_lora") or (look(sp.pref_key) if sp is not None and sp.pref_key else "")
    cache = look("cache_dir")
    audio = path("audio_vae")
    kw = {"dit_path": path(roles[0]), "vae_path": path("vae"), "text_encoder_path": path("text_encoder"),
          "device": "cuda", "speed_lora_path": turbo if turbo and os.path.exists(turbo) else "",
          "turbo_lora_strength": (desc.preview_speed_defaults() or (0, 0.75))[1],
          "te_cache_dir": os.path.join(cache, "te_prompts") if cache else "",
          "audio_vae_path": audio if audio and os.path.exists(audio) else "", "base_mode": base_mode}
    return kw, f"Loading {desc.display_name} (the base takes a minute)…", problems


def layout(desc, prefs, pref_path, samples=None):
    cats = dict(desc.block_categories or ())
    groups = []
    for g in desc.load_driver().block_map():
        groups.append({"label": g.label, "blocks": [
            {"id": b.id, "label": b.label, "cat": CAT_SHORT.get(cats.get(b.id), ""),
             "color": CAT_COLOR.get(cats.get(b.id), "")} for b in g.blocks]})
    if desc.category_masters:
        presets = [{"name": n, "keep": keep} for n, keep in KLEIN_PRESETS.items()]
    else:
        presets = [{"name": "✨Reset All", "keep": None}] + [{"name": n, "blocks": dict(b)} for n, b in
                                                            (desc.repair_presets or ())]
    look = lambda k: pref_path(prefs, k)
    have = lambda k: bool(k and look(k) and os.path.exists(look(k)))
    ck, sp = desc.preview_checkpoint(), desc.preview_speed()
    fast = (f"{ck[1].name} ({ck[1].steps}-step, default)" if ck is not None and have(ck[0].pref_key) else
            f"{sp.name} ({sp.settings.steps}-step, default)" if sp is not None and have(sp.pref_key) else "")
    ds = desc.default_sampling()
    kind = "video" if desc.video_workbench else "audio" if desc.audio_workbench else "stills"
    audio_vae = desc.model_path("audio_vae", look) if kind == "video" else ""
    vc = desc.video_controls
    return {"key": desc.key, "name": desc.display_name, "groups": groups, "kind": kind,
            "video": {"lengths": [n for n, _ in vc.lengths], "length_frames": dict(vc.lengths),
                      "length": vc.default_length or vc.lengths[0][0], "dims": [str(d) for d in vc.sizes],
                      "size": [str(d) for d in vc.default_size],
                      "scales": [n for n, _ in vc.scales], "scale_of": dict(vc.scales),
                      "models": [c.label for c in vc.checkpoints],
                      "references": {c.label: c.references for c in vc.checkpoints},
                      "bases": [n for n, _ in vc.bases], "speed_boost": bool(vc.speed_boost),
                      "sound": bool(audio_vae and os.path.exists(audio_vae))} if kind == "video" and vc else None,
            "song_lengths": SONG_LENGTHS if kind == "audio" else None,
            "prompt_blend": bool(getattr(desc.load_driver(), "supports_prompt_blend", False)),
            "model_eq": _model_eq(desc, look) if kind != "video" else "",
            "audio_source": bool(getattr(desc.load_driver(), "supports_audio_source", False)),
            "categories": {b: CAT_SHORT.get(c, "") for b, c in cats.items()},
            "masters": [{"cat": c, "label": CAT_SHORT[c], "color": CAT_COLOR[c]} for c in
                        ("style_composition", "style_ident_overlap", "identity", "ident_details_overlap", "details")]
            if desc.category_masters else [],
            "category_of": cats if desc.category_masters else {},
            "presets": presets, "lucky_zones": [list(z) for z in (desc.lucky_zones or ())],
            "detail_blocks": list(desc.detail_blocks or ()),
            "reference_kind": desc.reference_kind or "", "reference_strength": bool(desc.reference_strength),
            "size": str(desc.repair_size or 768), "sizes": SONG_LENGTHS if kind == "audio" else STILLS_SIZES,
            "default_strength": float(desc.repair_default_strength or 1.0),
            "negative_default": desc.preview_negative, "turbo_preview": bool(desc.activation_cache),
            "follows_samples": bool(desc.workbench_follows_samples),
            "dit_fast": fast, "dit_full": f"Full ({ds.steps}-step, slow)" if ds else "",
            "cfg_fast": float(ck[1].cfg) if ck is not None and have(ck[0].pref_key) else
            float(sp.settings.cfg) if fast and sp is not None else float(ds.cfg if ds else 1.0),
            "cfg_full": float(ds.cfg if ds else 1.0),
            # a model that follows its Samples tab: what Studio renders with (the negative only matters above CFG 1)
            "follow_cfg": _num((samples or {}).get("cfg"), 1.0), "follow_steps": str((samples or {}).get("steps") or ""),
            "follow_turbo": (samples or {}).get("turbo_name", "") if (samples or {}).get("turbo_live") else "",
            # the desktop's note: a preview checkpoint set in Preferences (Qwen's Turbo) renders at its own recipe
            "follow_ck": (f"Previews on the {ck[0].label.split(' (')[0]} - {ck[1].steps} steps, CFG {ck[1].cfg:g}. "
                          f"Clear it in Preferences to preview with the Samples tab's settings instead.")
            if ck is not None and have(ck[0].pref_key) else ""}


def _model_eq(desc, look):
    """The family's Model EQ LoRA (Preferences), when set and there."""
    try:
        p = desc.model_path("model_eq", look)
    except Exception:
        p = ""
    return p if p and os.path.isfile(p) else ""


def _num(x, d):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d
