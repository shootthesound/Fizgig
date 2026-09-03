r"""Repair Studio on MiniMax H3 renders CLIPS — the Clip row, the worker's clip path and the
in-app player exist only for that family; Klein / Krea 2 keep the still path untouched.

Pins (headless, no models):
  1. family gating: the Clip row (Length / Size / Regime / Sound) is managed only under
     minimax; the square Res combo hides there and comes back for Klein AND Krea 2 (the
     Krea 2 return path has no Turbo tick to pack `before=`);
  2. Size choices: long side 768, short side 768/704/640 in both orientations; 576/512 only
     with "allow lower"; un-ticking it with a now-locked rung selected falls back to square;
     the parser reads "W × H" strings;
  3. _run_preview_async under minimax takes width/height/frames from the Clip row and hands
     {frames, regime, with_audio} to the worker; under Klein it hands None;
  4. the worker: h3_opts -> baseline_clip + render_clip (never generate_preview);
     None -> generate_baseline + generate_preview (never render_clip);
  5. clips landing: both middle frames reach the panel, the player opens playing in
     lockstep, Swap trades sides, a frame step pauses and moves one frame, close tears down;
     a session reset closes the player and drops the clips.

Run: venv/Scripts/python.exe tests/test_repair_h3_video.py
"""
import os
import sys

os.environ["FIZGIG_NO_PERSIST"] = "1"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import tkinter as tk  # noqa: E402
from PIL import Image  # noqa: E402

import lora_trainer_gui as G  # noqa: E402

FAILS = []


def ck(name, cond, detail=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


G.LAST_USED_FILE = os.path.join(os.environ.get("TEMP", "/tmp"), "nope", ".last_used.json")
G.LoRATrainerGUI.save_prefs = lambda self, *a, **k: None
G.LoRATrainerGUI._save_training_queue = lambda self, *a, **k: None
G.LoRATrainerGUI._save_last_used_paths = lambda self, *a, **k: None

root = tk.Tk()
root.withdraw()
app = G.LoRATrainerGUI(root)
_err = G.messagebox.showerror
G.messagebox.showerror = lambda *a, **k: print("  (showerror suppressed)", a)


def fam(name):
    app.repair_family_var.set(name)
    app._on_repair_family_changed()


try:
    # --- 1. family gating -----------------------------------------------------------------
    fam("klein")
    ck("Klein: Clip row not managed", not app._repair_h3_row.winfo_manager())
    ck("Klein: Res combo packed", bool(app._repair_res_combo.winfo_manager()))
    ck("Klein: compare button is the still compare",
       app._repair_cmp_btn.cget("text").startswith("⧉"))
    fam("minimax")
    ck("H3: Clip row managed", bool(app._repair_h3_row.winfo_manager()))
    ck("H3: Res combo hidden", not app._repair_res_combo.winfo_manager())
    ck("H3: compare button is the player", app._repair_cmp_btn.cget("text").startswith("▶"))
    ck("H3: hint says play", "play" in app._repair_preview_hint.cget("text").lower())
    fam("krea2")
    ck("Krea 2 (from H3): Clip row hidden again", not app._repair_h3_row.winfo_manager())
    ck("Krea 2 (from H3): Res combo back (no Turbo tick to pack before)",
       bool(app._repair_res_combo.winfo_manager()))
    fam("klein")
    ck("Klein (from Krea 2): Res combo present, Turbo tick present",
       bool(app._repair_res_combo.winfo_manager()) and bool(app._repair_turbo_chk.winfo_manager()))

    # --- 2. size choices + parser ---------------------------------------------------------
    fam("minimax")
    app.repair_h3_lower_var.set(False)
    app._on_repair_h3_lower_toggled()
    vals = list(app._repair_h3_size_combo["values"])
    ck("default sizes: 768 square + 704/640 both ways, nothing lower",
       vals == ["768 × 768", "768 × 704  (landscape)", "704 × 768  (portrait)",
                "768 × 640  (landscape)", "640 × 768  (portrait)"], vals)
    app.repair_h3_lower_var.set(True)
    app._on_repair_h3_lower_toggled()
    vals = list(app._repair_h3_size_combo["values"])
    ck("allow lower adds 576 and 512 rungs",
       any(v.startswith("768 × 512") for v in vals) and any(v.startswith("576 × 768") for v in vals))
    app.repair_h3_size_var.set("768 × 512  (landscape)")
    ck("parser: landscape 768x512", app._repair_h3_size() == (768, 512))
    app.repair_h3_size_var.set("640 × 768  (portrait)")
    ck("parser: portrait 640x768", app._repair_h3_size() == (640, 768))
    app.repair_h3_size_var.set("768 × 512  (landscape)")
    app.repair_h3_lower_var.set(False)
    app._on_repair_h3_lower_toggled()
    ck("un-ticking with a locked rung selected falls back to square",
       app.repair_h3_size_var.get() == "768 × 768")
    app.repair_h3_frames_var.set("56 frames (~2.3s)")
    ck("length parser", app._repair_h3_frames() == 56)
    app.repair_h3_frames_var.set("Still (1 frame)")
    ck("length parser: still", app._repair_h3_frames() == 1)
    app.repair_h3_frames_var.set("22 frames (~1s)")

    # --- 3/4. worker dispatch --------------------------------------------------------------
    calls = []

    class FakeEngine:
        primary_network = object()
        primary_path = "x"
        donor_path = None
        on_step = None

        def clear_cancel(self):
            pass

        def reset(self):
            pass

        def generate_baseline(self, st):
            calls.append("generate_baseline"); return Image.new("RGB", (32, 32), "red")

        def generate_preview(self, st):
            calls.append("generate_preview"); return Image.new("RGB", (32, 32), "blue")

        def baseline_clip(self, st, **kw):
            calls.append(("baseline_clip", dict(kw), st.preview_width, st.preview_height,
                          st.preview_frames))
            return _clip("green", kw.get("frames", 22), kw.get("regime"))

        def render_clip(self, st, **kw):
            calls.append(("render_clip", dict(kw)))
            return _clip("white", kw.get("frames", 22), kw.get("regime"))

    def _clip(color, n, regime):
        frames = [Image.new("RGB", (64, 48), color) for _ in range(n)]
        return {"latent": None, "audio_rows": None, "frames": frames, "wav": None,
                "middle": frames[len(frames) // 2], "regime": regime, "steps": 4,
                "turbo_strength": 1.0, "frames_n": n}

    captured = {}
    _real_worker = app._repair_preview_worker
    app._repair_preview_worker = lambda snap, opts=None: captured.update(snap=snap, opts=opts)
    app.repair_engine = FakeEngine()
    app.repair_prompt_var.set("zwxem test prompt")
    app.repair_seed_var.set("7")
    app.repair_h3_size_var.set("768 × 640  (landscape)")
    app.repair_h3_regime_var.set("confirm")
    app.repair_h3_sound_var.set(True)
    app._repair_preview_in_flight = False
    app._run_preview_async()
    # (a thread ran the stub; give it a beat)
    import time; time.sleep(0.2)
    ck("H3 run_async: canvas from the Clip row, not Res",
       captured["snap"].preview_width == 768 and captured["snap"].preview_height == 640)
    ck("H3 run_async: frames on the snapshot", captured["snap"].preview_frames == 22)
    o = captured["opts"]
    ck("H3 run_async: opts carry frames/regime; sound only with an audio VAE configured",
       o == {"frames": 22, "regime": "confirm",
             "with_audio": bool(app._repair_h3_audio_vae_path())}, o)
    app.repair_h3_sound_var.set(False)
    app._repair_preview_in_flight = False
    app._run_preview_async()
    time.sleep(0.2)
    ck("H3 run_async: Sound unticked -> with_audio False", captured["opts"]["with_audio"] is False)
    app.repair_h3_sound_var.set(True)
    app._repair_preview_in_flight = False
    fam("klein")
    app.repair_engine = FakeEngine()
    app.repair_prompt_var.set("zwxem test prompt")
    app.repair_res_var.set("512")
    app._run_preview_async()
    time.sleep(0.2)
    ck("Klein run_async: no H3 opts, square Res", captured["opts"] is None
       and captured["snap"].preview_width == 512)
    app._repair_preview_worker = _real_worker
    app._repair_preview_in_flight = False

    # worker paths (run inline: the worker posts results via master.after)
    calls.clear()
    st = app.repair_state.copy()
    st.preview_width = st.preview_height = 64
    app._repair_preview_in_flight = True
    app._repair_preview_worker(st, None)
    root.update()
    ck("worker without opts: still path", calls == ["generate_baseline", "generate_preview"], calls)
    calls.clear()
    fam("minimax")
    app.repair_engine = FakeEngine()
    st = app.repair_state.copy()
    st.preview_width, st.preview_height, st.preview_frames = 64, 48, 5
    app._repair_preview_in_flight = True
    app._repair_preview_worker(st, {"frames": 5, "regime": "dial", "with_audio": False})
    root.update()
    ck("worker with opts: clip path only",
       [c[0] for c in calls] == ["baseline_clip", "render_clip"], calls)
    ck("worker with opts: baseline gets the same opts",
       calls[0][1] == {"frames": 5, "regime": "dial", "with_audio": False}, calls[0])

    # --- 5. clips landed + the player ------------------------------------------------------
    ck("clips landed: both sides stored, 5 frames each",
       set(app._repair_clips) == {"baseline", "tweaked"}
       and len(app._repair_clips["tweaked"]["frames"]) == 5)
    ck("panel shows the middle frames",
       app.repair_pil_images["tweaked"] is app._repair_clips["tweaked"]["middle"])
    ck("status names the regime + length", "Dial" in app.repair_status_var.get()
       and "5 frames" in app.repair_status_var.get(), app.repair_status_var.get())
    ck("in-flight cleared", not app._repair_preview_in_flight)
    app._repair_popout_preview()
    P = app._repair_player
    ck("player opened via the compare entry point", P is not None and P["win"].winfo_exists())
    ck("player is playing, 5 frames, sides baseline|tweaked",
       P["playing"] and P["n"] == 5 and P["sides"] == ["baseline", "tweaked"])
    ck("metrics strip registered on the player window",
       app._repair_popout_window is P["win"] and set(app._repair_popout_metric_lbls) == {
           "likeness", "grid", "texture", "clip", "sat"})
    app._repair_clip_player_swap()
    ck("swap trades sides", P["sides"] == ["tweaked", "baseline"])
    ck("titles follow the swap", "Tweaked" in P["titles"][0].cget("text")
       and "Baseline" in P["titles"][1].cget("text"))
    app._repair_clip_player_stop()
    app._repair_clip_player_paint(2)
    app._repair_clip_player_step(+1)
    ck("frame step: paused, one frame on", not P["playing"] and P["idx"] == 3)
    app._repair_clip_player_step(+1); app._repair_clip_player_step(+1)
    ck("frame step wraps", P["idx"] == 0)
    ck("pos label", P["pos_lbl"].cget("text") == "1 / 5", P["pos_lbl"].cget("text"))
    # a second open call raises the same window rather than making another
    app._repair_clip_player_open()
    ck("re-open raises, no second player", app._repair_player is P)
    # a new pair landing reloads the player in place
    app._repair_preview_in_flight = True
    app._set_repair_preview_clips(_clip("black", 3, "confirm"), _clip("gray", 3, "confirm"))
    ck("new clips reload the open player", app._repair_player is P and P["n"] == 3
       and "Confirm" in P["titles"][0].cget("text"))
    app._reset_repair_session()
    ck("session reset closes the player and drops the clips",
       app._repair_player is None and app._repair_clips == {})
    # Klein's still pop-out is unaffected
    fam("klein")
    app.repair_pil_images["baseline"] = Image.new("RGB", (32, 32), "red")
    app.repair_pil_images["tweaked"] = Image.new("RGB", (32, 32), "blue")
    app._repair_popout_preview()
    ck("Klein compare still opens the still pop-out",
       app._repair_player is None and app._repair_popout_window is not None
       and app._repair_popout_label is not None)
    app._repair_popout_window.destroy()
    app._repair_popout_window = None
finally:
    G.messagebox.showerror = _err
    try:
        root.destroy()
    except Exception:
        pass

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    sys.exit(1)
print("ALL PASS")
