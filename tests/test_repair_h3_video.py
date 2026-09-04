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
import shutil
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


def wait_for(cond, timeout=5.0):
    """Pump a real mainloop until cond() — after() from a worker thread needs the main
    thread to actually be in the loop, exactly as in the running app."""
    import time as _t
    deadline = _t.time() + timeout

    def _poll():
        if cond() or _t.time() > deadline:
            root.quit()
        else:
            root.after(20, _poll)

    root.after(20, _poll)
    root.mainloop()
    return cond()


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
    ck("un-ticking with a locked rung selected falls back to the default (768×640)",
       app.repair_h3_size_var.get() == "768 × 640  (landscape)")
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

        def mark_blocks_changed(self, blocks):
            pass

        def _invalidate_baseline_cache(self):
            self._baseline_clip_key = None
            self._baseline_clip = None

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
    ck("H3 run_async: opts carry frames/regime/early; sound only with an audio VAE configured",
       o == {"frames": 22, "regime": "confirm", "early_step": 2,
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
    ck("worker with opts: baseline gets the same opts (+ the cache slot)",
       calls[0][1] == {"frames": 5, "regime": "dial", "with_audio": False, "cache": None}, calls[0])

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

    # --- 6. render cache + block library: build after an exact Dial render, ticks, cache
    #        hit on [0], peek, show-early badge, no approximation anywhere ----------------
    import tempfile
    import torch
    cache_root = tempfile.mkdtemp(prefix="fizgig_rcache_gui_")
    app._repair_cache_root = lambda: cache_root
    ACTIVE = {"h3blk_3", "h3blk_21", "h3_rf_0"}
    renders = []

    class CacheEngine(FakeEngine):
        """Synthetic renders: latent = ones + (index+1) marker per active block at 1.0, so an
        off-render differs from the baseline in a known way. Real render_clip/baseline_clip
        semantics (cache hit / put, early hook) are exercised through the real engine
        methods by binding them onto this fake."""
        primary_block_ids = ACTIVE
        primary_hash = "deadbeef"
        donor_network = None
        donor_hash = None
        donor_path = None
        primary_path = "C:/fake/lora.safetensors"
        _steps = 6
        _turbo_net = object()
        _baseline_clip_key = None
        _baseline_clip = None
        last_resume_from = None

        def regime_params(self, regime):
            return (4, 1.0) if regime == "dial" else (6, 0.75)

        def request_cancel(self):
            pass

        @staticmethod
        def keyframe_signature(st):
            return ()

        def clip_key(self, st, **kw):
            return (st.seed, st.prompt, st.preview_width, st.preview_height, kw.get("frames"),
                    kw.get("steps"), kw.get("turbo_strength"))

        def cache_key_for(self, st, *, frames, regime, **_):
            from fizgig.repair_studio.h3_render_cache import setup_key
            steps, strength = self.regime_params(regime)
            return setup_key(primary_hash=self.primary_hash, donor_hash="", prompt=st.prompt,
                             seed=st.seed, frames=frames, width=st.preview_width,
                             height=st.preview_height, steps=steps, turbo_strength=strength,
                             keyframe_sig=())

        def _latent(self, st):
            lat = torch.ones(1, 24, 2, 4, 4)
            for i, bid in enumerate(sorted(ACTIVE)):
                bs = st.blocks[bid]
                m = bs.primary_strength if bs.primary_enabled else 0.0
                lat[:, i] += float(m) * (i + 1)
            return lat

        def render_latent(self, st, on_denoised=None, **kw):
            renders.append(("render_latent", kw.get("steps"), st.preview_width))
            if on_denoised is not None:
                for _s in (1, 2):
                    on_denoised(_s, kw.get("steps") or 4, self._latent(st) * 0.5)
            time.sleep(0.02)
            return self._latent(st), torch.zeros(4, 32)

        def decode_clip_frames(self, latent):
            return [Image.new("RGB", (32, 32), "black") for _ in range(3)]

        def decode_middle_frame_image(self, latent):
            renders.append(("early_decode",))
            return Image.new("RGB", (32, 32), "blue")

        def decode_audio(self, rows):
            return None

    from fizgig.repair_studio.h3_engine import H3RepairEngine as _H3E
    CacheEngine.render_clip = _H3E.render_clip
    CacheEngine.baseline_clip = _H3E.baseline_clip
    CacheEngine.describe_state = staticmethod(_H3E.describe_state)
    CacheEngine.clip_from_cache = _H3E.clip_from_cache

    fam("minimax")
    app.repair_engine = CacheEngine()
    app._refresh_block_slider_activity()
    app.repair_prompt_var.set("zwxem cache prompt")
    app.repair_seed_var.set("3")
    app.repair_h3_frames_var.set("22 frames (~1s)")
    app.repair_h3_size_var.set("768 × 640  (landscape)")
    app.repair_h3_dial_scale_var.set("⅔")
    app.repair_h3_regime_var.set("dial")
    app.repair_h3_sound_var.set(False)
    app.repair_h3_early_var.set(True)
    ck("library row managed under H3", bool(app._repair_cache_row.winfo_manager()))
    ck("dial canvas: ⅔ of 768x640 snapped to /32", app._repair_h3_canvas("dial") == (512, 416))
    ck("confirm canvas: the full Size", app._repair_h3_canvas("confirm") == (768, 640))
    app.repair_h3_dial_scale_var.set("½")
    ck("dial canvas: ½", app._repair_h3_canvas("dial") == (384, 320))
    app.repair_h3_dial_scale_var.set("Full")
    ck("dial canvas: Full = Size", app._repair_h3_canvas("dial") == (768, 640))
    app.repair_h3_dial_scale_var.set("⅔")
    ck("no cache before the first render", app._repair_cache is None)
    if app._repair_preview_after_id is not None:      # (donor-var traces, see above)
        root.after_cancel(app._repair_preview_after_id)
        app._repair_preview_after_id = None
    # first exact Dial render -> early look mid-render, then the library builds
    early_seen = []
    _orig_show_early = app._repair_show_early
    def _spy_early(img, step, n, gen):
        early_seen.append((step, n, app._repair_preview_in_flight))
        _orig_show_early(img, step, n, gen)
    app._repair_show_early = _spy_early
    app._repair_preview_in_flight = False
    app._run_preview_async()
    wait_for(lambda: (not app._repair_preview_in_flight and app._repair_cache is not None
                      and app._repair_cache.complete()
                      and not app._repair_cache_thread.is_alive()), timeout=15.0)
    cache = app._repair_cache
    ck("render bound a cache for the LoRA's blocks at the DIAL canvas",
       cache is not None and set(cache.block_ids) == ACTIVE
       and cache.meta.get("width") == 512 and cache.meta.get("height") == 416,
       None if cache is None else cache.meta)
    ck("snapshot rendered at the dial canvas",
       any(r[0] == "render_latent" and r[2] == 512 for r in renders), renders[:3])
    ck("default sliders: the tweaked side is the baseline -> served from cache, no early look",
       app._repair_clips["tweaked"].get("cached") is True and not early_seen, early_seen)
    ck("library complete: base + every block-off entry",
       cache.complete() and cache.n_entries() == 1 + len(ACTIVE))
    root.update()
    ck("ticks: ● on active blocks (clickable), blank on untouched",
       app.repair_block_vars["h3blk_21"]["cache_lbl"].cget("text") == "●"
       and app.repair_block_vars["h3blk_5"]["cache_lbl"].cget("text") == "")
    ck("status: complete + counts", "complete" in app.repair_cache_status_var.get()
       and "renders cached" in app.repair_cache_status_var.get(), app.repair_cache_status_var.get())
    ck("entries carry thumbs", os.path.isfile(cache.thumb_path("off:h3blk_21")))
    e = cache.get("off:h3blk_21")[0]; b = cache.get("base")[0]
    i21 = sorted(ACTIVE).index("h3blk_21")
    ck("block-21-off entry is that exact render",
       torch.allclose((b - e)[:, i21], torch.full_like(e[:, i21], float(i21 + 1))))
    # [0] on block 21 -> a cache HIT: no render, no early look, status says from cache
    renders.clear(); early_seen.clear()
    app.repair_block_vars["h3blk_21"]["primary_strength"].set(0.0)
    wait_for(lambda: app._repair_preview_after_id is None and not app._repair_preview_in_flight
             and "from cache" in app.repair_status_var.get(), timeout=6.0)
    ck("[0] on a library block is served from the cache (no render, no early look)",
       not any(r[0] == "render_latent" for r in renders) and not early_seen
       and app._repair_clips["tweaked"].get("cached") is True, renders)
    ck("status says exact, from cache", "exact, from cache" in app.repair_status_var.get(),
       app.repair_status_var.get())
    # a new state renders, is cached, and re-rendering the same state hits
    renders.clear()
    app.repair_block_vars["h3blk_21"]["primary_strength"].set(0.5)
    wait_for(lambda: app._repair_preview_after_id is None and not app._repair_preview_in_flight
             and app._repair_clips["tweaked"].get("cached") is False, timeout=6.0)
    ck("a new state renders (exact) and lands in the cache",
       any(r[0] == "render_latent" for r in renders) and cache.n_entries() == 2 + len(ACTIVE))
    ck("show early fired during that render with pass 2 of 4 (pass 1 is mush), while in flight",
       early_seen and early_seen[0][:2] == (2, 4) and early_seen[0][2] is True
       and any(r[0] == "early_decode" for r in renders), early_seen)
    ck("...and the finished clip cleared the early badge",
       app._repair_tweaked_title.cget("text") == "Tweaked (current sliders)")
    app.repair_block_vars["h3blk_21"]["primary_strength"].set(1.0)
    wait_for(lambda: app._repair_preview_after_id is None and not app._repair_preview_in_flight
             and app._repair_clips["tweaked"].get("cached") is True, timeout=6.0)
    ck("back to 1.0 = baseline state, served from cache", app._repair_clips["tweaked"].get("cached") is True)
    # peek: click ● on block 3 -> its off clip shows, sliders untouched
    renders.clear()
    app._repair_cache_peek("h3blk_3")
    wait_for(lambda: not app._repair_preview_in_flight and app._repair_peek is not None, timeout=6.0)
    ck("peek shows the library entry without touching the sliders",
       app._repair_peek == "Block 3 off"
       and app.repair_block_vars["h3blk_3"]["primary_strength"].get() == 1.0
       and "library" in app._repair_tweaked_title.cget("text")
       and not any(r[0] == "render_latent" for r in renders), (app._repair_peek, renders))
    # a slider move ends the peek
    app.repair_block_vars["h3blk_3"]["primary_strength"].set(0.8)
    wait_for(lambda: app._repair_preview_after_id is None and not app._repair_preview_in_flight
             and app._repair_peek is None, timeout=6.0)
    ck("a slider move ends the peek and restores the title",
       app._repair_peek is None and app._repair_tweaked_title.cget("text") == "Tweaked (current sliders)")
    # regime switch = another setup key; Confirm renders at the full Size and never builds
    app.repair_h3_regime_var.set("confirm")
    renders.clear()
    app._on_repair_h3_clip_changed()
    wait_for(lambda: app._repair_preview_after_id is None and not app._repair_preview_in_flight
             and any(r[0] == "render_latent" and r[2] == 768 for r in renders), timeout=6.0)
    ck("Confirm renders at the full Size", any(r[0] == "render_latent" and r[2] == 768 for r in renders))
    ck("Confirm setup is its own cache, and does not build a library",
       app._repair_cache is not cache and (app._repair_cache_thread is None
                                          or not app._repair_cache_thread.is_alive()))
    # restart: a fresh cache on the dial key finds everything
    from fizgig.repair_studio.h3_render_cache import RenderCache
    again = RenderCache(cache_root, cache.key, sorted(ACTIVE))
    ck("restart: the dial library reloads complete from disk", again.complete())
    # --- 7. first/last frame card + crop dialog + history strip ---------------------------
    ck("keyframe card managed under H3", bool(app._repair_kf_container.winfo_manager()))
    photo = os.path.join(cache_root, "photo.png")
    Image.new("RGB", (1000, 500), "purple").save(photo)      # 2:1 photo, clip is 768x640 (1.2:1)
    # the crop dialog is modal: press Return on it once it's up -> the default box
    def _accept():
        for w in app.master.winfo_children():
            if isinstance(w, tk.Toplevel) and hasattr(w, "_crop_ok"):
                w._crop_ok()
                return
        root.after(100, _accept)
    root.after(300, _accept)
    rect = app._repair_kf_crop_dialog(photo)
    ck("crop dialog default box: largest centred box of the clip's aspect",
       rect is not None and rect[3] - rect[1] == 500 and abs((rect[2] - rect[0]) - 600) <= 2
       and abs(rect[0] - 200) <= 2, rect)
    # a pre-set crop (re-crop) comes back as given
    root.after(300, _accept)
    rect2 = app._repair_kf_crop_dialog(photo, initial=(100, 50, 400, 300))
    ck("re-crop starts from the stored rect", rect2 == (100, 50, 400, 300), rect2)
    # set the first frame directly (browse = dialog + this)
    app._repair_h3_kf["first"] = {"path": photo, "rect": rect}
    app._repair_kf_refresh_thumbs()
    ck("thumb + info shown for the slot",
       app._repair_h3_kf_widgets["first"]["thumb"].image is not None
       and "photo.png" in app._repair_h3_kf_widgets["first"]["info"].cget("text"))
    encodes = []
    class KFEngine(CacheEngine):
        def encode_keyframe(self, img, w, h):
            encodes.append((img.size, w, h))
            return torch.full((1, 24, 1, h // 16, w // 16), 0.5)
    KFEngine.keyframe_signature = staticmethod(_H3E.keyframe_signature)
    KFEngine.clip_from_cache = _H3E.clip_from_cache
    KFEngine.cache_key_for = _H3E.cache_key_for
    KFEngine.primary_network = object()
    app.repair_engine = KFEngine()
    kf = app._repair_h3_prepare_keyframes(512, 416, 22)
    ck("keyframes: the first photo, cropped, encoded at the render canvas, index 0",
       kf is not None and len(kf) == 1 and kf[0][0] == 0 and encodes[-1] == ((600, 500), 512, 416),
       encodes)
    kf_b = app._repair_h3_prepare_keyframes(512, 416, 22)
    ck("second call is a cache hit (no re-encode)", len(encodes) == 1 and kf_b[0][1] is kf[0][1])
    app._repair_h3_kf["last"] = {"path": photo, "rect": rect}
    kf2 = app._repair_h3_prepare_keyframes(768, 640, 22)
    ck("last frame -> index frames-1, encoded at the Confirm canvas",
       [i for i, _ in kf2] == [0, 21] and encodes[-1] == ((600, 500), 768, 640))
    ck("a still takes the first photo only",
       [i for i, _ in app._repair_h3_prepare_keyframes(512, 416, 1)] == [0])
    st_kf = app.repair_state.copy(); st_kf.preview_width, st_kf.preview_height = 512, 416
    st_kf.keyframes = kf
    st_no = st_kf.copy(); st_no.keyframes = None
    ck("keyframes change the render-cache setup key",
       app.repair_engine.cache_key_for(st_kf, frames=22, regime="dial")
       != app.repair_engine.cache_key_for(st_no, frames=22, regime="dial"))
    app._repair_kf_clear("first"); app._repair_kf_clear("last")
    ck("clear empties the slots", app._repair_h3_prepare_keyframes(512, 416, 22) is None)
    if app._repair_preview_after_id is not None:
        root.after_cancel(app._repair_preview_after_id); app._repair_preview_after_id = None
    # history strip: one thumb per cached entry of the current setup
    app.repair_engine = CacheEngine()
    app._repair_cache = cache
    app._repair_history_shown = ()
    app._repair_history_refresh()
    root.update()
    cells = app._repair_history_inner.winfo_children()
    ck("history strip shows every cached render", len(cells) == cache.n_entries(), len(cells))
    # view an entry -> tweaked pane shows it as a peek with its label
    sig = "off:h3blk_21"
    app._repair_preview_in_flight = False
    app._repair_history_view(sig)
    wait_for(lambda: not app._repair_preview_in_flight and app._repair_peek is not None, 6.0)
    ck("history view shows the entry (sliders untouched)",
       app._repair_peek == "Block 21 off" and app._repair_clips["tweaked"].get("sig") == sig,
       app._repair_peek)
    # pin it as the baseline: the next render's baseline pane is that entry (same setup)
    app.repair_h3_regime_var.set("dial")
    app._repair_history_pin(sig)
    ck("pin marks the strip + the baseline title",
       app._repair_pinned_sig == sig and "Pinned" in app._repair_baseline_title.cget("text"))
    wait_for(lambda: app._repair_preview_after_id is None and not app._repair_preview_in_flight
             and app._repair_clips["baseline"].get("pinned") is True, 6.0)
    ck("baseline pane is the pinned entry after the re-render",
       app._repair_clips["baseline"].get("sig") == sig)
    # a setup change (Confirm regime) drops the pin instead of leaving the title lying
    app._repair_history_pin(sig, rerender=False)
    app.repair_h3_regime_var.set("confirm"); app._on_repair_h3_clip_changed()
    wait_for(lambda: app._repair_preview_after_id is None and not app._repair_preview_in_flight
             and app._repair_pinned_sig is None and "Pinned" not in app._repair_baseline_title.cget("text"), 6.0)
    ck("a setup change drops the pin and its title",
       app._repair_pinned_sig is None and "Pinned" not in app._repair_baseline_title.cget("text")
       and not app._repair_clips["baseline"].get("pinned"))
    app.repair_h3_regime_var.set("dial")
    fam("klein")
    ck("Klein: keyframe card + history strip hidden",
       not app._repair_kf_container.winfo_manager() and not app._repair_history_row.winfo_manager())
    fam("minimax")
    app.repair_engine = CacheEngine(); app._repair_cache = cache
    app.repair_h3_regime_var.set("dial")
    app._reset_repair_session()
    ck("reset drops the cache and the title", app._repair_cache is None
       and app._repair_tweaked_title.cget("text") == "Tweaked (current sliders)")
    shutil.rmtree(cache_root, ignore_errors=True)
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
