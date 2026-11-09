"""The web UI's Repair Studio worker: one process holding a family's workbench engine (fizgig.families.workbench, the
desktop's own) so the model stays loaded between renders. Commands arrive as JSON lines over a local socket (its port in the "ready" event); events leave as
"@@EV {json}" lines on stdout (anything else on stdout is the engine's own log). One render runs at a time in its own
thread, so a "cancel" is heard while it renders.

Commands (cmd=...): load {family, plan, primary, donor, turbo, preview_settings}, donor {path} / unload_donor,
settings {turbo, preview_settings}, render {gen, state, out_dir}, cancel, save {out, state, include_donor}, reset, quit;
royale / royale_travel (LoRA Royale), refmod (RefMod Toolkit).
Events (ev=...): status {text}, step {done, total}, loaded {...}, rendered {gen, baseline, tweaked},
aborted {gen}, error {where, msg}, saved {summary}, reset, donor {...}.
"""
import json
import os
import sys
import threading
import traceback

EV = "@@EV "
_out_lock = threading.Lock()


def emit(**d):
    with _out_lock:
        sys.stdout.write(EV + json.dumps(d, default=str) + "\n")
        sys.stdout.flush()


S = {"engine": None, "desc": None, "render": None}
_engine_lock = threading.RLock()


def _auto_swap():
    """The desktop's inference block-swap Auto: 16 GB and up none, 10-14 GB 12, under 10 GB 16."""
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return 0 if gb >= 15 else 12 if gb >= 10 else 16
    except Exception:
        pass
    return 0


def _blocks(eng):
    eq = {}
    try:                                         # a Model EQ row: its gain (the sliders' 0 / 2 = -g / +g)
        from fizgig.families.workbench import DONOR, PRIMARY
        gains = getattr(eng, "_eq", {}) or {}
        eq = {"primary": gains.get(PRIMARY), "donor": gains.get(DONOR)}
    except Exception:
        pass
    return {"primary_blocks": sorted(eng.primary_block_ids or ()), "donor_blocks": sorted(eng.donor_block_ids or ()),
            "hash": getattr(eng, "primary_hash", "") or "", "eq": eq}


def cmd_load(c):
    from fizgig.families.registry import get as get_family
    desc = get_family(c["family"])
    if desc is None:
        raise ValueError(f"Unknown model: {c['family']}")
    plan = dict(c["plan"])
    if plan.pop("use_checkpoint_sampling", False):
        ck = desc.preview_checkpoint()
        plan["preview_sampling"] = ck[1] if ck else None
    else:
        plan["preview_sampling"] = None
    if str(plan.get("blocks_to_swap", "")).lower().startswith("auto"):
        plan["blocks_to_swap"] = _auto_swap()
    with _engine_lock:
        eng = S["engine"]
        if eng is not None and (eng.desc is not desc and getattr(eng.desc, "key", None) != desc.key):
            eng.reset()
            eng = None
        if eng is not None and eng.primary_network is not None:
            eng.reset()                          # a new primary: a fresh session, as the desktop's Start does
        if eng is None:
            eng = desc.make_workbench_engine()
        S["engine"], S["desc"] = eng, desc
        S["plan_sig"] = json.dumps(c["plan"], sort_keys=True, default=str)
        S["nolora_out"] = S["nolora_song"] = None
        S["mix_files"] = {}
        eng.turbo_preview = bool(c.get("turbo", True))
        if getattr(desc, "video_workbench", False):
            # comfy-kitchen's INT8 attention where the model takes it (its description; H3 yes, Vimoo no)
            eng.int8_attention = bool(getattr(desc, "int8_attention", False))
            eng.on_status = lambda text: emit(ev="status", text=str(text))
            S["kf"] = {}                             # keyframe / reference latents, per slot, photo, crop and canvas
        if c.get("preview_settings") is not None:
            eng.preview_settings = c["preview_settings"]
        eng.on_step = lambda done, total: emit(ev="step", done=done, total=total)
        emit(ev="status", text=c.get("status") or f"Loading {desc.display_name}…")
        eng.plan_canvas = tuple(c["canvas"]) if c.get("canvas") else None   # H3 Auto: room for this clip
        eng.ensure_pipeline(**plan)
        emit(ev="status", text="Loading the LoRA…")
        eng.load_primary(c["primary"])
        if c.get("donor"):
            eng.load_donor(c["donor"])
        emit(ev="loaded", family=desc.key, primary=c["primary"], donor=c.get("donor") or "", **_blocks(eng))


def cmd_donor(c):
    with _engine_lock:
        eng = S["engine"]
        if eng is None or eng.primary_network is None:
            raise ValueError("Load a primary LoRA first.")
        S.setdefault("mix_files", {}).pop("donor", None)      # a new donor: its epoch blend starts off
        if eng.donor_network is not None:
            eng.unload_donor()
        if c.get("path"):
            eng.load_donor(c["path"])
        emit(ev="donor", donor=c.get("path") or "", **_blocks(eng))


def cmd_settings(c):
    eng = S["engine"]
    if eng is None:
        return
    if "turbo" in c:
        eng.turbo_preview = bool(c["turbo"])
    if c.get("preview_settings") is not None:
        eng.preview_settings = c["preview_settings"]
        eng._invalidate_baseline_cache()


def _state(d):
    from fizgig.repair_studio.state import BlockState, SliderState
    s = SliderState.from_json(d)
    eng = S["engine"]
    if eng is not None:                         # every block of the family, the ones the page left out at defaults
        for bid in eng.block_ids():
            s.blocks.setdefault(bid, BlockState())
    return s


def _aborted(e):
    names = {type(e).__name__}
    return bool(names & {"RenderCancelled", "SampleAborted", "PreviewAborted"})


def _render(c):
    gen = c["gen"]
    eng = S["engine"]
    try:
        with _engine_lock:
            if eng is None or eng.primary_network is None:
                raise ValueError("Load a LoRA first.")
            st = _state(c["state"])
            eng.clear_cancel()
            S["notes"] = {}
            if not getattr(S["desc"], "video_workbench", False):
                S["notes"].update(_apply_mix(eng, c.get("mix")))
            emit(ev="status", text=c.get("status") or "Generating preview…")
            out = c["out_dir"]
            os.makedirs(out, exist_ok=True)
            desc = S["desc"]
            nolora = None                        # the No-LoRA pane's render, after the pair is up
            same = False                         # the Tweaked render is the Baseline (nothing moved: Studio's maps)
            if getattr(desc, "video_workbench", False):
                paths, nolora = _render_clips(eng, desc, st, c, out, gen)
            elif getattr(desc, "audio_workbench", False):
                paths, nolora = _render_songs(eng, st, c, out, gen)
            else:
                base = eng.generate_baseline(st)
                paths = {"baseline": os.path.join(out, f"baseline_{gen}.png")}
                base.save(paths["baseline"])
                same = eng.same_as_baseline(st)                 # nothing moved: one render for both panes
                if not same:                                    # the Baseline shows while the Tweaked one renders
                    emit(ev="baseline_ready", gen=gen, baseline=paths["baseline"])
                tweak = base if same else eng.generate_preview(st)
                paths["tweaked"] = os.path.join(out, f"tweaked_{gen}.png")
                tweak.save(paths["tweaked"])
            emit(ev="rendered", gen=gen, nolora_pending=nolora is not None, notes=S.get("notes"), same=bool(same), **paths)
            _tidy(out, gen)
            if nolora is not None:               # as the desktop: the pair first, then the base model's own
                try:
                    emit(ev="nolora", gen=gen, **nolora())
                except Exception as e:           # a newer edit took the engine (or it failed: say so)
                    emit(ev="nolora", gen=gen, aborted=_aborted(e), msg="" if _aborted(e) else str(e))
            # every model (pictures and songs as clips): the render's cached memory goes back to the card between
            # renders, the model staying loaded - room for whatever else runs beside Studio
            import torch
            torch.cuda.empty_cache()
    except Exception as e:
        if _aborted(e):
            emit(ev="aborted", gen=gen)
        else:
            msg = str(e)
            emit(ev="error", where="render", gen=gen,
                 msg=(("Out of VRAM for this clip: fewer frames, a smaller canvas, or Base: Stream blocks (the exact "
                       "base with room for the biggest clips)." if getattr(S["desc"], "video_workbench", False) else
                       "Out of VRAM for this preview: a smaller Width / Height, or more inference block swap in Preferences.")
                      if "out of memory" in msg.lower() else msg), trace=traceback.format_exc()[-2000:])


def _cond(eng, c, width, height, frames):
    """(keyframes, references) for the clip, as the desktop's _repair_h3_prepare_cond: under First/last frame each
    photo's crop is a keyframe (first at frame 0, last at the final frame; a still takes the first only); under
    Reference each is a reference. Latents are kept per slot / photo / crop / canvas."""
    from PIL import Image
    spec = [k for k in (c.get("keyframes") or []) if k.get("path") and os.path.isfile(k["path"])]
    if not spec:
        return None, None
    ref = bool(c.get("ref_mode"))
    out = []
    from fizgig.utils.image import flatten
    from fizgig.utils.upright import open_upright
    for k in spec:
        slot, path = k.get("slot"), k["path"]
        if not ref and slot == "last" and int(frames) <= 1:
            continue
        img = flatten(open_upright(path))
        rect = k.get("rect")
        if rect:                                 # the box placed for another W x H: its shape now, centred inside it
            x0, y0, x1, y1 = (int(v) for v in rect)
            c = _centre_crop((x1 - x0, y1 - y0), width, height)
            rect = (x0 + c[0], y0 + c[1], x0 + c[2], y0 + c[3])
        else:
            rect = _centre_crop(img.size, width, height)
        key = ("ref" if ref else "kf", slot, path, os.path.getmtime(path), rect, int(width), int(height))
        hit = S["kf"].get(key)
        if hit is None:
            crop = img.crop(rect)
            hit = eng.encode_reference_image(crop, width, height) if ref else eng.encode_keyframe(crop, width, height)
            S["kf"][key] = hit
        out.append(hit if ref else (0 if slot == "first" else max(0, int(frames) - 1), hit))
    return (None, out or None) if ref else (out or None, None)


def _centre_crop(size, width, height):
    """The largest box of the clip's shape in the middle of the photo (the desktop's crop dialog starts there)."""
    w, h = size
    want = width / float(height)
    if w / float(h) > want:
        cw = int(round(h * want))
        x0 = (w - cw) // 2
        return (x0, 0, x0 + cw, h)
    ch = int(round(w / want))
    y0 = (h - ch) // 2
    return (0, y0, w, y0 + ch)


def _write_clip(clip, path_png, path_mp4, fps, rate):
    """The clip's middle frame as a picture and, for a clip, an MP4 with its sound (silent when it has none)."""
    clip["middle"].save(path_png)
    frames = clip.get("frames") or []
    if len(frames) <= 1:
        return ""
    import numpy as np
    import tempfile
    import torch
    from fizgig.minimax.common import _find_ffmpeg, write_preview_mp4, write_wav
    if not _find_ffmpeg():
        return ""
    arr = np.stack([np.asarray(f.convert("RGB")) for f in frames])
    ten = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0
    wav = clip.get("wav")
    if not rate:                    # a model with no sound (Vimoo: audio_rate 0): a silent MP4, no track
        write_preview_mp4(path_mp4, ten, None, fps=fps)
        return path_mp4
    wav_path = os.path.join(tempfile.gettempdir(), f"fizgig_studio_{os.getpid()}.wav")
    write_wav(wav_path, wav if wav is not None else torch.zeros(2, int(rate * len(frames) / fps)), sample_rate=rate)
    write_preview_mp4(path_mp4, ten, wav_path, fps=fps)
    return path_mp4


def _clip_same_as_baseline(eng, st):
    """The tweaked clip would be the baseline (the picture engines' same_as_baseline rule): every primary block on at
    1.0, and no donor contributing - none loaded, its strength 0, or each of its blocks off or at 0. The donor sliders
    count only when a donor is loaded: the page starts them at 1, so judging them without one never matched."""
    if any(not bs.primary_enabled or float(bs.primary_strength) != 1.0 for bs in st.blocks.values()):
        return False
    if getattr(eng, "donor_network", None) is None and not getattr(eng, "donor_path", None):
        return True
    if float(getattr(st, "donor_scale", 1.0)) == 0.0:
        return True
    donors = getattr(eng, "donor_block_ids", None) or set(st.blocks)
    return not any(bs.donor_enabled and float(bs.donor_strength) != 0.0 for b, bs in st.blocks.items() if b in donors)


def _render_clips(eng, desc, st, c, out, gen):
    """H3: the baseline and tweaked clips, as the desktop renders them (baseline_clip, then render_clip)."""
    v = c.get("video") or {}
    frames = int(v.get("frames") or 22)
    width, height = int(v.get("width") or 768), int(v.get("height") or 640)
    st.preview_width, st.preview_height, st.preview_frames = width, height, frames
    st.keyframes, st.references = _cond(eng, c, width, height, frames)
    eng.speed_boost = bool(v.get("speed_boost"))      # the Studio's H3 tick: the first half of the steps at half size
    eng.clip_negative = v.get("negative")             # the page's Negative (used when the render runs with CFG)
    opts = dict(frames=frames, regime="custom", steps=int(v.get("steps") or 4),
                turbo_strength=float(v.get("turbo") if v.get("turbo") not in (None, "") else 1.0),
                with_audio=bool(v.get("sound", True)))
    spec = getattr(desc, "clip_spec", None)
    fps, rate = (spec.fps, spec.audio_rate) if spec is not None else (24, 32000)
    import torch
    torch.cuda.reset_peak_memory_stats()
    rest = torch.cuda.memory_allocated() / 2**30
    base = eng.baseline_clip(st, cache=None, **opts)
    if _clip_same_as_baseline(eng, st):
        # nothing moved: the tweaked clip IS the baseline - one render for both panes, as the picture path does
        tweak = base
    else:
        tweak = None
    early = {}
    if v.get("early") and opts["steps"] > 2:     # Show early: the pass-2 estimate while the rest runs (the desktop's)
        def on_early(img, step, n):
            if img is None:
                emit(ev="status", text=f"Show early skipped this time: not enough free VRAM on the card for its decode "
                                       f"(~{getattr(eng, 'early_skipped_gb', 5):.0f} GB) - the clip renders on.")
                return
            p = os.path.join(out, f"early_{gen}.png")
            img.save(p)
            emit(ev="early", gen=gen, png=p, step=step, n=n)
        early = dict(early_step=2, on_early=on_early)
    paths = {}
    png = os.path.join(out, f"baseline_{gen}.png")
    paths["baseline"], paths["baseline_media"] = png, _write_clip(base, png, os.path.join(out, f"baseline_{gen}.mp4"),
                                                                  fps, rate)
    if tweak is None:                                   # the Baseline plays while the Tweaked clip renders
        emit(ev="baseline_ready", gen=gen, baseline=paths["baseline"], baseline_media=paths["baseline_media"])
        tweak = eng.render_clip(st, cache=None, **opts, **early)
    print(f"[studio] clip {width}x{height} x{frames}, {len(st.keyframes or [])} keyframe(s), "
          f"{len(st.references or [])} reference(s): peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GB allocated, "
          f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GB reserved, from {rest:.2f} GB at rest "
          f"(base {getattr(eng, 'base_plan', None)})", flush=True)
    png = os.path.join(out, f"tweaked_{gen}.png")
    paths["tweaked"], paths["tweaked_media"] = png, _write_clip(tweak, png, os.path.join(out, f"tweaked_{gen}.mp4"),
                                                                fps, rate)

    # hand the render's cached memory back (the model stays loaded): the page plays these clips next, and a card the
    # worker leaves full - Vimoo bf16 + the decode's cache held 30.5 of 32 GB - gives the video decode and the desktop
    # compositor nothing, which with Windows' DWM over-committed froze the whole PC on play (6 Nov 2026)
    torch.cuda.empty_cache()
    return paths, None                  # no No-LoRA clip: a third H3 render was too slow (ACE-Step's song keeps one)


def _apply_mix(eng, mix):
    """The primary's Epochs of this run / LoRA mix and the donor's epoch blend onto the engine, as the desktop's
    _repair_blend_moved / _repair_mix_push; the "Mix: ..." lines for the page."""
    from fizgig.families.workbench import DONOR, PRIMARY, ema_weights
    if not hasattr(eng, "set_blend"):
        return {}
    notes, mix = {}, mix or {}
    mem = S.setdefault("mix_files", {})
    for role, key in (("primary", PRIMARY), ("donor", DONOR)):
        net = eng.primary_network if role == "primary" else eng.donor_network
        m = mix.get(role)
        path = eng.primary_path if role == "primary" else eng.donor_path
        if net is None:
            mem.pop(role, None)
            continue
        if not m:
            if mem.get(role):
                eng.set_blend(None, role=key)
                mem.pop(role, None)
            continue
        if m.get("mode") == "epochs":
            if mem.get(role, (None,))[0] != ("epochs", path):
                mem[role] = (("epochs", path), eng.find_epochs(role=key))
            w = ema_weights(mem[role][1], int(m["upto"]), float(m["hist"]))
            eng.set_blend(w, role=key)
            top = sorted(w.items(), key=lambda kv: -kv[0])
            notes["mix_" + role] = ("Mix: " + " \u00b7 ".join(f"e{e} {v * 100:.0f}%" for e, v in top[:6])
                                    + (f" \u00b7 +{len(top) - 6} more" if len(top) > 6 else ""))
        else:
            paths = tuple(m.get("paths") or ())
            amounts = [round(float(a) / 0.05) * 0.05 for a in (m.get("amounts") or ())]
            if mem.get(role, (None,))[0] != ("mix", paths):
                eng.set_sources(dict(enumerate(paths)), role=key)
                mem[role] = (("mix", paths), None)
            eng.set_blend({i: a for i, a in enumerate(amounts) if a}, role=key)
            names = [os.path.splitext(os.path.basename(p))[0][:24] for p in paths]
            notes["mix_" + role] = (f"Mix: {' · '.join(f'{n} {a:.2f}' for n, a in zip(names, amounts))}  "
                                    f"(total {sum(amounts):.2f})" if len(paths) > 1 else
                                    "Add LoRAs to mix with the primary; each gets its own amount.")
    return notes


def _audio_inputs(eng, a, seconds):
    """Prompt B, the song to start from and the reference recording onto the engine (it reads them each render), as
    the desktop's _repair_pb_push / _repair_src_effective; returns the readouts beside its sliders."""
    pb = a.get("prompt_b") or None
    if hasattr(eng, "prompt_b"):
        eng.prompt_b, eng.prompt_blend = pb, (float(a.get("blend") or 0) if pb else 0.0)
    src = a.get("source") or {}
    path, mode = src.get("path") or None, src.get("mode") or "song"
    d, f, s = float(src.get("denoise", 1.0)), float(src.get("follow", 1.0)), float(src.get("strength", 0.5))
    shown_d, shown_f = d, f
    rp, dur = None, None
    live = bool(path and os.path.isfile(path))
    if live and mode != "song":
        try:
            from fizgig.acestep.driver import audio_seconds
            dur = float(audio_seconds(path))
        except Exception:
            dur = None
    if live and mode == "repaint":
        r = src.get("region")
        rp = (round(float(r[0]), 2), round(float(r[1]), 2)) if r else None
    elif live and mode == "extend":
        rp = (round(dur, 2), float(seconds)) if dur is not None and dur < seconds - 0.25 else None
    if mode != "song":
        d, f = 1.0, 0.0
        live = live and rp is not None
    elif d >= 1.0 and f <= 0.0:
        live = False
    if not live:
        path, d, f, rp = None, 1.0, 0.0, None
    ref = a.get("reference") or None
    if not (ref and os.path.isfile(ref)):
        ref = None
    if hasattr(eng, "source_path"):
        eng.source_path, eng.source_denoise, eng.source_follow = path, d, f
        eng.source_repaint, eng.repaint_strength = rp, s
        eng.reference_path = ref
    notes = {"strength": f"{s:.2f}"}
    if rp is not None:
        notes["strength"] += f" - remakes {rp[0]:.1f}\u2013{rp[1]:.1f} s, keeps the rest"
    elif mode == "repaint":
        notes["strength"] += " - drag across the Tweaked waveform to choose the section"
    elif mode == "extend":
        notes["strength"] += (" - pick a song" if dur is None else f" - set Length above the song's {dur:.0f} s to extend it")
    txt, ftxt = f"{shown_d:.2f}", f"{shown_f:.2f}"
    drv = getattr(eng, "driver", None)
    if drv is not None and hasattr(drv, "schedule") and hasattr(drv, "source_start"):
        try:
            ts = drv.schedule(drv.description.preview_steps)
            t0 = drv.source_start(ts, shown_d) if shown_d < 1.0 else ts[0]
            n = sum(1 for t in ts if t0 is not None and t <= t0 + 1e-9)
            txt += ((" - the source as it is" if t0 is None else f" - starts at {t0:.2f}, {n} of {len(ts)} steps")
                    if shown_d < 1.0 else " - starts from noise")
            ftxt += " - off" if shown_f <= 0.0 else f" - {int(n * shown_f)} of {n} steps follow"
        except Exception:
            pass
    notes.update(denoise=txt, follow=ftxt)
    return notes


def _write_activity(act, path, most=1500):
    """A song's block activity (description block_activity: {block id: push along the song}) as JSON for the page:
    each block's curve averaged down to `most` points, and the 98th percentile of every push - the glow's full scale,
    as the desktop scales it. "" without one."""
    if not act:
        return ""
    import numpy as np
    blocks = {}
    for bid, v in act.items():
        v = np.asarray(v, dtype=np.float32).ravel()
        if not len(v):
            continue
        k = -(-len(v) // most)
        if k > 1:
            v = np.pad(v, (0, k * (-(-len(v) // k)) - len(v)), mode="edge").reshape(-1, k).mean(1)
        blocks[str(bid)] = [round(float(x), 5) for x in v]
    if not blocks:
        return ""
    top = max(1e-6, float(np.percentile(np.concatenate([np.asarray(v) for v in blocks.values()]), 98)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"top": top, "blocks": blocks}, f)
    return path


def _render_songs(eng, st, c, out, gen):
    """ACE-Step: the baseline and tweaked songs (the length is the Res box, in seconds) as WAVs with their
    waveform pictures."""
    a = c.get("audio") or {}
    seconds = int(a.get("seconds") or st.preview_width or 30)
    try:
        eng.driver.set_options({"preview_seconds": seconds})
    except Exception:
        pass
    S["notes"].update(_audio_inputs(eng, a, seconds))
    base = eng.generate_baseline(st)
    tweak = base if eng.same_as_baseline(st) else eng.generate_preview(st)     # nothing moved: one render
    from fizgig.minimax.common import write_wav
    paths = {}
    for name, song in (("baseline", base), ("tweaked", tweak)):
        png = os.path.join(out, f"{name}_{gen}.png")
        wav = os.path.join(out, f"{name}_{gen}.wav")
        song.picture().save(png)
        write_wav(wav, song.wave, sample_rate=48000)
        paths[name], paths[name + "_media"] = png, wav
        paths[name + "_act"] = _write_activity(getattr(song, "activity", None), os.path.join(out, f"{name}_{gen}.json"))
    key = (eng.primary_path, st.seed, st.prompt, seconds,
           eng._prompt_b_key() if hasattr(eng, "_prompt_b_key") else None,
           eng._audio_key() if hasattr(eng, "_audio_key") else None)

    def nolora():
        hit = S.get("nolora_song")
        if hit and hit[0] == key and os.path.isfile(hit[1]["nolora"]):
            return hit[1]
        import copy
        base_st = copy.deepcopy(st)
        base_st.primary_scale, base_st.donor_scale = 0.0, 0.0
        song = eng.generate_preview(base_st)
        png, wav = os.path.join(out, f"nolora_{gen}.png"), os.path.join(out, f"nolora_{gen}.wav")
        song.picture().save(png)
        write_wav(wav, song.wave, sample_rate=48000)
        got = {"nolora": png, "nolora_media": wav}
        S["nolora_song"] = (key, got)
        return got
    return paths, nolora


def _tidy(out, gen):
    """Drop renders two or more generations old (the page shows the newest), keeping the No-LoRA files in use."""
    import re
    keep = {os.path.normcase(p) for k in ("nolora_out", "nolora_song") if S.get(k) for p in S[k][1].values() if p}
    try:
        for f in os.listdir(out):
            m = re.match(r"(baseline|tweaked|nolora)_(\d+)\.(png|mp4|wav|json)$", f)
            p = os.path.join(out, f)
            if m and int(m.group(2)) < gen - 1 and os.path.normcase(p) not in keep:
                os.remove(p)
    except OSError:
        pass


def cmd_render(c):
    t = S["render"]
    if t is not None and t.is_alive():         # the server sends one at a time; a stray one waits its turn
        t.join()
    S["render"] = threading.Thread(target=_render, args=(c,), daemon=True)
    S["render"].start()


def cmd_cancel(_c):
    eng = S["engine"]
    if eng is not None:
        eng.request_cancel()


def _plan_for(desc, plan):
    """A load plan as the engine takes it: the preview checkpoint's sampling, Auto block swap resolved."""
    plan = dict(plan)
    if plan.pop("use_checkpoint_sampling", False):
        ck = desc.preview_checkpoint()
        plan["preview_sampling"] = ck[1] if ck else None
    else:
        plan["preview_sampling"] = None
    if str(plan.get("blocks_to_swap", "")).lower().startswith("auto"):
        plan["blocks_to_swap"] = _auto_swap()
    return plan


def _engine_for(c, status_ev="royale_status"):
    """LoRA Royale's (and RefMod Toolkit's) engine: the family's preview model loaded with this plan (the one already
    here when it matches - a Studio session's or the last render's), no LoRA chosen yet. Under the engine lock."""
    from fizgig.families.registry import get as get_family
    desc = get_family(c["family"])
    if desc is None:
        raise ValueError(f"Unknown model: {c['family']}")
    plan = _plan_for(desc, c["plan"])
    sig = json.dumps(c["plan"], sort_keys=True, default=str)
    eng = S["engine"]
    if eng is not None and (getattr(eng.desc, "key", None) != desc.key or S.get("plan_sig") != sig):
        eng.reset()                              # another model, or this one on another plan: load it fresh
        eng = None
    if eng is None:
        eng = desc.make_workbench_engine()
    S["engine"], S["desc"], S["plan_sig"] = eng, desc, sig
    if getattr(desc, "video_workbench", False):
        eng.int8_attention = bool(getattr(desc, "int8_attention", False))
        eng.on_status = lambda text: emit(ev=status_ev, text=str(text))
        if c.get("canvas"):
            eng.plan_canvas = tuple(c["canvas"])  # H3 Auto: room for the clip about to render
    if c.get("settings") is not None:
        eng.preview_settings = c["settings"]
    eng.on_step = lambda done, total: emit(ev="step", done=done, total=total)
    eng.ensure_pipeline(**plan)                   # nothing to do when it is loaded already
    return eng, desc


def _point_at(eng, desc, path):
    """The engine on this LoRA (the desktop's _royale_load_or_swap_primary): its weights swapped in place when the
    shapes match (same-rank epochs), else a fresh load. Refuses a LoRA of another model first (issue #62)."""
    from fizgig.networks.lora import assert_lora_family_matches
    assert_lora_family_matches(path, desc.shares_prefs_with or desc.key, "Royale")
    if eng.primary_network is None:
        eng.load_primary(path)
    elif eng.primary_path != path and not eng.swap_primary_weights(path):
        eng.reset()
        eng.ensure_pipeline(**_plan_for(desc, json.loads(S["plan_sig"])))
        eng.load_primary(path)


def cmd_royale(c):
    """LoRA Royale: every chosen epoch rendered on one prompt and seed, each saved as a PNG and reported as it lands."""
    gen = c["gen"]
    out = c["out_dir"]
    os.makedirs(out, exist_ok=True)
    done, cancelled = 0, False
    try:
        with _engine_lock:
            emit(ev="royale_status", text="Loading the model…")
            eng, desc = _engine_for(c)
            eng.clear_cancel()
            items = c["items"]
            for i, (label, path) in enumerate(items):
                emit(ev="royale_status", text=f"Rendering {label if not str(label).isdigit() else 'epoch ' + str(label)} "
                                              f"({i + 1}/{len(items)})…")
                try:
                    _point_at(eng, desc, path)
                    st = eng.default_state()
                    st.prompt, st.seed = c["prompt"], int(c["seed"])
                    st.preview_width, st.preview_height = int(c["width"]), int(c["height"])
                    if c.get("ref") and os.path.exists(c["ref"]):
                        st.ref_image_path, st.ref_megapixels = c["ref"], 0.2
                        st.ref_strength = float(c.get("ref_strength", 1.0))
                    img = eng.generate_preview(st)
                except Exception as e:
                    if _aborted(e):
                        cancelled = True
                        break
                    emit(ev="royale_item", gen=gen, i=i, label=label, path=path, png="",
                         error=str(e), trace=traceback.format_exc()[-1500:])
                    continue
                png = os.path.join(out, f"royale_{gen}_{i}.png")
                img.save(png)
                done += 1
                emit(ev="royale_item", gen=gen, i=i, label=label, path=path, png=png)
            try:
                from fizgig.utils.device import clean_memory_on_device
                clean_memory_on_device(eng.pipeline.device)   # the render's working set back to the driver
            except Exception:
                pass
    except Exception as e:
        emit(ev="royale_done", gen=gen, done=done, cancelled=False, error=str(e), trace=traceback.format_exc()[-2000:])
        return
    emit(ev="royale_done", gen=gen, done=done, cancelled=cancelled)


def _travel_ref(st, p, i):
    """A travel frame's reference(s) on its state; returns the strength for the PREVIOUS frame's latent (0 = none) -
    the desktop's _royale_apply_travel_ref. Non-sequential: the chosen reference on every frame. Sequential: frame 0
    the chosen reference, later frames edit the previous frame through its clean latent; with Anchor to original the
    original stays on as a clean image reference too."""
    st.ref_megapixels = p.get("ref_mp", 0.2)
    st.ref2_path = ""
    orig = p.get("ref", "")
    orig = orig if (orig and os.path.exists(orig)) else ""
    if p.get("sequential") and i > 0:
        if p.get("anchor"):
            st.ref_image_path, st.ref_strength = orig, p.get("anchor_str", 1.0)
        else:
            st.ref_image_path, st.ref_strength = "", 0.0
        return float(p.get("ref_strength", 1.0))
    st.ref_image_path, st.ref_strength = orig, p.get("ref_strength", 1.0)
    return 0.0


def _strip_trigger(prompt, trigger):
    """The trigger dropped for the comparison sheet's no-LoRA column (the base model has never seen it)."""
    import re
    trigger = (trigger or "").strip()
    if not trigger:
        return prompt
    parts = [p for p in (seg.strip() for seg in prompt.split(",")) if p and p.lower() != trigger.lower()]
    out = ", ".join(parts)
    if out.lower() == prompt.strip().lower():
        out = re.sub(rf"\b{re.escape(trigger)}\b", "", prompt, flags=re.IGNORECASE)
        out = re.sub(r"\s{2,}", " ", out).strip(" ,")
    return out or prompt                # a row that is only the trigger keeps it (an empty prompt is no picture)


def cmd_royale_travel(c):
    """LoRA Royale's travel modes and comparison sheet (the desktop's workers): "seed" (slerp between seeds along a
    journey), "strength" (the LoRA ramped), "prompt" (the text embedding interpolated through waypoints) - each frame
    saved and reported - and "sheet" (prompts x columns, one labelled picture)."""
    gen, mode, p = c["gen"], c["mode"], c
    out = c["out_dir"]
    os.makedirs(out, exist_ok=True)
    n_done, cancelled = 0, False

    def frame(i, img, label):
        nonlocal n_done
        png = os.path.join(out, f"travel_{mode}_{gen}_{i}.png")
        img.save(png)
        n_done += 1
        emit(ev="travel_frame", gen=gen, mode=mode, i=i, png=png, label=label)
    try:
        with _engine_lock:
            emit(ev="travel_status", gen=gen, mode=mode, text="Loading the model…")
            eng, desc = _engine_for(c)
            eng.clear_cancel()

            def say(t):
                emit(ev="travel_status", gen=gen, mode=mode, text=t)
            try:
                if mode == "sheet":
                    from fizgig.lora_royale.export import build_comparison_grid
                    cols, prompts = p["cols"], p["prompts"]
                    grid = [[None] * len(cols) for _ in prompts]
                    total, k = len(cols) * len(prompts), 0
                    for ci, (clabel, cpath) in enumerate(cols):       # column-major: each LoRA loads once
                        base_col = p["sheet_mode"] != "Every epoch" and ci == 0
                        _point_at(eng, desc, cpath)
                        for ri, prompt in enumerate(prompts):
                            k += 1
                            say(f"Rendering {k}/{total}…")
                            st = eng.default_state()
                            st.prompt = _strip_trigger(prompt, p.get("trigger")) if base_col else prompt
                            st.seed = int(p["seed"])
                            st.preview_width, st.preview_height = int(p["width"]), int(p["height"])
                            for bs in st.blocks.values():
                                bs.primary_strength = 0.0 if base_col else 1.0
                            grid[ri][ci] = eng.generate_preview(st).copy()
                    if p["sheet_mode"] == "Every epoch":
                        headers = [f"epoch {l}" if str(l).isdigit() else str(l) for l, _ in cols]
                    else:
                        headers = ["no lora", f"{p.get('trigger') or 'LoRA'} lora"]
                    sheet = build_comparison_grid(grid, headers, row_labels=prompts if p.get("row_labels") else None,
                                                  cell=(int(p["width"]), int(p["height"])), brand=bool(p.get("brand")))
                    png = os.path.join(out, f"travel_sheet_{gen}.png")
                    sheet.save(png)
                    emit(ev="travel_done", gen=gen, mode=mode, n=1, sheet=png, size=[sheet.width, sheet.height])
                    return
                _point_at(eng, desc, p["path"])
                n = max(2, int(p["frames"]))
                prev = None
                if mode == "prompt":
                    from fizgig.lora_royale import prompt_travel as pt
                    say("Encoding prompts…")
                    ctx_list, neg = eng.encode_travel_prompts(pt.build_waypoint_prompts(p["base"], p["words"]))
                    interp = {"Linear": "lerp", "Norm-preserved": "norm", "Slerp": "slerp"}.get(p.get("interp"), "lerp")
                    drift = float(p.get("drift") or 0.0)
                    drift_seed = (int(p["seed"]) + 1013904223) % (2 ** 31) if drift > 0 else None
                if mode == "seed":
                    from fizgig.lora_royale import journey_seeds
                    seeds = journey_seeds(p["seed_a"], p["seed_b"], p.get("waypoints", 2))
                    nseg = len(seeds) - 1
                for i in range(n):
                    t = i / float(n - 1)
                    say(f"Rendering frame {i + 1}/{n}…")
                    st = eng.default_state()
                    st.preview_width, st.preview_height = int(p["width"]), int(p["height"])
                    if mode == "strength":
                        s = float(p["s_start"]) + (float(p["s_end"]) - float(p["s_start"])) * t
                        st.prompt, st.seed = p["prompt"], int(p["seed"])
                        for bs in st.blocks.values():                 # one strength on every block
                            bs.primary_strength = s
                        frame(i, eng.generate_preview(st), f"{s:.2f}×")
                        continue
                    if mode == "seed":
                        pos = t * nseg
                        si = min(int(pos), nseg - 1)
                        st.prompt, st.seed = p["prompt"], seeds[si]
                        pls = _travel_ref(st, p, i)
                        img = eng.generate_preview(st, seed_b=seeds[si + 1], travel_t=pos - si, prev_latent=prev,
                                                   prev_latent_strength=pls)
                        label = p["label_disp"].upper()
                    else:
                        ctx = eng.interp_waypoints(ctx_list, t, mode=interp)
                        st.prompt = p["base"]
                        st.seed = (int(p["seed"]) + i) if p.get("vary_seed") else int(p["seed"])
                        pls = _travel_ref(st, p, i)
                        kw = dict(override_ctx=ctx, override_neg_ctx=neg, prev_latent=prev, prev_latent_strength=pls)
                        if drift > 0 and not p.get("vary_seed"):
                            kw.update(seed_b=drift_seed, travel_t=drift * t)
                        img = eng.generate_preview(st, **kw)
                        label = pt.dominant_word(p["words"], t).upper()
                    frame(i, img, label)
                    if p.get("sequential"):
                        prev = eng._last_frame_latent
            except Exception as e:
                if not _aborted(e):
                    raise
                cancelled = True
            finally:
                try:
                    from fizgig.utils.device import clean_memory_on_device
                    clean_memory_on_device(eng.pipeline.device)
                except Exception:
                    pass
    except Exception as e:
        emit(ev="travel_done", gen=gen, mode=mode, n=n_done, error=str(e), trace=traceback.format_exc()[-2000:])
        return
    emit(ev="travel_done", gen=gen, mode=mode, n=n_done, cancelled=cancelled)


def _rm_rows(rows):
    """The page's Mods rows as refmod_apply.ModRows, each mod's latent read once (in the file's own dtype, the pack
    mixes in it) and kept while the file is unchanged."""
    from safetensors import safe_open
    from fizgig.minimax import refmod_apply as ra
    cache = S.setdefault("rm_latents", {})
    out = []
    for r in rows:
        m = r["meta"]
        key = (m["path"], str(m.get("tensor_key", "latent")), os.path.getmtime(m["path"]))
        z = cache.get(key)
        if z is None:
            with safe_open(m["path"], framework="pt", device="cpu") as f:
                z = f.get_tensor(str(m.get("tensor_key", "latent"))).clone()
            cache[key] = z
        out.append(ra.ModRow(z, m, value=float(r.get("value", 1.0)), copies=int(r.get("copies", 1)),
                             enabled=bool(r.get("on", True)), name=str(m.get("name", ""))))
    return out


def _rm_bundle(job):
    """(latents, describe, entries, step schedule) for one RefMod render, as the desktop's _rms_job builds them: the
    numbered-references path takes the Text Encode node's route (loader order, every copy, row strengths only)."""
    from fizgig.minimax import refmod_apply as ra
    rows = _rm_rows(job["rows"])
    if job.get("numbered"):
        latents, describe, entries = ra.build_bundle_entries(rows, retention=1.0, curve=None, scramble_seed=-1)
        return latents, describe, entries, None
    fc = tuple(job["frame_curve"])
    latents, describe, entries = ra.build_bundle_entries(rows, retention=float(job["retention"]), curve=fc,
                                                         scramble_seed=int(job.get("scramble", -1)))
    sched = ra.step_schedule(tuple(job["step_curve"]), latents) if job.get("step_on") else None
    return latents, describe, entries, sched


def _rm_sync_lora(eng, path):
    """The engine's LoRA is the page's (swapped in place when the structure matches, else the old one off and the
    new one wired), or none - the desktop's _rms_sync_lora."""
    cur = eng.primary_path if getattr(eng, "primary_network", None) is not None else None
    if not path:
        if cur:
            eng.unload_primary()
        return
    if cur and os.path.normcase(cur) == os.path.normcase(path):
        return
    emit(ev="rm_status", text=f"Loading the LoRA {os.path.basename(path)}…")
    if cur and eng.swap_primary_weights(path):
        return
    if cur:
        eng.unload_primary()
    eng.load_primary(path)


def cmd_refmod(c):
    """RefMod Toolkit: kind "render" = With mods, then (need_base) the comparison render; kind "sweep" = one short clip
    per job. Each lands as a PNG (+ MP4 for a clip) and an rm_item event."""
    gen, out, kind = c["gen"], c["out_dir"], c.get("kind", "render")
    os.makedirs(out, exist_ok=True)
    done, cancelled = 0, False
    try:
        with _engine_lock:
            emit(ev="rm_status", text="Loading the model (the base takes a minute)…")
            eng, desc = _engine_for(c, status_ev="rm_status")
            eng.clear_cancel()
            spec = getattr(desc, "clip_spec", None)
            fps, rate = (spec.fps, spec.audio_rate) if spec is not None else (24, 32000)
            _rm_sync_lora(eng, c.get("lora") or "")
            n = len(c["jobs"])
            for i, job in enumerate(c["jobs"]):
                latents, describe, entries, sched = _rm_bundle(job)
                items = None
                if job.get("numbered") and latents:
                    emit(ev="rm_status", text="Decoding the mods for the numbered references…")
                    items = eng.reference_items(latents, entries)
                kw = dict(seed=int(job["seed"]), prompt=job["prompt"], width=int(job["width"]),
                          height=int(job["height"]), frames=int(job["frames"]), regime="custom",
                          with_audio=bool(job.get("sound")), steps=int(job["steps"]),
                          turbo_strength=float(job["turbo"]))

                def early(img, step, n_, gen=gen):
                    if img is None:                  # Show early skipped (not enough free VRAM for its decode)
                        return
                    p = os.path.join(out, f"rm_{gen}_early.png")
                    img.save(p)
                    emit(ev="rm_early", gen=gen, png=p, step=step, n=n_)
                if kind == "render" and job.get("early") and int(job["steps"]) > 2:
                    kw.update(early_step=2, on_early=early)
                emit(ev="rm_status", text=("Rendering With mods…" if kind == "render" else
                                           f"Sweep: {job.get('label', '')} ({i + 1} of {n})…"))
                clip = eng.render_refmod(ref_latents=latents or None, ref_schedule=sched, ref_items=items, **kw)
                side = "tweaked" if kind == "render" else "chip"
                png = os.path.join(out, f"rm_{gen}_{side}_{i}.png")
                mp4 = _write_clip(clip, png, os.path.join(out, f"rm_{gen}_{side}_{i}.mp4"), fps, rate)
                done += 1
                emit(ev="rm_item", gen=gen, side=side, i=i, png=png, mp4=mp4, label=job.get("label", ""),
                     describe=[[nm, float(s)] for nm, s in describe], steps=clip.get("steps"),
                     turbo=clip.get("turbo_strength"), frames=int(job["frames"]))
                if kind == "render" and c.get("need_base"):
                    cmp_ = c.get("compare") or "nomod"
                    emit(ev="rm_status", text="With mods done - rendering the comparison…")
                    keep = cmp_ == "nolora"
                    kw.pop("early_step", None)
                    kw.pop("on_early", None)
                    b = eng.render_refmod(ref_latents=(latents or None) if keep else None,
                                          ref_schedule=sched if keep else None, ref_items=items if keep else None,
                                          no_lora=cmp_ in ("nolora", "neither"), **kw)
                    png = os.path.join(out, f"rm_{gen}_baseline_0.png")
                    mp4 = _write_clip(b, png, os.path.join(out, f"rm_{gen}_baseline_0.mp4"), fps, rate)
                    emit(ev="rm_item", gen=gen, side="baseline", i=0, png=png, mp4=mp4, label="", describe=[],
                         steps=b.get("steps"), turbo=b.get("turbo_strength"), frames=int(job["frames"]))
            try:
                from fizgig.utils.device import clean_memory_on_device
                clean_memory_on_device(eng.pipeline.device)
            except Exception:
                pass
    except Exception as e:
        if not _aborted(e):
            emit(ev="rm_done", gen=gen, done=done, cancelled=False, error=str(e) or type(e).__name__,
                 trace=traceback.format_exc()[-2000:])
            return
        cancelled = True
    emit(ev="rm_done", gen=gen, done=done, cancelled=cancelled)


def cmd_save(c):
    with _engine_lock:
        eng = S["engine"]
        if eng is None or eng.primary_network is None:
            raise ValueError("Load a LoRA first.")
        summary = eng.save_repaired(c["out"], _state(c["state"]), include_donor=bool(c.get("include_donor")))
    emit(ev="saved", out=c["out"], summary=summary)


def cmd_reset(_c):
    with _engine_lock:
        if S["engine"] is not None:
            S["engine"].reset()
        S["engine"] = S["desc"] = None
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass
    emit(ev="reset")


def cmd_map(c):
    """Studio's maps over the Tweaked render on screen (engine.lora_map): stills only."""
    gen = c.get("gen", 0)
    eng = S["engine"]
    try:
        with _engine_lock:
            if eng is None or eng.primary_network is None:
                raise ValueError("Load a LoRA first.")
            if getattr(S["desc"], "video_workbench", False) or getattr(S["desc"], "audio_workbench", False):
                raise ValueError("Maps are for pictures.")
            from PIL import Image
            with Image.open(c["picture"]) as im:
                pic = im.convert("RGB")
            emit(ev="status", text="Making the maps…")
            res = eng.lora_map(_state(c["state"]), pic, trigger=c.get("trigger") or "", blocks=bool(c.get("blocks")))
            import torch
            torch.cuda.empty_cache()
        emit(ev="map", gen=gen, rid=c.get("rid"), with_blocks=bool(c.get("blocks")), **res)
        emit(ev="status", text="Ready.")
    except Exception as e:
        emit(ev="map", gen=gen, rid=c.get("rid"), error=str(e) or type(e).__name__)
        emit(ev="status", text="Ready.")


HANDLERS = {"load": cmd_load, "donor": cmd_donor, "settings": cmd_settings, "render": cmd_render, "map": cmd_map,
            "cancel": cmd_cancel, "save": cmd_save, "reset": cmd_reset, "royale": cmd_royale,
            "royale_travel": cmd_royale_travel, "refmod": cmd_refmod}


def _run(c):
    try:
        HANDLERS[c["cmd"]](c)
    except Exception as e:
        emit(ev="error", where=c.get("cmd"), msg=str(e) or type(e).__name__, trace=traceback.format_exc()[-2000:])


def main():
    if os.environ.get("FIZGIG_STUDIO_DUMP"):          # diagnosing a hang: every thread's stack after N seconds
        import faulthandler
        faulthandler.dump_traceback_later(int(os.environ["FIZGIG_STUDIO_DUMP"]), exit=False, file=sys.stderr)
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.abspath(os.path.join(here, "..", "..")), os.path.abspath(os.path.join(here, "..", "..", ".."))):
        if p not in sys.path:
            sys.path.insert(0, p)
    # the heavy libraries load HERE, on the main thread, before it waits on stdin: loading a native library on a
    # command thread while the main thread sits in a blocking pipe read deadlocks on Windows (numpy hung for ever)
    emit(ev="status", text="Starting the Studio worker…")
    import numpy  # noqa: F401
    import torch  # noqa: F401
    import fizgig.families.workbench  # noqa: F401
    from fizgig.families.registry import training_families
    training_families()
    # commands come over a local socket, not stdin: a thread blocked reading stdin holds the C runtime's file lock,
    # and anything that loads a native library meanwhile (numpy, torch's kernels, PNG encode) waits on it for ever
    import socket
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    emit(ev="ready", port=srv.getsockname()[1])
    srv.settimeout(120)
    try:
        conn, _ = srv.accept()
    except OSError:
        return
    srv.close()
    for line in conn.makefile("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if c.get("cmd") == "quit":
            break
        if c.get("cmd") == "cancel":
            cmd_cancel(c)                        # heard at once, even while a load or render holds the engine
            continue
        # load / save / reset can take a while: their own thread, so a cancel still gets through
        threading.Thread(target=_run, args=(c,), daemon=True).start()
    cmd_reset({})


if __name__ == "__main__":
    main()
