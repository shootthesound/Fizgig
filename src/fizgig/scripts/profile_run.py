"""The web UI's Profiler run: the desktop Profiler's job in its own process - the weights report (instant), or with
Quick / Thorough the measured profile: the LoRA rendered whole, off, and with each block group off through the same
workbench engine as Studio (families/block_profile.py), with likeness and bleed from three subject photos when given.

Usage: python -m fizgig.scripts.profile_run <job.json>
The job: {family, lora, mode ("weights" | "quick" | "thorough"), prompt, class_prompt, baselines [3 paths], size,
plan (the engine's load arguments, fizgig.web.studio.plan's), settings (Samples-tab preview settings or null),
profiles_dir}. Progress leaves as "@@P {json}" lines, the end as "@@DONE {json}" (report, summary lines, the Studio
hand-off) or "@@FAIL {json}".
"""
import json
import os
import sys
import time
import traceback


def out(tag, **d):
    sys.stdout.write(f"@@{tag} " + json.dumps(d, default=str) + "\n")
    sys.stdout.flush()


def _auto_swap():
    try:
        import torch
        if torch.cuda.is_available():
            gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return 0 if gb >= 15 else 12 if gb >= 10 else 16
    except Exception:
        pass
    return 0


def _embedder(baselines):
    """(embed_path, embed_pil, label) on one face model: the largest face's embedding (the desktop's
    _ff_embed_cached / _repair_embed_pil). The look match is Image Prep's; with Auto it is decided from the subject
    photos (ArcFace for a person, DINOv2 for a furry / anime character), the same for every render."""
    from fizgig.face_utils import decide_look_match
    fe = decide_look_match(baselines).embedder()
    return fe.embed, fe.embed_image, fe.label


def main(job_path):
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    from fizgig.families.block_profile import (conditions, renders_needed, repair_settings, run_ablation, verdicts,
                                               weight_stats, write_report)
    from fizgig.families.registry import get as get_family
    desc = get_family(job["family"])
    lora, mode = job["lora"], job.get("mode", "quick")
    prompt, class_prompt = (job.get("prompt") or "").strip(), (job.get("class_prompt") or "").strip()
    size = int(job.get("size") or 768)
    seeds = (1234, 5678) if mode == "thorough" else (1234,)
    os.makedirs(job["profiles_dir"], exist_ok=True)
    stem = os.path.splitext(os.path.basename(lora))[0]
    report = os.path.join(job["profiles_dir"], f"{stem}_{desc.lora_name_suffix}_profile.html")
    out("P", text="Reading the LoRA's weights…")
    stats = weight_stats(desc, lora)
    labels = {b.id: b.label for g in desc.load_driver().block_map() for b in g.blocks}
    abl, log = None, []
    eng = None
    try:
        if mode != "weights":
            plan = dict(job["plan"])
            if plan.pop("use_checkpoint_sampling", False):
                ck = desc.preview_checkpoint()
                plan["preview_sampling"] = ck[1] if ck else None
            else:
                plan["preview_sampling"] = None
            if str(plan.get("blocks_to_swap", "")).lower().startswith("auto"):
                plan["blocks_to_swap"] = _auto_swap()
            out("P", text=f"Loading {desc.display_name}…")
            eng = desc.make_workbench_engine()
            if job.get("settings") is not None:
                eng.preview_settings = job["settings"]
            eng.ensure_pipeline(**plan)
            eng.load_primary(lora)
            embed, base_embs = None, []
            if job.get("baselines"):
                embed_path, embed_pil, embed_label = _embedder(job["baselines"])
                base_embs = [embed_path(p) for p in job["baselines"]]
                if any(e is None for e in base_embs):
                    log.append("No face found in one of the subject photos - likeness and bleed scores are off for "
                               "this run.")
                    base_embs = []
                else:
                    embed = embed_pil
            conds = dict((c, l) for c, l, _ in conditions(eng, mode == "thorough"))
            t0 = time.time()

            def prog(done, total, kind, cond):
                el = time.time() - t0
                eta = el / done * (total - done) if done else 0
                bc = " (bleed check)" if kind == "class" else ""
                out("P", done=done, total=total, text=f"Rendering {done}/{total}: {conds.get(cond, cond)}{bc} · about "
                                                     f"{int(eta // 60)}m {int(eta % 60)}s left")
            per_block = mode == "thorough"
            log.append(f"{len(seeds)} seed(s), {size}x{size}: "
                       f"{renders_needed(eng, len(seeds), bool(class_prompt), per_block)} renders.")
            out("P", text=log[-1])
            abl = run_ablation(eng, prompt=prompt, class_prompt=class_prompt, seeds=seeds, width=size, height=size,
                               embed=embed, baselines=base_embs, per_block=per_block, on_progress=prog)
            if embed is not None and isinstance(abl, dict):
                abl["embedder"] = embed_label          # the report names the embedder its likeness comes from
        html, sidecar = write_report(desc, lora, stats, abl, report, labels)
    finally:
        if eng is not None:
            try:
                eng.reset()
            except Exception:
                pass
    lines = log + [f"{desc.display_name} profile complete.",
                   f"Rank: {stats['max_rank']} in the file; 95% of the change fits in rank {stats['rank_for'][0.95]}, "
                   f"99% in rank {stats['rank_for'][0.99]}."]
    studio = None
    if abl is not None:
        vd = verdicts(abl)
        alone = abl["alone"]["trigger"]
        g = abl["gain"].get("trigger")
        if g is not None and g > 0.03:
            lines.append(f"Likeness the LoRA adds: +{100 * g:.0f} points.")
        gb = abl["gain"].get("class")
        if gb is not None:
            lines.append(f"Bleed into the plain prompt: {'+%.0f points' % (100 * gb) if gb > 0.03 else 'none measurable'}.")
        lines.append("Each group of blocks on its own (only that group switched on), as a share of what the whole LoRA "
                     "does:")
        cls = abl["alone"].get("class", {})
        for w, l, _bs in abl["windows"]:
            a = alone[w]
            parts = [f"changes the picture by {100 * a['change']:.0f}%"]
            if a.get("score") is not None:
                parts.append(f"gives {100 * max(a['score'], 0):.0f}% of the likeness")
            if cls.get(w, {}).get("score") is not None:
                parts.append(f"gives {100 * max(cls[w]['score'], 0):.0f}% of the bleed")
            lines.append(f"  {l}: {', '.join(parts)}. {vd[w][1]}.")
        settings, notes = repair_settings(abl)
        studio = {"family": desc.key, "lora": lora, "prompt": prompt, "class_prompt": class_prompt, "seed": seeds[0],
                  "size": size, "blocks": {b: list(v) for b, v in settings.items()}, "notes": list(notes)}
    out("DONE", report=html, sidecar=sidecar, lines=lines, studio=studio)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except BaseException as e:                   # noqa: B902 - the page shows why
        stopped = type(e).__name__ in ("RenderCancelled", "KeyboardInterrupt")
        out("FAIL", stopped=stopped, error="" if stopped else traceback.format_exc()[-3000:])
        sys.exit(1)
