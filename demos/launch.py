"""launch.py <n>: run demo n. Starts the language model and Fizgig when they aren't running, checks Affinity, sets up
the demo's starting document, sends the demo's sentence to Fizgig's Assistant tab, and opens it in the browser to
watch. The console shows each step too; a log goes to demos/logs."""
import json
import subprocess
import os
import sys
import time
import webbrowser

import kit

QUICK = ("with the Qwen 2.1 Fast Identity preset (20 epochs, save every 5, no previews)")

DEMOS = {
    1: ("Your photos in Affinity to a trained portrait",
        "Train a LoRA of me from the photos in my '{board}' document in Affinity. Make a training folder "
        "called 'Me from Affinity' from its layers, caption them with the trigger word {t}, then train a quick "
        "likeness LoRA called {t}_affinity " + QUICK + " and wait for it to finish. Then render me with the new "
        "LoRA as a professional magazine-cover studio portrait (start the prompt with '{t}, {who},') and put it into "
        "Affinity as a new document."),
    2: ("A transparent sticker straight into a design",
        "Make a sticker for my '{poster}' in Affinity: a small friendly red robot mascot holding a lightning "
        "bolt, flat vector style, with a transparent background so it sits on the poster as a cut-out. Use my "
        "{t}_affinity LoRA at strength 0 (the sticker isn't me). Place it in the top-right corner of the poster, "
        "about 240 pixels square, then show me the poster."),
    3: ("The model reads the layout and fills it",
        "Look at my '{poster}' in Affinity. Make a portrait of me ({t}, {who}) with my {t}_affinity LoRA for "
        "its empty Speaker photo box that suits the poster's style and colours, and put it in the box."),
    4: ("Four variations, laid out in Affinity",
        "Make me a social media pack with my {t}_affinity LoRA ({t}, {who}): four square professional photos of me - "
        "speaking on a conference stage, working at a desk with a laptop, a relaxed studio portrait, and talking "
        "with people at a tech expo booth. Lay them out as a 2x2 grid in a new 2048x2048 Affinity document called "
        "'Social pack'."),
    5: ("Trained on a rented cloud GPU, designed here",
        "My training folder 'Me from Affinity' is captioned with the trigger word {t} (it's me, {who}). Train a "
        "quick likeness of me called {t}_cloud on a rented cloud GPU - no more than $2 an hour - " + QUICK + ". "
        "Then make a professional keynote-speaker portrait of me for the empty photo box on my '{poster}' "
        "document in Affinity, matching the poster's colours, and put it in the box. Tell me what the "
        "cloud training cost."),
    6: ("A poster designed from nothing, the best portrait picked by scores",
        "Design me a keynote poster in Affinity from scratch, 1080x1350, called 'Keynote poster (designed by the "
        "assistant)'. It's for my talk 'Local AI, your way' at a tech conference; I'm {name}, {role}. Give it a bold "
        "modern look with a large photo frame for me. Then fill the frame with a professional keynote-speaker "
        "portrait of me, using my {t}_affinity LoRA (trigger {t}, {who}): render 4 at the frame's shape, score them "
        "against my 'Me from Affinity' training folder, and use the best one."),
}
FOLLOW_UP = {4: "Swap the laptop photo for one of me presenting at a whiteboard in a meeting room, in the same spot."}


def already_open(title):
    """The newest open document called title (or title 2, title 3 ... from an earlier run): its title, else ""."""
    import re
    from fizgig.bridge import affinity
    want = re.compile(re.escape(title.lower()) + r"( \d+)?$")
    hits = [d["title"].removesuffix(".af") for d in affinity.documents()
            if want.match(d["title"].lower().removesuffix(".af"))]
    return hits[-1] if hits else ""


def photo_board():
    """Demo 1's start: your photos as layers of an Affinity document (copies; your folder is never touched)."""
    from PIL import Image
    from fizgig.bridge import affinity
    if already_open("My photo board"):
        return already_open("My photo board")
    pics = sorted(os.path.join(kit.PHOTOS, f) for f in os.listdir(kit.PHOTOS)
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
    pics = [p for p in pics if min(Image.open(p).size) >= 480][:25]
    if len(pics) < 10:
        raise SystemExit(f"Put 15-25 photos of yourself (at least 480 px) in {kit.PHOTOS} first.")
    cols = 5
    rows = (len(pics) + cols - 1) // cols
    doc = affinity.new_document(cols * 512, rows * 512, "My photo board", background=[255, 255, 255])
    for i, p in enumerate(pics):
        print(f"  photo board: {i + 1}/{len(pics)}", end="\r", flush=True)
        affinity.place_image(p, doc["id"], name=f"photo {i + 1:02d}", box=[(i % cols) * 512, (i // cols) * 512, 512, 512])
    affinity.save(doc["id"])
    print()
    return doc["title"]


def poster(title, cloud=False):
    """Demos 2, 3 and 5 start from this keynote poster: background, type, an empty 'Speaker photo' box."""
    from fizgig.bridge import affinity
    if already_open(title):
        return already_open(title)
    c = kit.config()
    doc = affinity.new_document(1080, 1350, title)
    d = doc["id"]
    affinity.add_shape(d, [0, 0, 1080, 1350], [12, 14, 30], [120, 18, 32], 0, name="Background")
    white, red, grey = [255, 255, 255], [255, 90, 80], [210, 210, 220]
    affinity.add_text(d, "LOCAL AI,", 70, 150, 104, True, white, name="Headline 1")
    affinity.add_text(d, "YOUR WAY", 70, 260, 104, True, red, name="Headline 2")
    affinity.add_text(d, "Train in the cloud, render and design here" if cloud else
                      "Train, render and design on one machine", 72, 340, 34, False, [225, 225, 235], name="Subtitle")
    affinity.add_shape(d, [70, 420, 540, 760], [255, 255, 255, 18], outline=[255, 255, 255, 140], outline_width=3,
                       name="Speaker photo")
    affinity.add_text(d, "KEYNOTE", 660, 520, 40, True, red, name="Label")
    affinity.add_text(d, c.get("name") or "Your Name", 660, 580, 52, True, white, name="Speaker name")
    affinity.add_text(d, c.get("role") or "Creator", 660, 630, 30, False, grey, name="Speaker role")
    pitch = (["A model trained", "on your own photos,", "on a rented GPU,", "designed on your desk."] if cloud else
             ["A model trained", "on your own photos,", "in minutes,", "without the cloud."])
    for i, line in enumerate(pitch):
        affinity.add_text(d, line, 660, 760 + 40 * i, 30, False, grey, name=f"Pitch {i + 1}")
    affinity.add_text(d, "FIZGIG  +  AFFINITY  +  A LOCAL LLM" + ("  +  CLOUD GPU" if cloud else ""), 70, 1270, 28,
                      True, white, name="Footer")
    affinity.save(d)
    return doc["title"]


def has_lora(api, name):
    """Demo 1's LoRA exists: on this computer, or on the AMD pod when the demos run there."""
    import amd_pod
    if amd_pod.active():
        from fizgig.web.helpers import Client
        st = json.load(open(amd_pod.STATE))
        pod = Client(f"https://{st['pod']}-8081.proxy.runpod.net")
        pod.login(st["FIZGIG_PASSWORD"])
        r = pod.call("/api/runs")
    else:
        r = api("/api/runs")
    return any(x.get("lora") == name for x in (r if isinstance(r, list) else r.get("runs", [])))


def watch(api, since, log):
    """Print the turn's steps until the Assistant answers."""
    while True:
        r = api(f"/api/assistant?since={since}")
        for ev in r["events"]:
            since = max(since, ev["id"])
            ev.pop("srcs", None)
            log.append(ev)
            k, t = ev["kind"], ev.get("text") or ""
            if k == "tool":
                print(f"  -> {ev['name']}", flush=True)
            elif k in ("note", "answer", "gpu") or (k == "result" and ev.get("error")):
                print(f"  {'ERROR ' if ev.get('error') else ''}{t.strip()[:600]}", flush=True)
        if not r["busy"]:
            return since, r.get("stats") or {}
        time.sleep(1.5)


def stop():
    """Stop the demos' Fizgig and language model (whatever listens on their ports)."""
    for port in (kit.PORT, kit.LLM_PORT):
        subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue | "
                        "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"])
    print("Stopped Fizgig and the language model (Affinity stays open).")


def main():
    if sys.argv[1] == "stop":
        return stop()
    n = int(sys.argv[1])
    title, prompt = DEMOS[n]
    c = kit.config()
    if not c.get("trigger"):
        raise SystemExit("Run one of the '0 - Set up ...' launchers first.")
    import amd_pod
    on_amd = amd_pod.active()
    print(f"Demo {n}: {title}" + (" - on the rented AMD GPU" if on_amd else "") + "\n")
    if c.get("route") == "amd" and n == 5:
        raise SystemExit("Demo 5 trains on a rented NVIDIA GPU and renders on this computer's own GPU, so it's for the "
                         "NVIDIA setup. On the AMD GPU route, demos 1-4 and 6 cover the same work.")
    if c.get("route") == "amd" and not on_amd:
        raise SystemExit("On the AMD GPU route the demos run on the rented MI300X: run "
                         "'A - Run the demos on a rented AMD GPU (MI300X)' first.")
    if on_amd and n == 5:
        raise SystemExit("Demo 5 rents a cloud GPU to train on: with the demos already on the AMD pod, run it from "
                         "this computer ('B - Back to this computer' first).")
    if not on_amd:                                   # on the AMD pod the language model runs there
        kit.start_llm()
    kit.start_fizgig()
    api = kit.Fizgig()
    try:
        from fizgig.bridge import affinity
        affinity.status()
    except Exception:
        raise SystemExit("Affinity isn't answering: open Affinity by Canva (Edit > Settings > Model Context "
                         "Protocol: 'Enable Affinity MCP' and 'Access files on your Desktop' on).")
    t = c["trigger"]
    names = {}
    if n == 1:
        names["board"] = photo_board()
    else:
        if not has_lora(api, f"{t}_affinity"):
            raise SystemExit("Run demo 1 first: it trains the LoRA the other demos use.")
        if n in (2, 3):
            names["poster"] = poster("Keynote poster")
        if n == 5:
            names["poster"] = poster("Keynote poster (cloud)", cloud=True)
    text = prompt.format(t=t, who=c.get("who", "a man"), name=c.get("name", ""), role=c.get("role", ""), **names)
    api("/api/assistant/reset", {})
    since = 0
    webbrowser.open(kit.FIZGIG + "/?tab=assistant")
    log = []
    t0 = time.time()
    for msg in [text] + ([FOLLOW_UP[n]] if n in FOLLOW_UP else []):
        print(f"You: {msg}\n")
        api("/api/assistant/send", {"text": msg})
        time.sleep(1)
        since, stats = watch(api, since, log)
    took = time.time() - t0
    print(f"\nDone in {took // 60:.0f} min {took % 60:.0f} s.")
    os.makedirs(kit.LOGS, exist_ok=True)
    with open(os.path.join(kit.LOGS, f"demo {n} {time.strftime('%Y%m%d-%H%M%S')}.json"), "w", encoding="utf-8") as f:
        json.dump({"demo": n, "prompt": text, "seconds": round(took), "stats": stats, "events": log}, f, indent=1)


if __name__ == "__main__":
    main()
