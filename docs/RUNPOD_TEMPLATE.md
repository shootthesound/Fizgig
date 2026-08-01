# RunPod template README

Paste the section below into the **README** tab of the RunPod template.

Two rules if you edit it later. **This is read before deploying**, so the things that cost money or
lose data go above the fold and everything else links out — the full docs are one click away and do
not need repeating here. And it is **not a minimum-spec pitch**: "Krea 2 fits in 8 GB" belongs in
the project README, where someone is deciding whether Fizgig runs on hardware they already own.
Here they are choosing what to rent, and the appeal is getting a bigger card than they have.

---

# Fizgig — Klein 9B & Krea 2 LoRA Studio

Train, profile, repair and extract LoRAs for **Flux 2 Klein 9B** and **Krea 2** — the full desktop
app in your browser, on whatever GPU you feel like renting.

## Storage

**Give it 100 GB+, mounted at `/workspace`.** The models are ~45 GB, plus your datasets and the
LoRAs you train.

The default **Volume Disk** is fine. Just know that **stopping a pod keeps everything, terminating
it does not** — so stop between sessions and your models, datasets and LoRAs are waiting when you
come back. Download the models once and every future session reuses them.

A **Network Volume** is optional, and the upgrade if you want it: separate storage that outlives any
pod, so you can terminate freely and reattach it later. It is region-locked, though, and that region
may not have the GPU you want — if you can't get both, take the Volume Disk and stop rather than
terminate.

## What you get

- **The whole app** — Training, Repair Studio, LoRA the Explorer, LoRA Royale, Profiler, Extract
  and the sample gallery. Not a cut-down web version.
- **Drag-and-drop file transfer** on port 8080 — datasets in, LoRAs out, no terminal
- **One-click model downloads** — Krea 2 needs no HuggingFace account
- **Auto-stop** — optionally shut the pod down when a run finishes, so an overnight finish doesn't
  bill until morning

## Logging in

**Give it a few minutes first.** RunPod downloads the image before anything can start, and it's a
big one. Until that finishes the links are dead — the pod looks ready and both ports give you
nothing. That's normal on a first deploy, and the **Logs** tab shows the download running.

You're ready when the log reaches `[fizgig] Starting KasmVNC display`.

Both ports then ask for a username and password:

| | Port | Username | Password |
|---|---|---|---|
| **Fizgig** | 6080 | `fizgig` | your `VNC_PASSWORD` |
| **File manager** | 8080 | `admin` | the same one, zero-padded to 12 chars if shorter — see below |

**Set your own password when you deploy** — it saves you hunting for one later. On the deploy
screen expand **Edit Template → Environment Variables**, add `VNC_PASSWORD`, and pick something
**12+ characters**. Shorter ones get zero-padded for the file manager, which requires 12, so 12+
keeps both logins identical.

**Didn't set one?** Fizgig generates one per pod and prints it at the **end of the log**, in the
"Ready" banner. If the log seems to stop short of it, switch to another tab and back; RunPod's log
view sometimes needs a nudge before it shows the newest lines.

## First run

1. Connect on **port 6080** and log in
2. **Preferences → ⬇ Download models for me** (Krea 2 ~45 GB, no account needed)
3. Open **port 8080** and drag a dataset folder into `/workspace/datasets`
4. **Start tab → Browse** → pick it, then **Training → Start**

Closing the browser tab does **not** stop training. Fizgig runs on the pod — shut the tab, come
back later, the run is still going.

## Settings

Environment variables — set them on the deploy screen under **Edit Template**.

All optional — the defaults are the intended setup, and everything here can be done from inside the
app instead.

| Variable | Values | |
|---|---|---|
| `VNC_PASSWORD` | **12+ characters** | Desktop *and* file manager. Generated per pod if unset. |
| `HF_TOKEN` | `hf_…` | Only needed for Klein, which is gated. Krea 2 needs nothing. |
| `FETCH_MODELS` | `krea2`, `klein`, `tools` — comma-separated | Download at boot instead of clicking the button in Preferences. `tools` is the Florence-2 captioner, the EN→ZH translator and the face model. |
| `FIZGIG_REF` | branch or tag | Which Fizgig to run. Defaults to `master`, so the app updates itself at every pod start regardless of the image version. |

To enable auto-stop, paste a RunPod API key into **Preferences → RunPod** inside the app. Don't put
one in a template — template variables reach every container deployed from it.

## Storage

```
/workspace/datasets/      your training images (one folder per LoRA)
/workspace/models/        the weights
/workspace/output_loras/  finished LoRAs
```

Everything under `/workspace` persists. Anything outside it is wiped when the pod stops.

Prefer a terminal? `runpodctl` is preinstalled — `runpodctl send <path>` on the pod, then
`runpodctl receive <code>` on your machine. `scp` and `rsync` over SSH work too, and rsync is the
better bet for a large dataset since it resumes.

## More

- [Fizgig on GitHub](https://github.com/shootthesound/Fizgig)
- [Running on a rented GPU — full guide](https://github.com/shootthesound/Fizgig/blob/master/docker/README.md)
