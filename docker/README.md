# Running Fizgig on a rented GPU

[**⚡ Deploy Fizgig on RunPod →**](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)

The whole app in a browser tab — Training, Repair Studio, LoRA the Explorer, LoRA Royale, Profiler,
Extract and the sample gallery. Nothing to install, and your own GPU stays free while it trains.

*That link is a referral one — it supports Fizgig's development at no extra cost to you.*

---

## Storage

**Give it 100 GB+, and mount it at `/workspace`.** The models are ~45 GB, and you want room for
datasets and output LoRAs on top.

Then the one thing to know: **stopping a pod keeps everything; terminating it does not.** Stop
between sessions and your models, datasets and LoRAs are all still there when you start it again.
Terminate and the disk goes with the pod.

That's it for the default **Volume Disk**, which is what the template comes with and is perfectly
fine to use.

A **Network Volume** is the upgrade if you want one: it's a separate thing that outlives any pod, so
you can terminate freely and attach the same storage to a new pod later. The catch is that it's
region-locked, and the region you create it in may not have the GPU you want — so if you can't find
a region with both, don't fight it. Take the Volume Disk and just stop rather than terminate.

Either way, download the models once and every future session reuses them.

## Which GPU

The link defaults to an **RTX 5090**, and that's a deliberate choice rather than "the newest one".

Fizgig sizes Krea 2's block swap to your VRAM, and **at 32 GB it uses none at all**. Block swap
moves weights over PCIe every step and costs roughly **4× the step time**. When you're billed by the
hour, a card that's slightly dearer but four times faster is much cheaper per finished LoRA.

| | Cards | |
|---|---|---|
| **Best** | RTX 5090 (32 GB), L40S / A6000 (48 GB) | Krea 2 with no block swap |
| **Good value** | RTX 4090, 3090, A5000 (24 GB) | Swaps for Krea 2; ideal for Klein 9B |
| **Smallest worth renting** | 16 GB | Fine, but heavy swap makes it false economy by the hour |

H100 and A100 work but are poor value here — LoRA training never touches 80 GB, and you'd pay
several times more for it.

## Logging in

**Give it a few minutes first.** RunPod has to download the image before anything can start, and
it's a big one. Until that finishes the links are dead — the pod looks ready, the ports are listed,
and both give you nothing. That's normal on a first deploy; the **Logs** tab shows the download
running.

You're in business when the log reaches:

```
[fizgig] Starting KasmVNC display (1600x1400, resizes to your browser window)
```

| | Port | Username | Password |
|---|---|---|---|
| **Fizgig** | 6080 | `fizgig` | your `VNC_PASSWORD` |
| **File manager** | 8080 | `admin` | the same one, zero-padded to 12 chars if shorter — see below |

**Set your own password when you deploy.** On the deploy screen expand **Edit Template →
Environment Variables**, add `VNC_PASSWORD`, and pick something 12+ characters. Then you already
know it, and you never have to go looking.

Anything shorter gets zero-padded for the file manager, which needs 12 — so choose 12+ and both
logins stay identical.

**Didn't set one?** Fizgig generates one per pod and prints it at the **end of the log**, in the
"Ready" banner:

```
[fizgig]  Your browser will ask for a username and password:
[fizgig]      username: fizgig
[fizgig]      password: ...
```

If the log looks like it stops short of that, switch to another tab and back — RunPod's log view
sometimes needs a nudge before it shows the newest lines.

## First run

1. Connect on **port 6080** and log in
2. **Preferences → ⬇ Download models for me** — Krea 2 is ~45 GB and needs no HuggingFace account;
   Klein is gated by Black Forest Labs and will ask for a free token
3. Open **port 8080** and drag a dataset folder into `/workspace/datasets`
4. **Start tab → Browse** → pick it, then **Training → Start**

Closing the browser tab does **not** stop training. Fizgig runs on the pod — shut the tab, come back
later, the run is still going.

## Getting files in and out

**Port 8080** is a file manager rooted at `/workspace`. Drag a dataset folder in from your desktop,
download finished LoRAs from `output_loras/` the same way. No terminal, no SSH keys.

Prefer a terminal? `runpodctl` is preinstalled — `runpodctl send <path>` on the pod, then
`runpodctl receive <code>` on your machine. `scp` and `rsync` over SSH work too, and rsync is the
better bet for a large dataset since it resumes.

## Where things live

```
/workspace/datasets/      your training images (one folder per LoRA)
/workspace/models/        the weights
/workspace/output_loras/  finished LoRAs
```

Everything under `/workspace` persists. Anything outside it is wiped when the pod stops.

## Stopping the pod when a run finishes

A rented GPU bills by the hour, so a run that ends at 4am keeps billing until you notice.
**Preferences → RunPod → Stop this pod when a training run finishes** fixes that. You get a
two-minute countdown you can cancel, and it never fires after a Pause, a Stop or a failure — those
are exactly the times you want the machine alive.

It needs an API key, pasted into that same panel. Make one at **RunPod → Settings → API Keys**; the
key RunPod gives a pod automatically is pod-scoped and cannot stop pods, which is a RunPod
limitation rather than a Fizgig one. It's stored on your volume, not in any template.

## Settings

Environment variables, set on the deploy screen under **Edit Template**:

| Variable | Values | |
|---|---|---|
| `VNC_PASSWORD` | 12+ characters | Desktop *and* file manager. Generated per pod if unset. |
| `HF_TOKEN` | `hf_…` | Only needed for Klein, which is gated. Krea 2 needs nothing. |
| `FETCH_MODELS` | `krea2`, `klein`, `tools` — comma-separated | Download at boot instead of clicking the button in Preferences. `tools` is the Florence-2 captioner, the EN→ZH translator for bilingual captions, and the face model the Look Filter uses. |
| `FIZGIG_REF` | branch or tag | Which Fizgig to run. Defaults to `master`, so the app updates itself at every pod start. |

Nothing is compulsory — the defaults are the intended setup, and everything here can be done from
inside the app instead.

Three more exist for unusual cases: `FETCH_MODELS_EXTRA` passes extra flags to the downloader,
`FIZGIG_REPO` points the pod at a fork, and `SCREEN_W`/`SCREEN_H` set the desktop's *starting* size
(1600×1400) — which rarely matters, since it then resizes to your browser window.

## If something's wrong

The **pod log** is the first place to look — every step is narrated there, including storage and the
login details.

- **Both links dead just after deploying** — it's still downloading the image. Wait for
  `Starting KasmVNC display` in the log.
- **Storage shows ~25 GB** — the volume didn't mount. Check the mount path is `/workspace`.
- **Downloads fail with "no space left"** — same cause: models are landing on container disk.
- **Your models vanished** — the pod was *terminated* rather than *stopped*. A Volume Disk goes
  with its pod; stop it instead, or use a Network Volume if you need to terminate.

Fizgig's own version and the image's are both shown in **Preferences → RunPod**; quote both if you
report a problem, since the app updates itself independently of the image.
