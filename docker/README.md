# Running Fizgig on a rented GPU

[**⚡ Deploy Fizgig on RunPod →**](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=bdrdew7hxw&ref=vkb387ep)

> **Before you press Deploy Pod:** RunPod doesn't add a disk for you: on its page press **Add volume**, click **Volume disk** (it opens on Global volume, which doesn't work), keep **300 GB**, press **Add volume disk**, then **Deploy Pod**. [See an explanation with screenshots](#give-it-a-disk-on-runpods-deploy-page).

Fizgig in your browser: the pod runs the app and you open it as a web page, from your computer or your phone. Training, Studio, LoRA Royale, Profiler, Extract and the sample gallery are all there. Nothing to install, and your own GPU stays free while it trains.

*That link is a referral one — it supports Fizgig's development at no extra cost to you.*

Prefer Vast.ai? The same image runs there too: [Fizgig on Vast.ai](https://cloud.vast.ai/?ref_id=757773&creator_id=757773&name=Fizgig%20LoRa%20Training%20and%20Fine-tuning) (also a referral link) · [the Vast.ai guide](../docs/VAST.md).

## Quick start: nothing installed on your computer

You don't need Fizgig on your own computer for this. You need a RunPod account and a web browser.

1. **Click [Deploy on RunPod](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=bdrdew7hxw&ref=vkb387ep).** It opens RunPod with the Fizgig template and an RTX 5090 chosen. **Add a 300 GB Volume disk** before you press **Deploy Pod**: RunPod doesn't add one for you, and its dialog opens on the wrong kind ([see an explanation with screenshots](#give-it-a-disk-on-runpods-deploy-page)). (Other cards: [Which GPU](#which-gpu).)
2. **Wait for "Ready".** Open the pod's **Logs** tab. The first start downloads the image and takes a few minutes. It's ready when the log ends with `Ready. Open the pod's HTTP port 8081 for Fizgig.` and the password on the line under it.
3. **Open Fizgig in your browser.** In the pod's **Connect** menu, click **HTTP port 8081** and log in with that password (there's no user name). That page is Fizgig, the whole app.
4. **Download a model.** **Preferences → ⬇ Download models for me.** Krea 2 needs no Hugging Face account.
5. **Train.** On the **Start** tab, **Upload a training folder**, then on the **Training** tab press **Start**.

Close the tab whenever you like: the run carries on, and you can come back from any browser, your phone included. **Stop** the pod between sessions to keep everything on its disk; terminating it deletes the disk ([Storage](#storage)). The rest of this page has the detail on each step.

**Which Fizgig the pod runs:** opened from your own Fizgig, the pod runs your computer's version and keeps it; on its own it takes the newest at every start ([more](#which-fizgig-a-pod-runs)).

**Need the old pod?** The previous template, with the classic app in a remote desktop, stays available as **[Fizgig Classic v7.x](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)**: see [the classic template](#the-classic-template-remote-desktop) below. Pods you already have keep running as they are.

---

## Which Fizgig a pod runs

- **On its own** (used only in a browser), a pod takes the newest Fizgig each time it starts: stop and start it, or restart it, and it's up to date.
- **Opened from your own Fizgig** (8.9.93 or later), it runs your computer's version and keeps it, at every start ([below](#the-same-fizgig-as-your-computer)). This is all most people need: nothing changes under a job, and updating your computer brings the pod along.
- **Set to one release** with `FIZGIG_REF`, it runs that release whatever opens it ([below](#keep-a-pod-on-one-version)).

### The same Fizgig as your computer

When your own Fizgig opens a pod (in its app window, or a pod on its Pods tab), it puts the pod on the same version as your computer, so the two always work together: moving a run, helpers and the app window need that. The pod switches in a few seconds (a minute or two when that version needs other packages) and keeps the version at every start after that. It never switches while it's busy (a run, a download, captions, a move); it does once it's free. The pod's **About** box says which computer's version it runs, and **Use the newest Fizgig** there puts it back on the newest (the next time your Fizgig opens it, it's put back on yours).

### Keep a pod on one version

For a pod you use only in a browser, or to stay on one release whatever your computer runs: set the environment variable `FIZGIG_REF` to a release's version, and the pod runs that version at every start until you change it. A pod set this way doesn't follow your computer: when you update your computer, Fizgig says the two run different builds until you change `FIZGIG_REF` too.

- **On RunPod, before you deploy:** on the deploy screen expand **Edit Template → Environment Variables**, add `FIZGIG_REF` with a version, for example `v8.9.93`, then **Deploy Pod**.
- **On a pod you already have:** in the pod's menu choose **Edit Pod**, add `FIZGIG_REF` under **Environment Variables**, and save. RunPod restarts the pod, which starts on that version. Your volume and everything on it stay.
- **On Vast.ai:** add `FIZGIG_REF` with the version to the template's environment before you rent.

The versions are on the [Releases page](https://github.com/shootthesound/Fizgig/releases): use the tag as it's written there, with its `v`. To go back to updating at every start, set `FIZGIG_REF` to `master`, or remove it.

A run you paused or moved resumes fine on the same version. Your computer and the pod don't have to be on the same version, but Send to this pod, Move a run and Helpers can need features that a pinned older pod doesn't have yet.

## Storage

### Give it a disk on RunPod's deploy page

RunPod's deploy page doesn't add a disk by itself. On the right of the page, a box called **Nothing mounted at the template's path** says so:

<p align="center"><img src="../docs/assets/screenshots/runpod_storage_sidebar.png" alt="RunPod's deploy summary: Nothing mounted at the template's path, This template recommends 300 GB of persistent storage at /workspace, Add volume" width="300"></p>

1. Press **Add volume** in that box.
2. In **Create storage**, click **Volume disk** (marked Legacy). The dialog opens with **Global volume** selected, and that one doesn't work for Fizgig (see below).
3. Leave **Size** at **300 GB** and press **Add volume disk**.

<p align="center"><img src="../docs/assets/screenshots/runpod_create_storage.png" alt="RunPod's Create storage dialog: Global volume, Network volume, and Volume disk (Legacy) selected, Size 300 GB, Add volume disk" width="800"></p>

4. Press **Deploy Pod**.

Skip this and the pod runs on its small container disk: the models download into it and everything is lost when the pod stops. The pod's log warns you if that happened (`Storage: only … GB free … that looks like container disk`).

### How much, and which kind

**Give it 300 GB (the template's recommendation), mounted at `/workspace`.** Each model's files run to tens of GB,
so a few models, your datasets and their LoRAs fill 200 GB quickly; 100 GB is the least that works for one.

Then the one thing to know: **stopping a pod keeps everything; terminating it does not.** Stop
between sessions and your models, datasets and LoRAs are all still there when you start it again.
Terminate and the disk goes with the pod.

That's it for the **Volume disk**, the one the steps above add, and it's perfectly fine to use.

A **Network Volume** is the upgrade if you want one: it's a separate thing that outlives any pod, so
you can terminate freely and attach the same storage to a new pod later. The catch is that it's
region-locked, and the region you create it in may not have the GPU you want — so if you can't find
a region with both, don't fight it. Take the Volume Disk and just stop rather than terminate.

**Not a Global Volume.** RunPod's newer global volumes are object storage behind a mount, not a real
filesystem: by RunPod's own documentation they can't set file permission bits, don't support file
locking or atomic rename, and aren't recommended for workloads that write heavily. Fizgig needs all
three — git can't even create the checkout there, and training writes a checkpoint every epoch. Mount
one at `/workspace` and the pod restarts every few seconds.
Use a Volume Disk or a regional Network Volume.

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

**Let Fizgig pick.** The **Rent a pod** card on the **Pods** tab of Fizgig on your own computer lists the cards RunPod has free right now (with your RunPod API key saved there), filtered by memory, cloud and a top price per hour, cheapest per run first. Tell it how long your run takes on a 5090 and each row shows roughly what the same run costs on that card. **Deploy** opens RunPod's page with that card and this template chosen; you add the volume and press Deploy there. The speeds are a rough guide from public benchmarks; the prices and stock are RunPod's own, counted only on hosts whose driver runs this image. Set **GPUs** to 2 for a two-card pod: on the pod, **Helpers** > **Use it as a helper** puts the second card to work on a training share.

<p align="center"><img src="../docs/assets/screenshots/rent_a_pod.png" alt="Rent a pod: RunPod's free cards, cheapest per run first, each with Deploy" width="800"></p>

## Logging in

**Give it a few minutes first.** RunPod downloads the image before anything starts, and until then the links give you nothing even though the pod looks ready. The **Logs** tab shows the download. You're ready when the log ends with the "Ready" banner:

```
[fizgig]  Ready. Open the pod's HTTP port 8081 for Fizgig.
[fizgig]      password: ...
```

In the pod's **Connect** menu, open **HTTP port 8081** and log in with that password. There's no user name.

| | Port | Username | Password |
|---|---|---|---|
| **Fizgig** | 8081 | none | the pod's password |
| **File manager** | 8080 | `admin` | the same one* |

\* The file manager refuses very common passwords (`abc123`, `password1234`). For one of those it gets zeros added to the end, up to 12 characters, **for port 8080 only** (`abc123` becomes `abc123000000`); if it refuses that too, its login isn't set up. Fizgig's sidebar and the pod's log say which. Pick a less common password and both logins are identical.

**Choosing the password.** On the deploy screen expand **Edit Template → Environment Variables** and add `FIZGIG_PASSWORD` (12+ characters). If you don't, the pod generates one on its first start, prints it in the log, and keeps it on your volume, so it stays the same every time you stop and start the pod. Either way, Fizgig on your own computer finds it for you (with your RunPod API key saved there): see below. If the log seems to stop short of the banner, switch to another tab and back; RunPod's log view sometimes needs a nudge.

**Opening it from Fizgig on your own computer.** With your RunPod API key saved in Fizgig's **Pods** tab, the Fizgig window opens your pod already logged in, and **Helpers** adds the pod with no password typed. Fizgig reads the password each time and never saves it: from the pod's settings on RunPod (`FIZGIG_PASSWORD`), or, when the pod made up its own or keeps it as a RunPod secret, from the log the pod printed when it started.

<p align="center"><img src="../docs/assets/screenshots/pods.png" alt="The Pods tab: your pods and your other computers" width="800"></p>

### How the pod is secured

**Fizgig always needs a password on a pod.** There is no open mode: without `FIZGIG_PASSWORD` one is generated and printed in the log. An open Fizgig on a public URL is a shell on someone else's GPU. Wrong guesses are slowed down per visitor.

**SSH is off unless you ask for it.** Nothing listens on port 22 until `PUBLIC_KEY` is set, and then only your own key gets in. RunPod's "SSH Terminal Access" toggle sets it for you.

**Everything in the pod runs as root, deliberately.** A rented GPU has one tenant, you, and the password is what protects it. An unprivileged user would not change what a logged-in visitor can do and would break `/workspace` permissions on existing pods.

## First run

1. Open **port 8081** and log in.
2. **Preferences → ⬇ Download models for me.** Krea 2 is ~45 GB and needs no Hugging Face account; gated models (Klein, Flux 3 Image Klein) ask for a free token.
3. **Start tab → Upload a training folder**, or drag a folder into `/workspace/datasets` on **port 8080**.
4. **Training → Start.**

**Using Fizgig on your own computer too?** Its **Pods** tab opens this pod in the Fizgig window: paste the pod's ID, or add your RunPod API key and your pods appear by themselves, with Start and Stop. See [Pods](../docs/WEB_UI.md#pods-a-rented-gpu-from-your-own-fizgig).

Closing the browser tab does **not** stop training. Fizgig runs on the pod, so close the tab, come back later (from your phone if you like) and the run is still going. Finished LoRAs download from the Training tab and the gallery. More in [the web UI guide](../docs/WEB_UI.md).

## Getting files in and out

**From Fizgig on your own computer** (its Pods tab opens the pod in the Fizgig window): the Start tab's **From this computer** card sends any of your local training folders up in one click. Or upload a folder from the Start tab in any browser. Both keep a folder's sub-folders when **Also train on its sub-folders** is ticked on the Training tab ([Sub-folders](../docs/TRAINING.md#sub-folders)).

<p align="center"><img src="../docs/assets/screenshots/from_this_computer.png" alt="From this computer: your own training folders on the pod's page, ready to send" width="800"></p>

**Your LoRAs, home as they're saved:** on your own computer, **Helpers** > **Help another Fizgig with this GPU** > **Bring its checkpoints home as they're saved** fetches each checkpoint as the pod saves it, and can delete the pod once the run has finished and they're all home ([how](../docs/HELPERS.md#a-pods-run-home-as-it-saves)). A run on a pod keeps training with the window closed.

**Port 8080** is a file manager rooted at `/workspace`: the **File manager** button in Fizgig's sidebar opens it. Drag a dataset folder in from your desktop,
download finished LoRAs from `output_loras/` the same way. No terminal, no SSH keys.

<p align="center"><img src="../docs/assets/screenshots/file_manager.png" alt="The pod's file manager: a training folder's pictures and captions" width="800"></p>

Prefer a terminal? `runpodctl` is preinstalled — `runpodctl send <path>` on the pod, then
`runpodctl receive <code>` on your machine. `scp` and `rsync` over SSH work too, and rsync is the
better bet for a large dataset since it resumes.

## Your home GPU helping the pod

Fizgig on your own computer can render a pod run's previews, so the pod's time goes on training, or train a share of each epoch's images, and bring the run's checkpoints home to try your own prompts on them: **Helpers** > **Help another Fizgig with this GPU** on your computer. A training share pays on big datasets only, as your home upload sets the transfer time: see [Helpers](../docs/TRAINING_SHARE.md#is-it-worth-it).

<p align="center"><img src="../docs/assets/screenshots/helpers_pod.png" alt="Your home Fizgig helping a pod train, with its verdict, and the pod's checkpoints ready to bring home" width="800"></p>

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

**Stopping a pod yourself mid-run** loses what the run trained since its last save: the pod gets a few seconds'
notice, not the minutes an epoch can take. Press **Pause** first (it saves and stops at the end of the epoch), then
stop the pod.

## Settings

Environment variables, set on the deploy screen under **Edit Template**:

| Variable | Values | |
|---|---|---|
| `FIZGIG_PASSWORD` | your choice (not a very common one) | Fizgig *and* the file manager. Generated on the first start if unset, then kept. (`VNC_PASSWORD`, the classic template's name, works too.) |
| `HF_TOKEN` | `hf_…` | Only needed for Klein and Flux 3 Image Klein, which are gated. Krea 2 needs nothing. |
| `FETCH_MODELS` | `krea2`, `klein`, `flux3`, `minimax`, `tools` — comma-separated | Download at boot instead of clicking the button in Preferences. `minimax` includes the Circlestone and both Ostris training adapters. `tools` is the Florence-2 captioner, the EN→ZH translator for bilingual captions, and the face model the Look Filter uses. |
| `FIZGIG_REF` | branch or tag | Which Fizgig to run. Defaults to `master`: the newest at every start, or your computer's version once your Fizgig has opened the pod. A release's tag (`v8.9.92`) keeps the pod on that version whatever opens it ([Which Fizgig a pod runs](#which-fizgig-a-pod-runs)). |

Nothing is compulsory — the defaults are the intended setup, and everything here can be done from
inside the app instead.

Two more exist for unusual cases: `FETCH_MODELS_EXTRA` passes extra flags to the downloader, and `FIZGIG_REPO` points the pod at a fork.

## On Vast.ai

The same image runs on Vast.ai, with its own template: see [Fizgig on Vast.ai](../docs/VAST.md).

## The classic template (remote desktop)

For a while there are two templates. The one above is the new Fizgig in a browser tab. **[Fizgig Classic v7.x](https://console.runpod.io/deploy?type=GPU&gpu=RTX+5090&count=1&template=faoq8ed6um&ref=vkb387ep)** is the previous pod: the classic desktop app in a remote desktop on port 6080, for anyone partway through a big job who wants to finish it on what they know.

- **Logging in:** port **6080**, user `fizgig`, and the pod's `VNC_PASSWORD` (generated and printed in the log if you didn't set one; on Classic it changes each time the pod starts). The file manager is on 8080 as above, and the new web page on 8081 with the same password.
- **First run:** Preferences → **⬇ Download models for me**, drag a dataset into `/workspace/datasets` on port 8080, then Start tab → **Browse** and **Training → Start**.
- **Settings:** the same as above, with `VNC_PASSWORD` for the password and `SCREEN_W` / `SCREEN_H` for the desktop's starting size (1600×1400; it then resizes to your browser window).
- **Moving over:** the two use the same `/workspace` layout, so a volume from a Classic pod works on the new one. Models, datasets, LoRAs and run history carry across.
- **Pods you already have** keep the image they were made with and carry on as Classic pods. Deploy the new template when you want the new one.

## If something's wrong

The **pod log** is the first place to look — every step is narrated there, including storage and the
password.

- **Both links dead just after deploying** — it's still downloading the image. Wait for the
  "Ready" banner in the log.
- **Storage shows ~25 GB** — the volume didn't mount. Check the mount path is `/workspace`.
- **Downloads fail with "no space left"** — same cause: models are landing on container disk.
- **Your models vanished** — the pod was *terminated* rather than *stopped*. A Volume Disk goes
  with its pod; stop it instead, or use a Network Volume if you need to terminate.
- **Restarts every few seconds** — `/workspace` is a Global
  Volume, which git can't clone onto (see Storage above). Switch it for a Volume Disk or a regional
  Network Volume.

Fizgig's own version and the image's are both shown in **Preferences → RunPod**; quote both if you
report a problem, since the app updates itself independently of the image.
