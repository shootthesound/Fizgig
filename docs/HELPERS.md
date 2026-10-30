# Helpers: other GPUs that take work off yours

[← Back to the README](../README.md)

A helper is another computer running Fizgig (a second computer on your network, a RunPod pod, or a computer on Tailscale) that does part of a run's work for yours. Everything is on the **Helpers** tab.

## How do I…?

### …add a helper?

1. On the helper, set a web password (**Preferences > Web UI**) and press **Restart Fizgig now**. On Windows, allow Fizgig through the firewall when asked.
2. Make sure both computers run the same Fizgig version (**About** > **Check for updates**).
3. On your main computer, open **Helpers** > **Add a helper** and pick it from the list: computers on your network and your running pods are listed. Give its password once. A pod of yours needs none: Fizgig finds its password itself, with your RunPod API key on the Pods tab. A Tailscale computer is added by its address.

Details: [Set one up](#set-one-up).

### …use a second GPU in this computer?

Open **Helpers** > **Add a helper** and press **Use it as a helper** beside **This computer's other GPU**. A second Fizgig starts on that card and joins your helpers, ready for previews, a training share or a moved run. See [This computer's other GPU](#this-computers-other-gpu).

### …stop training pausing for previews?

Add a helper and switch **Previews** on, on its tile. See [Previews on a helper](HELPER_PREVIEWS.md).

### …train faster with two GPUs?

Add a helper and switch **Training share** on, on its tile. Each GPU trains its own copy of the LoRA on its part of the images, and the copies' changes are added into one LoRA at every combine, so you still get one LoRA that learned from every image. On a fast network (two GPUs in one computer, or your home network) combine every epoch for the closest result to one GPU; otherwise every 3 epochs, the default. It pays most on long runs, and the tile tells you whether it does. See [Training share](TRAINING_SHARE.md).

### …use my computer for something else in the middle of a run?

Press **Move to…** on the run card and pick a helper. The run carries on there and comes back when you press **Bring it back**. See [Move a run](MOVE_RUN.md).

### …let my home GPU help a run on a pod?

On the home computer, open **Helpers** > **Help another Fizgig with this GPU**, then press **Help it with previews** or **Help it train** beside the pod. See [Your home GPU helping a pod](#your-home-gpu-helping-a-pod).

### …try my own prompts on a pod run's epochs at home?

On the home computer, open **Helpers** > **Help another Fizgig with this GPU** > **Its checkpoints…** beside the pod, tick the epochs, then press **Get them and open LoRA Royale**. See [A pod's checkpoints at home](#a-pods-checkpoints-at-home).

### …have a pod's checkpoints come home by themselves?

On the home computer, open **Helpers** > **Help another Fizgig with this GPU** and tick **Bring its checkpoints home as they're saved** beside the pod (and **its previews too**, if you want them). Tick **Delete the pod once its run has finished and they're all home** to stop it billing at the end. See [A pod's run, home as it saves](#a-pods-run-home-as-it-saves).

### …close the window while a pod trains?

Go ahead: the run trains on the pod, so it keeps going with the window closed or your computer off. Open the pod again to see it.

### …make sure a helper that fails can't ruin my run?

There's nothing to set: Fizgig already handles it. See [When a helper fails or goes away](#when-a-helper-fails-or-goes-away).

---

## What a helper can do

| Job | What it does | Guide |
|---|---|---|
| **Previews** | Renders your run's previews, so training never stops for them | [Previews on a helper](HELPER_PREVIEWS.md) |
| **Training share** (experimental) | Trains part of each epoch's images alongside your GPU | [Training share](TRAINING_SHARE.md) |
| **Move a run** | Takes over a run while you need your computer, and gives it back | [Move a run](MOVE_RUN.md) |
| **A pod's checkpoints at home** | Brings a pod run's epochs home to try your own prompts on them | [Below](#a-pods-checkpoints-at-home) |
| **A pod's run, home as it saves** | Brings each checkpoint home as it's saved, then deletes the pod if you tick that | [Below](#a-pods-run-home-as-it-saves) |

<p align="center"><img src="assets/screenshots/helpers.png" alt="The Helpers tab: a laptop rendering previews, a second computer training a share and saying it's worth it" width="800"></p>

## Set one up

1. **On the helper**, set a web password (**Preferences > Web UI**) and press **Restart Fizgig now**, so Fizgig listens on your network. On Windows, allow Fizgig through the firewall when asked. Fizgig uses port 8081 unless you changed it.
2. **Use the same Fizgig version on both.** **About** shows the version, and **Check for updates** there (or `update_fizgig.bat`) updates to the latest. A pod takes the latest version each time it starts, so update your home computer to match.
3. **On your main computer**, open **Helpers** > **Add a helper**. Computers on your network are listed, and so are the pods running on your RunPod account (with your API key on the **Pods** tab; without it, the pods saved there). Give the helper's password once: Fizgig keeps the login, not the password. A pod of yours is added with no password typed: Fizgig reads it from the pod's settings on RunPod (FIZGIG_PASSWORD or VNC_PASSWORD), or, when the pod made up its own, from the log the pod printed when it started. Add a Tailscale computer, or anything not listed, by its address.
4. **Switch on what it should do**, on its tile: **Previews** or **Training share**. Moving a run and fetching checkpoints need no switch.

### This computer's other GPU

A computer (or pod) with two NVIDIA GPUs lists the second under **Add a helper**: **This computer's other GPU**. **Use it as a helper** starts a second Fizgig on that card and adds it to **Your helpers**, where you switch on what it does, as with any helper. It uses the same model files as this Fizgig, keeps its own settings and caches (in `user_data/other_gpu_<n>`), and only answers this computer. It starts again whenever Fizgig starts and closes with it; **Stop using it** closes it now. This Fizgig keeps the first card (or the one picked in Preferences).

On one computer the transfers cost next to nothing, so a training share pays even on short epochs: on a pod with two RTX 4090s, MOSS-Image epochs went from about 15.5 s to about 9 s. To rent a pod with two cards, set **GPUs** to 2 in the **Pods** tab's card list; its run costs count the second card's help.

### Your home GPU helping a pod

A pod can't reach your home computer, so your home computer connects to it instead. On the home computer, open **Helpers** > **Help another Fizgig with this GPU**, then press **Help it with previews** or **Help it train** beside the pod. The pod shows there once it's on the **Pods** tab. Leave its password box empty: with your RunPod API key on the Pods tab, Fizgig finds the pod's password itself. Starting a run on the home computer stops its helping, and the pod's run carries on alone.

<p align="center"><img src="assets/screenshots/helpers_pod.png" alt="Help another Fizgig with this GPU: home helping a pod train, its verdict, and the pod's checkpoints ready to bring home" width="800"></p>

## Models

A helper that lacks a model file a job needs downloads it from Hugging Face before the job starts. **Get them now** on its tile fetches them ahead of time. Wherever the result has to match your computer's (a training share, a moved run), the helper uses the same model files as yours, checked by hash: its own copy when that's the same file, else a download of the exact file you use.

A fresh pod therefore spends its first minutes downloading. Starting it a little early, or pressing **Get them now**, saves the wait.

## When a helper fails or goes away

A helper that fails never costs you the run:

- **Previews** render on your own GPU again from the next round.
- **A training share** stops, and the run trains on your GPU alone.
- **A moved run** stays safe at home until the helper has really taken it over. If the helper goes away afterwards, **Carry on here** on the run card takes the run back from your computer's last save.

Each guide says exactly what happens. A helper that misses three checks in a row (asleep, switched off, off the network) counts as gone.

## A pod's checkpoints at home

On the home computer, open **Helpers** > **Help another Fizgig with this GPU** and press **Its checkpoints…** beside the pod. Tick the epochs you want, then press **Get them and open LoRA Royale**: they come home and open in LoRA Royale, where you try your own prompts on them on your GPU while the pod keeps training. If this computer has no text encoder for that model, a lighter build is fetched for trying prompts, and Preferences is pointed at it. The result line says when that happens.

## A pod's run, home as it saves

On the home computer, open **Helpers** > **Help another Fizgig with this GPU** and tick **Bring its checkpoints home as they're saved** beside the pod. Each checkpoint the pod's run saves comes to your LoRA output folder, under **From pods** and the pod's name, where Runs & outputs and LoRA Royale find it. Tick **its previews too** to bring the run's previews as well, into the same folder's `sample` folder. This computer does the fetching, so it has to be on, with Fizgig running; it carries on after a restart.

**Delete the pod once its run has finished and they're all home** deletes the pod, its disk too, so it stops billing. It does that only when:

- the run has finished, with nothing left in the pod's queue, and
- every checkpoint here matches the pod's copy, checked by hash.

Before it deletes the pod, it also brings home every other LoRA on the pod (earlier runs' too, into **From pods\\<pod>\\other runs**), and it waits while the pod still works for a computer (previews, a training share, a move on its way). A run that's paused, stopped or failed leaves the pod as it is. If not everything has come home 15 minutes after the run finished, the pod is stopped instead, with its files still on it. Deleting needs your RunPod API key (**Pods** tab).

**A run you moved to that pod** comes home into its own output folder, not **From pods**. With delete ticked, when it finishes there it's brought back as **Bring its LoRAs here** does (the run card then shows it finished here), and the pod goes after that.

A pod's own **Stop the pod when the run ends** waits while this computer brings the last files home, then the pod is stopped from here. If this computer goes quiet (asleep, restarted, logged out), the pod's own stop goes ahead by itself within 30 minutes, so a pod never bills on for nobody.

## Not yet

What helpers don't do yet:

- **Some previews stay on your own GPU.** A helper renders the previews of a normal LoRA run. For a slider, an edit LoRA, a full fine-tune, or a run that previews on a different model from the one it trains on (Klein previewing on Distilled while it trains on Base), the previews render on your GPU even with a helper's **Previews** on. Training is the same either way.
- **A run shares with one helper.** A training share splits the images between your GPU and one helper, not several. And a helper trains for one computer at a time: if a second computer asks while it's busy, it's told the helper is taken.
- **A fine-tune can't move.** Every LoRA run can, with all its folders: see [Move a run](MOVE_RUN.md#what-can-move).
