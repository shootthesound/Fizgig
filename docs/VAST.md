# Fizgig on Vast.ai

[← Back to the RunPod guide](../docker/README.md)

Vast.ai rents GPUs from a marketplace of hosts, often for less than RunPod. Fizgig's pod image runs there as it does on RunPod: you open Fizgig in your browser, and training runs on the rented card. The app's **Pods** tab deploys RunPod pods for you; on Vast you start the instance on Vast's site and open it by its address.

**[Fizgig's template on Vast.ai](https://cloud.vast.ai/?ref_id=757773&creator_id=757773&name=Fizgig%20LoRa%20Training%20and%20Fine-tuning)**: new to Vast.ai? Signing up through this link supports Fizgig: Vast.ai pays a small referral share from its own side. It costs you nothing, and you pay the same rates.

## How do I…?

### …start Fizgig on Vast.ai?

1. Sign up (through the link above, if you're new) and add credit under **Billing**: Vast is prepaid, so it won't rent until you have some.
2. Open [Fizgig's template](https://cloud.vast.ai/?ref_id=757773&creator_id=757773&name=Fizgig%20LoRa%20Training%20and%20Fine-tuning). Pick **On-demand**, not Interruptible (an interruptible instance can be taken back mid-run), and press **Rent** on a card with 24 GB or more (32 GB trains Krea 2 with no block swap).
3. To choose your own password, add **FIZGIG_PASSWORD** to the template's environment before you rent: 8 characters or more, and one you don't use anywhere else. Otherwise the instance makes one up at its first start and keeps it. Opened from your own Fizgig, the instance runs the same version as your computer; on its own it takes the newest at each start. To keep it on one release instead, add **FIZGIG_REF** with the release's tag, for example `v8.9.93` ([more](../docker/README.md#which-fizgig-a-pod-runs)).
4. Wait until the instance is running (the first start downloads the image, a few minutes). Its password, if it made one up, is in its log: the instance card's log icon, the line `password: …`.

### …open it?

Open its **https address**: the instance's log shows it a minute or so after it starts, as `Open Fizgig (https): https://<name>.trycloudflare.com`, and Fizgig shows it too (Preferences > Vast.ai). Log in with your password. It's a new address each time the instance starts.

The plain address works too: press **Open** on the instance, or open its **IP & port info** (Fizgig is the port mapped to **8081**, at `http://<its IP>:<that port>`). Prefer the https one: your password travels encrypted there.

### …get the models?

In Fizgig, **Preferences** > **Download models for me**. Vast's hosts download fast (over 100 MB/s in testing).

### …get my training pictures there?

Start tab > **Upload a training folder**, or drag them into the file manager (Fizgig's **File manager** button opens its https address; or the port mapped to **8080**), user `admin`, the same password.

### …train?

As on your own computer: pick the folder on the Start tab, then **Training** > **Start**. Closing the browser tab doesn't stop a run; Fizgig runs on the instance.

### …use it as a helper from my own computer?

On your computer, add the instance on the **Pods** tab (paste `http://<its IP>:<the 8081 port>` into **Add a pod**), then **Helpers** > **Add a helper** lists it: give its password once. It can then render your previews, train a share of each epoch, or take a run with **Move to…** ([Helpers](HELPERS.md)), and **Help another Fizgig with this GPU** lets your home GPU help it, or bring its checkpoints home as they're saved.

### …get my LoRAs home?

**Runs & outputs** downloads them, or the file manager (`/workspace/output_loras`). Do it before you stop or destroy the instance.

### …update Fizgig there?

Restart the instance: Fizgig updates itself every time it starts.

### …stop paying for it?

Destroy the instance on Vast.ai once your LoRAs are home. A destroyed one costs nothing, and its files are gone. A stopped one still bills for its disk (Vast's disk rates vary by host: check the instance's price), and may not start again while someone else is renting its GPU, so don't leave files you need on a stopped instance.

---

## Good to know

- **https and plain http.** Vast serves an instance's ports as plain http at its public address. Fizgig's Vast image adds an https address for Fizgig and one for the file manager (free Cloudflare tunnels, new each time it starts): use those. The plain ports stay open too, so use a strong password you don't use anywhere else, and remember that the host's owner can reach the container, as with any rented machine.
- **A new address after a restart.** A stopped and started instance can come back with a new IP or port. Paste the new address into **Add a pod** and add it as a helper again: Fizgig recognises it and moves its login, its bring-home setting and a run you moved there across to the new address.
- **Disk.** The template asks for 300 GB, as RunPod's does: one model family's files are 20 to 40 GB, and an instance's disk can't grow after it's made. Vast bills the disk until the instance is destroyed, stopped or not. Destroying the instance deletes everything on it.
- **Coming back to the same machine often?** Add a volume at `/workspace` when you rent: it keeps your models and files between instances on that machine. A Vast volume works only on the machine it was made on, and it bills until you delete it yourself, even after the instance is destroyed.
- **CUDA.** The image needs a host whose driver supports CUDA 12.8 or newer; the template's filter asks for that.
- **Stop when done.** **Preferences > Vast.ai > Stop this instance when a training run finishes** stops it after a run finishes, with a two-minute countdown you can cancel. No key needed: the instance stops itself. That ends the GPU billing; its disk bills until you destroy the instance, so bring your LoRAs home and destroy it when you're done.
- **Stopping it mid-run** loses what the run trained since its last save. Press **Pause** first (it saves and stops at the end of the epoch), then stop the instance.

Measured on a Vast RTX 4090 (22 Oct 2026): MOSS-Image trained at about 1.75 it/s, the same as a RunPod 4090.
