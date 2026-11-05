# Move a run to another computer

[← Back to Helpers](HELPERS.md)

Need your computer for a game, or for someone else, in the middle of a run? Move the run to a helper (a pod, or another computer on your network). It carries on training there, and comes back when you want it.

## How do I…?

### …move a run off this computer?

1. Add the other computer on the **Helpers** tab ([how](HELPERS.md#set-one-up)). It must be on, not training, and on the same Fizgig version.
2. On the run card (Training tab), press **Move to…** and pick the computer.
3. Press **At the end of this epoch** (or **Move it** for a paused run). It pauses here, goes over, and carries on there.

### …move it straight away?

Press **Now** instead of **At the end of this epoch**. It stops at once and moves from its last save, so the epochs since then are trained again there.

### …move it to a pod?

Start the pod from the **Pods** tab, add it on the **Helpers** tab, then use **Move to…** as above. A fresh pod downloads the model files first, and the run waits paused here while it does (about 10 minutes for Krea 2 in testing). See [A fresh pod](#a-fresh-pod).

### …make it faster on a big pod?

Tick **Train several pictures a step there** in **Move to…**. The pod works out the largest Batch size its card fits at the run's precision without swapping more blocks, and the run card says how many it took (Training on *pod*, 3 pictures a step). Each epoch makes fewer steps, each averaging that many pictures, as [Gradient Accumulation](CLI.md#batch-size) does. Brought back, the run carries on at this computer's own Batch size. The tick is greyed out for a run that trains one picture a step: an edit, a slider, a fine-tune, DOP, Fizgig LoRA Isolation, Start at a smaller size, or keeping the model's own picture outside the mask.

### …see how it's doing over there?

Watch the run card here: it shows the epoch and time left there. Press **Open** *(computer name)* to see that computer's Fizgig.

### …bring it back?

Press **Bring it back** on the run card. It pauses there at the end of its epoch and carries on here. **Bring it back now** stops it there at once and comes back from its last save. Once it has finished there, the button reads **Bring its LoRAs here**.

### …get it back if the other computer is gone?

Press **Carry on here** on the run card (a deleted pod, a laptop that's been shut down). The run carries on here from this computer's last save, and the epochs trained over there are lost. If the other computer still answers, the run is stopped there.

---

## What goes over, and what comes back

**Going:** the training folder (pictures and captions), the run's saved state, its loss log and its previews so far, and a Context LoRA. The other computer trains on the same model files as yours, checked by hash: its own copy, or a download of the exact file. It uses the same precision, plans block swap for its own card, and makes the run's cache itself from the pictures.

**Coming back:** the new checkpoints, previews, loss log and saved state, plus the training folder's captions and `fizgig_*.json` files. A caption rewritten there (by hand or by auto-recaption) and an image excluded there arrive here too. Your older saved states here are kept. Once the run is back, the other computer deletes its copy.

## While it's away

- The run card shows how far it has got there, with a link to open that computer's Fizgig. On the other computer it shows as its own run: its previews, saves and Pause work there as usual.
- **Start**, **Resume** and the queue are blocked here, so the run can't train in two places at once. Bring it back, or press **Carry on here**, first.
- The move survives a restart of Fizgig here.

## A fresh pod

1. **Start the pod** on the **Pods** tab (on RunPod's page, add a 300 GB **Volume disk** before **Deploy Pod**: [how](../docker/README.md#give-it-a-disk-on-runpods-deploy-page)) and wait until it's running Fizgig.
2. **Add it as a helper** on the **Helpers** tab, with its password.
3. **Press Move to…** The run pauses here. Then the pod checks which model files it has, downloads any it's missing, and receives the run. The run counts as moved only once the pod has resumed it.

A pod takes the latest Fizgig each time it starts, so update your own computer first: both must run the same version.

## When something goes wrong

The run stays safe on your computer until the other computer has really taken it over:

| What happens | What you see | What to do |
|---|---|---|
| The pod never starts, or never answers | It can't be added, or **Move to…** doesn't list it | Nothing changes here: carry on, or try another pod |
| A download, the transfer or the start fails over there | The move shows as failed, and the run is paused here | Press **OK**, then **Resume** to train here, or move again. In case it got going over there, it's told to stop |
| It fails or is stopped over there while training | "It failed on…" or "It was stopped on…" on the run card | Press **Bring it back** to carry on here from its last save, or **Carry on here** |
| The other computer goes away while training | "It isn't answering just now" on the run card | Wait for it, or press **Carry on here** |
| Bringing it back fails partway | The run card says so and keeps watching | Try **Bring it back** again, or press **Carry on here** |

## What can move

Any LoRA run: a standard LoRA, an edit LoRA or a slider, with more than one subject, **Leave the rest of the model alone** or **Fizgig LoRA Isolation**. Every folder the run trains from goes with it (its Training folder and, when **Also train on its sub-folders** is ticked, its sub-folders, Subjects 2 to 4, an edit run's originals and extras, a slider's other end), and so do the files it names (an edit run's test photo, the Samples tab's reference picture, a thumbnail). Captions you edit over there come back into each folder. A fine-tune can't move. The other computer must not be training a run of its own.
