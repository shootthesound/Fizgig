# Fizgig v8.9.97: pods log you in by themselves, and moved runs come home as they train

**No more typing a pod's password.** With your RunPod API key saved on the Pods tab, Fizgig now finds any of your pods' passwords itself, including a pod deployed from Fizgig's link with no password set, where the pod makes one up when it first starts. It reads it from the pod's settings on RunPod, or from the log the pod printed when it started, each time, and never saves it. So the app window opens your pods already logged in, and on the Helpers tab you add a pod, help it, or bring its checkpoints home with the password box left empty. A pod helper whose login runs out logs in again by itself. A computer on your network still takes its own password, as before, and every password box says which kind it is.

**A moved run comes home as it trains.** **Move to…** has a new tick, on by default: **Bring its checkpoints and previews back here as they're saved**. Each one arrives in the run's own folder on this computer while it trains on the pod, queued if your connection is slower than the saves. **Bring it back** then has only the last few to fetch, and its new **Delete the pod once it's all here** tick (on by default, for RunPod pods) deletes the pod once everything is back, so it stops billing. A move also takes the Samples tab's live preview changes with it, so the run previews the same on the pod.

**A clearer Helpers tab.** The tab now says plainly what a helper is, what it can do, how to add one and when you need a password. On **Help another Fizgig with this GPU**, the card for a run you moved there says its move takes care of it.

**Fixes**

- **Help it with previews did nothing** when this computer was training its own run. It now says why, in its own card: to have the pod make this run's previews, turn on **Previews** on the pod's tile under **Your helpers**. With nothing training on the pod yet, it says it will start when a run does.
- **A pod's previews looked stuck on the first round.** Every step showed "Rendering", including the first round's model load, which takes minutes for a big model on a fresh pod. Each step now says what it is: sending the LoRA, loading the model there, waiting its turn, rendering.
- **Checkpoints brought home from a pod went into a folder of their own** under the LoRA output folder, even for a run that started on this computer. A run moved from here now comes home into its own folder, also after the move is over.
- **Open on the run card** for a run on a pod opened a web browser. In the app window it now switches over to the pod, logged in, as the Pods tab's **Open** does.
- **Switching the app window between a pod and this computer** waited out the loading screen's 3.5-second minimum every time. The loading screen now shows only while the page loads.
- **A pod's checkpoint list could fail** while a run's files were being cleared from it, after the run moved back.
- **The training share tile** now says that switched on while a run trains, a share joins at the next epoch it's ready for.

**Updating:** update Fizgig on your computers, and restart your pods.
