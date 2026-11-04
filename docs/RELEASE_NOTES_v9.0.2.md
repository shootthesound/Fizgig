# Fizgig v9.0.2: RunPod's new API, SDXL runs that move, and a clip size fix

**Fizgig talks to RunPod through its REST API.** RunPod is turning off the GraphQL API that the Pods tab, pod auto-stop and pod passwords used, in January 2027. Every call now goes through the REST API instead, so the Pods tab keeps working after that date; older versions of Fizgig lose it then. Along the way:
- A pod that is still booting shows as Starting... until it is up, instead of a "didn't start" note after 90 s.
- Stop on a pod that is already stopped just counts as done, instead of an error.

**SDXL runs move and share.** Move a run (and training share with a helper) refused an SDXL run with the text encoder slot left empty, as it is by default, then waited for a download that never came. SDXL's checkpoint holds its own VAE and text encoders, so a pod or helper now takes them from the checkpoint, never from a separate VAE or encoder file it happens to have set.

**Fixes**
- The Training tab shows the clip size row for a training folder whose clips are all in sub-folders, and the run uses it (thanks @spaghettified26, #278).

**Updating:** update Fizgig on your computers, and restart your pods.
