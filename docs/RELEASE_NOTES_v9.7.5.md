# Fizgig v9.7.5: MiniMax H3 at int8 on any pruned file, and H3 Image on 12 GB cards

**MiniMax H3 and H3 Image train at int8 on any pruned file.** A pruned bf16 or fp8 file (Comfy-Org's pruned builds, or a fine-tune or merge saved that way) now trains at int8: Fizgig converts its block weights as it loads to the same int8 storage the pruned int8 files carry, which takes a minute or two longer. Before, Auto picked int8 for such a file on a card big enough for it, and the run stopped at load with "needs a pre-quantized checkpoint". In our testing, H3's pruned bf16 file converts to Comfy-Org's pruned int8 file value for value, and H3 Image's matches its int8 file to within 2 steps on every weight. A pruned fp8 file trains at 4-bit too (it stopped with "unsupported comfy quant format"). The full unpruned checkpoint still trains at 4-bit, as int8 would keep about 26 GB of its timestep layers at full size; pick int8 for it and Start says to pick 4-bit or a pruned file (#357, thanks @Aymenbayoss).

**MiniMax H3 Image on 12 GB cards.** Auto now swaps blocks for H3 Image's 4-bit base on a small card. Before, on a 12 GB card it planned 4-bit with no blocks swapped, which needed 12.8 GB and on Windows spilled into system memory at about 40 s a step (16 blocks swapped by hand brought the same run to 4.1 s). Auto now plans from memory figures measured for small cards: 4-bit with 15 blocks swapped at 0.25 MP on a 12 GB card. Out-of-memory recovery raises block swap for it too. 16 GB cards still plan 4-bit with no swap (#354, thanks @howDoILearnPython).

**The Linux AMD installer works on Arch and CachyOS.** The ROCm installer looked for libnuma only where Debian and Ubuntu keep it, so on Arch-based systems it tried to install it, found no apt, dnf or yum, and stopped. It now finds libnuma wherever the system keeps it, and installs it with pacman or zypper as well (#356, thanks @xGnome).

**The Training tab lists optimizers without loading them.** Drawing the optimizer list imported packages such as bitsandbytes, loading their native libraries into the app just to show their names. It now checks what's installed without loading it, on NVIDIA and AMD alike (#346, thanks @xGnome).

**Big datasets on a pod.** The pod guide (`docker/README.md`) has a new section on downloading a very big dataset straight onto the pod from Hugging Face, Google Drive or a direct link, instead of sending it up from your computer (#308, thanks @girzai).

**Gizmo shows why a recording take failed.** A failed take left the status line as it was; it now shows the error.

**To update:** run the updater as usual.
