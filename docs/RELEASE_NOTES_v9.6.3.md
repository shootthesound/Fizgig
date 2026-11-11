# Fizgig v9.6.3: training suddenly slow on Windows? Turn off Sysmem Fallback

**PSA for anyone training on Windows with an NVIDIA card:** if training is suddenly several times slower than it should be, especially after a driver or Windows update, it's probably the driver spilling part of the run into system RAM. Nothing crashes, it just crawls.

**Fix:** NVIDIA Control Panel → Manage 3D settings → **Global Settings** → set **CUDA - Sysmem Fallback Policy** to **Prefer No Sysmem Fallback** → Apply, then restart Fizgig. Use Global Settings, not Program Settings: Fizgig's Python there is only a launcher, so the per-program setting never reaches it.

A few people went from 10–15 s/it back to about 2.5 s/it with this. With it off, a run too big for your card stops with an out-of-memory message instead of slowing down, and Fizgig retries it lighter by itself.

**Fizgig now shows these steps when it starts** on Windows with an NVIDIA card, until you tick **Don't show this again**. More in the [install guide](https://github.com/shootthesound/Fizgig/blob/master/docs/INSTALL.md#training-suddenly-several-times-slower-windows-shared-gpu-memory).
