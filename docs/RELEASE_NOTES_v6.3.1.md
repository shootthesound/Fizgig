# Fizgig v6.3.1

A maintenance release: the Windows updater no longer trips antivirus, and a Krea 2 preview failure can no longer slow the rest of a training run.

## The updater no longer looks like malware to antivirus

Reported by **@VRAM-Hoarder** (#157), who also traced the cause and proposed the fix. To survive `git pull` rewriting it mid-run, `update_fizgig.bat` copied itself to `%TEMP%` and re-ran the copy. Antivirus tools treat "copies itself to TEMP and runs the copy" as dropper behaviour and blocked the updater.

All the update logic now lives in `update_fizgig.py`. `update_fizgig.bat` and `update_fizgig_rocm.bat` are a few static lines that run it, so nothing copies itself anywhere. The copy the old updater left in `%TEMP%` is deleted the next time you update.

The rewrite also fixed a few older rough edges:

- A venv with a missing or broken PyTorch used to be reported as "an AMD ROCm install". `update_fizgig.bat` now reinstalls PyTorch instead; `update_fizgig_rocm.bat` tells you to re-run `install_fizgig_rocm.bat`.
- If `git` isn't on your PATH, the update says so and carries on with the rest, as before.
- Both updaters now report failure through their exit code, and pause on errors so the window doesn't close before you can read them.

**What to do:** run `update_fizgig.bat` (or `update_fizgig_rocm.bat`) as usual. If your antivirus already blocks the old updater, run `git pull` once in the Fizgig folder, then use the updater as normal from then on.

## Krea 2: a failed preview decode no longer strands the model on the CPU

Spotted by **@Linkram** while reviewing PR #152. On smaller cards, Krea 2 previews move the training model off the GPU while the image is decoded. If that decode failed (usually out of memory), the model was never moved back, and training carried on much more slowly. It is now always moved back, whether the decode succeeds or not.
