# Fizgig v5.3.2

A fix for fresh installs: triton is pinned to the version torch pairs with, and a mismatched triton no longer hangs Krea 2 previews.

## triton pinned to torch

`triton-windows` was unpinned, and PyPI recently moved to a 3.8 build made for a newer torch than the 2.10 Fizgig ships. A fresh install picked it up, and with Compile Blocks on, Krea 2 training previews stalled inside torch.compile with nothing in the log (reported on YouTube — thank you). Installs made before that got a matching triton and never saw it.

Fixed two ways. The requirement now reads `triton-windows>=3.5.1,<3.7`, the range torch 2.10 works with, so the installer gets the right one and the updater downgrades an install that has 3.8. And the compile guard now checks that triton actually pairs with torch before compiling anything: if it doesn't, training runs uncompiled and the console says exactly which triton to install. That also covers Auto, which no longer chooses compile on a mismatched machine.

If you installed recently and saw previews hang during Krea 2 training, update and the updater will put triton right; or run `pip install "triton-windows>=3.5.1,<3.7"` in the venv yourself.
