# Fizgig v9.2.1: Python 3.11 or newer

**Fizgig now needs Python 3.11 or newer.** PyTorch 2.15, which 9.2 moved to, has no Python 3.10 build, so updating an install made with Python 3.10 couldn't install it. Installs on Python 3.11, 3.12 or 3.13 aren't affected, and AMD ROCm installs keep their own PyTorch.

**If your install uses Python 3.10,** the updater now says so and stops before changing anything. To fix it:
1. Install Python 3.12 from https://www.python.org/downloads/ (on Windows, tick "Add Python to PATH").
2. Run `install_fizgig.bat` (Linux: `python3.12 install_fizgig.py`). It rebuilds Fizgig's Python environment on 3.12. Your models, datasets, LoRAs and settings are kept.

If the 9.2 update stopped with "Failed to install/update dependencies", this is why: follow the two steps above.

**Pods:** the pod image moves to Ubuntu 24.04 (Python 3.12) so it can carry PyTorch 2.15.
