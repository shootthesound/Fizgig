# Fizgig - Independent LoRA Training Framework for Klein 9B
__version__ = "0.1.0"

import os as _os
# No telemetry: onnxruntime's Linux builds (1.29+) report a device ID and machine details to Microsoft unless this is
# set, and Hugging Face's libraries send usage pings. Set before anything loads them; an existing value wins.
# The Windows builds' own switch is utils.ort_telemetry.off(), called where onnxruntime loads.
_os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# torch.compile's on-disk cache (compiled kernels, keyed by GPU, torch version and code) on a pod's persistent volume:
# its default is /tmp on the container disk, which a restart, Pause / Resume or Move wipes, so every resume compiled
# from nothing. On a PC the default (%TEMP% / /tmp) already survives between runs. An existing value wins.
if _os.environ.get("FIZGIG_POD") == "1":
    _os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _os.path.join(
        _os.environ.get("FIZGIG_HOME") or "/workspace", ".cache", "torchinductor"))

# bitsandbytes ships libraries for some CUDA versions only (0.50.2: up to 13.2): on PyTorch's CUDA 13.4 it logs "No
# prebuilt binary for CUDA 13.4, loading CUDA 13.2 instead" and loads that one, which works (same CUDA major). Drop
# just that line. (Setting BNB_CUDA_VERSION instead makes it print a longer override warning.)
import logging as _logging
_logging.getLogger("bitsandbytes.cextension").addFilter(
    lambda r: "No prebuilt binary for CUDA" not in r.getMessage())

# AMD on Linux: expandable segments under ROCm's own name (it ignores PYTORCH_CUDA_ALLOC_CONF), set before the lines
# below first touch the GPU - the allocator reads it then. Only on AMD's datacenter cards: on consumer Radeon cards
# they corrupt memory silently (utils/rocm_alloc.py, #363). Without them a 16 GB-capped MI350X's Krea 2 LoKR run at
# 0.5 MP ran out with 1.8 GB held in unusable pieces (#341). FIZGIG_NO_EXPANDABLE=1 opts out.
if _os.environ.get("FIZGIG_GPU_BACKEND") == "rocm" and _os.name != "nt":
    from fizgig.utils.rocm_alloc import expandable_ok as _expandable_ok
    if _expandable_ok():
        _os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")
# AMD with built-in graphics listed first: the card is the one used - the launcher's pick even where the ROCm build
# ignores HIP_VISIBLE_DEVICES, and the card itself when nothing picked one (#342). Only on AMD launches.
if _os.environ.get("FIZGIG_GPU_BACKEND") == "rocm":
    from fizgig.utils.gpu_backend import pin_rocm_device as _pin
    _pin()
# AMD: every process says which devices it sees and which it uses, on stderr - so each stage's line is in its run log
# (#342: a card mix-up in one stage's process was invisible)
if _os.environ.get("FIZGIG_GPU_BACKEND") == "rocm":
    from fizgig.utils.gpu_backend import report_rocm_devices as _report
    _report()
# AMD: ROCm's fused attention mishandles a key-padding mask that broadcasts over the queries (#341, #343); expand it
# first (gpu_backend.fix_rocm_sdpa_masks; families/driver.py installs it too, for a launch without this variable)
if _os.environ.get("FIZGIG_GPU_BACKEND") == "rocm":
    from fizgig.utils.gpu_backend import fix_rocm_sdpa_masks as _masks
    _masks()
# AMD: MIOpen off for convolutions when Preferences or FIZGIG_MIOPEN=0 say so (gpu_backend.miopen_switch, #389)
if _os.environ.get("FIZGIG_GPU_BACKEND") == "rocm":
    from fizgig.utils.gpu_backend import miopen_switch as _miopen
    _miopen()

from fizgig.utils.container_ram import install as _container_ram
_container_ram()     # a pod's RAM checks read its own limit, not the host's (Linux containers only)
