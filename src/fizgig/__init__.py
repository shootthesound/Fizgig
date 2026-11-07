# Fizgig - Independent LoRA Training Framework for Klein 9B
__version__ = "0.1.0"

import os as _os
# No telemetry: onnxruntime's Linux builds (1.29+) report a device ID and machine details to Microsoft unless this is
# set, and Hugging Face's libraries send usage pings. Set before anything loads them; an existing value wins.
# The Windows builds' own switch is utils.ort_telemetry.off(), called where onnxruntime loads.
_os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
_os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from fizgig.utils.container_ram import install as _container_ram
_container_ram()     # a pod's RAM checks read its own limit, not the host's (Linux containers only)
