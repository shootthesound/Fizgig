"""Stream a plain Krea 2 checkpoint into NF4 without a full BF16 CPU copy."""
import logging

import torch

from fizgig.krea2.model import SingleStreamDiT
from fizgig.krea2.safetensors_utils import MemoryEfficientSafeOpen
from fizgig.krea2.utils import (
    KREA2_FP8_OPTIMIZATION_TARGET_KEYS,
    KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS,
    single_mmdit_large_wide,
)
from fizgig.modules.nf4 import nf4_linear_forward_patch

logger = logging.getLogger(__name__)


def load_nf4_streamed(path, device="cuda", dtype=torch.bfloat16, config=single_mmdit_large_wide):
    from bitsandbytes.functional import quantize_nf4

    with torch.device("meta"):
        model = SingleStreamDiT(config)
    expected = model.state_dict()
    count = 0
    with MemoryEfficientSafeOpen(path, disable_numpy_memmap=True) as source:
        if set(source.keys()) != set(expected):
            raise ValueError("Streamed NF4 loading requires a plain, complete Krea 2 checkpoint")
        # Validate before allocating GPU memory; packed fp8 requires its own loader.
        for key, tensor in expected.items():
            info = source.header[key]
            if tuple(info["shape"]) != tuple(tensor.shape) or info["dtype"] not in ("BF16", "F16", "F32"):
                raise ValueError(f"Unsupported Krea 2 tensor for streamed NF4: {key}")
        for key in source.keys():
            name, leaf = key.rsplit(".", 1)
            module = model.get_submodule(name)
            weight = source.get_tensor(key, device=torch.device(device), dtype=dtype)
            quantize = (leaf == "weight" and isinstance(module, torch.nn.Linear)
                        and any(t in name for t in KREA2_FP8_OPTIMIZATION_TARGET_KEYS)
                        and not any(t in name for t in KREA2_FP8_OPTIMIZATION_EXCLUDE_KEYS))
            if quantize:
                # Match apply_nf4_quantization: BF16 source, no nested statistics.
                packed, state = quantize_nf4(weight.to(torch.bfloat16).contiguous(), compress_statistics=False)
                module._nf4_packed, module._nf4_state = packed, state
                module._is_nf4 = True
                module.weight = torch.nn.Parameter(torch.empty(0, device=device, dtype=torch.bfloat16), requires_grad=False)
                module.forward = nf4_linear_forward_patch.__get__(module, type(module))
                count += 1
            else:
                setattr(module, leaf, torch.nn.Parameter(weight, requires_grad=False))
            del weight
    model._nf4_quantized = True
    logger.info("[rocm] streamed %d frozen Linears into NF4; no full BF16 host copy", count)
    return model
