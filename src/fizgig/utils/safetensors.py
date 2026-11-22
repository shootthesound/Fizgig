"""Safetensors I/O utilities for memory-efficient loading and saving."""

from dataclasses import dataclass
import json
import os
import re
import struct
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from safetensors.torch import load_file

from fizgig.utils.device import synchronize_device

import logging

logger = logging.getLogger(__name__)


def warm_file_cache(path: str) -> None:
    """Stream a model file once so the safetensors mmap hits page cache.

    Pod-only (FIZGIG_POD=1): RunPod container disks read ~600 MB/s sequentially but serve
    mmap page-faults at high per-request latency, so loading a 26 GB DiT through mmap took
    20+ minutes and looked like a hang. One sequential pass fills the page cache (~45 s cold,
    ~2 s if already cached) and the mmap load behind it becomes memory-speed. On desktops the
    pref is a no-op — local NVMe serves mmap faults fine and the extra read would only cost.
    Never fatal: any error just falls through to the normal load path.
    """
    if os.environ.get("FIZGIG_POD") != "1":
        return
    try:
        size = os.path.getsize(path)
        chunk = 64 * 1024 * 1024
        logged = 0
        import time as _time
        t0 = _time.time()
        with open(path, "rb", buffering=0) as f:
            done = 0
            while True:
                b = f.read(chunk)
                if not b:
                    break
                done += len(b)
                if done - logged >= 8 * chunk:  # every ~512 MB
                    logged = done
                    logger.info(f"[warm] caching model file {done / 1e9:.1f}/{size / 1e9:.1f} GB")
        dt = _time.time() - t0
        if dt > 2.0:
            logger.info(f"[warm] {size / 1e9:.1f} GB cached in {dt:.0f} s "
                        f"({size / 1e9 / max(dt, 0.001):.1f} GB/s) — load follows from RAM")
    except Exception as e:  # a failed warm must never block the real load
        logger.warning(f"[warm] skipped ({e}) — loading directly")


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    """Memory-efficient save to safetensors format.

    Writes tensors directly from GPU to disk when possible, avoiding unnecessary copies.
    """
    _TYPES = {
        torch.float64: "F64",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.int16: "I16",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256

    def validate_metadata(md: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in md.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)

    for k, v in tensors.items():
        if v.numel() == 0:
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset]}
        else:
            size = v.numel() * v.element_size()
            header[k] = {"dtype": _TYPES[v.dtype], "shape": list(v.shape), "data_offsets": [offset, offset + size]}
            offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)

        for k, v in tensors.items():
            if v.numel() == 0:
                continue
            if v.is_cuda:
                with torch.cuda.device(v.device):
                    if v.dim() == 0:
                        v = v.unsqueeze(0)
                    tensor_bytes = v.contiguous().view(torch.uint8)
                    tensor_bytes.cpu().numpy().tofile(f)
            else:
                if v.dim() == 0:
                    v = v.unsqueeze(0)
                v.contiguous().view(torch.uint8).numpy().tofile(f)


class MemoryEfficientSafeOpen:
    """Memory-efficient reader for safetensors files.

    Uses numpy memory mapping for large tensors and avoids unnecessary copies.
    """

    def __init__(self, filename, disable_numpy_memmap=False):
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()
        self.disable_numpy_memmap = disable_numpy_memmap

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        return self.header.get("__metadata__", {})

    def _read_header(self):
        from fizgig.utils.st_meta import _header      # refuses a length no header has (a GGUF's reads as ~14 GB)
        header_size, header = _header(self.file)
        return header, header_size

    def get_tensor(self, key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Load a tensor from the file.

        Note: If device is 'cuda', transfer uses pinned memory and non-blocking copy.
        Call torch.cuda.synchronize() before using the tensor.
        """
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]
        num_bytes = offset_end - offset_start

        original_dtype = self._get_torch_dtype(metadata["dtype"])
        target_dtype = dtype if dtype is not None else original_dtype

        if num_bytes == 0:
            return torch.empty(metadata["shape"], dtype=target_dtype, device=device)

        non_blocking = device is not None and device.type == "cuda"
        tensor_offset = self.header_size + 8 + offset_start

        # Use memmap for large tensors going to GPU
        if not self.disable_numpy_memmap and num_bytes > 10 * 1024 * 1024 and device is not None and device.type != "cpu":
            mm = np.memmap(self.filename, mode="c", dtype=np.uint8, offset=tensor_offset, shape=(num_bytes,))
            byte_tensor = torch.from_numpy(mm)
            del mm
            cpu_tensor = self._deserialize_tensor(byte_tensor, metadata)
            del byte_tensor
            gpu_tensor = cpu_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
            del cpu_tensor
            return gpu_tensor

        # Standard file reading for smaller tensors or CPU target
        self.file.seek(tensor_offset)
        numpy_array = np.fromfile(self.file, dtype=np.uint8, count=num_bytes)
        byte_tensor = torch.from_numpy(numpy_array)
        del numpy_array
        deserialized = self._deserialize_tensor(byte_tensor, metadata)
        del byte_tensor
        return deserialized.to(device=device, dtype=target_dtype, non_blocking=non_blocking)

    def _deserialize_tensor(self, byte_tensor: torch.Tensor, metadata: Dict):
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
            "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
            "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch)")


def load_safetensors(
    path: str,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
    disable_numpy_memmap: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load a safetensors file, optionally with memory-efficient loading."""
    if disable_mmap:
        state_dict = {}
        device = torch.device(device) if device is not None else None
        with MemoryEfficientSafeOpen(path, disable_numpy_memmap=disable_numpy_memmap) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key, device=device, dtype=dtype)
        synchronize_device(device)
        return state_dict
    else:
        try:
            state_dict = load_file(path, device=device)
        except Exception:
            state_dict = load_file(path)
        if dtype is not None:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(dtype=dtype)
        return state_dict


def get_split_weight_filenames(file_path: str) -> Optional[List[str]]:
    """Get list of split weight filenames if the file name matches 00001-of-00004 pattern."""
    basename = os.path.basename(file_path)
    match = re.match(r"^(.*?)(\d+)-of-(\d+)\.safetensors$", basename)
    if match:
        prefix = basename[: match.start(2)]
        count = int(match.group(3))
        filenames = []
        for i in range(count):
            filename = f"{prefix}{i + 1:05d}-of-{count:05d}.safetensors"
            filepath = os.path.join(os.path.dirname(file_path), filename)
            if os.path.exists(filepath):
                filenames.append(filepath)
            else:
                raise FileNotFoundError(f"File {filepath} not found")
        return filenames
    return None


def load_split_weights(
    file_path: str, device: Union[str, torch.device] = "cpu", disable_mmap: bool = False, dtype: Optional[torch.dtype] = None
) -> Dict[str, torch.Tensor]:
    """Load split weight files. Handles both single and multi-part safetensors."""
    device = torch.device(device)
    split_filenames = get_split_weight_filenames(file_path)
    if split_filenames is not None:
        state_dict = {}
        for filename in split_filenames:
            state_dict.update(load_safetensors(filename, device=device, disable_mmap=disable_mmap, dtype=dtype))
    else:
        state_dict = load_safetensors(file_path, device=device, disable_mmap=disable_mmap, dtype=dtype)
    return state_dict


def find_key(safetensors_file: str, starts_with: Optional[str] = None, ends_with: Optional[str] = None) -> Optional[str]:
    """Find a key in a safetensors file matching the given prefix/suffix."""
    with MemoryEfficientSafeOpen(safetensors_file) as f:
        for key in f.keys():
            if (starts_with is None or key.startswith(starts_with)) and (ends_with is None or key.endswith(ends_with)):
                return key
    return None


@dataclass
class WeightTransformHooks:
    split_hook: Optional[callable] = None
    concat_hook: Optional[callable] = None


class TensorWeightAdapter:
    """Adapter for applying split/concat weight transformations during safetensors loading."""

    def __init__(self, weight_convert_hook: WeightTransformHooks, original_f: MemoryEfficientSafeOpen):
        self.original_f = original_f
        self.new_key_to_original_key_map: Dict[str, Union[str, List[str]]] = {}
        self.concat_key_set = set()
        self.new_keys = []
        self.tensor_cache = {}
        self.split_hook = weight_convert_hook.split_hook
        self.concat_hook = weight_convert_hook.concat_hook

        for key in self.original_f.keys():
            if self.split_hook is not None:
                converted_keys, _ = self.split_hook(key, None)
                if converted_keys is not None:
                    for new_key in converted_keys:
                        self.new_key_to_original_key_map[new_key] = key
                    self.new_keys.extend(converted_keys)
                    continue

            if self.concat_hook is not None:
                converted_key, _ = self.concat_hook(key, None)
                if converted_key is not None:
                    if converted_key not in self.concat_key_set:
                        self.concat_key_set.add(converted_key)
                        self.new_key_to_original_key_map[converted_key] = []
                    self.new_key_to_original_key_map[converted_key].append(key)
                    self.new_keys.append(converted_key)
                    continue

            self.new_keys.append(key)

    def keys(self) -> List[str]:
        return self.new_keys

    def get_tensor(self, new_key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if new_key not in self.new_key_to_original_key_map:
            return self.original_f.get_tensor(new_key, device=device, dtype=dtype)

        if new_key not in self.concat_key_set:
            # Split hook
            original_key = self.new_key_to_original_key_map[new_key]
            if original_key not in self.tensor_cache:
                original_tensor = self.original_f.get_tensor(original_key, device=device, dtype=dtype)
                new_keys, new_tensors = self.split_hook(original_key, original_tensor)
                for k, t in zip(new_keys, new_tensors):
                    self.tensor_cache[k] = t
            return self.tensor_cache.pop(new_key)

        else:
            # Concat hook
            tensors = {}
            for original_key in self.new_key_to_original_key_map[new_key]:
                tensor = self.original_f.get_tensor(original_key, device=device, dtype=dtype)
                tensors[original_key] = tensor
            _, concatenated_tensors = self.concat_hook(self.new_key_to_original_key_map[new_key][0], tensors)
            return concatenated_tensors
