# Adapted from musubi-tuner (https://github.com/kohya-ss/musubi-tuner), Apache-2.0.
# Modified for Fizgig. See docs/THIRD_PARTY_NOTICES.md.
from dataclasses import dataclass
import mmap
import os
import re
import numpy as np
import torch
import json
import struct
from typing import Dict, Any, Union, Optional

from safetensors.torch import load_file

from fizgig.krea2.device_utils import synchronize_device


def release_mapped_pages(weights):
    """Drop resident pages of immutable checkpoint views; the file remains unchanged."""
    seen = set()
    for weight in weights:
        mapping = getattr(weight.untyped_storage(), "_fizgig_mmap", None)
        if mapping is not None and id(mapping) not in seen:
            seen.add(id(mapping))
            try:
                mapping.madvise(mmap.MADV_DONTNEED)
            except OSError:
                pass  # The OS can still reclaim these clean file-backed pages.


def move_frozen_weights(module, device, masters):
    """Stage file-backed CPU weights so ROCm never pins their private file mappings."""
    if not masters or torch.device(device).type != "cuda":
        return
    for parameter in module.parameters():
        source = masters.get(parameter)
        if source is not None and parameter.device.type == "cpu":
            staging = source.pin_memory(device=device)
            release_mapped_pages((source,))
            parameter.data = staging.to(device)  # Blocking: the pinned buffer can then be reused safely.


def mem_eff_save_file(tensors: Dict[str, torch.Tensor], filename: str, metadata: Dict[str, Any] = None):
    """
    memory efficient save file
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

    def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
        validated = {}
        for key, value in metadata.items():
            if not isinstance(key, str):
                raise ValueError(f"Metadata key must be a string, got {type(key)}")
            if not isinstance(value, str):
                print(f"Warning: Metadata value for key '{key}' is not a string. Converting to string.")
                validated[key] = str(value)
            else:
                validated[key] = value
        return validated

    # print(f"Using memory efficient save file: {filename}")

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = validate_metadata(metadata)
    for k, v in tensors.items():
        if v.numel() == 0:  # empty tensor
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
                # Direct GPU to disk save
                with torch.cuda.device(v.device):
                    if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                        v = v.unsqueeze(0)
                    tensor_bytes = v.contiguous().view(torch.uint8)
                    tensor_bytes.cpu().numpy().tofile(f)
            else:
                # CPU tensor save
                if v.dim() == 0:  # if scalar, need to add a dimension to work with view
                    v = v.unsqueeze(0)
                v.contiguous().view(torch.uint8).numpy().tofile(f)


def stream_save_file(specs, filename: str, metadata: Dict[str, Any] = None):
    """Write a safetensors file from LAZY tensor producers — one tensor in memory at a time.

    mem_eff_save_file above streams the WRITE but still takes the whole dict of tensors in
    RAM. Rotation fine-tune saves can't afford that: overlaying a ~21 GB checkpoint while a
    ~38 GB bf16 master is resident would crowd a 64 GB box. The safetensors format needs the
    full header (dtype/shape/offsets) before the first payload byte, but shapes are known
    without materializing anything — so the caller declares them up front and each tensor is
    produced only when it is that key's turn to hit the disk, then freed.

    specs: ordered dict {key: (torch_dtype, shape_tuple, producer)} where producer is a
    zero-arg callable returning the tensor (any device; CUDA tensors are copied via CPU on
    write). Produced dtype/shape are asserted against the declaration — a mismatch would
    silently corrupt every later offset in the file.

    Atomicity is the CALLER's job (write to path+'.tmp', os.replace) — same contract as
    _save_full_checkpoint."""
    _TYPES = {
        torch.float64: "F64", torch.float32: "F32", torch.float16: "F16",
        torch.bfloat16: "BF16", torch.int64: "I64", torch.int32: "I32",
        torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
        getattr(torch, "float8_e5m2", None): "F8_E5M2",
        getattr(torch, "float8_e4m3fn", None): "F8_E4M3",
    }
    _ALIGN = 256
    _ELEM = {torch.float64: 8, torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
             torch.int64: 8, torch.int32: 4, torch.int16: 2, torch.int8: 1,
             torch.uint8: 1, torch.bool: 1,
             getattr(torch, "float8_e5m2", None): 1,
             getattr(torch, "float8_e4m3fn", None): 1}

    header = {}
    offset = 0
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    for k, (dt, shape, _p) in specs.items():
        numel = 1
        for d in shape:
            numel *= int(d)
        size = numel * _ELEM[dt]
        header[k] = {"dtype": _TYPES[dt], "shape": [int(d) for d in shape],
                     "data_offsets": [offset, offset + size]}
        offset += size

    hjson = json.dumps(header).encode("utf-8")
    hjson += b" " * (-(len(hjson) + 8) % _ALIGN)

    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        for k, (dt, shape, producer) in specs.items():
            v = producer()
            if v.dtype != dt or tuple(v.shape) != tuple(shape):
                raise RuntimeError(
                    f"stream_save_file: producer for {k!r} returned "
                    f"{v.dtype}/{tuple(v.shape)}, declared {dt}/{tuple(shape)} — "
                    "every later offset in the file would be wrong")
            if v.numel() == 0:
                continue
            if v.dim() == 0:
                v = v.unsqueeze(0)
            v = v.contiguous()
            if v.is_cuda:
                v = v.cpu()
            v.view(torch.uint8).numpy().tofile(f)
            del v


class MemoryEfficientSafeOpen:
    """Memory-efficient reader for safetensors files.

    This class provides a memory-efficient way to read tensors from safetensors files
    by using memory mapping for large tensors and avoiding unnecessary copies.
    """

    def __init__(self, filename, disable_numpy_memmap=False, mmap_cpu=False):
        """Initialize the SafeTensor reader.

        Args:
            filename (str): Path to the safetensors file to read.
            disable_numpy_memmap (bool): If True, disable numpy memory mapping for large tensors, using standard file read instead.
            mmap_cpu (bool): Keep frozen inference weights in private file-backed CPU mappings.
        """
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()
        self.disable_numpy_memmap = disable_numpy_memmap
        self.mmap_cpu = mmap_cpu

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close file."""
        self.file.close()

    def keys(self):
        """Get all tensor keys in the file.

        Returns:
            list: List of tensor names (excludes metadata).
        """
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        """Get metadata from the file.

        Returns:
            Dict[str, str]: Metadata dictionary.
        """
        return self.header.get("__metadata__", {})

    def _read_header(self):
        """Read and parse the header from the safetensors file.

        Returns:
            tuple: (header_dict, header_size) containing parsed header and its size.
        """
        from fizgig.utils.st_meta import _header      # refuses a length no header has (a GGUF's reads as ~14 GB)
        header_size, header = _header(self.file)
        return header, header_size

    def get_tensor(self, key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Load a tensor from the file with memory-efficient strategies.

        **Note:**
        If device is 'cuda' , the transfer to GPU is done efficiently using pinned memory and non-blocking transfer.
        So you must ensure that the transfer is completed before using the tensor (e.g., by `torch.cuda.synchronize()`).

        If the tensor is large (>10MB) and the target device is CUDA, memory mapping with numpy.memmap is used to avoid intermediate copies.

        Args:
            key (str): Name of the tensor to load.
            device (Optional[torch.device]): Target device for the tensor.
            dtype (Optional[torch.dtype]): Target dtype for the tensor.

        Returns:
            torch.Tensor: The loaded tensor.

        Raises:
            KeyError: If the tensor key is not found in the file.
        """
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]
        num_bytes = offset_end - offset_start

        original_dtype = self._get_torch_dtype(metadata["dtype"])
        target_dtype = dtype if dtype is not None else original_dtype

        # Handle empty tensors
        if num_bytes == 0:
            return torch.empty(metadata["shape"], dtype=target_dtype, device=device)

        # Determine if we should use pinned memory for GPU transfer
        non_blocking = device is not None and device.type == "cuda"

        # Calculate absolute file offset
        tensor_offset = self.header_size + 8 + offset_start  # adjust offset by header size

        # CPU targets normally use reads; frozen inference explicitly requests private mappings.
        cpu_mapping = self.mmap_cpu and (device is None or device.type == "cpu")
        if not self.disable_numpy_memmap and (cpu_mapping or
                (num_bytes > 10 * 1024 * 1024 and device is not None and device.type != "cpu")):
            # Create memory map for zero-copy reading
            mm = np.memmap(self.filename, mode="c", dtype=np.uint8, offset=tensor_offset, shape=(num_bytes,))
            mapping = mm._mmap
            byte_tensor = torch.from_numpy(mm)  # zero copy
            del mm

            # Deserialize tensor (view and reshape)
            cpu_tensor = self._deserialize_tensor(byte_tensor, metadata)  # view and reshape
            if cpu_mapping:
                cpu_tensor.untyped_storage()._fizgig_mmap = mapping
            del byte_tensor

            # Transfer to target device and dtype
            gpu_tensor = cpu_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
            del cpu_tensor
            return gpu_tensor

        # Standard file reading strategy for smaller tensors or CPU target
        # seek to the specified position
        self.file.seek(tensor_offset)

        # read directly into a numpy array by numpy.fromfile without intermediate copy
        numpy_array = np.fromfile(self.file, dtype=np.uint8, count=num_bytes)
        byte_tensor = torch.from_numpy(numpy_array)
        del numpy_array

        # deserialize (view and reshape)
        deserialized_tensor = self._deserialize_tensor(byte_tensor, metadata)
        del byte_tensor

        # cast to target dtype and move to device
        return deserialized_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)

    def _deserialize_tensor(self, byte_tensor: torch.Tensor, metadata: Dict):
        """Deserialize byte tensor to the correct shape and dtype.

        Args:
            byte_tensor (torch.Tensor): Raw byte tensor from file.
            metadata (Dict): Tensor metadata containing dtype and shape info.

        Returns:
            torch.Tensor: Deserialized tensor with correct shape and dtype.
        """
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        # Handle special float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # Standard conversion: view as target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        """Convert string dtype to PyTorch dtype.

        Args:
            dtype_str (str): String representation of the dtype.

        Returns:
            torch.dtype: Corresponding PyTorch dtype.
        """
        # Standard dtype mappings
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # Add float8 types if available in PyTorch version
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        """Convert byte tensor to float8 format if supported.

        Args:
            byte_tensor (torch.Tensor): Raw byte tensor.
            dtype_str (str): Float8 dtype string ("F8_E5M2" or "F8_E4M3").
            shape (tuple): Target tensor shape.

        Returns:
            torch.Tensor: Tensor with float8 dtype.

        Raises:
            ValueError: If float8 type is not supported in current PyTorch version.
        """
        # Convert to specific float8 types if available
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            # Float8 not supported in this PyTorch version
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")


class ShardedSafeOpen:
    """A directory of sharded safetensors behind the MemoryEfficientSafeOpen interface.

    Hugging Face serves large checkpoints as `model-0000N-of-0000M.safetensors` plus an index,
    while every streaming loader here takes a single file. This adapts one to the other, so a
    loader can accept either by dispatching on os.path.isdir.

    Shards are opened lazily and kept open: a loader that walks named_parameters() in module
    order touches each shard's keys contiguously, so this is a handful of file handles, not a
    reopen per tensor. The index is trusted when present; without one the headers are read
    (cheap — a header is a few tens of KB regardless of shard size).
    """

    def __init__(self, dirname: str, disable_numpy_memmap: bool = False):
        self.dirname = dirname
        self.disable_numpy_memmap = disable_numpy_memmap
        self._readers: Dict[str, "MemoryEfficientSafeOpen"] = {}

        entries = sorted(os.listdir(dirname))
        index = [e for e in entries if e.endswith(".safetensors.index.json")]
        if index:
            with open(os.path.join(dirname, index[0]), "r", encoding="utf-8") as fh:
                self.weight_map = json.load(fh)["weight_map"]
        else:
            shards = [e for e in entries if e.endswith(".safetensors")]
            if not shards:
                raise FileNotFoundError(f"{dirname}: no .safetensors shards")
            self.weight_map = {}
            for shard in shards:
                with MemoryEfficientSafeOpen(os.path.join(dirname, shard)) as f:
                    for key in f.keys():
                        self.weight_map[key] = shard

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for reader in self._readers.values():
            reader.file.close()
        self._readers.clear()

    def _reader(self, shard: str) -> "MemoryEfficientSafeOpen":
        reader = self._readers.get(shard)
        if reader is None:
            reader = MemoryEfficientSafeOpen(os.path.join(self.dirname, shard), self.disable_numpy_memmap)
            self._readers[shard] = reader
        return reader

    def keys(self):
        return list(self.weight_map.keys())

    def metadata(self) -> Dict[str, str]:
        merged: Dict[str, str] = {}
        for shard in dict.fromkeys(self.weight_map.values()):
            merged.update(self._reader(shard).metadata())
        return merged

    def get_tensor(self, key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"{key} is not in any shard of {self.dirname}")
        return self._reader(shard).get_tensor(key, device=device, dtype=dtype)


def load_safetensors(
    path: str,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    dtype: Optional[torch.dtype] = None,
    disable_numpy_memmap: bool = False,
) -> dict[str, torch.Tensor]:
    if disable_mmap:
        # return safetensors.torch.load(open(path, "rb").read())
        # use experimental loader
        # logger.info(f"Loading without mmap (experimental)")
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
        except:
            state_dict = load_file(path)  # prevent device invalid Error
        if dtype is not None:
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(dtype=dtype)
        return state_dict


def get_split_weight_filenames(file_path: str) -> Optional[list[str]]:
    """
    Get the list of split weight filenames (full paths) if the file name ends with 00001-of-00004 etc.
    Returns None if the file is not split.
    """
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
    else:
        return None


def load_split_weights(
    file_path: str, device: Union[str, torch.device] = "cpu", disable_mmap: bool = False, dtype: Optional[torch.dtype] = None
) -> Dict[str, torch.Tensor]:
    """
    Load split weights from a file. If the file name ends with 00001-of-00004 etc, it will load all files with the same prefix.
    dtype is as is, no conversion is done.
    """
    device = torch.device(device)

    # if the file name ends with 00001-of-00004 etc, we need to load the files with the same prefix
    split_filenames = get_split_weight_filenames(file_path)
    if split_filenames is not None:
        state_dict = {}
        for filename in split_filenames:
            state_dict.update(load_safetensors(filename, device=device, disable_mmap=disable_mmap, dtype=dtype))
    else:
        state_dict = load_safetensors(file_path, device=device, disable_mmap=disable_mmap, dtype=dtype)
    return state_dict


def find_key(safetensors_file: str, starts_with: Optional[str] = None, ends_with: Optional[str] = None) -> Optional[str]:
    """
    Find a key in a safetensors file that starts with `starts_with` and ends with `ends_with`.
    If `starts_with` is None, it will match any key.
    If `ends_with` is None, it will match any key.
    Returns the first matching key or None if no key matches.
    """
    with MemoryEfficientSafeOpen(safetensors_file) as f:
        for key in f.keys():
            if (starts_with is None or key.startswith(starts_with)) and (ends_with is None or key.endswith(ends_with)):
                return key
    return None


def find_keys(safetensors_file: str, starts_with: Optional[str] = None, ends_with: Optional[str] = None) -> list[str]:
    """
    Find all keys in a safetensors file that start with `starts_with` and end with `ends_with`.
    If `starts_with` is None, it will match any key. If `ends_with` is None, it will match any key.

    The matching keys are returned sorted, so callers get a deterministic order regardless of the
    order the keys happen to be stored in the file header. This matters when the keys are folded into
    a bucket key (e.g. ``latents_control_{i}``): otherwise two files holding the same tensors in a
    different header order, or with the index<->shape association swapped, could end up in different
    buckets or be wrongly batched together.
    """
    with MemoryEfficientSafeOpen(safetensors_file) as f:
        keys = [
            key
            for key in f.keys()
            if (starts_with is None or key.startswith(starts_with)) and (ends_with is None or key.endswith(ends_with))
        ]
    return sorted(keys)


@dataclass
class WeightTransformHooks:
    split_hook: Optional[callable] = None
    concat_hook: Optional[callable] = None


class TensorWeightAdapter:
    """
    A wrapper for weight conversion hooks (split and concat) to be used with MemoryEfficientSafeOpen.
    This wrapper adapts the original MemoryEfficientSafeOpen to apply the provided split and concat hooks
    when loading tensors.

    split_hook: A callable that takes (original_key: str, original_tensor: torch.Tensor) and returns (new_keys: list[str], new_tensors: list[torch.Tensor]).
    concat_hook: A callable that takes (original_key: str, tensors: dict[str, torch.Tensor]) and returns (new_key: str,  concatenated_tensor: torch.Tensor).

    If tensors is None, the hook should return only the new keys (for split) or new key (for concat), without tensors.

    No need to implement __enter__ and __exit__ methods, as they are handled by the original MemoryEfficientSafeOpen.
    Do not use this wrapper as a context manager directly, like `with WeightConvertHookWrapper(...) as f:`.

    **concat_hook is not tested yet.**
    """

    def __init__(self, weight_convert_hook: WeightTransformHooks, original_f: MemoryEfficientSafeOpen):
        self.original_f = original_f
        self.new_key_to_original_key_map: dict[
            str, Union[str, list[str]]
        ] = {}  # for split: new_key -> original_key; for concat: new_key -> list of original_keys
        self.concat_key_set = set()  # set of keys that are created by concat_hook, to distinguish from split_hook
        self.new_keys = []
        self.tensor_cache = {}  # cache for split tensors
        self.split_hook = weight_convert_hook.split_hook
        self.concat_hook = weight_convert_hook.concat_hook

        for key in self.original_f.keys():
            if self.split_hook is not None:
                converted_keys, _ = self.split_hook(key, None)  # get new keys only
                if converted_keys is not None:
                    for new_key in converted_keys:
                        self.new_key_to_original_key_map[new_key] = key
                    self.new_keys.extend(converted_keys)
                    continue  # skip concat_hook if split_hook is applied

            if self.concat_hook is not None:
                converted_key, _ = self.concat_hook(key, None)  # get new key only
                if converted_key is not None:
                    if converted_key not in self.concat_key_set:  # first time seeing this concatenated key
                        self.concat_key_set.add(converted_key)
                        self.new_key_to_original_key_map[converted_key] = []

                    # multiple original keys map to the same concatenated key
                    self.new_key_to_original_key_map[converted_key].append(key)

                    self.new_keys.append(converted_key)
                    continue  # skip to next key

            # direct mapping
            self.new_keys.append(key)

    def keys(self) -> list[str]:
        return self.new_keys

    def get_tensor(self, new_key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # load tensor by new_key, applying split or concat hooks as needed
        if new_key not in self.new_key_to_original_key_map:
            # direct mapping
            return self.original_f.get_tensor(new_key, device=device, dtype=dtype)

        elif new_key not in self.concat_key_set:
            # split hook: split key is requested multiple times, so we cache the result
            original_key = self.new_key_to_original_key_map[new_key]
            if original_key not in self.tensor_cache:  # not yet split
                original_tensor = self.original_f.get_tensor(original_key, device=device, dtype=dtype)
                new_keys, new_tensors = self.split_hook(original_key, original_tensor)  # apply split hook
                for k, t in zip(new_keys, new_tensors):
                    self.tensor_cache[k] = t
            return self.tensor_cache.pop(new_key)  # return and remove from cache

        else:
            # concat hook: concatenated key is requested only once, so we do not cache the result
            tensors = {}
            for original_key in self.new_key_to_original_key_map[new_key]:
                tensor = self.original_f.get_tensor(original_key, device=device, dtype=dtype)
                tensors[original_key] = tensor
            _, concatenated_tensors = self.concat_hook(self.new_key_to_original_key_map[new_key][0], tensors)  # apply concat hook
            return concatenated_tensors
