"""A .safetensors file's metadata, read and rewritten without torch: the format is an 8-byte little-endian header
length, a JSON header (its "__metadata__" holds the string metadata) and the raw tensor bytes, which a new header
leaves untouched (their offsets count from the end of the header). So a metadata edit is a header swap and a byte
copy - fast for a 20 GB checkpoint, and every tensor stays bit-identical."""

import json
import os
import struct

_COPY = 64 << 20


def _header(f):
    """(length, header) - ValueError when the length can't be a header's: over 100 MB or past the end of the file. A
    GGUF's magic and version read as a length of ~14 GB, and reading that much first cost seconds and gigabytes."""
    n = struct.unpack("<Q", f.read(8))[0]
    if n <= 0 or n > 100 << 20 or n > os.fstat(f.fileno()).st_size - 8:
        raise ValueError("Not a .safetensors file.")
    return n, json.loads(f.read(n).decode("utf-8"))


def read_header(path):
    """The file's whole JSON header (tensor names, dtypes, shapes, "__metadata__"), without reading any weights.
    ValueError / OSError / struct.error when it can't be read as a .safetensors file."""
    with open(path, "rb") as f:
        return _header(f)[1]


def read_metadata(path):
    """{key: value} of the file's metadata ({} without any)."""
    with open(path, "rb") as f:
        _n, head = _header(f)
    return dict(head.get("__metadata__") or {})


def write_metadata(src, dest, metadata):
    """Write `src` to `dest` (may be the same file) with `metadata` in place of its own, through a temp file swapped in
    at the end - an interrupted save never leaves a half-written file."""
    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    tmp = dest + ".tmp"
    with open(src, "rb") as f:
        n, head = _header(f)
        head.pop("__metadata__", None)
        if meta:
            head = {"__metadata__": meta, **head}
        raw = json.dumps(head, separators=(",", ":")).encode("utf-8")
        raw += b" " * (-len(raw) % 8)               # the header padded to 8 bytes, as the safetensors writer does
        try:
            with open(tmp, "wb") as out:
                out.write(struct.pack("<Q", len(raw)))
                out.write(raw)
                f.seek(8 + n)
                while chunk := f.read(_COPY):
                    out.write(chunk)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
    os.replace(tmp, dest)
