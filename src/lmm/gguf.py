"""Read GGUF headers (metadata + tensor names) without loading weight data."""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# GGUF metadata value types -> struct format for scalars.
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I",
           5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_TYPE_STRING = 8
_TYPE_ARRAY = 9


class GGUFError(Exception):
    """Raised when a file is not valid GGUF or cannot be parsed."""


@dataclass
class GGUFInfo:
    version: int
    metadata: dict
    tensor_names: list[str]


class _Reader:
    def __init__(self, mm: mmap.mmap):
        self.mm = mm
        self.off = 0

    def take(self, fmt: str):
        size = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, self.mm, self.off)
        self.off += size
        return vals

    def string(self) -> str:
        (length,) = self.take("<Q")
        s = self.mm[self.off:self.off + length].decode("utf-8", "replace")
        self.off += length
        return s

    def value(self, vtype: int):
        if vtype in _SCALAR:
            return self.take(_SCALAR[vtype])[0]
        if vtype == _TYPE_STRING:
            return self.string()
        if vtype == _TYPE_ARRAY:
            (elem_type,) = self.take("<I")
            (count,) = self.take("<Q")
            if elem_type == _TYPE_STRING:
                for _ in range(count):
                    self.string()
            elif elem_type in _SCALAR:
                self.off += struct.calcsize(_SCALAR[elem_type]) * count
            else:
                raise GGUFError(f"unsupported array element type {elem_type}")
            return {"__array__": True, "elem_type": elem_type, "count": count}
        raise GGUFError(f"unsupported value type {vtype}")


def read_gguf(path: str | Path) -> GGUFInfo:
    path = Path(path)
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            r = _Reader(mm)
            (magic,) = r.take("<I")
            if magic != GGUF_MAGIC:
                raise GGUFError(f"{path}: not a GGUF file")
            (version,) = r.take("<I")
            (n_tensors,) = r.take("<Q")
            (n_kv,) = r.take("<Q")
            metadata: dict = {}
            for _ in range(n_kv):
                key = r.string()
                (vtype,) = r.take("<I")
                metadata[key] = r.value(vtype)
            tensor_names: list[str] = []
            for _ in range(n_tensors):
                name = r.string()
                (n_dims,) = r.take("<I")
                r.off += 8 * n_dims        # dims (uint64 each)
                r.take("<I")               # ggml type
                r.take("<Q")               # offset
                tensor_names.append(name)
            return GGUFInfo(version=version, metadata=metadata, tensor_names=tensor_names)
        except struct.error as e:
            raise GGUFError(f"{path}: truncated or malformed GGUF ({e})") from e
        finally:
            mm.close()
