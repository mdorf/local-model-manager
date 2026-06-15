"""Architecture-aware memory estimation for llama.cpp model serving."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_CACHE_BYTES = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0,
                "q5_1": 0.6875, "q5_0": 0.6875, "q4_1": 0.5625, "q4_0": 0.5}

_OVERHEAD_BYTES = 1 * 1024**3  # 1 GiB allowance for compute buffers, SSM state, scratch


def bytes_per_element(cache_type: str) -> float:
    try:
        return _CACHE_BYTES[cache_type.lower()]
    except KeyError as e:
        raise ValueError(f"unknown cache type: {cache_type!r}") from e


def _as_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def kv_cache_bytes(arch: str, metadata: dict, context_len: int, cache_type: str) -> int:
    """KV-cache size in bytes. Counts only full-attention layers
    (block_count // full_attention_interval) when the interval is present, else
    all block_count layers. Returns 0 if required attention metadata is absent.
    """
    block_count = _as_int(metadata.get(f"{arch}.block_count"))
    head_count_kv = _as_int(metadata.get(f"{arch}.attention.head_count_kv"))
    key_length = _as_int(metadata.get(f"{arch}.attention.key_length"))
    value_length = _as_int(metadata.get(f"{arch}.attention.value_length"))
    if not (block_count and head_count_kv and key_length and value_length):
        return 0
    interval = _as_int(metadata.get(f"{arch}.full_attention_interval"))
    attn_layers = block_count // interval if interval else block_count
    per_token = attn_layers * head_count_kv * (key_length + value_length)
    return int(per_token * bytes_per_element(cache_type) * context_len)


def weights_bytes(shards: list[Path]) -> int:
    total = 0
    for p in shards:
        try:
            total += os.path.getsize(p)
        except OSError:
            continue
    return total


@dataclass
class MemoryEstimate:
    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_cache_bytes + self.overhead_bytes


def estimate_memory(arch: str, metadata: dict, shards: list[Path],
                    context_len: int, cache_type: str) -> MemoryEstimate:
    return MemoryEstimate(
        weights_bytes=weights_bytes(shards),
        kv_cache_bytes=kv_cache_bytes(arch, metadata, context_len, cache_type),
        overhead_bytes=_OVERHEAD_BYTES,
    )
