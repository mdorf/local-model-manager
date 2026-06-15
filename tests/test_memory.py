import pytest

from lmm.memory import (
    MemoryEstimate,
    bytes_per_element,
    estimate_memory,
    kv_cache_bytes,
    weights_bytes,
)


def _qwen_meta():
    return {
        "qwen35.block_count": 65,
        "qwen35.full_attention_interval": 4,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
    }


def test_bytes_per_element():
    assert bytes_per_element("f16") == 2.0
    assert bytes_per_element("q8_0") == 1.0
    assert bytes_per_element("q4_0") == 0.5
    with pytest.raises(ValueError):
        bytes_per_element("nonsense")


def test_kv_cache_is_architecture_aware():
    got = kv_cache_bytes("qwen35", _qwen_meta(), context_len=131072, cache_type="q8_0")
    assert got == 16 * 4 * (256 + 256) * 1 * 131072  # 65//4=16 attn layers


def test_kv_cache_without_interval_uses_all_layers():
    meta = {"qwen35.block_count": 10, "qwen35.attention.head_count_kv": 2,
            "qwen35.attention.key_length": 64, "qwen35.attention.value_length": 64}
    got = kv_cache_bytes("qwen35", meta, context_len=1000, cache_type="f16")
    assert got == 10 * 2 * (64 + 64) * 2 * 1000


def test_kv_cache_returns_zero_when_metadata_missing():
    assert kv_cache_bytes("qwen35", {}, context_len=1000, cache_type="q8_0") == 0


def test_weights_bytes_sums_shard_sizes(tmp_path):
    a = tmp_path / "a.gguf"
    a.write_bytes(b"\x00" * 1000)
    b = tmp_path / "b.gguf"
    b.write_bytes(b"\x00" * 2345)
    assert weights_bytes([a, b]) == 3345


def test_estimate_memory_totals(tmp_path):
    f = tmp_path / "w.gguf"
    f.write_bytes(b"\x00" * 5000)
    est = estimate_memory("qwen35", _qwen_meta(), shards=[f],
                          context_len=131072, cache_type="q8_0")
    assert isinstance(est, MemoryEstimate)
    assert est.weights_bytes == 5000
    assert est.kv_cache_bytes == 16 * 4 * 512 * 131072
    assert est.total_bytes == est.weights_bytes + est.kv_cache_bytes + est.overhead_bytes
    assert est.overhead_bytes > 0
