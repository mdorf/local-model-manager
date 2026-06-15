import struct

import pytest

from lmm.gguf import GGUFError, read_gguf


def test_reads_metadata_scalars_and_strings(qwen_like):
    info = read_gguf(qwen_like)
    assert info.version == 3
    assert info.metadata["general.architecture"] == "qwen35"
    assert info.metadata["general.name"] == "Qwen3.6-27B"
    assert info.metadata["qwen35.block_count"] == 65
    assert info.metadata["qwen35.context_length"] == 262144


def test_reads_tensor_names(qwen_like):
    info = read_gguf(qwen_like)
    assert "blk.0.attn_q.weight" in info.tensor_names
    assert "blk.64.nextn.eh_proj.weight" in info.tensor_names


def test_skips_array_values_without_crashing(qwen_like):
    info = read_gguf(qwen_like)
    assert info.metadata["tokenizer.ggml.tokens"]["__array__"] is True
    assert info.metadata["tokenizer.ggml.tokens"]["count"] == 3


def test_rejects_non_gguf(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\x00" * 32)
    with pytest.raises(GGUFError):
        read_gguf(bad)


# --- Error-hardening tests (TDD: written before the fix) ---

def test_missing_path_raises_gguferror(tmp_path):
    with pytest.raises(GGUFError):
        read_gguf(tmp_path / "nope.gguf")


def test_directory_raises_gguferror(tmp_path):
    with pytest.raises(GGUFError):
        read_gguf(tmp_path)


def test_empty_file_raises_gguferror(tmp_path):
    empty = tmp_path / "empty.gguf"
    empty.write_bytes(b"")
    with pytest.raises(GGUFError):
        read_gguf(empty)


def test_truncated_header_raises_gguferror(tmp_path):
    truncated = tmp_path / "truncated.gguf"
    # Magic + version only (8 bytes) — missing n_tensors and n_kv counts
    truncated.write_bytes(b"GGUF" + b"\x00" * 4)
    with pytest.raises(GGUFError):
        read_gguf(truncated)


def test_overlong_string_length_raises_gguferror(tmp_path):
    bad = tmp_path / "overlong.gguf"
    # Minimal valid header: magic, version=3, n_tensors=1, n_kv=1
    # Then a KV entry whose key length claims 10**9 bytes (far beyond buffer)
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + struct.pack("<Q", 10**9)
    bad.write_bytes(data)
    with pytest.raises(GGUFError):
        read_gguf(bad)
