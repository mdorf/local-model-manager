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
