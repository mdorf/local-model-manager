import struct
from pathlib import Path

import pytest

GGUF_MAGIC = b"GGUF"


def _wstr(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _wval(value) -> bytes:
    # Returns (value_type uint32) + encoded value. Supports int->UINT32,
    # float->FLOAT32, str->STRING, list[str]->ARRAY of STRING.
    if isinstance(value, bool):
        return struct.pack("<I", 7) + struct.pack("<?", value)
    if isinstance(value, int):
        return struct.pack("<I", 4) + struct.pack("<I", value)
    if isinstance(value, float):
        return struct.pack("<I", 6) + struct.pack("<f", value)
    if isinstance(value, str):
        return struct.pack("<I", 8) + _wstr(value)
    if isinstance(value, list):  # array of strings
        out = struct.pack("<I", 9) + struct.pack("<I", 8) + struct.pack("<Q", len(value))
        for item in value:
            out += _wstr(item)
        return out
    raise TypeError(f"unsupported fixture value: {value!r}")


def write_minimal_gguf(path: Path, metadata: dict, tensor_names: list[str]) -> Path:
    body = bytearray()
    body += GGUF_MAGIC
    body += struct.pack("<I", 3)                      # version
    body += struct.pack("<Q", len(tensor_names))      # tensor_count
    body += struct.pack("<Q", len(metadata))          # metadata_kv_count
    for key, value in metadata.items():
        body += _wstr(key) + _wval(value)
    for name in tensor_names:
        body += _wstr(name)
        body += struct.pack("<I", 1)                  # n_dims = 1
        body += struct.pack("<Q", 1)                  # dims[0] = 1
        body += struct.pack("<I", 0)                  # ggml type = F32
        body += struct.pack("<Q", 0)                  # offset
    path.write_bytes(bytes(body))
    return path


@pytest.fixture(autouse=True)
def _isolate_shared_state_dir(monkeypatch, tmp_path):
    """Keep the suite hermetic on hosts where the daemon is actually installed.

    `state.state_dir()` resolves to /Users/Shared/local-model-manager when that
    dir holds a daemon.json. The tests were written assuming a pristine host;
    point SHARED_STATE_DIR at an absent path so a real install never leaks into
    state resolution. Tests that exercise resolution re-set it themselves.
    """
    import lmm.gguf as gguf
    import lmm.state as state
    monkeypatch.setattr(state, "SHARED_STATE_DIR", tmp_path / "no-shared-state")
    # The GGUF header cache is process-global; clear it so cached parses from a
    # prior test can't leak across tests that reuse a path.
    gguf.clear_gguf_cache()
    # Also pin LMM_STATE_DIR to a per-test tmp so state_dir() never resolves to a
    # real ~/Library/... dir holding a stray daemon.json (which would make the CLI
    # think a daemon is running). Resolution tests override/delenv this themselves.
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "state"))


@pytest.fixture
def qwen_like(tmp_path):
    """A Qwen3.6-style hybrid model with an MTP head."""
    meta = {
        "general.architecture": "qwen35",
        "general.name": "Qwen3.6-27B",
        "general.basename": "Qwen3.6-27B",
        "general.size_label": "27B",
        "general.file_type": 7,  # Q8_0
        "general.license.link": "https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE",
        "qwen35.block_count": 65,
        "qwen35.context_length": 262144,
        "qwen35.nextn_predict_layers": 1,
        "tokenizer.ggml.tokens": ["<a>", "<b>", "<c>"],  # exercises array skip
    }
    tensors = ["blk.0.attn_q.weight", "blk.64.nextn.eh_proj.weight"]
    return write_minimal_gguf(tmp_path / "Qwen3.6-27B-Q8_0.gguf", meta, tensors)
