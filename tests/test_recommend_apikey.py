from lmm.daemonconfig import load_or_create_config
from lmm.hardware import HardwareInfo
from lmm.models import Model
from lmm.recommend import recommend_config

_META = {"qwen35.block_count": 65, "qwen35.full_attention_interval": 4,
         "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256,
         "qwen35.attention.value_length": 256}
_SUP = {"-m", "-ngl", "-fa", "--cache-type-k", "--cache-type-v", "-t", "-c",
        "--host", "--port", "--alias", "--api-key"}


def _model(tmp_path):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"\x00" * 100)
    return Model(path=f, arch="qwen35", name="m", family="q", size_label="27B",
                 quant="Q8_0", block_count=65, context_length=262144,
                 has_mtp=False, hf_base_repo=None, shards=[f], sidecars=[])


def _hw():
    return HardwareInfo(total_ram_bytes=64 * 1024**3, logical_cores=14,
                        perf_cores=10, platform="Darwin", has_metal=True)


def test_recommend_adds_api_key_when_given(tmp_path):
    cfg = recommend_config(_model(tmp_path), _META, _hw(), supported=_SUP,
                           api_key="sek-123")
    assert "--api-key" in cfg.flags
    i = cfg.flags.index("--api-key")
    assert cfg.flags[i + 1] == "sek-123"


def test_recommend_omits_api_key_when_absent(tmp_path):
    cfg = recommend_config(_model(tmp_path), _META, _hw(), supported=_SUP)
    assert "--api-key" not in cfg.flags


def test_config_has_inference_key(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    c = load_or_create_config()
    assert len(c.inference_key) >= 16
    assert c.inference_key != c.token
