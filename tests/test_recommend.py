from lmm.hardware import HardwareInfo
from lmm.models import Model
from lmm.recommend import LaunchConfig, choose_context, recommend_config

_META = {
    "qwen35.block_count": 65,
    "qwen35.full_attention_interval": 4,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
}

_SUPPORTED = {"-m", "-ngl", "-fa", "--cache-type-k", "--cache-type-v",
              "--spec-type", "--spec-draft-n-max", "-t", "-c", "--host",
              "--port", "--alias"}


def _model(tmp_path, *, has_mtp, size_bytes):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"\x00" * size_bytes)
    return Model(path=f, arch="qwen35", name="m", family="qwen3.6",
                 size_label="27B", quant="Q8_0", block_count=65,
                 context_length=262144, has_mtp=has_mtp, hf_base_repo=None,
                 shards=[f], sidecars=[])


def _hw(total_gib):
    return HardwareInfo(total_ram_bytes=total_gib * 1024**3, logical_cores=14,
                        perf_cores=10, platform="Darwin", has_metal=True)


def test_choose_context_picks_largest_that_fits():
    ctx = choose_context("qwen35", _META, weights=1, model_max=262144,
                         usable_ram_bytes=64 * 1024**3, cache_type="q8_0")
    assert ctx == 262144


def test_choose_context_reduces_when_budget_small():
    # weights=32 GiB forces reduction: 32+8+1=41 GiB > 40 GiB at ctx=262144,
    # but 32+4+1=37 GiB ≤ 40 GiB at ctx=131072 → should pick 131072.
    ctx = choose_context("qwen35", _META, weights=32 * 1024**3, model_max=262144,
                         usable_ram_bytes=40 * 1024**3, cache_type="q8_0")
    assert ctx < 262144
    assert ctx in (131072, 65536, 32768, 16384, 8192)


def test_recommend_includes_core_flags_and_mtp(tmp_path):
    m = _model(tmp_path, has_mtp=True, size_bytes=1000)
    cfg = recommend_config(m, _META, _hw(64), supported=_SUPPORTED)
    assert isinstance(cfg, LaunchConfig)
    assert "-ngl" in cfg.flags and "999" in cfg.flags
    assert "-fa" in cfg.flags
    assert "--spec-type" in cfg.flags and "draft-mtp" in cfg.flags
    assert "-t" in cfg.flags and "10" in cfg.flags
    assert "-c" in cfg.flags
    assert cfg.fit.level in ("comfortable", "tight", "wont_load")


def test_recommend_omits_mtp_when_model_lacks_head(tmp_path):
    m = _model(tmp_path, has_mtp=False, size_bytes=1000)
    cfg = recommend_config(m, _META, _hw(64), supported=_SUPPORTED)
    assert "--spec-type" not in cfg.flags


def test_recommend_drops_unsupported_flags_with_warning(tmp_path):
    m = _model(tmp_path, has_mtp=True, size_bytes=1000)
    supported = _SUPPORTED - {"--spec-type", "--spec-draft-n-max"}
    cfg = recommend_config(m, _META, _hw(64), supported=supported)
    assert "--spec-type" not in cfg.flags
    assert any("spec-type" in w for w in cfg.warnings)
