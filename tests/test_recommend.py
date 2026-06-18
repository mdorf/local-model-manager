from lmm.hardware import HardwareInfo
from lmm.models import Model
from lmm.recommend import (
    LaunchConfig,
    choose_context,
    context_budget_bytes,
    recommend_config,
)

_META = {
    "qwen35.block_count": 65,
    "qwen35.full_attention_interval": 4,
    "qwen35.attention.head_count_kv": 4,
    "qwen35.attention.key_length": 256,
    "qwen35.attention.value_length": 256,
}

_SUPPORTED = {"-m", "-ngl", "-fa", "--jinja", "--cache-type-k", "--cache-type-v",
              "--spec-type", "--spec-draft-n-max", "-t", "-c", "--host",
              "--port", "--alias"}


def _model(tmp_path, *, has_mtp, size_bytes, has_chat_template=False):
    f = tmp_path / "m.gguf"
    f.write_bytes(b"\x00" * size_bytes)
    return Model(path=f, arch="qwen35", name="m", family="qwen3.6",
                 size_label="27B", quant="Q8_0", block_count=65,
                 context_length=262144, has_mtp=has_mtp, hf_base_repo=None,
                 shards=[f], sidecars=[], has_chat_template=has_chat_template)


def _hw(total_gib):
    return HardwareInfo(total_ram_bytes=total_gib * 1024**3, logical_cores=14,
                        perf_cores=10, platform="Darwin", has_metal=True)


def test_context_budget_prefers_gpu_working_set():
    # When the GPU working set is known, budget against it (× safety), NOT 70% RAM.
    hw = HardwareInfo(total_ram_bytes=64 * 1024**3, logical_cores=14, perf_cores=10,
                      platform="Darwin", has_metal=True,
                      gpu_working_set_bytes=48 * 1024**3)
    assert context_budget_bytes(hw) == int(48 * 1024**3 * 0.90)


def test_context_budget_falls_back_to_ram_fraction_when_gpu_unknown():
    # Non-Metal / undetectable (gpu_working_set_bytes=0) → 70% of total RAM.
    hw = HardwareInfo(total_ram_bytes=64 * 1024**3, logical_cores=14, perf_cores=10,
                      platform="Linux", has_metal=False)
    assert context_budget_bytes(hw) == int(64 * 1024**3 * 0.70)


def test_gpu_aware_budget_steps_context_down_below_ram_fraction():
    # A model that fits under 70%-of-RAM (44.8 GiB) but NOT under a constrained
    # GPU working set must step the context down — that's the cliff the tweak guards.
    weights = 32 * 1024**3
    big = HardwareInfo(total_ram_bytes=64 * 1024**3, logical_cores=14, perf_cores=10,
                       platform="Darwin", has_metal=True,
                       gpu_working_set_bytes=48 * 1024**3)
    small = HardwareInfo(total_ram_bytes=64 * 1024**3, logical_cores=14, perf_cores=10,
                         platform="Darwin", has_metal=True,
                         gpu_working_set_bytes=38 * 1024**3)  # e.g. a lowered wired limit
    ctx_big = choose_context("qwen35", _META, weights, 262144,
                             context_budget_bytes(big), "q8_0")
    ctx_small = choose_context("qwen35", _META, weights, 262144,
                               context_budget_bytes(small), "q8_0")
    assert ctx_small < ctx_big  # tighter GPU ceiling → smaller context


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


def test_recommend_splits_tuning_from_plumbing(tmp_path):
    m = _model(tmp_path, has_mtp=True, size_bytes=1000)
    cfg = recommend_config(m, _META, _hw(64), supported=_SUPPORTED,
                           host="0.0.0.0", port=8080, alias="m", api_key="K")
    # tuning_flags = editable knobs only, no deployment plumbing
    assert "-c" in cfg.tuning_flags and "-ngl" in cfg.tuning_flags
    for plumb in ("--host", "--port", "--api-key", "-m", "--alias"):
        assert plumb not in cfg.tuning_flags
    # full flags carry the plumbing (for the local CLI / spawn)
    assert "--host" in cfg.flags and "0.0.0.0" in cfg.flags
    assert "-m" in cfg.flags and "--api-key" in cfg.flags


def test_recommend_omits_mtp_when_model_lacks_head(tmp_path):
    m = _model(tmp_path, has_mtp=False, size_bytes=1000)
    cfg = recommend_config(m, _META, _hw(64), supported=_SUPPORTED)
    assert "--spec-type" not in cfg.flags


def test_recommend_adds_jinja_when_embedded_chat_template(tmp_path):
    # The model ships a jinja chat template → render it faithfully (correct
    # tool-calling + lets --chat-template-kwargs apply). It's an editable knob.
    m = _model(tmp_path, has_mtp=False, size_bytes=1000, has_chat_template=True)
    cfg = recommend_config(m, _META, _hw(64), supported=_SUPPORTED)
    assert "--jinja" in cfg.tuning_flags


def test_recommend_omits_jinja_without_chat_template(tmp_path):
    m = _model(tmp_path, has_mtp=False, size_bytes=1000, has_chat_template=False)
    cfg = recommend_config(m, _META, _hw(64), supported=_SUPPORTED)
    assert "--jinja" not in cfg.flags


def test_recommend_drops_jinja_when_unsupported(tmp_path):
    m = _model(tmp_path, has_mtp=False, size_bytes=1000, has_chat_template=True)
    supported = _SUPPORTED - {"--jinja"}
    cfg = recommend_config(m, _META, _hw(64), supported=supported)
    assert "--jinja" not in cfg.flags
    assert any("jinja" in w for w in cfg.warnings)


def test_recommend_drops_unsupported_flags_with_warning(tmp_path):
    m = _model(tmp_path, has_mtp=True, size_bytes=1000)
    supported = _SUPPORTED - {"--spec-type", "--spec-draft-n-max"}
    cfg = recommend_config(m, _META, _hw(64), supported=supported)
    assert "--spec-type" not in cfg.flags
    assert any("spec-type" in w for w in cfg.warnings)


def test_recommend_drops_only_the_unsupported_group(tmp_path):
    # Removing only -ngl must drop that group alone; other groups survive.
    m = _model(tmp_path, has_mtp=True, size_bytes=1000)
    supported = _SUPPORTED - {"-ngl"}
    cfg = recommend_config(m, _META, _hw(64), supported=supported)
    assert "-ngl" not in cfg.flags
    assert "999" not in cfg.flags                 # its value dropped with it
    assert "--spec-type" in cfg.flags             # unrelated group unaffected
    assert any("-ngl" in w for w in cfg.warnings)


def test_recommend_keeps_full_window_when_roomy(tmp_path, monkeypatch):
    # Plenty of RAM (weights tiny): the model's full 262144 window fits within the
    # ~30%-free budget, so it's kept.
    monkeypatch.setattr("lmm.recommend.weights_bytes", lambda shards: 1000)
    m = _model(tmp_path, has_mtp=True, size_bytes=1000)
    cfg = recommend_config(m, _META, _hw(64), supported=_SUPPORTED)
    assert int(cfg.flags[cfg.flags.index("-c") + 1]) == 262144


def test_recommend_steps_down_context_for_tight_headroom(tmp_path, monkeypatch):
    # Big weights: the full 262144 window would breach the ~30%-free budget even
    # though it'd still "fit" usable RAM — so context steps down to leave headroom.
    monkeypatch.setattr("lmm.recommend.weights_bytes", lambda shards: 40 * 1024**3)
    m = _model(tmp_path, has_mtp=True, size_bytes=1000)
    cfg = recommend_config(m, _META, _hw(64), supported=_SUPPORTED)
    ctx = int(cfg.flags[cfg.flags.index("-c") + 1])
    assert ctx < 262144 and ctx in (131072, 65536, 32768, 16384, 8192)
