from lmm.hermes import bind, unbind

_SAMPLE = """\
model:
  default: moonshotai/kimi-k2.6
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
providers: {}
"""


def test_unbind_restores_original(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_SAMPLE)
    bind(cfg, base_url="http://127.0.0.1:8080/v1", model_id="m")
    assert cfg.read_text() != _SAMPLE
    assert unbind(cfg) is True
    assert cfg.read_text() == _SAMPLE


def test_unbind_without_backup_returns_false(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_SAMPLE)
    assert unbind(cfg) is False
    assert cfg.read_text() == _SAMPLE


def test_unbind_after_rebind_restores_pristine(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_SAMPLE)
    bind(cfg, base_url="http://127.0.0.1:8080/v1", model_id="modelA")
    bind(cfg, base_url="http://127.0.0.1:8081/v1", model_id="modelB")  # re-bind
    assert unbind(cfg) is True
    assert cfg.read_text() == _SAMPLE       # must be the PRISTINE original, not modelA state
