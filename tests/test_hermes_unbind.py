from ruamel.yaml import YAML

from lmm.hermes import bind, unbind

_SAMPLE = """\
model:
  default: moonshotai/kimi-k2.6
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
providers: {}
"""


def _load(path):
    return YAML().load(path.read_text())


def test_unbind_preserves_unrelated_edits_made_after_bind(tmp_path):
    # The clobber risk: bind → user edits config for unrelated reasons → unbind.
    # A wholesale restore would discard those edits; surgical unbind must keep them.
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_SAMPLE)
    bind(cfg, base_url="http://127.0.0.1:8080/v1", model_id="m")
    # user adds an unrelated top-level key AND their own provider after binding
    cfg.write_text(cfg.read_text() + "\nlogging:\n  level: debug\n")
    d = _load(cfg)
    d["providers"]["openrouter"] = {"base_url": "https://openrouter.ai/api/v1"}
    YAML().dump(d, cfg.open("w"))

    assert unbind(cfg) is True
    out = _load(cfg)
    # unrelated edits survive:
    assert out["logging"]["level"] == "debug"
    assert "openrouter" in out["providers"]
    # the bind itself is reverted:
    assert out["model"]["provider"] == "openrouter"
    assert out["model"]["default"] == "moonshotai/kimi-k2.6"
    assert "local" not in out["providers"]


def test_unbind_restores_original(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_SAMPLE)
    bind(cfg, base_url="http://127.0.0.1:8080/v1", model_id="m")
    assert cfg.read_text() != _SAMPLE
    assert unbind(cfg) is True
    assert cfg.read_text() == _SAMPLE


def test_unbind_removes_backup_residue(tmp_path):
    # "complete removal": after revert, no .lmm-prev backup is left behind.
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_SAMPLE)
    bind(cfg, base_url="http://127.0.0.1:8080/v1", model_id="m")
    backup = tmp_path / "config.yaml.lmm-prev"
    assert backup.exists()
    assert unbind(cfg) is True
    assert not backup.exists()


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
