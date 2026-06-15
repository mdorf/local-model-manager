from pathlib import Path

from lmm.hermes import bind

_SAMPLE = """\
# Hermes config (sample)
model:
  default: moonshotai/kimi-k2.6
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
providers: {}
fallback_providers: []
# keep this comment
toolsets:
- hermes-cli
"""


def _write(tmp_path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(_SAMPLE)
    return p


def test_bind_sets_provider_and_model(tmp_path):
    cfg = _write(tmp_path)
    bind(cfg, provider_name="local", base_url="http://127.0.0.1:8080/v1",
         model_id="Qwen3.6-27B-Q8_0")
    import ruamel.yaml
    data = ruamel.yaml.YAML().load(cfg.read_text())
    assert data["model"]["provider"] == "custom:local"
    assert data["model"]["default"] == "Qwen3.6-27B-Q8_0"
    assert data["model"]["base_url"] == "http://127.0.0.1:8080/v1"
    assert data["providers"]["local"]["base_url"] == "http://127.0.0.1:8080/v1"
    assert data["providers"]["local"]["api_key"]
    assert data["providers"]["local"]["default_model"] == "Qwen3.6-27B-Q8_0"


def test_bind_preserves_comments_and_other_keys(tmp_path):
    cfg = _write(tmp_path)
    bind(cfg, provider_name="local", base_url="http://127.0.0.1:8080/v1", model_id="m")
    text = cfg.read_text()
    assert "# keep this comment" in text
    assert "fallback_providers" in text
    assert "hermes-cli" in text


def test_bind_writes_backup(tmp_path):
    cfg = _write(tmp_path)
    bind(cfg, provider_name="local", base_url="http://127.0.0.1:8080/v1", model_id="m")
    backup = Path(str(cfg) + ".lmm-prev")
    assert backup.exists()
    assert backup.read_text() == _SAMPLE
