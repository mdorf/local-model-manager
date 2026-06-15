from lmm.cli import build_parser, cmd_bind
from lmm.daemonconfig import load_or_create_config


def _cfg(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("model:\n  default: x\n  provider: openrouter\n  base_url: u\nproviders: {}\n")
    return p


def test_bind_uses_inference_key_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    cfg_file = _cfg(tmp_path)
    key = load_or_create_config().inference_key
    cmd_bind(build_parser().parse_args(["bind", "m.gguf", "--hermes-config", str(cfg_file)]))
    import ruamel.yaml
    data = ruamel.yaml.YAML().load(cfg_file.read_text())
    assert data["providers"]["local"]["api_key"] == key


def test_bind_api_key_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    cfg_file = _cfg(tmp_path)
    cmd_bind(build_parser().parse_args(
        ["bind", "m.gguf", "--hermes-config", str(cfg_file), "--api-key", "override-key"]))
    import ruamel.yaml
    data = ruamel.yaml.YAML().load(cfg_file.read_text())
    assert data["providers"]["local"]["api_key"] == "override-key"
