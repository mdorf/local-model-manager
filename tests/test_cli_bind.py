from lmm.cli import build_parser, cmd_bind, cmd_unbind

_SAMPLE = """\
model:
  default: moonshotai/kimi-k2.6
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
providers: {}
"""


def test_parser_has_bind_unbind():
    p = build_parser()
    a = p.parse_args(["bind", "Qwen3.6-27B-Q8_0.gguf", "--port", "8080",
                      "--hermes-config", "/x/config.yaml"])
    assert a.func is cmd_bind
    assert a.model == "Qwen3.6-27B-Q8_0.gguf"
    assert a.port == 8080
    b = p.parse_args(["unbind", "--hermes-config", "/x/config.yaml"])
    assert b.func is cmd_unbind


def test_cmd_bind_writes_config(tmp_path, capsys):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_SAMPLE)
    rc = cmd_bind(build_parser().parse_args(
        ["bind", "Qwen3.6-27B-Q8_0.gguf", "--port", "8080",
         "--hermes-config", str(cfg)]))
    assert rc == 0
    out = capsys.readouterr().out
    import ruamel.yaml
    data = ruamel.yaml.YAML().load(cfg.read_text())
    assert data["model"]["provider"] == "custom:local"
    assert data["model"]["default"] == "Qwen3.6-27B-Q8_0"
    assert data["providers"]["local"]["base_url"] == "http://127.0.0.1:8080/v1"
    assert "max_tokens" in out.lower() or "reasoning" in out.lower()


def test_cmd_bind_missing_config_errors(tmp_path, capsys):
    rc = cmd_bind(build_parser().parse_args(
        ["bind", "m.gguf", "--hermes-config", str(tmp_path / "nope.yaml")]))
    assert rc == 1
    assert "not found" in capsys.readouterr().out.lower()


def test_cmd_unbind(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(_SAMPLE)
    cmd_bind(build_parser().parse_args(
        ["bind", "m.gguf", "--hermes-config", str(cfg)]))
    rc = cmd_unbind(build_parser().parse_args(["unbind", "--hermes-config", str(cfg)]))
    assert rc == 0
    assert cfg.read_text() == _SAMPLE
