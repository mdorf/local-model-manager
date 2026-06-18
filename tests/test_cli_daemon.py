from lmm.cli import build_parser, cmd_daemon, cmd_token


def test_cmd_daemon_pins_config_host_to_actual_bind(monkeypatch, tmp_path):
    # Regression: the daemon binds to --host (e.g. 0.0.0.0 from the plist) but the
    # model-launch builder reads config.host. Pin config.host to the real bind so a
    # stale daemon.json can't make models launch on loopback while the daemon is LAN.
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    import lmm.api
    import lmm.server
    import uvicorn
    captured = {}
    monkeypatch.setattr(lmm.api, "create_app",
                        lambda config, **kw: captured.update(host=config.host, port=config.port))
    monkeypatch.setattr(lmm.server, "autodetect_servers", lambda *a, **k: [])
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    rc = cmd_daemon(build_parser().parse_args(["daemon", "--host", "0.0.0.0", "--port", "8770"]))
    assert rc == 0
    assert captured["host"] == "0.0.0.0"   # → llama-server also binds 0.0.0.0 + gets an api-key


def test_parser_has_daemon_and_token():
    p = build_parser()
    a = p.parse_args(["daemon", "--host", "0.0.0.0", "--port", "8771"])
    assert a.func.__name__ == "cmd_daemon"
    assert a.host == "0.0.0.0"
    assert a.port == 8771
    b = p.parse_args(["token"])
    assert b.func is cmd_token


def test_token_prints_token(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    rc = cmd_token(build_parser().parse_args(["token"]))
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert len(out) >= 16


def test_token_rotate_changes_token_and_preserves_other_fields(monkeypatch, tmp_path, capsys):
    import json
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "st"))
    cmd_token(build_parser().parse_args(["token"]))  # create the config
    cfg_path = tmp_path / "st" / "daemon.json"
    before = json.loads(cfg_path.read_text())
    capsys.readouterr()  # drain

    rc = cmd_token(build_parser().parse_args(["token", "--rotate"]))
    new_printed = capsys.readouterr().out.strip()
    after = json.loads(cfg_path.read_text())
    assert rc == 0
    assert after["token"] != before["token"]            # rotated
    assert after["token"] == new_printed                 # prints the new one
    assert after["inference_key"] == before["inference_key"]  # other secrets preserved
    assert after["roots"] == before["roots"]
    # plain `token` (no --rotate) must NOT rotate
    cmd_token(build_parser().parse_args(["token"]))
    assert json.loads(cfg_path.read_text())["token"] == after["token"]
