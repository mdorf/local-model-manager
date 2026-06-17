from lmm.cli import build_parser, cmd_token


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
