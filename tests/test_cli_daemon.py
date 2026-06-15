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
