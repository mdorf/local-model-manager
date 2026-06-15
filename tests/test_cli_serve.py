from lmm.cli import build_parser, cmd_status, cmd_stop


def test_parser_has_lifecycle_subcommands():
    p = build_parser()
    a = p.parse_args(["serve", "m.gguf", "--root", "/x"])
    assert a.func.__name__ == "cmd_serve"
    b = p.parse_args(["stop", "--port", "8080"])
    assert b.func is cmd_stop
    c = p.parse_args(["status"])
    assert c.func is cmd_status


def test_status_empty_is_clean(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "state"))
    rc = cmd_status(build_parser().parse_args(["status"]))
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "no" in out


def test_stop_unknown_port_reports(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "state"))
    rc = cmd_stop(build_parser().parse_args(["stop", "--port", "59999"]))
    out = capsys.readouterr().out.lower()
    assert rc == 0
    assert "no server" in out or "not" in out
