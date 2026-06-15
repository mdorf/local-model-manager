# tests/test_cli_daemon_client.py
import lmm.cli as cli
from lmm.cli import build_parser, cmd_serve, cmd_status

def test_serve_routes_through_daemon(monkeypatch, capsys):
    calls = {}
    monkeypatch.setattr(cli.daemon_client, "daemon_available",
                        lambda: {"base": "http://h:8770", "token": "tk"})
    def _fake_start(base, token, model, port):
        calls["start"] = (base, token, model, port)
        return {"port": port, "status": "ready"}
    monkeypatch.setattr(cli.daemon_client, "start", _fake_start)
    rc = cmd_serve(build_parser().parse_args(["serve", "m.gguf", "--port", "8080"]))
    assert rc == 0
    assert calls["start"] == ("http://h:8770", "tk", "m.gguf", 8080)

def test_serve_direct_when_no_daemon(monkeypatch):
    monkeypatch.setattr(cli.daemon_client, "daemon_available", lambda: None)
    # direct path: model not found short-circuits before spawning anything
    monkeypatch.setattr(cli, "_find_model", lambda root, name: None)
    assert cmd_serve(build_parser().parse_args(["serve", "missing.gguf"])) == 1

def test_status_routes_through_daemon(monkeypatch, capsys):
    monkeypatch.setattr(cli.daemon_client, "daemon_available",
                        lambda: {"base": "http://h:8770", "token": "tk"})
    monkeypatch.setattr(cli.daemon_client, "status",
                        lambda base, token: {"servers": [
                            {"port": 8080, "status": "ready", "pid": 9, "model": "x.gguf",
                             "external": False}]})
    assert cmd_status(build_parser().parse_args(["status"])) == 0
    assert "8080" in capsys.readouterr().out

def test_serve_surfaces_daemon_error_cleanly(monkeypatch, capsys):
    # routed serve against a bad model → daemon 404 → clean message + exit 1 (no traceback)
    monkeypatch.setattr(cli.daemon_client, "daemon_available",
                        lambda: {"base": "http://h:8770", "token": "tk"})
    def boom(*a, **k):
        raise cli.daemon_client.DaemonError("daemon error 404: model not found")
    monkeypatch.setattr(cli.daemon_client, "start", boom)
    rc = cmd_serve(build_parser().parse_args(["serve", "missing.gguf"]))
    assert rc == 1
    out = capsys.readouterr().out
    assert "404" in out and "model not found" in out
