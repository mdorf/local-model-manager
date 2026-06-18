import sys
from pathlib import Path

from fastapi.testclient import TestClient

from lmm.api import create_app
from lmm.daemonconfig import DaemonConfig
from lmm.ports import pick_free_port

FAKE = Path(__file__).parent / "fake_llama.py"
_H = {"Authorization": "Bearer t"}


def _client(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "state"))
    cfg = DaemonConfig(host="127.0.0.1", port=8770, token="t", roots=[str(tmp_path)])

    def builder(model_name, port, tuning=None):
        return [sys.executable, str(FAKE), "--port", str(port)], f"/m/{model_name}"

    return TestClient(create_app(cfg, command_builder=builder))


def test_start_then_list_then_stop(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    port = pick_free_port(start=49900)
    r = c.post("/api/servers", json={"model": "x.gguf", "port": port}, headers=_H)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "ready"
    listed = c.get("/api/servers", headers=_H).json()["servers"]
    assert any(s["port"] == port for s in listed)
    d = c.request("DELETE", f"/api/servers/{port}", headers=_H)
    assert d.status_code == 200
    assert all(s["port"] != port for s in c.get("/api/servers", headers=_H).json()["servers"])


def test_start_requires_token(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    r = c.post("/api/servers", json={"model": "x.gguf", "port": 49950})
    assert r.status_code == 401


def test_switch_replaces(monkeypatch, tmp_path):
    c = _client(monkeypatch, tmp_path)
    p1 = pick_free_port(start=49960)
    c.post("/api/servers", json={"model": "a.gguf", "port": p1}, headers=_H)
    p2 = pick_free_port(start=p1 + 1)
    r = c.post("/api/servers/switch", json={"model": "b.gguf", "port": p2}, headers=_H)
    try:
        assert r.status_code == 200
        ports = {s["port"] for s in c.get("/api/servers", headers=_H).json()["servers"]}
        assert p2 in ports and p1 not in ports
    finally:
        c.request("DELETE", f"/api/servers/{p2}", headers=_H)
