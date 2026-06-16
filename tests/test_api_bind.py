from fastapi.testclient import TestClient
from lmm.api import create_app
from lmm.daemonconfig import DaemonConfig
from lmm.server import ServerInstance

_SAMPLE = "model:\n  default: x\n  provider: openrouter\nproviders: {}\n"
H = {"Authorization": "Bearer t"}


class FakeManager:
    def __init__(self, instances=None):
        self._inst = list(instances or [])
    def status(self):
        return list(self._inst)
    def list(self):
        return list(self._inst)


def _running(port=8080, model="/m/Qwen3.6-27B-Q8_0.gguf"):
    return ServerInstance(port=port, pid=1, model_path=model, started_at=0.0,
                          status="ready", external=False)


def _app(manager, host="127.0.0.1"):
    cfg = DaemonConfig(host=host, token="t", inference_key="sek", roots=["/x"])
    return create_app(cfg, manager=manager, command_builder=lambda *a: None)


def test_bind_remote_is_forbidden():
    # default TestClient client host is "testclient" → non-loopback → 403
    app = _app(FakeManager([_running()]))
    r = TestClient(app).post("/api/bind", json={}, headers=H)
    assert r.status_code == 403


def test_bind_loopback_writes_running_model(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(_SAMPLE)
    app = _app(FakeManager([_running()]), host="0.0.0.0")  # LAN → key is written
    c = TestClient(app, client=("127.0.0.1", 12345))
    r = c.post("/api/bind", json={"hermes_config": str(cfg_file)}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["bound"] is True
    assert body["model"] == "Qwen3.6-27B-Q8_0"  # the RUNNING model's id
    import ruamel.yaml
    data = ruamel.yaml.YAML().load(cfg_file.read_text())
    assert data["model"]["default"] == "Qwen3.6-27B-Q8_0"
    assert data["providers"]["local"]["base_url"] == "http://127.0.0.1:8080/v1"
    assert data["providers"]["local"]["api_key"] == "sek"


def test_bind_no_running_server_conflict(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(_SAMPLE)
    app = _app(FakeManager([]))
    c = TestClient(app, client=("127.0.0.1", 12345))
    r = c.post("/api/bind", json={"hermes_config": str(cfg_file)}, headers=H)
    assert r.status_code == 409


def test_bind_status_reports_bound(tmp_path, monkeypatch):
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    (hermes / "config.yaml").write_text(
        "model:\n  default: Qwen3.6-27B-Q8_0\n  base_url: http://127.0.0.1:8080/v1\n"
        "providers: {}\n")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    app = _app(FakeManager([_running()]))
    c = TestClient(app, client=("127.0.0.1", 12345))
    b = c.get("/api/bind-status", headers=H).json()
    assert b["bound"] is True and b["model_id"] == "Qwen3.6-27B-Q8_0"


def test_bind_status_false_when_url_differs(tmp_path, monkeypatch):
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    (hermes / "config.yaml").write_text("model:\n  base_url: http://elsewhere/v1\nproviders: {}\n")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    app = _app(FakeManager([_running()]))
    c = TestClient(app, client=("127.0.0.1", 12345))
    assert c.get("/api/bind-status", headers=H).json()["bound"] is False


def test_bind_status_false_when_model_differs(tmp_path, monkeypatch):
    # The switch scenario: same port/base_url, but Hermes still names the OLD
    # model. The badge must NOT claim "bound" — the config is stale.
    hermes = tmp_path / ".hermes"
    hermes.mkdir()
    (hermes / "config.yaml").write_text(
        "model:\n  default: Some-Other-Model\n  base_url: http://127.0.0.1:8080/v1\n"
        "providers: {}\n")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    app = _app(FakeManager([_running()]))  # running model id == Qwen3.6-27B-Q8_0
    c = TestClient(app, client=("127.0.0.1", 12345))
    b = c.get("/api/bind-status", headers=H).json()
    assert b["bound"] is False and b["model_id"] is None


def test_bind_status_remote_false():
    app = _app(FakeManager([_running()]))
    assert TestClient(app).get("/api/bind-status", headers=H).json()["bound"] is False
