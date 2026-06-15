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
