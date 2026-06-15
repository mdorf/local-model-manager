# tests/test_api_bodies.py
from fastapi.testclient import TestClient
from lmm.api import create_app
from lmm.daemonconfig import DaemonConfig
from lmm.server import ServerInstance


class FakeManager:
    def __init__(self, instances=None):
        self._inst = list(instances or [])
        self.calls = []
    def status(self): return list(self._inst)
    def list(self): return list(self._inst)
    def start(self, command, *, port, model_path):
        self.calls.append(("start", port, model_path))
        inst = ServerInstance(port=port, pid=4242, model_path=model_path,
                              started_at=0.0, status="ready", external=False)
        self._inst.append(inst)
        return inst
    def switch(self, command, *, port, model_path):
        self.calls.append(("switch", port, model_path))
        self._inst = []
        return self.start(command, port=port, model_path=model_path)
    def stop(self, port):
        self.calls.append(("stop", port))
        self._inst = [i for i in self._inst if i.port != port]
        return True


def fake_builder(model_name, port):
    return ["llama-server", "-m", model_name, "--port", str(port)], f"/models/{model_name}"


def _client():
    cfg = DaemonConfig(host="127.0.0.1", port=8770, token="t", roots=["/x"])
    app = create_app(cfg, manager=FakeManager(), command_builder=fake_builder)
    return TestClient(app)


H = {"Authorization": "Bearer t"}


def test_start_requires_model_422():
    r = _client().post("/api/servers", json={"port": 8080}, headers=H)
    assert r.status_code == 422  # Pydantic: missing 'model'


def test_start_valid_body_200():
    r = _client().post("/api/servers", json={"model": "m.gguf", "port": 8081}, headers=H)
    assert r.status_code == 200
    assert r.json()["port"] == 8081


def test_switch_defaults_port_8080():
    r = _client().post("/api/servers/switch", json={"model": "m.gguf"}, headers=H)
    assert r.status_code == 200
    assert r.json()["port"] == 8080
