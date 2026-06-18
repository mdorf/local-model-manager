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


def fake_builder(model_name, port, tuning=None):
    # mimic the real builder: daemon-owned plumbing + (recommended-or-overridden) tuning
    cmd = ["llama-server", "-m", model_name, "--host", "0.0.0.0", "--port", str(port)]
    cmd += tuning if tuning is not None else ["-ngl", "999"]
    return cmd, f"/models/{model_name}"


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


def test_start_port_in_use_returns_409():
    # A port-in-use RuntimeError from the manager must surface as a clean 409,
    # not a raw 500 (regression: switching onto an occupied port 500'd).
    class Busy(FakeManager):
        def start(self, command, *, port, model_path):
            raise RuntimeError(f"port {port} is already in use / managed")

    cfg = DaemonConfig(host="127.0.0.1", port=8770, token="t", roots=["/x"])
    app = create_app(cfg, manager=Busy(), command_builder=fake_builder)
    r = TestClient(app).post("/api/servers", json={"model": "m", "port": 8080}, headers=H)
    assert r.status_code == 409
    assert "in use" in r.json()["detail"]


def test_switch_port_in_use_returns_409():
    class Busy(FakeManager):
        def switch(self, command, *, port, model_path):
            raise RuntimeError(f"port {port} is already in use / managed")

    cfg = DaemonConfig(host="127.0.0.1", port=8770, token="t", roots=["/x"])
    app = create_app(cfg, manager=Busy(), command_builder=fake_builder)
    r = TestClient(app).post("/api/servers/switch", json={"model": "m", "port": 8080}, headers=H)
    assert r.status_code == 409


def test_start_with_flags_override_passes_tuning_to_builder():
    # Edited flags are TUNING only; the endpoint passes them to the builder, which
    # composes the daemon-owned plumbing around them (so host/port can't be edited).
    captured = {}

    class Cap(FakeManager):
        def start(self, command, *, port, model_path):
            captured["command"] = command
            return super().start(command, port=port, model_path=model_path)

    cfg = DaemonConfig(host="127.0.0.1", port=8770, token="t", roots=["/x"])
    app = create_app(cfg, manager=Cap(), command_builder=fake_builder)
    tuning = ["-ngl", "999", "-c", "131072"]
    r = TestClient(app).post("/api/servers",
                             json={"model": "Foo", "port": 8080, "flags": tuning}, headers=H)
    assert r.status_code == 200
    # tuning reached the launch, and the daemon's plumbing (--host) is present
    assert "-c" in captured["command"] and "131072" in captured["command"]
    assert "--host" in captured["command"]


def test_switch_with_flags_override_passes_tuning_to_builder():
    captured = {}

    class Cap(FakeManager):
        def switch(self, command, *, port, model_path):
            captured["command"] = command
            return super().switch(command, port=port, model_path=model_path)

    cfg = DaemonConfig(host="127.0.0.1", port=8770, token="t", roots=["/x"])
    app = create_app(cfg, manager=Cap(), command_builder=fake_builder)
    tuning = ["-c", "65536"]
    r = TestClient(app).post("/api/servers/switch",
                             json={"model": "Bar", "port": 8080, "flags": tuning}, headers=H)
    assert r.status_code == 200
    assert "-c" in captured["command"] and "65536" in captured["command"]
    assert "--host" in captured["command"]
