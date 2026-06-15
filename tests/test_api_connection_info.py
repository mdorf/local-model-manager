# tests/test_api_connection_info.py
from fastapi.testclient import TestClient
from lmm.api import create_app
from lmm.daemonconfig import DaemonConfig
from lmm.server import ServerInstance
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
        self._inst.append(inst); return inst
    def switch(self, command, *, port, model_path):
        self.calls.append(("switch", port, model_path))
        self._inst = []
        return self.start(command, port=port, model_path=model_path)
    def stop(self, port):
        self.calls.append(("stop", port))
        self._inst = [i for i in self._inst if i.port != port]; return True


def test_connection_info_with_running_server():
    inst = ServerInstance(port=8080, pid=1, model_path="/models/Qwen3.6-27B-Q8_0.gguf",
                          started_at=0.0, status="ready", external=False)
    # base_url is a computed @property (http://127.0.0.1:<port>) -> connection-info adds /v1
    cfg = DaemonConfig(token="t", inference_key="sekret", roots=["/x"])
    app = create_app(cfg, manager=FakeManager([inst]), command_builder=lambda *a: None)
    r = TestClient(app).get("/api/connection-info", headers={"Authorization": "Bearer t"})
    assert r.status_code == 200
    body = r.json()
    assert body["base_url"].endswith("/v1")
    assert body["inference_key"] == "sekret"
    assert body["model_id"] == "Qwen3.6-27B-Q8_0"


def test_connection_info_requires_token():
    cfg = DaemonConfig(token="t", inference_key="sekret", roots=["/x"])
    app = create_app(cfg, manager=FakeManager(), command_builder=lambda *a: None)
    assert TestClient(app).get("/api/connection-info").status_code == 401
