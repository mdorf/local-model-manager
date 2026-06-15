# tests/test_api_stream.py
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


def _app(tmp_path, token="t"):
    log = tmp_path / "server-8080.log"
    log.write_text("hello-from-llama\nstarting up\n")
    inst = ServerInstance(port=8080, pid=1, model_path="/m/x.gguf", started_at=0.0,
                          status="ready", external=False)
    mgr = FakeManager([inst])
    cfg = DaemonConfig(token=token, roots=["/x"])
    app = create_app(cfg, manager=mgr, command_builder=lambda *a: None)
    app.state.log_dir = tmp_path  # the WS reads server-{port}.log from here
    return app


def test_stream_sends_initial_log_tail_and_status(tmp_path):
    app = _app(tmp_path)
    client = TestClient(app)
    with client.websocket_connect("/api/stream", subprotocols=["lmm.bearer.t"]) as ws:
        frames = [ws.receive_json() for _ in range(3)]
    types = [f["type"] for f in frames]
    assert "status" in types
    log_lines = [f["line"] for f in frames if f["type"] == "log"]
    assert "hello-from-llama" in log_lines


def test_stream_rejects_bad_token(tmp_path):
    app = _app(tmp_path)
    client = TestClient(app)
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises((WebSocketDisconnect, Exception)):
        with client.websocket_connect("/api/stream", subprotocols=["lmm.bearer.WRONG"]):
            pass
