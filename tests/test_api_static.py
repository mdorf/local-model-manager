# tests/test_api_static.py
from fastapi.testclient import TestClient
from lmm.api import create_app, _inject_token
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


def test_inject_token_only_for_loopback():
    html = "<head></head><body></body>"
    out = _inject_token(html, "tok", "127.0.0.1")
    assert 'window.LMM_TOKEN="tok"' in out
    assert _inject_token(html, "tok", "10.0.0.5") == html  # remote: untouched


def test_index_served():
    cfg = DaemonConfig(token="t", roots=["/x"])
    app = create_app(cfg, manager=FakeManager(), command_builder=lambda *a: None)
    r = TestClient(app).get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()


def test_ui_assets_sent_no_cache():
    cfg = DaemonConfig(token="t", roots=["/x"])
    app = create_app(cfg, manager=FakeManager(), command_builder=lambda *a: None)
    c = TestClient(app)
    assert c.get("/").headers.get("cache-control") == "no-cache"
    # API responses aren't forced no-cache by the UI middleware
    assert "no-cache" not in (c.get("/api/health").headers.get("cache-control") or "")


def test_api_js_asset_gets_no_cache():
    # /api.js is a UI ASSET, not an /api/ endpoint — its name starts with "/api"
    # but it must still be no-cache, or the browser serves a stale api.js after a
    # reinstall (causing "api.setHomepage is not a function" style mismatches).
    cfg = DaemonConfig(token="t", roots=["/x"])
    app = create_app(cfg, manager=FakeManager(), command_builder=lambda *a: None)
    assert TestClient(app).get("/api.js").headers.get("cache-control") == "no-cache"
