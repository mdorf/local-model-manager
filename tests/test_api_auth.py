from fastapi.testclient import TestClient

from lmm.api import create_app
from lmm.daemonconfig import DaemonConfig


def _client(token="secret-token"):
    cfg = DaemonConfig(host="127.0.0.1", port=8770, token=token, roots=["/tmp"])
    return TestClient(create_app(cfg))


def test_health_is_open():
    r = _client().get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_protected_route_rejects_without_token():
    r = _client().get("/api/servers")
    assert r.status_code == 401


def test_protected_route_accepts_valid_token():
    c = _client(token="abc123")
    r = c.get("/api/servers", headers={"Authorization": "Bearer abc123"})
    assert r.status_code == 200


def test_protected_route_rejects_wrong_token():
    c = _client(token="abc123")
    r = c.get("/api/servers", headers={"Authorization": "Bearer WRONG"})
    assert r.status_code == 401
