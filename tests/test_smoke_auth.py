import lmm.health as health
from lmm.server import _api_key_from_command


class _Resp:
    status = 200
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False
    def read(self):
        return b"{}"


def test_smoke_test_sends_bearer_when_key(monkeypatch):
    seen = {}

    def fake(req, timeout=0):
        seen["auth"] = req.headers.get("Authorization")
        return _Resp()

    monkeypatch.setattr(health.urllib.request, "urlopen", fake)
    assert health.smoke_test("http://127.0.0.1:8080", api_key="sek") is True
    assert seen["auth"] == "Bearer sek"


def test_smoke_test_no_auth_header_without_key(monkeypatch):
    seen = {}

    def fake(req, timeout=0):
        seen["auth"] = req.headers.get("Authorization")
        return _Resp()

    monkeypatch.setattr(health.urllib.request, "urlopen", fake)
    assert health.smoke_test("http://127.0.0.1:8080") is True
    assert seen["auth"] is None


def test_api_key_from_command():
    assert _api_key_from_command(
        ["llama-server", "-m", "x", "--api-key", "abc", "--port", "8080"]) == "abc"
    assert _api_key_from_command(["llama-server", "-m", "x"]) is None
    assert _api_key_from_command(["llama-server", "--api-key"]) is None  # flag w/o value
