# tests/test_daemon_client.py
import io
import json
import lmm.daemon_client as dc

class _Resp(io.BytesIO):
    status = 200
    def __enter__(self): return self
    def __exit__(self, *a): return False

def test_daemon_available_reads_json_and_pings(monkeypatch, tmp_path):
    (tmp_path / "daemon.json").write_text(json.dumps(
        {"host": "127.0.0.1", "port": 8770, "token": "tk"}))
    monkeypatch.setattr(dc, "state_dir", lambda: tmp_path)
    monkeypatch.setattr(dc.urllib.request, "urlopen",
                        lambda url, timeout=0: _Resp(b'{"status":"ok"}'))
    info = dc.daemon_available()
    assert info == {"base": "http://127.0.0.1:8770", "token": "tk"}

def test_daemon_available_none_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setattr(dc, "state_dir", lambda: tmp_path)
    assert dc.daemon_available() is None

def test_start_posts_body(monkeypatch):
    seen = {}
    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["method"] = req.get_method()
        seen["data"] = req.data
        seen["auth"] = req.headers.get("Authorization")
        return _Resp(b'{"port":8080,"status":"ready"}')
    monkeypatch.setattr(dc.urllib.request, "urlopen", fake_urlopen)
    out = dc.start("http://h:8770", "tk", "m.gguf", 8080)
    assert out["status"] == "ready"
    assert seen["url"].endswith("/api/servers") and seen["method"] == "POST"
    assert json.loads(seen["data"]) == {"model": "m.gguf", "port": 8080}
    assert seen["auth"] == "Bearer tk"
