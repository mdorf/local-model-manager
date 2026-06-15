"""Minimal stdlib HTTP client for talking to a running lmm daemon."""
from __future__ import annotations
import json
import urllib.error
import urllib.request
from lmm.state import state_dir


class DaemonError(Exception):
    """A routed CLI request to the daemon failed (HTTP error or unreachable)."""


def _info() -> dict | None:
    try:
        return json.loads((state_dir() / "daemon.json").read_text())
    except (OSError, ValueError):
        return None


def daemon_available(timeout: float = 1.0) -> dict | None:
    info = _info()
    if not info:
        return None
    base = f"http://{info.get('host', '127.0.0.1')}:{info.get('port', 8770)}"
    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout=timeout) as r:
            if getattr(r, "status", 200) == 200:
                return {"base": base, "token": info.get("token", "")}
    except (urllib.error.URLError, OSError):
        return None
    return None


def _request(method: str, base: str, path: str, token: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{base}{path}", data=data, method=method,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = (json.loads(e.read() or b"{}") or {}).get("detail", "")
        except ValueError:
            pass
        raise DaemonError(f"daemon error {e.code}: {detail or e.reason}") from None
    except urllib.error.URLError as e:
        raise DaemonError(f"daemon unreachable: {e.reason}") from None
    return json.loads(raw) if raw else None


def start(base, token, model, port):
    return _request("POST", base, "/api/servers", token, {"model": model, "port": port})


def switch(base, token, model, port):
    return _request("POST", base, "/api/servers/switch", token, {"model": model, "port": port})


def stop(base, token, port):
    return _request("DELETE", base, f"/api/servers/{port}", token)


def status(base, token):
    return _request("GET", base, "/api/servers", token)
