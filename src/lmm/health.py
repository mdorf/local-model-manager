"""HTTP readiness and smoke-test probes for an OpenAI-compatible server."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def _get_status(url: str, timeout: float) -> int | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, OSError):
        return None


def is_healthy(base_url: str, timeout: float = 2.0) -> bool:
    return _get_status(base_url.rstrip("/") + "/health", timeout) == 200


def wait_for_health(base_url: str, timeout: float = 120.0, interval: float = 0.5) -> bool:
    """Poll /health until it returns 200 or the timeout elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_healthy(base_url, timeout=min(interval + 1.0, timeout)):
            return True
        time.sleep(interval)
    return False


def smoke_test(base_url: str, timeout: float = 30.0) -> bool:
    """Send a 1-token chat completion; True iff the server returns HTTP 200."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = json.dumps({"model": "smoke",
                          "messages": [{"role": "user", "content": "ping"}],
                          "max_tokens": 1}).encode()
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False
