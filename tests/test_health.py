import subprocess
import sys
from pathlib import Path

import pytest

from lmm.health import is_healthy, smoke_test, wait_for_health
from lmm.ports import pick_free_port

FAKE = Path(__file__).parent / "fake_llama.py"


@pytest.fixture
def fake_server():
    port = pick_free_port(start=49600)
    proc = subprocess.Popen([sys.executable, str(FAKE), "--port", str(port)])
    base = f"http://127.0.0.1:{port}"
    try:
        assert wait_for_health(base, timeout=10.0)
        yield base
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_wait_for_health_true_when_up(fake_server):
    assert is_healthy(fake_server) is True


def test_wait_for_health_false_when_nothing_there():
    assert wait_for_health("http://127.0.0.1:49999", timeout=1.0) is False


def test_smoke_test_passes_against_fake(fake_server):
    assert smoke_test(fake_server) is True
