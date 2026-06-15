import subprocess
import sys
from pathlib import Path

import pytest

from lmm.ports import pick_free_port
from lmm.server import ServerInstance, ServerManager

FAKE = Path(__file__).parent / "fake_llama.py"


@pytest.fixture
def mgr(monkeypatch, tmp_path):
    monkeypatch.setenv("LMM_STATE_DIR", str(tmp_path / "state"))
    return ServerManager(log_dir=tmp_path / "logs")


def _fake_cmd(port):
    return [sys.executable, str(FAKE), "--port", str(port)]


def test_start_reaches_ready_and_persists(mgr):
    port = pick_free_port(start=49700)
    inst = mgr.start(_fake_cmd(port), port=port, model_path="/m/x.gguf",
                     ready_timeout=10.0)
    try:
        assert isinstance(inst, ServerInstance)
        assert inst.status == "ready"
        assert any(r.port == port for r in ServerManager(log_dir=mgr.log_dir).list())
    finally:
        mgr.stop(port)


def test_stop_removes_instance(mgr):
    port = pick_free_port(start=49720)
    mgr.start(_fake_cmd(port), port=port, model_path="/m/x.gguf", ready_timeout=10.0)
    assert mgr.stop(port) is True
    assert all(r.port != port for r in mgr.list())


def test_start_refuses_occupied_port(mgr):
    port = pick_free_port(start=49740)
    mgr.start(_fake_cmd(port), port=port, model_path="/m/a.gguf", ready_timeout=10.0)
    try:
        with pytest.raises(RuntimeError, match="in use|already"):
            mgr.start(_fake_cmd(port), port=port, model_path="/m/b.gguf",
                      ready_timeout=3.0)
    finally:
        mgr.stop(port)


def test_switch_replaces_running_model(mgr):
    p1 = pick_free_port(start=49760)
    mgr.start(_fake_cmd(p1), port=p1, model_path="/m/a.gguf", ready_timeout=10.0)
    p2 = pick_free_port(start=p1 + 1)
    inst = mgr.switch(_fake_cmd(p2), port=p2, model_path="/m/b.gguf",
                      ready_timeout=10.0)
    try:
        assert inst.status == "ready"
        ports = {r.port for r in mgr.list()}
        assert p2 in ports and p1 not in ports
    finally:
        mgr.stop(p2)


def test_adopt_external_server(mgr):
    port = pick_free_port(start=49780)
    proc = subprocess.Popen(_fake_cmd(port))
    try:
        from lmm.health import wait_for_health
        assert wait_for_health(f"http://127.0.0.1:{port}", timeout=10.0)
        inst = mgr.adopt(port)
        assert inst is not None
        assert inst.external is True
        assert inst.status in ("ready", "running")
        assert any(r.port == port and r.external for r in mgr.list())
    finally:
        proc.terminate()
        proc.wait(timeout=5)
        mgr.forget(port)


def test_adopt_refuses_already_managed_port(mgr):
    port = pick_free_port(start=49820)
    mgr.start(_fake_cmd(port), port=port, model_path="/m/x.gguf", ready_timeout=10.0)
    try:
        # adopting a port we already manage must NOT clobber the managed record
        assert mgr.adopt(port) is None
        rec = next(r for r in mgr.list() if r.port == port)
        assert rec.external is False
        assert rec.model_path == "/m/x.gguf"
    finally:
        assert mgr.stop(port) is True          # real stop still works (no leak)
        assert all(r.port != port for r in mgr.list())


def test_switch_same_port_succeeds(mgr):
    # The real CLI `switch --port 8080` reuses the same port: stop then start
    # on the identical port must succeed (listen socket freed on process exit).
    port = pick_free_port(start=49850)
    mgr.start(_fake_cmd(port), port=port, model_path="/m/a.gguf", ready_timeout=10.0)
    inst = mgr.switch(_fake_cmd(port), port=port, model_path="/m/b.gguf",
                      ready_timeout=10.0)
    try:
        assert inst.status == "ready"
        recs = mgr.list()
        assert [r.port for r in recs] == [port]          # exactly one, same port
        assert recs[0].model_path == "/m/b.gguf"          # now the new model
    finally:
        mgr.stop(port)


def test_status_marks_crashed_when_pid_gone(mgr):
    port = pick_free_port(start=49800)
    inst = mgr.start(_fake_cmd(port), port=port, model_path="/m/x.gguf",
                     ready_timeout=10.0)
    from lmm.process import terminate_pid
    terminate_pid(inst.pid, timeout=5)
    fresh = ServerManager(log_dir=mgr.log_dir)
    statuses = {s.port: s.status for s in fresh.status()}
    assert statuses.get(port) in ("crashed", "stopped")
    mgr.forget(port)
