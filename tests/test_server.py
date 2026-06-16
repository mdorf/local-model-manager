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
        assert inst.pid == proc.pid          # captured the real listening pid
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


# --- adopt model-id capture + auto-detect on startup ---

def test_adopt_records_explicit_model_path(mgr, monkeypatch):
    import lmm.server as s
    monkeypatch.setattr(s, "is_healthy", lambda base, **k: True)
    inst = mgr.adopt(48080, model_path="/models/Qwen3.6-27B-Q8_0.gguf")
    assert inst is not None and inst.external is True
    assert inst.model_path == "/models/Qwen3.6-27B-Q8_0.gguf"
    assert mgr.list()[0].model_path == "/models/Qwen3.6-27B-Q8_0.gguf"


def test_adopt_detects_served_model_id(mgr, monkeypatch):
    import lmm.server as s
    monkeypatch.setattr(s, "is_healthy", lambda base, **k: True)
    monkeypatch.setattr(s, "served_model_id", lambda base, **k: "Qwen3.6-27B-Q8_0")
    inst = mgr.adopt(48081)  # no explicit path → detects the served id
    assert inst.model_path == "Qwen3.6-27B-Q8_0"  # not "(external)"


def test_autodetect_adopts_and_resolves_to_file_path(mgr, monkeypatch):
    import lmm.server as s

    class _M:
        path = Path("/Users/Shared/models/Qwen3.6-27B-Q8_0.gguf")

    monkeypatch.setattr(s, "is_healthy", lambda base, **k: True)
    # only :8080 is serving; its alias is the gguf stem
    monkeypatch.setattr(s, "served_model_id",
                        lambda base, **k: "Qwen3.6-27B-Q8_0" if ":8080" in base else None)
    monkeypatch.setattr(s, "discover_models", lambda roots: [_M()])
    adopted = s.autodetect_servers(mgr, roots=["/Users/Shared/models"], ports=[8080, 8081])
    assert len(adopted) == 1
    # resolved the served alias back to the real file path (sidebar matches by name)
    assert mgr.list()[0].model_path == "/Users/Shared/models/Qwen3.6-27B-Q8_0.gguf"


def test_autodetect_skips_already_managed_port(mgr, monkeypatch):
    import lmm.server as s
    monkeypatch.setattr(s, "is_healthy", lambda base, **k: True)
    monkeypatch.setattr(s, "served_model_id", lambda base, **k: "Whatever")
    mgr.adopt(8080, model_path="/models/Already.gguf")  # pre-existing managed record
    adopted = s.autodetect_servers(mgr, roots=["/x"], ports=[8080])
    assert adopted == []  # didn't clobber the managed port
    assert mgr.list()[0].model_path == "/models/Already.gguf"


# --- adopted servers are now fully manageable (real pid → explicit stop/switch kills them) ---

def test_listening_pid_finds_the_listener(mgr):
    from lmm.ports import listening_pid
    port = pick_free_port(start=49840)
    proc = subprocess.Popen(_fake_cmd(port))
    try:
        from lmm.health import wait_for_health
        assert wait_for_health(f"http://127.0.0.1:{port}", timeout=10.0)
        assert listening_pid(port) == proc.pid
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_stop_terminates_adopted_by_pid(mgr, monkeypatch):
    # An adopted server has a real pid → explicit stop must terminate it by pid.
    # (Monkeypatched so we don't create an unreapable zombie: in production the
    # adopted process is reparented to launchd, which reaps it on death.)
    import lmm.server as s
    killed = []
    monkeypatch.setattr(s, "is_healthy", lambda base, **k: True)
    monkeypatch.setattr(s, "listening_pid", lambda port: 4242)
    monkeypatch.setattr(s, "terminate_pid", lambda pid, timeout=10.0: killed.append(pid) or True)
    inst = mgr.adopt(48090, model_path="/m/x.gguf")
    assert inst.pid == 4242                       # adopt captured the listening pid
    assert mgr.stop(48090) is True
    assert killed == [4242]                       # terminated by the captured pid
    assert all(r.port != 48090 for r in mgr.list())


def test_switch_terminates_adopted_then_starts(mgr, monkeypatch):
    # Switching while an adopted server is recorded must stop (kill) it, not 500.
    import lmm.server as s
    killed = []
    monkeypatch.setattr(s, "is_healthy", lambda base, **k: True)
    monkeypatch.setattr(s, "listening_pid", lambda port: 4242)
    monkeypatch.setattr(s, "terminate_pid", lambda pid, timeout=10.0: killed.append(pid) or True)
    mgr.adopt(48091, model_path="/m/old.gguf")    # adopted record present
    p = pick_free_port(start=49900)               # start the replacement on a real port
    inst = mgr.switch(_fake_cmd(p), port=p, model_path="/m/new.gguf", ready_timeout=10.0)
    try:
        assert inst.status == "ready"
        assert 4242 in killed                     # the adopted server was terminated
        assert all(r.port != 48091 for r in mgr.list())   # its record is gone
    finally:
        mgr.stop(p)
