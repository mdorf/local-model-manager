import sys

from lmm.process import pid_alive, spawn, stop_proc, terminate_pid


def test_spawn_runs_and_logs(tmp_path):
    import time
    log = tmp_path / "out.log"
    proc = spawn([sys.executable, "-c", "print('hello'); import time; time.sleep(30)"], log)
    try:
        assert pid_alive(proc.pid) is True
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if log.exists() and "hello" in log.read_text():
                break
            time.sleep(0.02)
        assert "hello" in log.read_text()
    finally:
        stop_proc(proc, timeout=5)
    assert pid_alive(proc.pid) is False


def test_stop_proc_terminates(tmp_path):
    proc = spawn([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "l.log")
    assert pid_alive(proc.pid) is True
    assert stop_proc(proc, timeout=5) is True
    assert pid_alive(proc.pid) is False


def test_terminate_pid_on_dead_pid_is_true(tmp_path):
    proc = spawn([sys.executable, "-c", "pass"], tmp_path / "l.log")
    proc.wait(timeout=5)
    assert terminate_pid(proc.pid, timeout=2) is True


def test_pid_alive_false_for_unused_pid():
    assert pid_alive(999999) is False
