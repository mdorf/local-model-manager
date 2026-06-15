"""Spawn and stop detached child processes (llama-server in production)."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path


def spawn(command: list[str], log_path: str | Path) -> subprocess.Popen:
    """Start `command` detached (new session), appending stdout+stderr to log_path.

    Waits briefly for the OS to schedule the child before returning, so callers
    can safely inspect liveness or early log output immediately.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 - handed to the child; closed on exit
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)
    log.close()  # parent's copy not needed; child inherited the fd
    # Give the OS a moment to schedule the child so liveness checks and early log
    # reads work reliably immediately after spawn returns.
    time.sleep(0.05)
    return proc


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_gone(pid: int, timeout: float, interval: float = 0.1) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(interval)
    return not pid_alive(pid)


def stop_proc(proc: subprocess.Popen, timeout: float = 10.0) -> bool:
    """Terminate a process we own (have the Popen for). SIGTERM, then SIGKILL.
    Reaps the child via Popen.wait so no zombie remains.
    """
    if proc.poll() is not None:
        return True
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        return True


def terminate_pid(pid: int, timeout: float = 10.0) -> bool:
    """Terminate by pid (cross-invocation; no Popen handle). SIGTERM then SIGKILL.
    The OS reaps the reparented child.
    """
    if not pid_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    if _wait_gone(pid, timeout):
        return True
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return _wait_gone(pid, timeout)
