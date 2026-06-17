"""Spawn and stop detached child processes (llama-server in production)."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from pathlib import Path


def process_argv(pid: int) -> list[str] | None:
    """The argv of a running process, so the daemon can report what a server was
    actually launched with — including an adopted one we didn't spawn. None if it
    can't be read. `ps` space-joins argv; for our flag set (no spaces in paths)
    shlex.split reconstructs it faithfully enough for display."""
    if pid <= 0:
        return None
    # absolute path first: a launchd daemon's PATH may lack /bin (cf. the sysctl/lsof lessons)
    for exe in ("/bin/ps", "ps"):
        try:
            out = subprocess.run([exe, "-ww", "-o", "command=", "-p", str(pid)],
                                 capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        line = out.stdout.strip()
        if not line:
            return None
        try:
            return shlex.split(line)
        except ValueError:
            return line.split()
    return None


def spawn(command: list[str], log_path: str | Path) -> subprocess.Popen:
    """Start `command` detached (new session), appending stdout+stderr to log_path."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "ab")  # noqa: SIM115 - handed to the child; closed on exit
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env=env)
    log.close()  # parent's copy not needed; child inherited the fd
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
    Worst-case duration is ~2×timeout (SIGTERM wait then SIGKILL wait).
    """
    if proc.poll() is not None:
        return True
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()  # SIGKILL is guaranteed to terminate; blocking wait reaps the child
        return True


def terminate_pid(pid: int, timeout: float = 10.0) -> bool:
    """Terminate by pid (cross-invocation; no Popen handle). SIGTERM then SIGKILL.
    The OS reaps the reparented child.
    Worst-case duration is ~2×timeout (SIGTERM wait then SIGKILL wait).
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
