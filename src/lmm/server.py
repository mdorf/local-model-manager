"""Orchestrate the llama-server lifecycle: start, stop, switch, adopt, status."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from lmm.discovery import discover_models
from lmm.health import is_healthy, served_model_id, smoke_test, wait_for_health
from lmm.ports import is_port_in_use, listening_pid
from lmm.process import pid_alive, spawn, stop_proc, terminate_pid
from lmm.state import InstanceRecord, load_instances, mutate_instances, state_dir


def _api_key_from_command(command: list[str]) -> str | None:
    """Pull the value of `--api-key` out of a llama-server command, if present."""
    if "--api-key" in command:
        i = command.index("--api-key")
        if i + 1 < len(command):
            return command[i + 1]
    return None


@dataclass
class ServerInstance:
    port: int
    pid: int
    model_path: str
    started_at: float
    status: str               # starting|ready|unhealthy|running|crashed|stopped
    external: bool = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@dataclass
class ServerManager:
    log_dir: Path = field(default_factory=lambda: state_dir() / "logs")

    def __post_init__(self):
        self.log_dir = Path(self.log_dir)
        self._procs: dict[int, subprocess.Popen] = {}

    def list(self) -> list[InstanceRecord]:
        return load_instances()

    def _upsert(self, rec: InstanceRecord) -> None:
        def apply(recs):
            return [r for r in recs if r.port != rec.port] + [rec]
        mutate_instances(apply)

    def forget(self, port: int) -> None:
        mutate_instances(lambda recs: [r for r in recs if r.port != port])
        self._procs.pop(port, None)

    def start(self, command: list[str], *, port: int, model_path: str,
              ready_timeout: float = 120.0) -> ServerInstance:
        if any(r.port == port for r in load_instances()) or is_port_in_use(port):
            raise RuntimeError(f"port {port} is already in use / managed")
        log_path = self.log_dir / f"server-{port}.log"
        proc = spawn(command, log_path)
        self._procs[port] = proc
        started_at = time.time()
        self._upsert(InstanceRecord(port=port, pid=proc.pid,
                                    model_path=model_path, started_at=started_at))
        base = f"http://127.0.0.1:{port}"
        api_key = _api_key_from_command(command)
        if wait_for_health(base, timeout=ready_timeout):
            status = "ready" if smoke_test(base, api_key=api_key) else "unhealthy"
        else:
            status = "unhealthy"
        return ServerInstance(port=port, pid=proc.pid, model_path=model_path,
                              started_at=started_at, status=status)

    def stop(self, port: int, timeout: float = 10.0) -> bool:
        rec = next((r for r in load_instances() if r.port == port), None)
        ok = True
        proc = self._procs.get(port)
        if proc is not None:
            ok = stop_proc(proc, timeout=timeout)
        elif rec is not None and rec.pid > 0:
            # Cross-process or adopted server we have a real pid for — an
            # explicit stop/switch terminates it (adopt captures the pid). A
            # pid-less record (-1) is just forgotten; we can't kill what we can't find.
            ok = terminate_pid(rec.pid, timeout=timeout)
        self.forget(port)
        return ok

    def switch(self, command: list[str], *, port: int, model_path: str,
               ready_timeout: float = 120.0) -> ServerInstance:
        # Single-model policy: stop every running server (including an adopted
        # one occupying the target port) before starting the replacement.
        for rec in load_instances():
            self.stop(rec.port)
        return self.start(command, port=port, model_path=model_path,
                          ready_timeout=ready_timeout)

    def adopt(self, port: int, model_path: str | None = None) -> ServerInstance | None:
        if any(r.port == port for r in load_instances()):
            return None                      # already managed — don't clobber
        base = f"http://127.0.0.1:{port}"
        if not is_healthy(base):
            return None
        # Capture the real served model id (its --alias) so the UI shows the live
        # model, not "(external)". Caller may pass a resolved file path instead.
        if model_path is None:
            model_path = served_model_id(base) or "(external)"
        # Capture the real listening pid so an explicit stop/switch can terminate
        # this server (falls back to -1 = "known to be running, pid unknown").
        pid = listening_pid(port) or -1
        rec = InstanceRecord(port=port, pid=pid, model_path=model_path,
                             started_at=time.time(), external=True)
        self._upsert(rec)
        return ServerInstance(port=port, pid=pid, model_path=model_path,
                              started_at=rec.started_at, status="ready",
                              external=True)

    def status(self) -> list[ServerInstance]:
        out: list[ServerInstance] = []
        for r in load_instances():
            if r.external:
                status = "ready" if is_healthy(f"http://127.0.0.1:{r.port}") else "stopped"
            elif not is_port_in_use(r.port):
                # Port gone → server has exited (pid may still be a zombie).
                # This port-first classification is safe because start() blocks
                # on wait_for_health before persisting; a future concurrent
                # daemon should special-case very-recently-started records.
                status = "crashed"
            elif is_healthy(f"http://127.0.0.1:{r.port}"):
                status = "ready"
            elif pid_alive(r.pid):
                status = "running"
            else:
                status = "crashed"
            out.append(ServerInstance(port=r.port, pid=r.pid,
                                      model_path=r.model_path,
                                      started_at=r.started_at, status=status,
                                      external=r.external))
        return out


def autodetect_servers(manager: ServerManager, roots: list[str],
                       ports: list[int]) -> list[ServerInstance]:
    """Adopt any healthy llama-server on `ports` the daemon doesn't already manage.

    Lets the UI reflect a model that's running but wasn't started by this daemon
    (e.g. a manually-launched server, or one that outlived a daemon restart). The
    served model id is resolved back to a discovered model file when possible, so
    the UI matches it to a model in the sidebar by filename.
    """
    managed = {r.port for r in manager.list()}
    discovered = None
    adopted: list[ServerInstance] = []
    for port in ports:
        if port in managed:
            continue
        served = served_model_id(f"http://127.0.0.1:{port}")
        if not served:
            continue
        if discovered is None:
            discovered = discover_models(roots)
        match = next((m for m in discovered if m.path.stem == served), None)
        inst = manager.adopt(port, model_path=str(match.path) if match else served)
        if inst is not None:
            adopted.append(inst)
    return adopted
