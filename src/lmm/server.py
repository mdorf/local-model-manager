"""Orchestrate the llama-server lifecycle: start, stop, switch, adopt, status."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from lmm.health import is_healthy, smoke_test, wait_for_health
from lmm.ports import is_port_in_use
from lmm.process import pid_alive, spawn, stop_proc, terminate_pid
from lmm.state import InstanceRecord, load_instances, save_instances, state_dir


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

    def _save(self, records: list[InstanceRecord]) -> None:
        save_instances(records)

    def _upsert(self, rec: InstanceRecord) -> None:
        recs = [r for r in load_instances() if r.port != rec.port]
        recs.append(rec)
        self._save(recs)

    def forget(self, port: int) -> None:
        self._save([r for r in load_instances() if r.port != port])
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
        if wait_for_health(base, timeout=ready_timeout):
            status = "ready" if smoke_test(base) else "unhealthy"
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
        elif rec is not None and not rec.external:
            ok = terminate_pid(rec.pid, timeout=timeout)
        self.forget(port)
        return ok

    def switch(self, command: list[str], *, port: int, model_path: str,
               ready_timeout: float = 120.0) -> ServerInstance:
        for rec in load_instances():
            if not rec.external:
                self.stop(rec.port)
        return self.start(command, port=port, model_path=model_path,
                          ready_timeout=ready_timeout)

    def adopt(self, port: int) -> ServerInstance | None:
        base = f"http://127.0.0.1:{port}"
        if not is_healthy(base):
            return None
        rec = InstanceRecord(port=port, pid=-1, model_path="(external)",
                             started_at=time.time(), external=True)
        self._upsert(rec)
        return ServerInstance(port=port, pid=-1, model_path="(external)",
                              started_at=rec.started_at, status="ready",
                              external=True)

    def status(self) -> list[ServerInstance]:
        out: list[ServerInstance] = []
        for r in load_instances():
            if r.external:
                status = "ready" if is_healthy(f"http://127.0.0.1:{r.port}") else "stopped"
            elif not is_port_in_use(r.port):
                # Port gone → server has exited (pid may still be a zombie)
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
