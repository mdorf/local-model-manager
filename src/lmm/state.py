"""Persist running-server records so separate CLI invocations can manage them."""

from __future__ import annotations

import fcntl
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


def state_dir() -> Path:
    override = os.environ.get("LMM_STATE_DIR")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "local-model-manager"
    xdg = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".local" / "state"
    return base / "local-model-manager"


def _instances_file() -> Path:
    return state_dir() / "instances.json"


@dataclass
class InstanceRecord:
    port: int
    pid: int
    model_path: str
    started_at: float
    external: bool = False


def load_instances() -> list[InstanceRecord]:
    path = _instances_file()
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(raw, list):
        return []
    out: list[InstanceRecord] = []
    for item in raw:
        try:
            out.append(InstanceRecord(**item))
        except (TypeError, ValueError):
            continue
    return out


def save_instances(records: list[InstanceRecord]) -> None:
    path = _instances_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps([asdict(r) for r in records], indent=2))
    tmp.replace(path)  # atomic


def _lock_file() -> Path:
    return state_dir() / "instances.lock"


def mutate_instances(mutator):
    """Atomically read -> mutator(list) -> save, under a cross-process exclusive lock.

    `mutator` receives the current list of InstanceRecord and returns the new list.
    Serializes concurrent writers (daemon threadpool + any direct CLI invocation)
    so no read-modify-write update is lost.
    """
    lock_path = _lock_file()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            new_records = mutator(load_instances())
            save_instances(new_records)
            return new_records
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
