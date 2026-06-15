"""Daemon configuration: bind host/port, shared token, model roots."""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lmm.state import state_dir

DEFAULT_PORT = 8770


def _default_roots() -> list[str]:
    override = os.environ.get("LMM_MODELS_DIR")
    if override:
        return [override]
    return [str(Path.home() / "models")]


@dataclass
class DaemonConfig:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PORT
    token: str = ""
    roots: list[str] = field(default_factory=_default_roots)


def _config_file() -> Path:
    return state_dir() / "daemon.json"


def load_or_create_config() -> DaemonConfig:
    path = _config_file()
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    cfg = DaemonConfig(
        host=data.get("host", "127.0.0.1"),
        port=int(data.get("port", DEFAULT_PORT)),
        token=data.get("token") or secrets.token_hex(24),
        roots=data.get("roots") or _default_roots(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2))
    return cfg
