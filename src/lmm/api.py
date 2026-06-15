"""FastAPI control-plane app wrapping discovery/recommendation/lifecycle."""

from __future__ import annotations

import threading
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException

from lmm.daemonconfig import DaemonConfig
from lmm.server import ServerInstance, ServerManager


def _instance_dict(inst: ServerInstance) -> dict:
    return {"port": inst.port, "pid": inst.pid, "model_path": inst.model_path,
            "model": Path(inst.model_path).name, "status": inst.status,
            "external": inst.external, "base_url": inst.base_url,
            "started_at": inst.started_at}


def _make_auth(config: DaemonConfig):
    def require_token(authorization: str | None = Header(default=None)):
        if not config.token:
            return
        if authorization != f"Bearer {config.token}":
            raise HTTPException(status_code=401, detail="invalid or missing token")
    return require_token


def create_app(config: DaemonConfig, manager: ServerManager | None = None,
               command_builder=None) -> FastAPI:
    app = FastAPI(title="local-model-manager")
    app.state.config = config
    app.state.manager = manager or ServerManager()
    app.state.command_builder = command_builder
    app.state.lock = threading.Lock()
    auth = _make_auth(config)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/servers", dependencies=[Depends(auth)])
    def list_servers():
        return {"servers": [_instance_dict(s) for s in app.state.manager.status()]}

    return app
