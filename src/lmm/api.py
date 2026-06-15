"""FastAPI control-plane app wrapping discovery/recommendation/lifecycle."""

from __future__ import annotations

import secrets
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from lmm.daemonconfig import DaemonConfig
from lmm.discovery import discover_models
from lmm.gguf import read_gguf
from lmm.hardware import detect_hardware
from lmm.llama import get_supported_flags
from lmm.models import Model
from lmm.recommend import recommend_config
from lmm.server import ServerInstance, ServerManager


class StartServerRequest(BaseModel):
    model: str
    port: int | None = None
    flags: list[str] | None = None


class SwitchServerRequest(BaseModel):
    model: str
    port: int | None = None


def _model_dict(m: Model) -> dict:
    return {"name": m.path.name, "path": str(m.path), "arch": m.arch,
            "family": m.family, "size_label": m.size_label, "quant": m.quant,
            "context_length": m.context_length, "has_mtp": m.has_mtp,
            "hf_base_repo": m.hf_base_repo}


def _instance_dict(inst: ServerInstance) -> dict:
    return {"port": inst.port, "pid": inst.pid, "model_path": inst.model_path,
            "model": Path(inst.model_path).name, "status": inst.status,
            "external": inst.external, "base_url": inst.base_url,
            "started_at": inst.started_at}


def _default_command_builder(config: DaemonConfig):
    def build(model_name: str, port: int):
        for m in discover_models(config.roots):
            if m.path.name == model_name or str(m.path) == model_name:
                metadata = read_gguf(m.shards[0]).metadata
                cfg = recommend_config(m, metadata, detect_hardware(),
                                       supported=get_supported_flags() or None,
                                       port=port, alias=m.path.stem,
                                       api_key=config.inference_key)
                return ["llama-server", *cfg.flags], str(m.path)
        raise HTTPException(status_code=404, detail="model not found")
    return build


def _make_auth(config: DaemonConfig):
    def require_token(authorization: str | None = Header(default=None)):
        if not config.token:
            return  # empty token = auth disabled (dev only); real configs auto-generate one
        expected = f"Bearer {config.token}"
        if authorization is None or not secrets.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="invalid or missing token")
    return require_token


def create_app(config: DaemonConfig, manager: ServerManager | None = None,
               command_builder=None) -> FastAPI:
    app = FastAPI(title="local-model-manager")
    app.state.config = config
    app.state.manager = manager or ServerManager()
    app.state.command_builder = command_builder or _default_command_builder(config)
    app.state.lock = threading.Lock()
    auth = _make_auth(config)

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/servers", dependencies=[Depends(auth)])
    def list_servers():
        return {"servers": [_instance_dict(s) for s in app.state.manager.status()]}

    def _find(name: str):
        for m in discover_models(config.roots):
            if m.path.name == name or str(m.path) == name:
                return m
        return None

    @app.get("/api/models", dependencies=[Depends(auth)])
    def list_models():
        return {"models": [_model_dict(m) for m in discover_models(config.roots)]}

    @app.get("/api/models/{name}/recommend", dependencies=[Depends(auth)])
    def recommend_for(name: str):
        model = _find(name)
        if model is None:
            raise HTTPException(status_code=404, detail="model not found")
        metadata = read_gguf(model.shards[0]).metadata
        cfg = recommend_config(model, metadata, detect_hardware(),
                               supported=get_supported_flags() or None,
                               port=config.port, alias=model.path.stem)
        return {"model": model.path.name, "context": cfg.context,
                "cache_type": cfg.cache_type, "flags": cfg.flags,
                "warnings": cfg.warnings,
                "fit": {"level": cfg.fit.level, "fits": cfg.fit.fits,
                        "message": cfg.fit.message}}

    @app.post("/api/servers", dependencies=[Depends(auth)])
    def start_server(body: StartServerRequest):
        port = body.port or 8080
        command, model_path = app.state.command_builder(body.model, port)
        with app.state.lock:
            inst = app.state.manager.start(command, port=port, model_path=model_path)
        return _instance_dict(inst)

    @app.delete("/api/servers/{port}", dependencies=[Depends(auth)])
    def stop_server(port: int):
        with app.state.lock:
            ok = app.state.manager.stop(port)
        return {"stopped": ok, "port": port}

    @app.post("/api/servers/switch", dependencies=[Depends(auth)])
    def switch_server(body: SwitchServerRequest):
        port = body.port or 8080
        command, model_path = app.state.command_builder(body.model, port)
        with app.state.lock:
            inst = app.state.manager.switch(command, port=port, model_path=model_path)
        return _instance_dict(inst)

    return app
