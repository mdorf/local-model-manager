"""FastAPI control-plane app wrapping discovery/recommendation/lifecycle."""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lmm.daemonconfig import DaemonConfig
from lmm.discovery import discover_models
from lmm.gguf import read_gguf
from lmm.hardware import detect_hardware
from lmm.hermes import bind as hermes_bind
from lmm.llama import get_supported_flags
from lmm.logtail import read_log_tail, tail_new_lines
from lmm.models import Model
from lmm.net import is_loopback
from lmm.recommend import recommend_config
from lmm.server import ServerInstance, ServerManager
from lmm.state import state_dir


_WEBUI_DIR = Path(__file__).parent / "webui"


def _inject_token(html: str, token: str, client_host: str | None) -> str:
    if client_host in ("127.0.0.1", "::1", "localhost") and token:
        return html.replace("</head>", f'<script>window.LMM_TOKEN={json.dumps(token)}</script></head>', 1)
    return html


class StartServerRequest(BaseModel):
    model: str
    port: int | None = None
    flags: list[str] | None = None


class SwitchServerRequest(BaseModel):
    model: str
    port: int | None = None


class BindRequest(BaseModel):
    provider_name: str = "local"
    hermes_config: str | None = None  # default: the daemon user's ~/.hermes/config.yaml


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


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
                # bind the inference server to the daemon's host; enforce an
                # api-key only when that's LAN-exposed (loopback = local-only,
                # no key — so llama-server's own UI works without one).
                lan = not is_loopback(config.host)
                cfg = recommend_config(m, metadata, detect_hardware(),
                                       supported=get_supported_flags() or None,
                                       host=config.host, port=port, alias=m.path.stem,
                                       api_key=config.inference_key if lan else None)
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
    app.state.log_dir = state_dir() / "logs"
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

    @app.get("/api/connection-info", dependencies=[Depends(auth)])
    def connection_info():
        running = app.state.manager.status()
        inst = running[0] if running else None
        if inst is not None:
            base_url = inst.base_url.rstrip("/") + "/v1"
            model_id = Path(inst.model_path).stem
        else:
            host = config.host if config.host not in ("0.0.0.0", "::") else "127.0.0.1"
            base_url = f"http://{host}:8080/v1"
            model_id = None
        # a loopback (local-only) server runs without --api-key, so report no key
        lan = not is_loopback(config.host)
        return {"base_url": base_url,
                "inference_key": config.inference_key if lan else "",
                "model_id": model_id}

    @app.post("/api/bind", dependencies=[Depends(auth)])
    def bind_hermes(body: BindRequest, request: Request):
        # Loopback-only: the daemon (running as the owning user) can write that
        # user's local ~/.hermes. It cannot write a remote client's machine, so
        # remote callers use the `lmm bind` command instead.
        client = request.client.host if request.client else None
        if client not in _LOOPBACK_HOSTS:
            raise HTTPException(status_code=403,
                                detail="bind is only available to the local host operator")
        running = app.state.manager.status()
        inst = running[0] if running else None
        if inst is None:
            raise HTTPException(status_code=409, detail="no server is running to bind to")
        model_id = Path(inst.model_path).stem
        base_url = f"http://127.0.0.1:{inst.port}/v1"
        lan = not is_loopback(config.host)
        api_key = config.inference_key if lan else "local"  # keyless server ignores it
        config_path = (Path(body.hermes_config) if body.hermes_config
                       else Path.home() / ".hermes" / "config.yaml")
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Hermes config not found: {config_path}")
        info = hermes_bind(config_path, base_url=base_url, model_id=model_id,
                           provider_name=body.provider_name, api_key=api_key)
        return {"bound": True, **info}

    SUBPROTO_PREFIX = "lmm.bearer."

    @app.websocket("/api/stream")
    async def stream(ws: WebSocket):
        # auth via subprotocol: browsers can't set headers on WS
        protos = [p.strip() for p in
                  (ws.headers.get("sec-websocket-protocol") or "").split(",") if p.strip()]
        token = next((p[len(SUBPROTO_PREFIX):] for p in protos
                      if p.startswith(SUBPROTO_PREFIX)), None)
        if config.token and (token is None or
                             not secrets.compare_digest(token, config.token)):
            await ws.close(code=1008)
            return
        accept_proto = next((p for p in protos if p.startswith(SUBPROTO_PREFIX)), None)
        await ws.accept(subprotocol=accept_proto)

        log_dir = Path(app.state.log_dir)
        offsets: dict[int, int] = {}
        # initial tail of each running server's log
        for inst in app.state.manager.status():
            path = log_dir / f"server-{inst.port}.log"
            for line in read_log_tail(path, max_lines=200):
                await ws.send_json({"type": "log", "port": inst.port, "line": line})
            offsets[inst.port] = path.stat().st_size if path.exists() else 0
        await ws.send_json({"type": "status",
                            "servers": [_instance_dict(s) for s in app.state.manager.status()]})
        try:
            while True:
                await asyncio.sleep(1.0)
                for inst in app.state.manager.status():
                    path = log_dir / f"server-{inst.port}.log"
                    prev = offsets.get(inst.port, 0)
                    # log truncated/rotated (e.g. on switch) → restart from the top
                    if path.exists() and path.stat().st_size < prev:
                        prev = 0
                    lines, offsets[inst.port] = tail_new_lines(path, prev)
                    for line in lines:
                        await ws.send_json({"type": "log", "port": inst.port, "line": line})
                await ws.send_json({"type": "status",
                                    "servers": [_instance_dict(s) for s in app.state.manager.status()]})
        except Exception:
            return

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        html = (_WEBUI_DIR / "index.html").read_text()
        host = request.client.host if request.client else None
        return HTMLResponse(_inject_token(html, config.token, host))

    if _WEBUI_DIR.exists():
        app.mount("/", StaticFiles(directory=str(_WEBUI_DIR)), name="webui")

    return app
