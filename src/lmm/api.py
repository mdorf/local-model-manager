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
from lmm.hermes import list_profiles as hermes_list_profiles
from lmm.hermes import profiles_bound_to as hermes_profiles_bound_to
from lmm.llama import get_supported_flags
from lmm.logtail import read_log_tail, tail_new_lines
from lmm.models import Model
from lmm.net import is_loopback
from lmm.recommend import plumbing_flags, recommend_config
from lmm.server import ServerInstance, ServerManager, context_length_from_command
from lmm.state import load_homepages, set_homepage as state_set_homepage, state_dir


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
    flags: list[str] | None = None


class BindRequest(BaseModel):
    provider_name: str = "local"
    hermes_config: str | None = None  # default: the daemon user's ~/.hermes/config.yaml


class HomepageRequest(BaseModel):
    url: str | None = None  # blank/None clears the override


_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _model_dict(m: Model, homepage_override: str | None = None) -> dict:
    return {"name": m.path.name, "path": str(m.path), "arch": m.arch,
            "family": m.family, "size_label": m.size_label, "quant": m.quant,
            "context_length": m.context_length, "has_mtp": m.has_mtp,
            "hf_base_repo": m.hf_base_repo, "license": m.license,
            "quantized_by": m.quantized_by, "has_chat_template": m.has_chat_template,
            "author": m.author, "display_name": m.name, "sampling": m.sampling,
            "homepage_override": homepage_override}


def _instance_dict(inst: ServerInstance) -> dict:
    # the live launch flags (argv minus the binary) so the UI can show "current
    # run params"; None when unknown (e.g. an adopted server whose pid we lost).
    cmd = inst.command
    flags = cmd[1:] if cmd and len(cmd) > 1 else None
    return {"port": inst.port, "pid": inst.pid, "model_path": inst.model_path,
            "model": Path(inst.model_path).name, "status": inst.status,
            "external": inst.external, "base_url": inst.base_url,
            "started_at": inst.started_at, "flags": flags}


def _default_command_builder(config: DaemonConfig):
    def build(model_name: str, port: int, tuning: list[str] | None = None):
        """Compose the launch command: daemon-owned plumbing + tuning knobs.
        `tuning` is the user's editable-flag override; None = use the recommended
        tuning. The plumbing (-m/--host/--port/--alias/--api-key) ALWAYS comes from
        the daemon's config, so a stale/edited host can't reach llama-server."""
        for m in discover_models(config.roots):
            if m.matches(model_name):
                if tuning is None:
                    metadata = read_gguf(m.shards[0]).metadata
                    tuning = recommend_config(
                        m, metadata, detect_hardware(),
                        supported=get_supported_flags() or None).tuning_flags
                # bind to the daemon's host; add an api-key only when LAN-exposed
                # (loopback = local-only, no key — llama-server's own UI stays open).
                lan = not is_loopback(config.host)
                plumbing = plumbing_flags(str(m.path), host=config.host, port=port,
                                          alias=m.path.stem,
                                          api_key=config.inference_key if lan else None)
                return ["llama-server", *plumbing, *tuning], str(m.path)
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

    @app.middleware("http")
    async def _no_cache_ui_assets(request: Request, call_next):
        # UI assets must revalidate so a reinstall's new JS/CSS is picked up
        # without a manual hard-reload (StaticFiles alone allows heuristic caching).
        response = await call_next(request)
        if not request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/servers", dependencies=[Depends(auth)])
    def list_servers():
        return {"servers": [_instance_dict(s) for s in app.state.manager.status()]}

    def _find(name: str):
        for m in discover_models(config.roots):
            if m.matches(name):
                return m
        return None

    @app.get("/api/models", dependencies=[Depends(auth)])
    def list_models():
        overrides = load_homepages()
        return {"models": [_model_dict(m, overrides.get(m.path.name))
                           for m in discover_models(config.roots)]}

    @app.put("/api/models/{name}/homepage", dependencies=[Depends(auth)])
    def set_homepage(name: str, body: HomepageRequest):
        # User-set model-card link for GGUFs whose metadata can't derive the repo.
        # Token-gated; resolved by filename. Blank url clears it.
        model = _find(name)
        if model is None:
            raise HTTPException(status_code=404, detail="model not found")
        state_set_homepage(model.path.name, body.url)
        return {"name": model.path.name,
                "homepage_override": load_homepages().get(model.path.name)}

    @app.get("/api/models/{name}/recommend", dependencies=[Depends(auth)])
    def recommend_for(name: str):
        model = _find(name)
        if model is None:
            raise HTTPException(status_code=404, detail="model not found")
        metadata = read_gguf(model.shards[0]).metadata
        # Return only the editable TUNING flags. The daemon owns and derives the
        # plumbing (-m/--host/--port/--alias/--api-key) from its own config at launch,
        # so it's never exposed for editing — a user (or a stale default) can't make
        # the model bind the wrong host.
        cfg = recommend_config(model, metadata, detect_hardware(),
                               supported=get_supported_flags() or None)
        return {"model": model.path.name, "context": cfg.context,
                "cache_type": cfg.cache_type, "flags": cfg.tuning_flags,
                "warnings": cfg.warnings,
                "fit": {"level": cfg.fit.level, "fits": cfg.fit.fits,
                        "message": cfg.fit.message}}

    @app.post("/api/servers", dependencies=[Depends(auth)])
    def start_server(body: StartServerRequest):
        port = body.port or 8080
        # body.flags = the user's edited TUNING flags (None = use recommended); the
        # daemon always supplies the plumbing, so the host/key can't be overridden.
        command, model_path = app.state.command_builder(body.model, port, body.flags)
        try:
            with app.state.lock:
                inst = app.state.manager.start(command, port=port, model_path=model_path)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        return _instance_dict(inst)

    @app.delete("/api/servers/{port}", dependencies=[Depends(auth)])
    def stop_server(port: int):
        with app.state.lock:
            ok = app.state.manager.stop(port)
        return {"stopped": ok, "port": port}

    @app.post("/api/servers/switch", dependencies=[Depends(auth)])
    def switch_server(body: SwitchServerRequest):
        port = body.port or 8080
        # body.flags = edited TUNING flags (None = recommended); plumbing is daemon-owned.
        command, model_path = app.state.command_builder(body.model, port, body.flags)
        try:
            with app.state.lock:
                inst = app.state.manager.switch(command, port=port, model_path=model_path)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
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
    def bind_hermes(body: BindRequest):
        # Binds the SERVER HOST's ~/.hermes (the daemon runs as the owning user) to
        # the running model at 127.0.0.1. Token-gated; usable over the LAN so the
        # operator can bind the server's Hermes from another machine (it configures
        # the SERVER, not the caller — a remote caller's own machine is bound via
        # the `lmm bind` command instead).
        running = app.state.manager.status()
        inst = running[0] if running else None
        if inst is None:
            raise HTTPException(status_code=409, detail="no server is running to bind to")
        model_id = Path(inst.model_path).stem
        base_url = f"http://127.0.0.1:{inst.port}/v1"
        lan = not is_loopback(config.host)
        api_key = config.inference_key if lan else "local"  # keyless server ignores it
        # Pin Hermes's context window to the server's actual -c (Hermes can't read
        # it from /v1/models). Pulled from the launch command — no extra probe.
        context_length = context_length_from_command(inst.command)
        config_path = (Path(body.hermes_config) if body.hermes_config
                       else Path.home() / ".hermes" / "config.yaml")
        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"Hermes config not found: {config_path}")
        info = hermes_bind(config_path, base_url=base_url, model_id=model_id,
                           provider_name=body.provider_name, api_key=api_key,
                           context_length=context_length)
        return {"bound": True, **info}

    @app.get("/api/hermes-profiles", dependencies=[Depends(auth)])
    def hermes_profiles():
        # Token-gated; lists the SERVER host's Hermes profiles (names + paths only,
        # no config contents) for both the in-app bind and the remote command.
        return {"profiles": hermes_list_profiles()}

    @app.get("/api/bind-status", dependencies=[Depends(auth)])
    def bind_status():
        # Which of the server host's Hermes profiles point at the running model
        # server? Token-gated; readable over the LAN so the badge works there too.
        # Connection is base_url/port-based (not model.default) — see hermes
        # helpers — so it stays accurate across model switches.
        running = app.state.manager.list()  # records: no health probe (fast)
        inst = running[0] if running else None
        if inst is None:
            return {"bound": False, "profiles": [], "model_id": None}
        profiles = hermes_profiles_bound_to(inst.port)
        return {"bound": bool(profiles), "profiles": profiles,
                "model_id": Path(inst.model_path).stem}

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
        # status() does blocking /health probes — run it OFF the event loop
        # (via a thread) so it can't stall every other request for ~seconds.
        servers_now = await asyncio.to_thread(app.state.manager.status)
        for inst in servers_now:
            path = log_dir / f"server-{inst.port}.log"
            for line in read_log_tail(path, max_lines=200):
                await ws.send_json({"type": "log", "port": inst.port, "line": line})
            offsets[inst.port] = path.stat().st_size if path.exists() else 0
        await ws.send_json({"type": "status",
                            "servers": [_instance_dict(s) for s in servers_now]})
        try:
            while True:
                await asyncio.sleep(1.0)
                servers_now = await asyncio.to_thread(app.state.manager.status)
                for inst in servers_now:
                    path = log_dir / f"server-{inst.port}.log"
                    prev = offsets.get(inst.port, 0)
                    # log truncated/rotated (e.g. on switch) → restart from the top
                    if path.exists() and path.stat().st_size < prev:
                        prev = 0
                    lines, offsets[inst.port] = tail_new_lines(path, prev)
                    for line in lines:
                        await ws.send_json({"type": "log", "port": inst.port, "line": line})
                await ws.send_json({"type": "status",
                                    "servers": [_instance_dict(s) for s in servers_now]})
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
