"""Command-line entrypoint for local-model-manager."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from lmm.daemonconfig import load_or_create_config
from lmm.discovery import discover_models
from lmm.gguf import read_gguf
from lmm.hardware import detect_hardware
from lmm.llama import get_supported_flags
from lmm.recommend import recommend_config
from lmm.server import ServerManager


def cmd_models(args: argparse.Namespace) -> int:
    models = discover_models(args.root)
    if not models:
        print("No models found.")
        return 0
    for m in sorted(models, key=lambda x: (x.family, x.size_label, x.path.name)):
        flags = " ".join(f for f in ["MTP" if m.has_mtp else ""] if f)
        ctx = f"{m.context_length // 1024}K" if m.context_length is not None else "?"
        print(f"{m.path.name}  [{m.arch} {m.size_label} {m.quant} ctx={ctx}] {flags}".rstrip())
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    matches = [m for m in discover_models(args.root)
               if m.path.name == args.model or str(m.path) == args.model]
    if not matches:
        print(f"Model not found: {args.model}")
        return 1
    model = matches[0]
    metadata = read_gguf(model.shards[0]).metadata
    hardware = detect_hardware()
    supported = get_supported_flags() or None
    cfg = recommend_config(model, metadata, hardware, supported=supported,
                           alias=model.path.stem)
    print(f"Recommended for {model.path.name} "
          f"(ctx {cfg.context}, cache {cfg.cache_type}):")
    print("llama-server " + " ".join(cfg.flags))
    print(f"\nFit: {cfg.fit.level} — {cfg.fit.message}")
    for w in cfg.warnings:
        print(f"warning: {w}")
    return 0


def _find_model(root, name):
    for m in discover_models(root):
        if m.path.name == name or str(m.path) == name:
            return m
    return None


def cmd_serve(args: argparse.Namespace) -> int:
    model = _find_model(args.root, args.model)
    if model is None:
        print(f"Model not found: {args.model}")
        return 1
    metadata = read_gguf(model.shards[0]).metadata
    cfg = recommend_config(model, metadata, detect_hardware(),
                           supported=get_supported_flags() or None,
                           port=args.port, alias=model.path.stem)
    for w in cfg.warnings:
        print(f"warning: {w}")
    if cfg.fit.level == "wont_load":
        print(f"Refusing to start — {cfg.fit.message}")
        return 1
    mgr = ServerManager()
    command = ["llama-server", *cfg.flags]
    print(f"Starting {model.path.name} on port {args.port} (ctx {cfg.context})...")
    inst = mgr.start(command, port=args.port, model_path=str(model.path))
    print(f"Status: {inst.status}  ({inst.base_url})")
    return 0 if inst.status == "ready" else 1


def cmd_stop(args: argparse.Namespace) -> int:
    mgr = ServerManager()
    if not any(r.port == args.port for r in mgr.list()):
        print(f"No server managed on port {args.port}.")
        return 0
    ok = mgr.stop(args.port)
    print(f"Stopped server on port {args.port}." if ok else
          f"Failed to fully stop port {args.port}.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    instances = ServerManager().status()
    if not instances:
        print("No running servers.")
        return 0
    for s in instances:
        tag = " (external)" if s.external else ""
        print(f"port {s.port}: {s.status}{tag}  pid={s.pid}  {Path(s.model_path).name}")
    return 0


def cmd_switch(args: argparse.Namespace) -> int:
    model = _find_model(args.root, args.model)
    if model is None:
        print(f"Model not found: {args.model}")
        return 1
    metadata = read_gguf(model.shards[0]).metadata
    cfg = recommend_config(model, metadata, detect_hardware(),
                           supported=get_supported_flags() or None,
                           port=args.port, alias=model.path.stem)
    for w in cfg.warnings:
        print(f"warning: {w}")
    mgr = ServerManager()
    command = ["llama-server", *cfg.flags]
    print(f"Switching to {model.path.name} on port {args.port}...")
    inst = mgr.switch(command, port=args.port, model_path=str(model.path))
    print(f"Status: {inst.status}  ({inst.base_url})")
    return 0 if inst.status == "ready" else 1


def cmd_token(args: argparse.Namespace) -> int:
    print(load_or_create_config().token)
    return 0


def cmd_daemon(args: argparse.Namespace) -> int:
    import uvicorn

    from lmm.api import create_app
    config = load_or_create_config()
    host = args.host or config.host
    port = args.port or config.port
    print(f"Starting daemon on http://{host}:{port}  (auth token via: lmm token)")
    uvicorn.run(create_app(config), host=host, port=port, log_level="info")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lmm", description="local-model-manager")
    sub = parser.add_subparsers(dest="command", required=True)
    p_models = sub.add_parser("models", help="list discovered local models")
    p_models.add_argument("--root", action="append", default=None,
                          help="model root dir (repeatable); defaults to ~/models")
    p_models.set_defaults(func=cmd_models)
    p_rec = sub.add_parser("recommend", help="recommend a llama-server config for a model")
    p_rec.add_argument("model", help="model filename (or full path) to recommend for")
    p_rec.add_argument("--root", action="append", default=None,
                       help="model root dir (repeatable); defaults to ~/models")
    p_rec.set_defaults(func=cmd_recommend)

    p_serve = sub.add_parser("serve", help="start a llama-server for a model")
    p_serve.add_argument("model", help="model filename (or full path)")
    p_serve.add_argument("--root", action="append", default=None,
                         help="model root dir (repeatable); defaults to ~/models")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.set_defaults(func=cmd_serve)

    p_stop = sub.add_parser("stop", help="stop a managed llama-server")
    p_stop.add_argument("--port", type=int, default=8080)
    p_stop.set_defaults(func=cmd_stop)

    p_status = sub.add_parser("status", help="show managed servers")
    p_status.set_defaults(func=cmd_status)

    p_switch = sub.add_parser("switch", help="switch the running model")
    p_switch.add_argument("model", help="model filename (or full path)")
    p_switch.add_argument("--root", action="append", default=None,
                          help="model root dir (repeatable); defaults to ~/models")
    p_switch.add_argument("--port", type=int, default=8080)
    p_switch.set_defaults(func=cmd_switch)

    p_daemon = sub.add_parser("daemon", help="run the control-plane HTTP daemon")
    p_daemon.add_argument("--host", default=None, help="bind host (default 127.0.0.1)")
    p_daemon.add_argument("--port", type=int, default=None, help="bind port (default 8770)")
    p_daemon.set_defaults(func=cmd_daemon)

    p_token = sub.add_parser("token", help="print the daemon auth token")
    p_token.set_defaults(func=cmd_token)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if getattr(args, "root", None) is None:
        args.root = [str(Path.home() / "models")]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
