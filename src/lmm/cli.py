"""Command-line entrypoint for local-model-manager."""

from __future__ import annotations

import argparse
import getpass
import json
import logging
import os
import pwd
import secrets
import subprocess
import sys
from pathlib import Path

from lmm import daemon_client, deploy
from lmm.daemonconfig import load_or_create_config, rotate_token
from lmm.discovery import discover_models
from lmm.gguf import read_gguf
from lmm.hardware import detect_hardware
from lmm.hermes import (
    bind as hermes_bind,
    profile_config_path as hermes_profile_config_path,
    unbind as hermes_unbind,
)
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
    matches = [m for m in discover_models(args.root) if m.matches(args.model)]
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
        if m.matches(name):
            return m
    return None


def cmd_serve(args: argparse.Namespace) -> int:
    d = daemon_client.daemon_available()
    if d:
        try:
            out = daemon_client.start(d["base"], d["token"], args.model, args.port) or {}
        except daemon_client.DaemonError as e:
            print(e)
            return 1
        print(f"Status: {out.get('status')}  (port {out.get('port')})")
        return 0 if out.get("status") == "ready" else 1
    # ---- existing direct-mode code unchanged below ----
    model = _find_model(args.root, args.model)
    if model is None:
        print(f"Model not found: {args.model}")
        return 1
    metadata = read_gguf(model.shards[0]).metadata
    # dev `serve` is loopback / current-user — no --api-key (host defaults to
    # 127.0.0.1, so llama-server stays open locally like a hand-run server).
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
    d = daemon_client.daemon_available()
    if d:
        try:
            out = daemon_client.stop(d["base"], d["token"], args.port)
        except daemon_client.DaemonError as e:
            print(e)
            return 1
        print(f"Stopped server on port {args.port}." if out and out.get("stopped")
              else f"No server on port {args.port}.")
        return 0
    # ---- existing direct-mode code unchanged ----
    mgr = ServerManager()
    if not any(r.port == args.port for r in mgr.list()):
        print(f"No server managed on port {args.port}.")
        return 0
    ok = mgr.stop(args.port)
    print(f"Stopped server on port {args.port}." if ok else
          f"Failed to fully stop port {args.port}.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    d = daemon_client.daemon_available()
    if d:
        try:
            servers = (daemon_client.status(d["base"], d["token"]) or {}).get("servers", [])
        except daemon_client.DaemonError as e:
            print(e)
            return 1
        if not servers:
            print("No running servers.")
            return 0
        for s in servers:
            tag = " (external)" if s.get("external") else ""
            print(f"port {s['port']}: {s['status']}{tag}  pid={s.get('pid')}  {s.get('model')}")
        return 0
    # ---- existing direct-mode code unchanged ----
    instances = ServerManager().status()
    if not instances:
        print("No running servers.")
        return 0
    for s in instances:
        tag = " (external)" if s.external else ""
        print(f"port {s.port}: {s.status}{tag}  pid={s.pid}  {Path(s.model_path).name}")
    return 0


def cmd_switch(args: argparse.Namespace) -> int:
    d = daemon_client.daemon_available()
    if d:
        try:
            out = daemon_client.switch(d["base"], d["token"], args.model, args.port) or {}
        except daemon_client.DaemonError as e:
            print(e)
            return 1
        print(f"Status: {out.get('status')}  (port {out.get('port')})")
        return 0 if out.get("status") == "ready" else 1
    # ---- existing direct-mode code unchanged ----
    model = _find_model(args.root, args.model)
    if model is None:
        print(f"Model not found: {args.model}")
        return 1
    metadata = read_gguf(model.shards[0]).metadata
    # dev `switch` is loopback / current-user — no --api-key (see cmd_serve)
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
    if getattr(args, "rotate", False):
        print(rotate_token())
        print("Token rotated. Restart the daemon for it to take effect "
              "(sudo \"$(command -v lmm)\" service restart), then re-enter the new "
              "token on every client.", file=sys.stderr)
        return 0
    print(load_or_create_config().token)
    return 0


def _detect_served_model(host: str, port: int) -> str | None:
    """Ask the running server for the model id it advertises (its --alias)."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=5) as r:
            data = json.loads(r.read() or "{}")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    models = data.get("data") or []
    return models[0].get("id") if models else None


def _resolve_hermes_config(args: argparse.Namespace) -> Path:
    """An explicit --hermes-config path wins; otherwise resolve --profile by name
    (portable across machines). Defaults to the active ~/.hermes/config.yaml."""
    if args.hermes_config:
        return Path(args.hermes_config)
    return hermes_profile_config_path(getattr(args, "profile", None))


def cmd_bind(args: argparse.Namespace) -> int:
    config_path = _resolve_hermes_config(args)
    if not config_path.exists():
        print(f"Hermes config not found: {config_path}")
        return 1
    if args.model:
        model_id = Path(args.model).stem
    else:
        model_id = _detect_served_model(args.host, args.port)
        if not model_id:
            print(f"No running model detected on {args.host}:{args.port} — "
                  "start one first, or pass the model name explicitly.")
            return 1
    base_url = f"http://{args.host}:{args.port}/v1"
    api_key = args.api_key or load_or_create_config().inference_key
    info = hermes_bind(config_path, base_url=base_url, model_id=model_id,
                       provider_name=args.provider_name, api_key=api_key)
    print(f"Bound {config_path} -> {info['provider']} / {info['model']} @ {info['base_url']}")
    print("note: reasoning models (e.g. Qwen3.6) need a generous max_tokens — "
          "set it in your Hermes client if replies come back empty.")
    revert = (f"--profile {args.profile}" if getattr(args, "profile", None) and not args.hermes_config
              else f"--hermes-config {config_path}")
    print(f"revert with: lmm unbind {revert}")
    return 0


def cmd_unbind(args: argparse.Namespace) -> int:
    config_path = _resolve_hermes_config(args)
    if hermes_unbind(config_path):
        print(f"Reverted {config_path} from its pre-bind backup.")
    else:
        print(f"No pre-bind backup found next to {config_path}; nothing to revert.")
    return 0


SHARED_DIR = "/Users/Shared/local-model-manager"
_DAEMON_PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _install_user(args: argparse.Namespace) -> str:
    """The user the daemon runs as: --user, else the sudo invoker, else current."""
    return args.user or os.environ.get("SUDO_USER") or getpass.getuser()


def _resolve_project_dir(args: argparse.Namespace) -> str | None:
    """Directory to `uv pip install` the package from (must hold pyproject.toml).
    Explicit --project-dir, else the editable-dev src layout
    (<clone>/src/lmm/cli.py -> <clone>). A uv-tool-installed cli.py lives in
    site-packages, so the src-layout guess does NOT resolve there -- that's why
    --project-dir is required for an installed CLI (both install and reinstall)."""
    if args.project_dir:
        return args.project_dir
    cand = Path(__file__).resolve().parents[2]
    return str(cand) if (cand / "pyproject.toml").exists() else None


def _write_daemon_config(path: Path, *, host: str, port: int, models_dir: str) -> None:
    """Write daemon.json for an install. Preserves the token + inference_key across
    re-installs (so clients keep working), but always refreshes host/port/roots to
    match THIS install — otherwise `--reinstall --host 0.0.0.0` would update the
    launchd plist while daemon.json (which decides the model's bind host + whether
    an inference api-key is set) stayed stale, so models would launch on loopback."""
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError):
            existing = {}
    path.write_text(json.dumps({
        "host": host, "port": port,
        "token": existing.get("token") or secrets.token_hex(24),
        "inference_key": existing.get("inference_key") or secrets.token_hex(24),
        "roots": [models_dir]}, indent=2))


def _install_child_path(user: str) -> str:
    """PATH for install-time subprocesses (uv, llama-server). sudo strips the
    caller's PATH, so prepend the invoking user's bin dirs — lets
    `sudo "$(command -v lmm)" install` work without a manual `env PATH=...`."""
    return ":".join([f"/Users/{user}/.local/bin", "/opt/homebrew/bin",
                     "/usr/local/bin", os.environ.get("PATH", "")])


def cmd_install(args: argparse.Namespace) -> int:
    user = _install_user(args)
    project_dir = _resolve_project_dir(args)
    if not project_dir or not (Path(project_dir) / "pyproject.toml").exists():
        print("could not locate the local-model-manager source to install from.\n"
              "Re-run with --project-dir pointing at your clone, e.g.:\n"
              '  sudo env "PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH" \\\n'
              '    lmm install --project-dir "$(pwd)" --models-dir /path/to/models')
        return 1
    exec_path = deploy.shared_venv_exec(SHARED_DIR)
    # Run as the owning user: set HOME so the daemon resolves ~/.hermes (for
    # one-click bind) and its state dir correctly.
    env = {"LMM_STATE_DIR": SHARED_DIR, "HOME": f"/Users/{user}", "PATH": _DAEMON_PATH}
    steps = deploy.install_steps(user=user, host=args.host, port=args.port,
                                 shared_dir=SHARED_DIR, project_dir=project_dir,
                                 reinstall=args.reinstall)
    plist_xml = deploy.launchd_plist(exec_path=exec_path, host=args.host,
                                     port=args.port, user=user, env=env)
    if args.dry_run:
        print(f"# daemon will run as user: {user}")
        print(f"# would write {deploy.plist_install_path()} :")
        print(plist_xml)
        print(f"# would write {SHARED_DIR}/daemon.json (fresh token + inference_key)")
        print("# would run (as root):")
        for s in steps:
            print(f"  {s}")
        return 0
    if os.geteuid() != 0:
        print("install must run as root — re-run: sudo lmm install "
              "(or preview with: lmm install --dry-run)")
        return 1
    if user == "root":
        print("refusing to run the daemon as root — run `sudo lmm install` as a "
              "normal user, or pass --user <name>.")
        return 1
    try:
        pwd.getpwnam(user)
    except KeyError:
        print(f"target user '{user}' does not exist — pass an existing --user.")
        return 1

    existing = deploy.existing_install_artifacts(shared_dir=SHARED_DIR)
    if existing and not args.reinstall:
        print(f"{deploy.LABEL} appears already installed: {', '.join(existing)}.")
        print("Run `sudo lmm uninstall` first, or re-run with --reinstall to "
              "replace it in place.")
        return 1

    child_env = {**os.environ, "PATH": _install_child_path(user)}

    def _run(cmds, *, critical):
        for c in cmds:
            subprocess.run(c, shell=True, check=critical, env=child_env)

    # Ordered phases: plist before bootstrap; daemon.json (in the shared dir,
    # later chowned to the user by shared_venv_steps) before the daemon starts.
    if args.reinstall:
        subprocess.run(f"launchctl bootout system {deploy.plist_install_path()}",
                       shell=True, check=False)
    Path(deploy.plist_install_path()).write_text(plist_xml)
    _run(deploy.shared_setup_steps(user=user, shared_dir=SHARED_DIR), critical=True)
    daemon_json = Path(SHARED_DIR) / "daemon.json"
    _write_daemon_config(daemon_json, host=args.host, port=args.port,
                         models_dir=args.models_dir)
    _run(deploy.shared_venv_steps(shared_dir=SHARED_DIR, project_dir=project_dir,
                                  user=user, clear=args.reinstall), critical=True)
    _run(deploy.plist_steps(user=user), critical=True)
    _run(deploy.firewall_steps(exec_path=exec_path), critical=False)
    print(f"Installed {deploy.LABEL} (runs as user={user}). "
          f"Verify: sudo launchctl print system/{deploy.LABEL}")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    steps = deploy.uninstall_steps(shared_dir=SHARED_DIR)
    if args.dry_run:
        print("# would run (as root):")
        for s in steps:
            print(f"  {s}")
        return 0
    if os.geteuid() != 0:
        print("uninstall must run as root — re-run: sudo lmm uninstall")
        return 1
    for s in steps:
        subprocess.run(s, shell=True, check=False)
    print(f"Uninstalled {deploy.LABEL}.")
    return 0


def _probe_health(host: str, port: int) -> bool:
    """Is the control daemon answering on /api/health? (open endpoint, no token)."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=3) as r:
            return json.loads(r.read() or "{}").get("status") == "ok"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _service_status() -> int:
    if not Path(deploy.plist_install_path()).exists():
        print(f"{deploy.LABEL}: not installed (no LaunchDaemon).")
        print("Run it in the foreground with `uv run lmm daemon`, or install the "
              "always-on service with `sudo lmm install`.")
        return 0
    # The installer records host/port in the shared daemon.json (world-readable).
    host, port = "127.0.0.1", 8770
    dj = Path(SHARED_DIR) / "daemon.json"
    if dj.exists():
        try:
            cfg = json.loads(dj.read_text())
            host, port = cfg.get("host") or host, int(cfg.get("port") or port)
        except (ValueError, OSError):
            pass
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    running = _probe_health(probe_host, port)
    print(f"{deploy.LABEL}: installed ({deploy.plist_install_path()})")
    print(f"  control daemon: {'running' if running else 'NOT responding'} "
          f"(http://{probe_host}:{port})")
    if not running:
        print("  start it with: sudo lmm service start")
    return 0


def cmd_service(args: argparse.Namespace) -> int:
    if args.action == "status":
        return _service_status()  # read-only, no privileges
    steps = {
        "stop": deploy.service_stop_steps(),
        "start": deploy.service_start_steps(),
        "restart": deploy.service_restart_steps(),
    }[args.action]
    if args.dry_run:
        print("# would run (as root):")
        for s in steps:
            print(f"  {s}")
        return 0
    if os.geteuid() != 0:
        print(f"`lmm service {args.action}` must run as root — re-run: "
              f"sudo lmm service {args.action} (or preview with --dry-run)")
        return 1
    if not Path(deploy.plist_install_path()).exists():
        print(f"{deploy.LABEL} is not installed — run `sudo lmm install` first.")
        return 1
    for s in steps:
        subprocess.run(s, shell=True, check=False)
    print(f"Service {args.action}: done for {deploy.LABEL}.")
    return 0


# v1 serves a single model on 8080; widen this when multi-model lands.
_MODEL_PORTS = [8080]


def cmd_daemon(args: argparse.Namespace) -> int:
    import uvicorn

    from lmm.api import create_app
    from lmm.server import ServerManager, autodetect_servers
    config = load_or_create_config()
    host = args.host or config.host
    port = args.port or config.port
    # The model-launch builder reads config.host to decide where to bind llama-server
    # and whether to set an inference api-key. Pin it to the host we ACTUALLY bind to
    # (e.g. --host 0.0.0.0 from the plist), so a stale daemon.json can't make models
    # launch on loopback while the daemon itself is LAN-exposed.
    config.host = host
    config.port = port
    # Detect a model server that's already running (manually launched, or one
    # that outlived a daemon restart) so the UI reflects reality on startup.
    manager = ServerManager()
    for inst in autodetect_servers(manager, config.roots, _MODEL_PORTS):
        print(f"Detected running model on :{inst.port} → {Path(inst.model_path).name} (adopted)")
    print(f"Starting daemon on http://{host}:{port}  (auth token via: lmm token)")
    uvicorn.run(create_app(config, manager=manager), host=host, port=port, log_level="info")
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
    p_token.add_argument("--rotate", action="store_true",
                         help="generate a new token (requires a daemon restart + client re-entry)")
    p_token.set_defaults(func=cmd_token)

    p_bind = sub.add_parser("bind", help="point a Hermes config at a local server")
    p_bind.add_argument("model", nargs="?", default=None,
                        help="model filename (its stem is the served id); "
                             "omit to auto-detect the running model on --port")
    p_bind.add_argument("--port", type=int, default=8080)
    p_bind.add_argument("--host", default="127.0.0.1")
    p_bind.add_argument("--provider-name", default="local")
    p_bind.add_argument("--profile", default=None,
                        help="Hermes profile name to bind (e.g. qwen-herm); "
                             "'default'/omit = the active ~/.hermes/config.yaml")
    p_bind.add_argument("--hermes-config", default=None,
                        help="explicit path to a Hermes config.yaml (overrides --profile)")
    p_bind.add_argument("--api-key", default=None,
                        help="inference key (default: this host's inference_key)")
    p_bind.set_defaults(func=cmd_bind)

    p_unbind = sub.add_parser("unbind", help="revert a Hermes config bound by lmm")
    p_unbind.add_argument("--profile", default=None,
                          help="Hermes profile name to revert (default = active config)")
    p_unbind.add_argument("--hermes-config", default=None,
                          help="explicit path (overrides --profile)")
    p_unbind.set_defaults(func=cmd_unbind)

    p_install = sub.add_parser("install", help="install the daemon as a system service (sudo)")
    p_install.add_argument("--dry-run", action="store_true", help="print steps, do nothing")
    p_install.add_argument("--user", default=None,
                           help="user to run the daemon as (default: the sudo invoker)")
    p_install.add_argument("--host", default="127.0.0.1")
    p_install.add_argument("--port", type=int, default=8770)
    p_install.add_argument("--models-dir", default="/Users/Shared/models")
    p_install.add_argument("--project-dir", default=None,
                           help="source dir to install lmm from (default: repo root)")
    p_install.add_argument("--reinstall", "--force", action="store_true",
                           help="replace an existing install in place (bootout + rebuild venv)")
    p_install.set_defaults(func=cmd_install)

    p_uninstall = sub.add_parser("uninstall", help="remove the daemon system service (sudo)")
    p_uninstall.add_argument("--dry-run", action="store_true")
    p_uninstall.set_defaults(func=cmd_uninstall)

    p_service = sub.add_parser(
        "service", help="control the installed daemon: status / stop / start / restart")
    p_service.add_argument("action", choices=["status", "stop", "start", "restart"],
                           help="status is read-only; stop/start/restart need sudo")
    p_service.add_argument("--dry-run", action="store_true",
                           help="print the privileged steps without running them")
    p_service.set_defaults(func=cmd_service)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    if getattr(args, "root", None) is None:
        args.root = [str(Path.home() / "models")]
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
