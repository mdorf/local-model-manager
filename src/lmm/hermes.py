"""Bind/unbind a Hermes Agent config.yaml to a locally-served model."""

from __future__ import annotations

import io
from pathlib import Path
from urllib.parse import urlparse

from ruamel.yaml import YAML

DEFAULT_HERMES_CONFIG = Path.home() / ".hermes" / "config.yaml"
_BACKUP_SUFFIX = ".lmm-prev"
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    return y


def profile_config_path(name: str | None, hermes_dir: str | Path | None = None) -> Path:
    """Resolve a Hermes profile NAME to its config path. ``None``/``""``/``default``
    → the root ``~/.hermes/config.yaml``; any other name →
    ``~/.hermes/profiles/<name>/config.yaml``. Name-based (not absolute path) so the
    same command is portable across machines (each resolves its own ~/.hermes)."""
    base = Path(hermes_dir) if hermes_dir else Path.home() / ".hermes"
    if name in (None, "", "default"):
        return base / "config.yaml"
    return base / "profiles" / name / "config.yaml"


def list_profiles(hermes_dir: str | Path | None = None) -> list[dict]:
    """List the operator's Hermes profiles so the bind UI can target a specific
    one (not just the active config). Returns the default profile (the root
    ``~/.hermes/config.yaml``) plus each ``~/.hermes/profiles/<name>/config.yaml``.
    Each entry is ``{"name", "path"}``. Only paths/names are returned — no config
    contents are read, so no secrets are exposed."""
    base = Path(hermes_dir) if hermes_dir else Path.home() / ".hermes"
    profiles: list[dict] = []
    root = base / "config.yaml"
    if root.is_file():
        profiles.append({"name": "default", "path": str(root)})
    pdir = base / "profiles"
    if pdir.is_dir():
        for d in sorted(pdir.iterdir()):
            cfg = d / "config.yaml"
            if cfg.is_file():
                profiles.append({"name": d.name, "path": str(cfg)})
    return profiles


def _points_at_local_server(data: dict, port: int) -> bool:
    """Does this Hermes config's ACTIVE provider point at the local model server
    on `port`? Resolves model.provider=custom:<name> → providers[<name>].base_url
    (Hermes's real lookup), falling back to model.base_url. 'Connected' = host is
    loopback + port matches — NOT model.default (llama-server ignores the request
    model id, so a profile keeps working across switches even if its label is
    stale)."""
    model = data.get("model") or {}
    base = model.get("base_url")
    prov = str(model.get("provider") or "")
    if prov.startswith("custom:"):
        name = prov.split(":", 1)[1]
        entry = (data.get("providers") or {}).get(name) or {}
        base = entry.get("base_url") or base
    if not base:
        return False
    try:
        u = urlparse(str(base))
    except ValueError:
        return False
    return u.port == port and u.hostname in _LOCAL_HOSTS


def profiles_bound_to(port: int, hermes_dir: str | Path | None = None) -> list[str]:
    """Names of the operator's Hermes profiles currently pointed at the local model
    server on `port` (see _points_at_local_server). Used by the UI to say WHICH
    profiles are connected, not just whether one is."""
    out: list[str] = []
    for prof in list_profiles(hermes_dir):
        try:
            data = _yaml().load(Path(prof["path"]).read_text()) or {}
        except Exception:
            continue
        if _points_at_local_server(data, port):
            out.append(prof["name"])
    return out


def bind(config_path: str | Path, *, base_url: str, model_id: str,
         provider_name: str = "local", api_key: str = "local",
         context_length: int | None = None) -> dict:
    """Point a Hermes config at a local server. Preserves comments/other keys;
    writes a <config>.lmm-prev backup before editing.

    When ``context_length`` is given it's written as ``model.context_length`` —
    Hermes can't reliably read a local llama.cpp server's runtime ``-c`` (it
    name-guesses or falls back to the trained max), so pinning the real window
    makes Hermes compress/size at the correct threshold.
    """
    config_path = Path(config_path)
    original = config_path.read_text()
    backup = Path(str(config_path) + _BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(original)   # preserve the pristine pre-lmm config across re-binds

    yaml = _yaml()
    data = yaml.load(original) or {}

    existing = data.get("providers")
    providers = dict(existing) if isinstance(existing, dict) else {}
    providers[provider_name] = {"base_url": base_url, "api_key": api_key,
                                "default_model": model_id}
    data["providers"] = providers

    model = data.get("model")
    if not isinstance(model, dict):
        model = {}
    model["provider"] = f"custom:{provider_name}"
    model["default"] = model_id
    model["base_url"] = base_url
    if isinstance(context_length, int) and context_length > 0:
        model["context_length"] = context_length
    data["model"] = model

    buf = io.StringIO()
    yaml.dump(data, buf)
    config_path.write_text(buf.getvalue())
    return {"provider": f"custom:{provider_name}", "model": model_id,
            "base_url": base_url, "context_length": context_length,
            "config": str(config_path)}


def unbind(config_path: str | Path) -> bool:
    """Reverse a bind, preserving any *unrelated* edits made since. Restores only
    the keys bind changes — model.provider/default/base_url and the provider entry
    it added — to their pre-bind values (read from <config>.lmm-prev); every other
    key in the current config is left untouched. A wholesale restore would discard
    edits made after binding. Returns False if no backup exists (nothing to revert).
    """
    config_path = Path(config_path)
    backup = Path(str(config_path) + _BACKUP_SUFFIX)
    if not backup.exists():
        return False

    yaml = _yaml()
    current = yaml.load(config_path.read_text()) or {}
    prev = yaml.load(backup.read_text()) or {}
    prev_model = prev.get("model") if isinstance(prev.get("model"), dict) else {}
    prev_providers = prev.get("providers") if isinstance(prev.get("providers"), dict) else {}

    model = current.get("model")
    if isinstance(model, dict):
        # the provider bind registered (model.provider == "custom:<name>"); fall
        # back to bind's default name if it's been changed away from custom:*.
        ref = model.get("provider")
        name = (ref.split(":", 1)[1]
                if isinstance(ref, str) and ref.startswith("custom:") else "local")
        for key in ("provider", "default", "base_url", "context_length"):
            if key in prev_model:
                model[key] = prev_model[key]
            else:
                model.pop(key, None)
        if not model and "model" not in prev:
            current.pop("model", None)

        providers = current.get("providers")
        if isinstance(providers, dict):
            if name in prev_providers:
                providers[name] = prev_providers[name]  # restore a pre-existing entry
            else:
                providers.pop(name, None)               # drop the one bind added
            if not providers and "providers" not in prev:
                current.pop("providers", None)

    buf = io.StringIO()
    yaml.dump(current, buf)
    config_path.write_text(buf.getvalue())
    backup.unlink()  # revert is complete — leave no .lmm-prev residue behind
    return True
