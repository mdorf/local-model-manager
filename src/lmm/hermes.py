"""Bind/unbind a Hermes Agent config.yaml to a locally-served model."""

from __future__ import annotations

import io
from pathlib import Path

from ruamel.yaml import YAML

DEFAULT_HERMES_CONFIG = Path.home() / ".hermes" / "config.yaml"
_BACKUP_SUFFIX = ".lmm-prev"


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    return y


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


def bind(config_path: str | Path, *, base_url: str, model_id: str,
         provider_name: str = "local", api_key: str = "local") -> dict:
    """Point a Hermes config at a local server. Preserves comments/other keys;
    writes a <config>.lmm-prev backup before editing.
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
    data["model"] = model

    buf = io.StringIO()
    yaml.dump(data, buf)
    config_path.write_text(buf.getvalue())
    return {"provider": f"custom:{provider_name}", "model": model_id,
            "base_url": base_url, "config": str(config_path)}


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
        for key in ("provider", "default", "base_url"):
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
