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
    """Restore the config from the <config>.lmm-prev backup. Returns False if
    no backup exists (nothing to revert)."""
    config_path = Path(config_path)
    backup = Path(str(config_path) + _BACKUP_SUFFIX)
    if not backup.exists():
        return False
    config_path.write_text(backup.read_text())
    backup.unlink()  # revert is complete — leave no .lmm-prev residue behind
    return True
