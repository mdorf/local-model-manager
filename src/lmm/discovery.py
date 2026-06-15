"""Recursively discover and classify models under root directories."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from lmm.gguf import GGUFError, read_gguf
from lmm.models import Model, classify

log = logging.getLogger("lmm.discovery")

_SHARD_RE = re.compile(r"^(?P<base>.+)-\d{5}-of-\d{5}\.gguf$")


def collapse_shards(filenames: list[str]) -> dict[str, list[str]]:
    """Group shard filenames under a single logical '<base>.gguf' key.

    Non-sharded files map to themselves. Shard lists are sorted.
    """
    groups: dict[str, list[str]] = {}
    for name in filenames:
        m = _SHARD_RE.match(name)
        key = f"{m.group('base')}.gguf" if m else name
        groups.setdefault(key, []).append(name)
    return {k: sorted(v) for k, v in groups.items()}


def _is_sidecar(path: Path) -> bool:
    """Return True if this file should be treated as a sidecar (not a main model)."""
    return path.name.startswith("mmproj") or path.suffix == ".jinja"


def _sidecars(directory: Path, main_names: set[str]) -> list[Path]:
    """Collect sidecar files in *directory*.

    A file is a sidecar if it matches the sidecar pattern (starts with 'mmproj'
    or has a '.jinja' suffix). Sidecar-pattern files are included regardless of
    whether they are also listed in main_names, because a .gguf that failed to
    parse should still appear as a sidecar for the successfully-parsed neighbor.

    Non-sidecar-pattern files that share a name with a main model are excluded.
    """
    out: list[Path] = []
    for p in sorted(directory.iterdir()):
        if not p.is_file():
            continue
        if _is_sidecar(p):
            # Always include sidecar-pattern files
            out.append(p)
        elif p.name not in main_names:
            out.append(p)
    return out


def discover_models(roots: list[str | Path]) -> list[Model]:
    models: list[Model] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            log.warning("root not found, skipping: %s", root)
            continue
        by_dir: dict[Path, list[str]] = {}
        # NOTE: rglob does not descend directory symlinks, so models reached
        # only via a symlinked directory are not discovered. Symlink-following
        # policy is intentionally deferred (see ROADMAP).
        for p in root.rglob("*.gguf"):
            # Don't treat sidecar-pattern files as candidate models at all
            if not _is_sidecar(p):
                by_dir.setdefault(p.parent, []).append(p.name)
            else:
                # Ensure the directory is known even if it only has sidecars
                by_dir.setdefault(p.parent, [])
        for directory, names in by_dir.items():
            groups = collapse_shards(names)
            main_names = set(groups.keys()) | {n for v in groups.values() for n in v}
            for logical_name, shard_names in groups.items():
                shard_paths = [directory / n for n in shard_names]
                first = shard_paths[0]
                try:
                    info = read_gguf(first)
                except (GGUFError, OSError) as e:
                    log.warning("skipping unreadable model %s: %s", first, e)
                    continue
                models.append(classify(
                    info, first,
                    shards=shard_paths,
                    sidecars=_sidecars(directory, main_names),
                ))
    return models
