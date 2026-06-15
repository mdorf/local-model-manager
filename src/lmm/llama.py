"""Locate the installed llama-server and parse its supported flags."""

from __future__ import annotations

import re
import shutil
import subprocess

# Matches CLI flag tokens like -ngl, --n-gpu-layers, --cache-type-k. A flag is a
# leading '-' or '--' followed by a letter, then letters/digits/hyphens.
_FLAG_RE = re.compile(r"(?<![\w-])(--?[A-Za-z][A-Za-z0-9-]*)")


def find_llama_server() -> str | None:
    return shutil.which("llama-server")


def supported_flags(help_text: str) -> set[str]:
    """Parse the set of flag tokens from `llama-server --help` output."""
    return {m.group(1) for m in _FLAG_RE.finditer(help_text)}


def get_supported_flags(binary: str | None = None) -> set[str]:
    """Run `<binary> --help` and parse its flags. Empty set if unavailable."""
    binary = binary or find_llama_server()
    if not binary:
        return set()
    try:
        out = subprocess.run([binary, "--help"], capture_output=True,
                             text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return set()
    return supported_flags(out.stdout + out.stderr)
