"""Local TCP port helpers."""

from __future__ import annotations

import socket
import subprocess


def listening_pid(port: int) -> int | None:
    """PID of the process listening on `port` (via lsof), or None if none/unknown.

    Lets the daemon adopt an externally-started server with a real pid so an
    explicit stop/switch can terminate it. Try absolute paths first — lsof is in
    /usr/sbin on macOS (NOT /usr/bin), which a launchd daemon's minimal PATH may
    not include — then fall back to PATH lookup.
    """
    for exe in ("/usr/sbin/lsof", "/usr/bin/lsof", "lsof"):
        try:
            out = subprocess.run([exe, "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                                 capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        pids = [int(x) for x in out.stdout.split() if x.strip().isdigit()]
        return pids[0] if pids else None
    return None


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def pick_free_port(start: int = 8080, host: str = "127.0.0.1", limit: int = 200) -> int:
    """Return the first free port at or after `start` (scans up to `limit` ports)."""
    for port in range(start, min(start + limit, 65536)):
        if not is_port_in_use(port, host):
            return port
    raise RuntimeError(f"no free port found in [{start}, {start + limit})")
