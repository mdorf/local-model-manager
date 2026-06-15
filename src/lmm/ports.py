"""Local TCP port helpers."""

from __future__ import annotations

import socket


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
