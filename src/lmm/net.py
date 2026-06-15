"""Small network helpers shared across the daemon, CLI, and recommender."""

from __future__ import annotations

_LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}


def is_loopback(host: str | None) -> bool:
    """True if `host` is loopback-only (or unset) — i.e. not exposed to the LAN.

    Used to decide when the inference server needs an `--api-key`: a loopback
    server is reachable only by the host operator (trusted), so no key; a
    LAN-bound server (e.g. 0.0.0.0) must be key-gated.
    """
    return (host or "").strip() in _LOOPBACK
