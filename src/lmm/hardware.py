"""Detect the host's memory, CPU, and accelerator characteristics."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

USABLE_FRACTION = 0.85  # leave headroom for OS/other apps; live free-RAM is deferred


@dataclass
class HardwareInfo:
    total_ram_bytes: int
    logical_cores: int
    perf_cores: int
    platform: str
    has_metal: bool
    usable_fraction: float = USABLE_FRACTION

    @property
    def usable_ram_bytes(self) -> int:
        return int(self.total_ram_bytes * self.usable_fraction)


def _sysctl_int(key: str) -> int | None:
    # Use the absolute path first: sysctl lives in /usr/sbin, which isn't on a
    # launchd daemon's minimal PATH — relying on PATH made RAM read as 0 there.
    for exe in ("/usr/sbin/sysctl", "sysctl"):
        try:
            out = subprocess.run([exe, "-n", key], capture_output=True,
                                 text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            continue
        out_str = out.stdout.strip()
        if out_str.isdigit():
            return int(out_str)
    return None


def detect_hardware() -> HardwareInfo:
    system = platform.system()
    if system == "Darwin":
        total = _sysctl_int("hw.memsize") or 0
        logical = _sysctl_int("hw.logicalcpu") or os.cpu_count() or 1
        perf = _sysctl_int("hw.perflevel0.logicalcpu") or logical
        return HardwareInfo(total_ram_bytes=total, logical_cores=logical,
                            perf_cores=perf, platform=system, has_metal=True)
    total = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    logical = os.cpu_count() or 1
    return HardwareInfo(total_ram_bytes=total, logical_cores=logical,
                        perf_cores=logical, platform=system, has_metal=False)
