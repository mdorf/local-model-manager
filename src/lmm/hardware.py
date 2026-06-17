"""Detect the host's memory, CPU, and accelerator characteristics."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

USABLE_FRACTION = 0.85  # leave headroom for OS/other apps; live free-RAM is deferred
# macOS default Metal working set when iogpu.wired_limit_mb is unset (0): the GPU
# can hold ~75% of unified RAM resident. Beyond it, weights+KV spill off-GPU and
# decode slows hard — so it's the real ceiling for context sizing, not total RAM.
_DEFAULT_METAL_WORKING_SET_FRACTION = 0.75


@dataclass
class HardwareInfo:
    total_ram_bytes: int
    logical_cores: int
    perf_cores: int
    platform: str
    has_metal: bool
    gpu_working_set_bytes: int = 0  # 0 = unknown (non-Metal/undetectable) → RAM-fraction fallback
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


def _metal_working_set_bytes(total_ram_bytes: int) -> int:
    """Bytes the Metal GPU can hold resident. `iogpu.wired_limit_mb` overrides it
    (settable via sysctl for power users); 0/unset → macOS default ~75% of RAM."""
    wired_mb = _sysctl_int("iogpu.wired_limit_mb") or 0
    if wired_mb > 0:
        return wired_mb * 1024 * 1024
    return int(total_ram_bytes * _DEFAULT_METAL_WORKING_SET_FRACTION)


def detect_hardware() -> HardwareInfo:
    system = platform.system()
    if system == "Darwin":
        total = _sysctl_int("hw.memsize") or 0
        logical = _sysctl_int("hw.logicalcpu") or os.cpu_count() or 1
        perf = _sysctl_int("hw.perflevel0.logicalcpu") or logical
        return HardwareInfo(total_ram_bytes=total, logical_cores=logical,
                            perf_cores=perf, platform=system, has_metal=True,
                            gpu_working_set_bytes=_metal_working_set_bytes(total))
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
