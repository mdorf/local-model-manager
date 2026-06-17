"""Recommend a llama-server launch configuration for a model + this hardware."""

from __future__ import annotations

from dataclasses import dataclass, field

from lmm.fitcheck import FitResult, assess_fit
from lmm.hardware import HardwareInfo
from lmm.memory import MemoryEstimate, estimate_memory, weights_bytes
from lmm.models import Model

_CONTEXT_LADDER = [131072, 65536, 32768, 16384, 8192]
_DEFAULT_CACHE = "q8_0"
# Budget the context's footprint (weights+KV+overhead) against the GPU working
# set when we know it: on unified-memory Macs that's the real cliff — once
# weights+KV exceed what the GPU can hold resident, decode slows ~2x. Stay just
# under it (safety margin leaves room for compute scratch). When the GPU limit
# is unknown (non-Metal/undetectable), fall back to a fraction of total RAM.
_GPU_BUDGET_SAFETY = 0.90
_CONTEXT_RAM_FRACTION = 0.70


def context_budget_bytes(hardware: HardwareInfo) -> int:
    """Memory budget for the context's footprint. Prefer the GPU working-set
    limit (the real performance cliff on unified memory); else a RAM fraction."""
    if hardware.gpu_working_set_bytes > 0:
        return int(hardware.gpu_working_set_bytes * _GPU_BUDGET_SAFETY)
    return int(hardware.total_ram_bytes * _CONTEXT_RAM_FRACTION)


def choose_context(arch: str, metadata: dict, weights: int, model_max: int,
                   usable_ram_bytes: int, cache_type: str) -> int:
    """Largest context (capped at model_max) whose total estimate fits the budget.
    Falls back to the smallest ladder value if none fit (fit-check then warns).
    """
    candidates = sorted({c for c in [model_max, *_CONTEXT_LADDER] if c <= model_max},
                        reverse=True)
    for ctx in candidates:
        est = estimate_memory(arch, metadata, [], ctx, cache_type)
        if weights + est.kv_cache_bytes + est.overhead_bytes <= usable_ram_bytes:
            return ctx
    return candidates[-1] if candidates else 8192


@dataclass
class LaunchConfig:
    model_path: str
    context: int
    cache_type: str
    flags: list[str]
    estimate: MemoryEstimate
    fit: FitResult
    warnings: list[str] = field(default_factory=list)


def recommend_config(model: Model, metadata: dict, hardware: HardwareInfo, *,
                     supported: set[str] | None = None,
                     cache_type: str = _DEFAULT_CACHE,
                     host: str = "127.0.0.1", port: int = 8080,
                     alias: str | None = None,
                     api_key: str | None = None) -> LaunchConfig:
    weights = weights_bytes(model.shards)
    model_max = model.context_length or 8192
    # Select context against the GPU working-set budget (the real cliff); classify
    # fit against the (more permissive) usable-RAM figure so it reads as comfortable.
    context_budget = context_budget_bytes(hardware)
    context = choose_context(model.arch, metadata, weights, model_max,
                             context_budget, cache_type)
    estimate = estimate_memory(model.arch, metadata, model.shards, context, cache_type)
    fit = assess_fit(estimate.total_bytes, weights, hardware.usable_ram_bytes)

    groups: list[list[str]] = [["-m", str(model.path)]]
    if hardware.has_metal:
        groups.append(["-ngl", "999"])
    groups.append(["-fa", "on"])
    groups.append(["--cache-type-k", cache_type])
    groups.append(["--cache-type-v", cache_type])
    if api_key:
        groups.append(["--api-key", api_key])
    if model.has_mtp:
        groups.append(["--spec-type", "draft-mtp"])
        groups.append(["--spec-draft-n-max", "2"])
    groups.append(["-t", str(hardware.perf_cores)])
    groups.append(["-c", str(context)])
    groups.append(["--host", host])
    groups.append(["--port", str(port)])
    if alias:
        groups.append(["--alias", alias])

    flags: list[str] = []
    warnings: list[str] = []
    for group in groups:
        flag = group[0]
        if supported is not None and flag not in supported:
            warnings.append(f"dropped unsupported flag {flag} (not in installed "
                            f"llama-server)")
            continue
        flags.extend(group)
    return LaunchConfig(model_path=str(model.path), context=context,
                        cache_type=cache_type, flags=flags, estimate=estimate,
                        fit=fit, warnings=warnings)
