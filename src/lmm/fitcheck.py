"""Graduated, actionable memory fit assessment. Never hard-blocks."""

from __future__ import annotations

from dataclasses import dataclass

_TIGHT_FRACTION = 0.8  # comfortable below this share of the budget


def _gib(n: int) -> str:
    return f"{n / 1024**3:.1f} GiB"


@dataclass
class FitResult:
    level: str            # "comfortable" | "tight" | "wont_load"
    fits: bool            # False only for wont_load
    message: str


def assess_fit(total_bytes: int, weights_bytes: int, usable_ram_bytes: int) -> FitResult:
    """Graduated: weights alone over budget OR total over budget -> wont_load;
    total over 80% of budget -> tight; otherwise comfortable. Messages name the
    culprit and stay human-readable.
    """
    budget = usable_ram_bytes
    if weights_bytes > budget:
        return FitResult("wont_load", False,
                         f"Weights alone need {_gib(weights_bytes)} but only "
                         f"{_gib(budget)} is usable — this model will not load.")
    if total_bytes > budget:
        return FitResult("wont_load", False,
                         f"Needs ~{_gib(total_bytes)} but only {_gib(budget)} is "
                         f"usable. Reduce context (most of the gap is KV-cache) "
                         f"or choose a smaller quant.")
    if total_bytes > budget * _TIGHT_FRACTION:
        return FitResult("tight", True,
                         f"Tight: ~{_gib(total_bytes)} of {_gib(budget)} usable — "
                         f"may swap or slow down under load.")
    return FitResult("comfortable", True,
                     f"Fits comfortably: ~{_gib(total_bytes)} of {_gib(budget)} usable.")
