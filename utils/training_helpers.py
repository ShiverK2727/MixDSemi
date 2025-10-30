"""Auxiliary helpers for training scripts."""

from __future__ import annotations

from typing import Iterable

import torch

from . import ramps


def parse_alpha_schedule(raw_schedule: Iterable[float] | str) -> list[float]:
    """Parse user-provided alpha schedule into a list of floats."""

    if isinstance(raw_schedule, (tuple, list)):
        values = [float(v) for v in raw_schedule]
    else:
        parts = [segment.strip() for segment in str(raw_schedule).split(",") if segment.strip()]
        values = [float(part) for part in parts]

    if not values:
        raise ValueError("Alpha schedule must contain at least one float value")

    return values


def compute_piecewise_threshold(current_stage: int, num_stages: int, tau_min: float, tau_max: float) -> float:
    """Linearly interpolate threshold between ``tau_min`` and ``tau_max``."""

    if num_stages <= 1:
        return tau_max

    stage = max(0, min(int(current_stage), num_stages - 1))
    return tau_min + (stage / (num_stages - 1)) * (tau_max - tau_min)


def compute_consistency_weight(epoch: float, base_weight: float, rampup_length: float) -> float:
    """Return ramped-up consistency weight following Mean Teacher schedule."""

    if base_weight <= 0:
        return 0.0

    if rampup_length <= 0:
        return base_weight

    ramp = ramps.sigmoid_rampup(epoch, rampup_length)
    return float(base_weight * ramp)


def update_ema_variables(model: torch.nn.Module, ema_model: torch.nn.Module, alpha: float, global_step: int) -> None:
    """Update ``ema_model`` parameters using EMA of ``model`` parameters."""

    effective_alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(effective_alpha).add_(param.data, alpha=1 - effective_alpha)
